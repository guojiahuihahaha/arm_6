# -*- coding: utf-8 -*-
"""
给定 TCP 坐标控制 6 轴达妙机械臂运动。

算法：自写正运动学 + 几何雅可比 + 阻尼最小二乘（DLS）。
驱动：复用现有 DM_Motor_Device.MultiMotorController 和 DM_CAN。

直接运行程序即可。程序会先询问运行模式，再提示输入目标坐标：
    python coordinate_move_dls.py

运行模式由文件顶部 REAL_CONTROL 控制：
    REAL_CONTROL = None   # 每次启动时询问：只计算 / 真实控制
    REAL_CONTROL = False  # 始终只计算，不连接电机
    REAL_CONTROL = True   # 始终连接并控制真实电机

重要安全说明：
1. 0x02 已知：竖直向上为 0 rad，范围 [-0.70, 1.57] rad；
2. 其余关节限位、零位、正方向尚未标定，默认禁止真实运动；
3. 首次真实运行请悬空机械臂、降低速度，并随时准备断电；
4. 本程序退出时只失能电机，不自动回零，也不会重新设置零位。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence

# 限制 NumPy/BLAS 线程，减少 CPU 占用并避免某些 Windows 环境出现长时间等待。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from DM_CAN import Control_Type, DM_Motor_Type
from lightweight_dls_ik import LightweightArmIK, IKResult, transform


# ============================================================
# 1. 硬件与机械臂参数：必须根据实物继续测量、修改
# ============================================================
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
BAUDRATE = 921600

# ============================================================
# 最直观的运行模式开关：通常只需要修改这里
# ============================================================
# None  ：每次启动程序时询问“只计算”还是“真实控制”（推荐）
# False ：始终只计算目标角度，不连接、不控制真实电机
# True  ：始终连接 /dev/ttyACM0 并允许真实运动
REAL_CONTROL: Optional[bool] = None

# 当前除 0x02 外，其余关节限位尚未测量。
# True：临时允许真实控制；False：存在未测量限位时禁止真实控制。
# 等所有关节限位测量完成后，建议改成 False。
ALLOW_UNMEASURED_LIMITS_FOR_REAL = True

# True：逆解完成后，必须输入 MOVE 才实际运动（推荐保留 True）
REQUIRE_MOVE_CONFIRMATION = True

MOTOR_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]  # 从底部向上
MASTER_IDS = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16]
MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
MOTOR_TYPE = DM_Motor_Type.DM4340
CONTROL_MODE = Control_Type.POS_VEL

# 坐标系：目标 x/y/z 均表达在机械臂基坐标系中，单位 m。
# BASE_TO_SHOULDER_Z：基坐标原点到 0x02 肩关节轴线的竖直距离。
# 若你把肩关节轴线交点直接作为坐标原点，这里改为 0.0。
BASE_TO_SHOULDER_Z = 0.10
L1 = 0.25          # 0x02 肩关节轴 -> 0x03 肘关节轴，大臂长度
L2 = 0.20          # 0x03 肘关节轴 -> 0x04 手腕关节轴，小臂长度
TOOL_LENGTH = 0.12 # 最后关节 -> 夹爪 TCP

# 当前假设的关节旋转轴。必须与实物逐轴核对。
# 0x01: RZ，0x02~0x04: RY，0x05: RX，0x06: RZ
JOINT_AXES_LOCAL = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)

# 电机角度和数学模型角度之间的关系：
# q_motor = MOTOR_ZERO_OFFSET + MOTOR_DIRECTION * q_model
# 正方向相反的关节，将对应 MOTOR_DIRECTION 改成 -1。
# 电机零位与模型零位不一致时，填写对应 MOTOR_ZERO_OFFSET。
MOTOR_DIRECTION = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
MOTOR_ZERO_OFFSET = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

# 电机坐标系下的真实机械限位。None 表示尚未测量。
# 已确认：0x02 以竖直向上为 0，范围为 [-0.70, 1.57] rad。
MEASURED_MOTOR_LIMITS: list[Optional[tuple[float, float]]] = [
    None,
    (-0.70, 1.57),
    None,
    None,
    None,
    None,
]

# 未测量关节仅用于离线求解的占位范围，不代表真实机械安全范围。
UNMEASURED_FALLBACK_LIMIT = (-math.pi, math.pi)

# 真实运动参数
COMMAND_RATE_HZ = 40.0
DEFAULT_JOINT_SPEED = 0.15       # rad/s，首次调试保持较低
MIN_MOVE_DURATION = 1.0         # s
MAX_MOVE_DURATION = 20.0        # s
POSITION_TOLERANCE = 0.003      # m
MAX_IK_ITERATIONS = 150


# ============================================================
# 2. 模型角度 / 电机角度映射
# ============================================================
def model_to_motor(q_model: Sequence[float]) -> np.ndarray:
    q_model = np.asarray(q_model, dtype=float)
    return MOTOR_ZERO_OFFSET + MOTOR_DIRECTION * q_model


def motor_to_model(q_motor: Sequence[float]) -> np.ndarray:
    q_motor = np.asarray(q_motor, dtype=float)
    return MOTOR_DIRECTION * (q_motor - MOTOR_ZERO_OFFSET)


def effective_motor_limits() -> tuple[np.ndarray, np.ndarray]:
    """返回求解时使用的电机角度上下限；未测量关节使用占位范围。"""
    q_min = np.empty(6, dtype=float)
    q_max = np.empty(6, dtype=float)
    for i, limit in enumerate(MEASURED_MOTOR_LIMITS):
        lo, hi = UNMEASURED_FALLBACK_LIMIT if limit is None else limit
        q_min[i] = lo
        q_max[i] = hi
    return q_min, q_max


def model_limits_from_motor_limits() -> tuple[np.ndarray, np.ndarray]:
    """把电机坐标系下的限位转换到数学模型坐标系。"""
    motor_min, motor_max = effective_motor_limits()
    model_a = motor_to_model(motor_min)
    model_b = motor_to_model(motor_max)
    return np.minimum(model_a, model_b), np.maximum(model_a, model_b)


def validate_motor_target(q_motor: Sequence[float], allow_unmeasured_limits: bool) -> None:
    q_motor = np.asarray(q_motor, dtype=float)
    if q_motor.shape != (6,):
        raise ValueError("目标关节角必须包含 6 个元素")
    if not np.all(np.isfinite(q_motor)):
        raise ValueError("目标关节角包含 NaN 或无穷大")

    for i, value in enumerate(q_motor):
        measured = MEASURED_MOTOR_LIMITS[i]
        if measured is None:
            if not allow_unmeasured_limits:
                raise RuntimeError(
                    f"电机 0x{MOTOR_IDS[i]:02X} 的机械限位尚未测量，已阻止真实运动。"
                )
            lo, hi = UNMEASURED_FALLBACK_LIMIT
        else:
            lo, hi = measured

        if not lo <= value <= hi:
            raise ValueError(
                f"电机 0x{MOTOR_IDS[i]:02X} 目标 {value:.4f} rad 超出范围 "
                f"[{lo:.4f}, {hi:.4f}] rad"
            )


# ============================================================
# 3. 构建轻量级机械臂模型
# ============================================================
def build_arm_model() -> LightweightArmIK:
    model_q_min, model_q_max = model_limits_from_motor_limits()

    # 每个关节旋转后，到下一关节轴的固定平移。
    link_vectors = np.array(
        [
            [0.0, 0.0, BASE_TO_SHOULDER_Z],
            [0.0, 0.0, L1],
            [0.0, 0.0, L2],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    tool_T = transform(translation=[0.0, 0.0, TOOL_LENGTH])

    return LightweightArmIK(
        joint_axes_local=JOINT_AXES_LOCAL,
        link_vectors=link_vectors,
        q_min=model_q_min,
        q_max=model_q_max,
        tool_T=tool_T,
    )


def analytic_seed(target_xyz: Sequence[float], current_q_model: Sequence[float]) -> np.ndarray:
    """
    使用简化两连杆几何法生成 DLS 初值。

    只用于帮助数值迭代脱离全零奇异位形；最终结果仍由 DLS 求解。
    假设 0x02 的 0 rad 为竖直向上。
    """
    x, y, z = np.asarray(target_xyz, dtype=float)
    q = np.asarray(current_q_model, dtype=float).copy()

    r = math.hypot(x, y)
    z_rel = z - BASE_TO_SHOULDER_Z
    l2_effective = L2 + TOOL_LENGTH
    d2 = r * r + z_rel * z_rel

    q[0] = math.atan2(y, x)

    cos_q3 = (d2 - L1 * L1 - l2_effective * l2_effective) / (2.0 * L1 * l2_effective)
    cos_q3 = float(np.clip(cos_q3, -1.0, 1.0))

    # 优先采用肘部弯曲的解，另一组解由 DLS 和当前姿态进一步修正。
    q3 = -math.acos(cos_q3)
    target_angle_from_vertical = math.atan2(r, z_rel)
    correction = math.atan2(
        l2_effective * math.sin(q3),
        L1 + l2_effective * math.cos(q3),
    )
    q2 = target_angle_from_vertical - correction

    q[1] = q2
    q[2] = q3
    q[3] = 0.0

    model_q_min, model_q_max = model_limits_from_motor_limits()
    return np.clip(q, model_q_min, model_q_max)


# ============================================================
# 4. 达妙电机硬件适配层
# ============================================================
class DMArmHardware:
    def __init__(
        self,
        *,
        serial_port: str,
        dry_run: bool,
        allow_unmeasured_limits: bool,
    ) -> None:
        self.dry_run = dry_run
        self.allow_unmeasured_limits = allow_unmeasured_limits
        self.controller: Optional[object] = None
        self._dry_q_motor = np.zeros(6, dtype=float)

        if self.dry_run:
            print("[DRY_RUN] 不连接电机，只计算目标关节角。")
            return

        if not self.allow_unmeasured_limits:
            unknown_ids = [
                f"0x{MOTOR_IDS[i]:02X}"
                for i, limit in enumerate(MEASURED_MOTOR_LIMITS)
                if limit is None
            ]
            raise RuntimeError(
                "仍有未测量机械限位的关节："
                + ", ".join(unknown_ids)
                + "。真实运行必须补全限位，或显式使用 --allow-unmeasured-limits。"
            )

        # 延迟导入：离线求解不要求安装 pyserial，也不会打开串口。
        from DM_Motor_Device import MultiMotorController

        self.controller = MultiMotorController(serial_port, BAUDRATE)

        for motor_id, master_id, name in zip(MOTOR_IDS, MASTER_IDS, MOTOR_NAMES):
            self.controller.add_motor(MOTOR_TYPE, motor_id, master_id, name)

        # 不写入闪存，不重新设零位。
        for motor_id in MOTOR_IDS:
            ok = self.controller.setup_motor(motor_id, CONTROL_MODE, save_to_flash=False)
            if not ok:
                raise RuntimeError(f"电机 0x{motor_id:02X} 切换 POS_VEL 模式失败")

        self.controller.enable_all_motors(delay_seconds=0.3)
        time.sleep(0.5)
        print("达妙机械臂已连接并使能；程序没有重新设置任何电机零位。")

    def read_motor_positions(self) -> np.ndarray:
        if self.dry_run:
            return self._dry_q_motor.copy()

        assert self.controller is not None
        q = np.zeros(6, dtype=float)

        for index, motor_id in enumerate(MOTOR_IDS):
            status = None
            # 多读几次，降低串口一发一收时读取缓存旧值的概率。
            for _ in range(3):
                status = self.controller.get_motor_status(motor_id, auto_refresh=True)
                time.sleep(0.015)
            if status is None:
                raise RuntimeError(f"无法读取电机 0x{motor_id:02X} 状态")
            q[index] = float(status["position"])

        return q

    def send_motor_positions(self, q_motor: Sequence[float], velocity: float) -> None:
        q_motor = np.asarray(q_motor, dtype=float)
        validate_motor_target(q_motor, self.allow_unmeasured_limits)

        if self.dry_run:
            self._dry_q_motor = q_motor.copy()
            return

        assert self.controller is not None
        for motor_id, position in zip(MOTOR_IDS, q_motor):
            self.controller.control_pos_vel(motor_id, float(position), float(velocity))

    def move_smooth(self, q_motor_target: Sequence[float], joint_speed: float) -> None:
        q_start = self.read_motor_positions()
        q_target = np.asarray(q_motor_target, dtype=float)
        validate_motor_target(q_target, self.allow_unmeasured_limits)

        max_delta = float(np.max(np.abs(q_target - q_start)))
        duration = max(MIN_MOVE_DURATION, max_delta / max(joint_speed, 1e-3))
        duration = min(duration, MAX_MOVE_DURATION)
        steps = max(2, int(math.ceil(duration * COMMAND_RATE_HZ)))

        print(f"规划关节插值：持续 {duration:.2f} s，共 {steps} 步")

        start_time = time.perf_counter()
        for step in range(1, steps + 1):
            u = step / steps
            # 三次平滑步进：起点和终点速度为零，降低机械冲击。
            s = 3.0 * u * u - 2.0 * u * u * u
            q_cmd = q_start + s * (q_target - q_start)
            self.send_motor_positions(q_cmd, joint_speed)

            expected_time = start_time + step / COMMAND_RATE_HZ
            sleep_time = expected_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        self.send_motor_positions(q_target, joint_speed)

    def shutdown(self) -> None:
        if self.dry_run:
            return

        if self.controller is not None:
            try:
                # 退出时只失能，不调用 MultiMotorController.close()，避免其先命令所有轴回 0。
                self.controller.disable_all_motors()
            finally:
                serial_device = self.controller.serial_device
                if serial_device is not None and serial_device.is_open:
                    serial_device.close()
                print("所有电机已失能，串口已关闭。")


# ============================================================
# 5. 坐标运动控制器
# ============================================================
@dataclass
class CoordinateMoveResult:
    ik: IKResult
    q_motor_start: np.ndarray
    q_motor_target: np.ndarray


class CoordinateArmController:
    def __init__(
        self,
        *,
        serial_port: str,
        dry_run: bool,
        allow_unmeasured_limits: bool,
    ) -> None:
        self.model = build_arm_model()
        self.hardware = DMArmHardware(
            serial_port=serial_port,
            dry_run=dry_run,
            allow_unmeasured_limits=allow_unmeasured_limits,
        )
        self.dry_run = dry_run
        self.allow_unmeasured_limits = allow_unmeasured_limits

    def solve_xyz(self, target_xyz: Sequence[float], tool_down: bool) -> CoordinateMoveResult:
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("目标坐标必须为三个有限数字：x y z")

        shoulder = np.array([0.0, 0.0, BASE_TO_SHOULDER_Z])
        approximate_max_reach = L1 + L2 + TOOL_LENGTH
        if np.linalg.norm(target - shoulder) > approximate_max_reach + 0.03:
            raise ValueError(
                f"目标距离肩关节过远，超过近似最大臂展 {approximate_max_reach:.3f} m"
            )

        print("\n[1/4] 正在读取当前关节角...", flush=True)
        q_motor_start = self.hardware.read_motor_positions()
        q_model_start = motor_to_model(q_motor_start)
        print("当前电机角度(rad)：", np.round(q_motor_start, 5), flush=True)

        seed_preview = analytic_seed(target, q_model_start)
        print("解析几何初值(rad)：", np.round(seed_preview, 5), flush=True)

        mode = "position_tool_axis" if tool_down else "position"
        kwargs = {"target_tool_axis": [0.0, 0.0, -1.0]} if tool_down else {}

        print("[2/4] 开始第一次 DLS 逆运动学求解...", flush=True)
        solve_start = time.perf_counter()
        result = self.model.solve(
            target_position=target,
            q0=q_model_start,
            mode=mode,
            max_iterations=MAX_IK_ITERATIONS,
            position_tolerance=POSITION_TOLERANCE,
            orientation_tolerance=math.radians(3.0),
            damping=0.04,
            max_step=0.06,
            verbose=True,
            progress_interval=10,
            **kwargs,
        )
        print(
            f"第一次 DLS 完成：success={result.success}，耗时={time.perf_counter() - solve_start:.3f} s，"
            f"位置误差={result.position_error * 1000.0:.3f} mm",
            flush=True,
        )

        # 当前姿态求解失败时，用解析几何初值重试，避免全零或伸直姿态的奇异问题。
        if not result.success:
            print("[3/4] 第一次未收敛，使用解析几何初值重新求解...", flush=True)
            seed = seed_preview
            retry_start = time.perf_counter()
            result_retry = self.model.solve(
                target_position=target,
                q0=seed,
                mode=mode,
                max_iterations=MAX_IK_ITERATIONS,
                position_tolerance=POSITION_TOLERANCE,
                orientation_tolerance=math.radians(3.0),
                damping=0.04,
                max_step=0.06,
                verbose=True,
                progress_interval=10,
                **kwargs,
            )
            print(
                f"第二次 DLS 完成：success={result_retry.success}，耗时={time.perf_counter() - retry_start:.3f} s，"
                f"位置误差={result_retry.position_error * 1000.0:.3f} mm",
                flush=True,
            )
            if result_retry.success or result_retry.position_error < result.position_error:
                result = result_retry

        print("[4/4] 正在转换并检查电机目标角...", flush=True)
        q_motor_target = model_to_motor(result.q)
        validate_motor_target(q_motor_target, self.allow_unmeasured_limits or self.dry_run)
        print("目标电机角度(rad)：", np.round(q_motor_target, 5), flush=True)

        return CoordinateMoveResult(
            ik=result,
            q_motor_start=q_motor_start,
            q_motor_target=q_motor_target,
        )

    def move_to_xyz(
        self,
        target_xyz: Sequence[float],
        *,
        joint_speed: float,
        tool_down: bool,
        require_confirmation: bool,
    ) -> CoordinateMoveResult:
        result = self.solve_xyz(target_xyz, tool_down=tool_down)
        ik = result.ik

        print("\n========== 逆运动学结果 ==========")
        print(f"收敛状态: {ik.success}")
        print(f"原因: {ik.reason}")
        print(f"迭代次数: {ik.iterations}")
        print(f"位置误差: {ik.position_error * 1000.0:.2f} mm")
        if tool_down:
            print(f"工具轴方向误差: {math.degrees(ik.orientation_error):.2f} deg")

        for i, (start, target) in enumerate(zip(result.q_motor_start, result.q_motor_target)):
            limit_text = "未测量" if MEASURED_MOTOR_LIMITS[i] is None else str(MEASURED_MOTOR_LIMITS[i])
            print(
                f"0x{MOTOR_IDS[i]:02X}: {start:+.4f} -> {target:+.4f} rad, "
                f"限位={limit_text}"
            )

        if not ik.success:
            raise RuntimeError("逆运动学未收敛，已拒绝发送电机运动指令")

        if self.dry_run:
            print("\n[DRY_RUN] 计算完成，未向真实电机发送指令。")
            return result

        if require_confirmation:
            confirm = input("\n确认机械臂周围安全，输入 MOVE 执行运动：").strip()
            if confirm != "MOVE":
                raise RuntimeError("用户取消运动")

        self.hardware.move_smooth(result.q_motor_target, joint_speed=joint_speed)

        q_motor_final = self.hardware.read_motor_positions()
        q_model_final = motor_to_model(q_motor_final)
        T_final, _, _ = self.model.forward_kinematics(q_model_final)
        p_final = T_final[:3, 3]
        print(
            "运动完成；根据当前模型和电机反馈估算 TCP = "
            f"[{p_final[0]:.4f}, {p_final[1]:.4f}, {p_final[2]:.4f}] m"
        )
        return result

    def shutdown(self) -> None:
        self.hardware.shutdown()


# ============================================================
# 6. 命令行入口
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "给定 TCP 坐标控制达妙 6 轴机械臂。"
            "可以在命令后直接填写 x y z；不填写时，程序会提示输入目标坐标。"
        )
    )
    parser.add_argument("x", nargs="?", type=float, help="目标 X，单位 m")
    parser.add_argument("y", nargs="?", type=float, help="目标 Y，单位 m")
    parser.add_argument("z", nargs="?", type=float, help="目标 Z，单位 m")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT, help="串口设备")
    parser.add_argument(
        "--tool-down",
        action="store_true",
        help="除位置外，同时要求夹爪工具 Z 轴竖直向下",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_JOINT_SPEED,
        help=f"关节运动速度 rad/s，默认 {DEFAULT_JOINT_SPEED}",
    )
    return parser.parse_args()



def choose_real_control() -> bool:
    """
    根据 REAL_CONTROL 决定是否控制真实电机。

    返回：
        True  -> 连接并控制真实电机
        False -> 只计算，不连接电机
    """
    if REAL_CONTROL is True:
        print("\n当前运行模式：真实控制（REAL_CONTROL=True）")
        print(f"将连接串口：{DEFAULT_SERIAL_PORT}")
        return True

    if REAL_CONTROL is False:
        print("\n当前运行模式：只计算（REAL_CONTROL=False）")
        return False

    while True:
        print("\n========== 运行模式选择 ==========")
        print("1：只计算目标关节角，不连接电机")
        print(f"2：连接真实电机并运动（串口 {DEFAULT_SERIAL_PORT}）")
        print("q：退出程序")
        print("==================================")
        choice = input("请选择运行模式 [1/2/q] > ").strip().lower()

        if choice in {"1", "n", "no", "false", "计算", "只计算"}:
            print("已选择：只计算，不连接真实电机。")
            return False
        if choice in {"2", "y", "yes", "true", "真实", "真实控制"}:
            print("已选择：真实控制。")
            return True
        if choice in {"q", "quit", "exit"}:
            raise KeyboardInterrupt

        print("输入无效，请输入 1、2 或 q。")


def confirm_real_control_with_unmeasured_limits() -> None:
    """真实控制时，对尚未测量的关节限位再次进行显式确认。"""
    unknown_ids = [
        f"0x{MOTOR_IDS[i]:02X}"
        for i, limit in enumerate(MEASURED_MOTOR_LIMITS)
        if limit is None
    ]
    if not unknown_ids:
        return

    if not ALLOW_UNMEASURED_LIMITS_FOR_REAL:
        raise RuntimeError(
            "仍有未测量机械限位的关节：" + ", ".join(unknown_ids)
            + "。请先完成限位测量，或将 ALLOW_UNMEASURED_LIMITS_FOR_REAL 改为 True。"
        )

    print("\n警告：以下电机的机械限位尚未测量：" + ", ".join(unknown_ids))
    print("当前允许这些关节真实运动，可能发生碰撞或超限。")
    confirm = input("确认机械臂已悬空且可随时断电，输入 RISK 继续 > ").strip()
    if confirm != "RISK":
        raise RuntimeError("未输入 RISK，已取消真实控制")

def prompt_target_xyz() -> list[float]:
    """
    当命令行没有提供 x y z 时，引导用户输入目标 TCP 坐标。

    坐标单位为米，坐标原点和正方向由机械臂模型定义。
    输入示例：0.25 0.00 0.20
    """
    print("\n========== 目标坐标输入指引 ==========")
    print("请输入夹爪 TCP 想要到达的目标坐标，单位：米（m）")
    print("X：机械臂前后方向")
    print("Y：机械臂左右方向")
    print("Z：机械臂竖直方向")
    print("输入格式：x y z")
    print("示例：0.25 0.00 0.20")
    print("输入 q 可退出程序")
    print("======================================")

    while True:
        raw = input("\n目标 x y z > ").strip()

        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt

        parts = raw.replace(",", " ").split()
        if len(parts) != 3:
            print("输入错误：必须输入 3 个数字，并使用空格分隔，例如：0.25 0.00 0.20")
            continue

        try:
            target = [float(value) for value in parts]
        except ValueError:
            print("输入错误：坐标必须是数字，例如：0.25 0.00 0.20")
            continue

        if not all(math.isfinite(value) for value in target):
            print("输入错误：坐标不能包含 NaN 或无穷大")
            continue

        print(
            f"已输入目标坐标：X={target[0]:.4f} m，"
            f"Y={target[1]:.4f} m，Z={target[2]:.4f} m"
        )
        return target


def main() -> int:
    args = parse_args()

    try:
        real_control = choose_real_control()
        if real_control:
            confirm_real_control_with_unmeasured_limits()
    except KeyboardInterrupt:
        print("\n用户取消运行模式选择，程序退出。")
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if args.speed <= 0.0 or args.speed > 0.5:
        print("错误：首次调试建议速度范围为 (0, 0.5] rad/s", file=sys.stderr)
        return 2

    supplied_coordinates = [args.x, args.y, args.z]
    if all(value is None for value in supplied_coordinates):
        try:
            target_xyz = prompt_target_xyz()
        except KeyboardInterrupt:
            print("\n用户取消输入，程序退出。")
            return 130
    elif any(value is None for value in supplied_coordinates):
        print(
            "错误：命令行坐标必须同时提供 x、y、z 三个值；"
            "也可以完全不填写坐标，运行后按提示输入。",
            file=sys.stderr,
        )
        return 2
    else:
        target_xyz = [float(args.x), float(args.y), float(args.z)]

    controller: Optional[CoordinateArmController] = None
    try:
        controller = CoordinateArmController(
            serial_port=args.port,
            dry_run=not real_control,
            allow_unmeasured_limits=(
                ALLOW_UNMEASURED_LIMITS_FOR_REAL if real_control else True
            ),
        )
        controller.move_to_xyz(
            target_xyz,
            joint_speed=args.speed,
            tool_down=args.tool_down,
            require_confirmation=(REQUIRE_MOVE_CONFIRMATION and real_control),
        )
        return 0
    except KeyboardInterrupt:
        print("\n用户中断，准备失能电机。")
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
