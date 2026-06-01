#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinocchio 6轴机械臂位置逆解 + 达妙电机封装控制版本

使用前请先处理 arm_6.py：
1. 注释掉 ArmManager.__init__() 末尾的 time.sleep(999999)
2. 调试阶段建议注释掉 self.arm_set_zero_position()
3. 调试阶段建议把 setup_motor(..., save_to_flash=True) 改为 False

本文件默认 DRY_RUN=True，只计算不发给真实电机。
确认模型、零位、方向、限位正确后，再改成 DRY_RUN=False。
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin

# 调用你已有的机械臂封装类
from arm_6 import ArmManager


# =========================
# 1. 用户需要根据实物修改的参数
# =========================

@dataclass
class ArmGeometry:
    """机械臂几何参数，单位：米。必须按实物测量后修改。"""
    base_z: float = 0.10       # joint1 到 joint2 的竖直高度
    L1: float = 0.25           # 大臂长度：joint2 -> joint3
    L2: float = 0.20           # 小臂长度：joint3 -> joint4
    Lw: float = 0.06           # 腕部长度：joint4 -> joint5/joint6 附近
    tool: float = 0.10         # joint6 -> 夹爪末端 TCP


# 关节限位，单位 rad。先保守，后续按实物机械限位修改。
JOINT_LIMITS = np.array([
    [-2.80,  2.80],   # joint1 底座
    [-1.50,  1.50],   # joint2 肩
    [-2.40,  2.40],   # joint3 肘
    [-2.40,  2.40],   # joint4 腕俯仰
    [-1.80,  1.80],   # joint5 腕摆动
    [-3.14,  3.14],   # joint6 末端旋转
], dtype=float)

# 电机角 = MOTOR_SIGN * Pinocchio关节角 + MOTOR_OFFSET
# 这两个数组必须通过实物标定。默认只是假设：数学正方向 = 电机正方向，零位一致。
MOTOR_SIGN = np.array([1, 1, 1, 1, 1, 1], dtype=float)
MOTOR_OFFSET = np.array([0, 0, 0, 0, 0, 0], dtype=float)

# 为了让冗余自由度不要乱摆，给一个默认参考姿态。
Q_REF = np.array([0.0, 0.3, -0.8, 0.5, 0.0, 0.0], dtype=float)


# =========================
# 2. 基础工具函数
# =========================

def T_xyz(x: float, y: float, z: float) -> pin.SE3:
    return pin.SE3(np.eye(3), np.array([x, y, z], dtype=float))


def wrap_to_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def clip_joint_limits(q: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, JOINT_LIMITS[:, 0]), JOINT_LIMITS[:, 1])


def joint_to_motor(q_joint: np.ndarray) -> np.ndarray:
    """把 Pinocchio 关节角转换为真实电机目标角。"""
    q_joint = np.asarray(q_joint, dtype=float).reshape(6)
    return MOTOR_SIGN * q_joint + MOTOR_OFFSET


def motor_to_joint(q_motor: np.ndarray) -> np.ndarray:
    """把真实电机读回角转换为 Pinocchio 关节角。"""
    q_motor = np.asarray(q_motor, dtype=float).reshape(6)
    return MOTOR_SIGN * (q_motor - MOTOR_OFFSET)


# =========================
# 3. Pinocchio 模型
# =========================

def create_arm_model(geom: ArmGeometry) -> pin.Model:
    """
    创建一个 6R 串联机械臂模型。

    默认零位定义：
    - joint1 绕 Z 轴转，控制底座水平旋转；
    - joint2 / joint3 / joint4 绕 Y 轴转，控制大臂、小臂、腕部俯仰；
    - joint5 绕 X 轴转；
    - joint6 绕 Z 轴转；
    - 连杆在零位时主要沿 +X 方向伸出。

    如果你的实物零位不是“向 +X 伸直”，需要改这里的关节轴和 T_xyz() 平移方向。
    """
    model = pin.Model()

    # joint1：底座，绕世界/基座 Z 轴旋转
    j1 = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "joint1")

    # joint2：肩关节，安装在底座上方 base_z，绕 Y 轴俯仰
    j2 = model.addJoint(j1, pin.JointModelRY(), T_xyz(0.0, 0.0, geom.base_z), "joint2")

    # joint3：肘关节，大臂长度 L1，绕 Y 轴
    j3 = model.addJoint(j2, pin.JointModelRY(), T_xyz(geom.L1, 0.0, 0.0), "joint3")

    # joint4：腕部俯仰，小臂长度 L2，绕 Y 轴
    j4 = model.addJoint(j3, pin.JointModelRY(), T_xyz(geom.L2, 0.0, 0.0), "joint4")

    # joint5：腕部侧摆，腕部前伸 Lw，绕 X 轴
    j5 = model.addJoint(j4, pin.JointModelRX(), T_xyz(geom.Lw, 0.0, 0.0), "joint5")

    # joint6：末端自转，绕 Z 轴
    j6 = model.addJoint(j5, pin.JointModelRZ(), pin.SE3.Identity(), "joint6")

    # TCP / tool frame：真正要到达的夹爪末端点
    model.addFrame(
        pin.Frame(
            "tool",
            j6,
            0,
            T_xyz(geom.tool, 0.0, 0.0),
            pin.FrameType.OP_FRAME,
        )
    )

    return model


def forward_tool_position(model: pin.Model, data, tool_id: int, q: np.ndarray) -> np.ndarray:
    """正运动学：返回 tool 末端在世界坐标系的位置。

    注意：这里不再每次 createData()，因为部分 Windows/Pinocchio 环境在循环 createData/forwardKinematics 时会底层崩溃。
    """
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.array(data.oMf[tool_id].translation).reshape(3)


# =========================
# 4. 位置逆解：只约束 x,y,z
# =========================

def numeric_position_jacobian(model: pin.Model, data, tool_id: int, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """
    用有限差分计算 tool 位置雅可比，避免调用 Pinocchio 的 computeFrameJacobian。
    这样在 Windows 上更稳，虽然速度比解析雅可比慢一点，但 6 轴调试完全够用。
    """
    q = np.asarray(q, dtype=float).reshape(model.nq)
    p0 = forward_tool_position(model, data, tool_id, q)
    J = np.zeros((3, model.nv), dtype=float)

    for j in range(model.nv):
        q2 = q.copy()
        q2[j] += delta
        q2[j] = wrap_to_pi(q2[j])
        q2 = clip_joint_limits(q2)
        p1 = forward_tool_position(model, data, tool_id, q2)
        J[:, j] = (p1 - p0) / delta

    return J


def solve_ik_position(
    model: pin.Model,
    target_xyz: Tuple[float, float, float],
    q_init: Optional[np.ndarray] = None,
    q_ref: Optional[np.ndarray] = None,
    eps: float = 1e-4,
    max_iter: int = 300,
    dt: float = 0.45,
    damp: float = 1e-4,
    null_gain: float = 0.02,
    max_step: float = 0.06,
) -> Tuple[bool, np.ndarray, float, int]:
    """
    安全版位置 IK：
    - 只约束 TCP 的 x, y, z；
    - 不调用 pin.computeFrameJacobian；
    - 不调用 pin.integrate；
    - 直接用 q = q + dq 更新，因为当前模型 6 个关节都是一维转动关节。

    这个版本用于解决 Windows/PyCharm 下 Pinocchio 原生函数异常退出的问题。
    """
    target = np.array(target_xyz, dtype=float).reshape(3)
    data = model.createData()
    tool_id = model.getFrameId("tool")

    if q_init is None:
        q = pin.neutral(model).copy()
    else:
        q = np.array(q_init, dtype=float).reshape(model.nq).copy()

    if q_ref is None:
        q_ref = Q_REF.copy()
    else:
        q_ref = np.array(q_ref, dtype=float).reshape(model.nq)

    q = clip_joint_limits(q)
    q_ref = clip_joint_limits(q_ref)

    last_err_norm = float("inf")

    for i in range(max_iter):
        p = forward_tool_position(model, data, tool_id, q)
        err = target - p
        err_norm = float(norm(err))
        last_err_norm = err_norm

        if i % 20 == 0:
            print(f"IK迭代 {i:03d}: err={err_norm:.6f} m, p={np.round(p, 4).tolist()}")

        if err_norm < eps:
            return True, q, err_norm, i

        # 数值雅可比，避免 Pinocchio 原生雅可比接口崩溃
        J = numeric_position_jacobian(model, data, tool_id, q)

        # 阻尼最小二乘：dq = J.T (J J.T + λI)^-1 err
        A = J @ J.T + damp * np.eye(3)
        dq_task = J.T @ solve(A, err)

        # 零空间项：让关节尽量靠近参考姿态，避免腕部乱转
        pinvJ = J.T @ solve(A, np.eye(3))
        N = np.eye(model.nv) - pinvJ @ J
        dq_null = null_gain * (N @ (q_ref - q))

        dq = dq_task + dq_null

        # 单步限制，避免跳变太大
        step_norm = float(norm(dq, ord=np.inf))
        if step_norm > max_step:
            dq *= max_step / step_norm

        # 当前模型全部是一维 Revolute Joint，直接相加即可
        q = q + dq * dt
        q = np.array([wrap_to_pi(a) for a in q], dtype=float)
        q = clip_joint_limits(q)

    return False, q, last_err_norm, max_iter

# =========================
# 5. 调用你封装好的 ArmManager 控制真实电机
# =========================

def read_current_joint_position(arm: ArmManager) -> np.ndarray:
    """读取真实电机位置，并转换成 Pinocchio 关节角。"""
    motor_pos = np.array(arm.get_arm_position(1), dtype=float)
    return motor_to_joint(motor_pos)


def send_joint_position_smooth(
    arm: ArmManager,
    q_start: np.ndarray,
    q_goal: np.ndarray,
    velocity: float = 0.25,
    duration: float = 2.0,
    rate_hz: float = 30.0,
) -> None:
    """
    使用 arm.arm_move() 做简单线性插补，避免一次性跳到目标角。
    """
    q_start = np.asarray(q_start, dtype=float).reshape(6)
    q_goal = np.asarray(q_goal, dtype=float).reshape(6)

    steps = max(2, int(duration * rate_hz))
    dt = 1.0 / rate_hz

    for k in range(steps + 1):
        s = k / steps
        # 三次平滑插值，起末速度更小
        s = 3 * s * s - 2 * s * s * s
        q = (1.0 - s) * q_start + s * q_goal
        motor_cmd = joint_to_motor(q)

        arm.arm_move(
            float(motor_cmd[0]),
            float(motor_cmd[1]),
            float(motor_cmd[2]),
            float(motor_cmd[3]),
            float(motor_cmd[4]),
            float(motor_cmd[5]),
            velocity=velocity,
        )
        time.sleep(dt)


# =========================
# 6. 主程序
# =========================

def main():
    # 第一次调试务必保持 True：只计算、不发电机。
    DRY_RUN = True

    geom = ArmGeometry(
        base_z=0.10,
        L1=0.25,
        L2=0.20,
        Lw=0.06,
        tool=0.10,
    )
    model = create_arm_model(geom)

    print("Pinocchio 模型创建完成")
    print(f"nq={model.nq}, nv={model.nv}, frames={len(model.frames)}")

    arm = None
    if not DRY_RUN:
        print("正在初始化真实机械臂 ArmManager...")
        arm = ArmManager()
        q_now = read_current_joint_position(arm)
    else:
        q_now = Q_REF.copy()

    try:
        while True:
            print("\n请输入目标 TCP 坐标，单位 m。示例：0.25 0.00 0.20")
            text = input("target x y z > ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                break

            try:
                x, y, z = [float(v) for v in text.split()]
            except Exception:
                print("输入格式错误，请输入三个数字，例如：0.25 0.00 0.20")
                continue

            print("开始IK计算...")
            success, q_sol, err, iters = solve_ik_position(
                model,
                target_xyz=(x, y, z),
                q_init=q_now,
                q_ref=Q_REF,
            )

            print("IK 结果：", "成功" if success else "未完全收敛")
            print(f"迭代次数: {iters}, 末端误差: {err:.6f} m")
            print("Pinocchio关节角 rad:", np.round(q_sol, 4).tolist())
            print("电机目标角 rad:", np.round(joint_to_motor(q_sol), 4).tolist())
            data_fk = model.createData()
            tool_id_fk = model.getFrameId("tool")
            print("正解TCP位置:", np.round(forward_tool_position(model, data_fk, tool_id_fk, q_sol), 4).tolist())

            if not success:
                print("目标可能超出工作空间，或模型/零位/方向还没标定正确。不会发送电机指令。")
                continue

            if DRY_RUN:
                print("DRY_RUN=True：本次只计算，不发送真实电机。")
                q_now = q_sol
                continue

            assert arm is not None
            print("准备发送到真实机械臂。")
            confirm = input("确认发送？输入 y 继续，其它键取消 > ").strip().lower()
            if confirm != "y":
                print("已取消发送。")
                continue

            q_start = read_current_joint_position(arm)
            send_joint_position_smooth(
                arm,
                q_start=q_start,
                q_goal=q_sol,
                velocity=0.25,
                duration=2.5,
                rate_hz=30.0,
            )
            q_now = q_sol
            print("发送完成。")

    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        if arm is not None:
            print("正在失能机械臂。")
            arm.arm_disable()


if __name__ == "__main__":
    main()
