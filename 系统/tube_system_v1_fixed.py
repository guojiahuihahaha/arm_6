# -*- coding: utf-8 -*-
# tube_system_v1_fixed.py
# 试管抓取/清洗任务系统 v1.0
#
# 使用方法：
#   只计算，不连接电机：
#       python tube_system_v1_fixed.py --dry-run
#
#   真实控制：
#       python tube_system_v1_fixed.py --real --port /dev/ttyACM0
#
# 需要和 coordinate_move_dls.py、lightweight_dls_ik.py 放在同一个目录。

from __future__ import annotations

import argparse
import time
from typing import Protocol, Sequence

from coordinate_move_dls import CoordinateArmController, DEFAULT_SERIAL_PORT


# ============================================================
# 第 0 步：现在主要替换这里
# ============================================================

# 第 0.1 步：替换为真实试管抓取点坐标，单位 m
TUBE_PICK_XYZ = [0.25, 0.00, 0.12]

# 第 0.2 步：替换为真实清洗点坐标，单位 m
WASH_XYZ = [0.28, -0.10, 0.12]

# 第 0.3 步：目标点上方安全高度，单位 m
HOVER_DZ = 0.08

# 第 0.4 步：最低安全高度，防止撞桌面，单位 m
MIN_SAFE_Z = 0.03

# 第 0.5 步：临时工作空间限制，后面按实物范围修改
X_MIN, X_MAX = -0.45, 0.45
Y_MIN, Y_MAX = -0.45, 0.45

# 第 0.6 步：任务速度，首次真实运行建议 0.08~0.15 rad/s
TASK_SPEED = 0.12

# 第 0.7 步：清洗动作参数
WASH_UP_DZ = 0.04
WASH_CYCLES = 3
WASH_DWELL = 0.4

# 第 0.8 步：抓试管时建议工具轴向下
TOOL_DOWN = True

# 第 0.9 步：肘部配置，True=肘部向下；False=肘部向上；None=自动
ELBOW_DOWN = True


# ============================================================
# 第 1 步：夹爪统一接口
# ============================================================

class GripperInterface(Protocol):
    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def hold(self) -> None:
        ...


class VirtualGripper:
    # 当前机械爪还没设计好，先用虚拟夹爪跑通任务流程。

    def open(self) -> None:
        print("[GRIPPER] 虚拟夹爪：打开")
        time.sleep(0.5)

    def close(self) -> None:
        print("[GRIPPER] 虚拟夹爪：闭合")
        time.sleep(0.5)

    def hold(self) -> None:
        print("[GRIPPER] 虚拟夹爪：保持夹持")
        time.sleep(0.2)


# ============================================================
# 第 2 步：以后真实夹爪替换区
# ============================================================

class DMGripper:
    # 如果夹爪以后也是达妙电机控制，就用这个类。
    # 替换方式见 main() 里的“第 9.2 步”。

    def __init__(
        self,
        controller: CoordinateArmController,
        motor_id: int,
        open_pos: float = -1.20,
        close_pos: float = -0.30,
        speed: float = 0.10,
    ) -> None:
        self.controller = controller
        self.motor_id = motor_id
        self.open_pos = open_pos
        self.close_pos = close_pos
        self.speed = speed

    def _send(self, pos: float) -> None:
        if self.controller.dry_run:
            print(f"[GRIPPER-DRY] motor=0x{self.motor_id:02X}, pos={pos:.3f}")
            time.sleep(0.5)
            return

        dm = self.controller.hardware.controller
        if dm is None:
            raise RuntimeError("达妙控制器未初始化，无法控制夹爪")

        dm.control_pos_vel(self.motor_id, float(pos), float(self.speed))
        time.sleep(0.5)

    def open(self) -> None:
        print("[GRIPPER] 达妙夹爪：打开")
        self._send(self.open_pos)

    def close(self) -> None:
        print("[GRIPPER] 达妙夹爪：闭合")
        self._send(self.close_pos)

    def hold(self) -> None:
        print("[GRIPPER] 达妙夹爪：保持")
        time.sleep(0.2)


class ServoGripper:
    # 如果以后夹爪是 PWM 舵机，就改这个类里的 _send_pwm()。

    def __init__(self, open_angle: float = 90.0, close_angle: float = 30.0) -> None:
        self.open_angle = open_angle
        self.close_angle = close_angle

    def _send_pwm(self, angle: float) -> None:
        # 第 2B.1 步：以后在这里替换成真实 PWM 输出。
        # 例：pwm.set_servo_angle(channel=0, angle=angle)
        print(f"[SERVO-GRIPPER] 发送舵机角度：{angle:.1f} deg")
        time.sleep(0.5)

    def open(self) -> None:
        print("[GRIPPER] 舵机夹爪：打开")
        self._send_pwm(self.open_angle)

    def close(self) -> None:
        print("[GRIPPER] 舵机夹爪：闭合")
        self._send_pwm(self.close_angle)

    def hold(self) -> None:
        print("[GRIPPER] 舵机夹爪：保持")
        time.sleep(0.2)


class GPIOGripper:
    # 如果以后夹爪是气动夹爪/电磁阀，就改这个类里的 _set_gpio()。

    def _set_gpio(self, open_state: bool) -> None:
        # 第 2C.1 步：以后在这里替换成真实 GPIO 输出。
        # 例：GPIO.output(VALVE_PIN, GPIO.HIGH if open_state else GPIO.LOW)
        print(f"[GPIO-GRIPPER] 电磁阀状态：{'OPEN' if open_state else 'CLOSE'}")
        time.sleep(0.5)

    def open(self) -> None:
        print("[GRIPPER] 气动夹爪：打开")
        self._set_gpio(True)

    def close(self) -> None:
        print("[GRIPPER] 气动夹爪：闭合")
        self._set_gpio(False)

    def hold(self) -> None:
        print("[GRIPPER] 气动夹爪：保持")
        time.sleep(0.2)


# ============================================================
# 第 3 步：以后视觉替换区
# ============================================================

class FixedVision:
    # 当前没有相机，先返回固定试管坐标。

    def get_tube_position(self) -> list[float]:
        print("[VISION] 使用固定试管坐标，不使用摄像头")
        return list(TUBE_PICK_XYZ)


class CameraVisionPlaceholder:
    # 摄像头买好后，只需要实现 get_tube_position()，返回机械臂基坐标系下的 [x, y, z]。

    def get_tube_position(self) -> list[float]:
        raise NotImplementedError("摄像头视觉还没实现。当前请继续使用 FixedVision。")


# ============================================================
# 第 4 步：安全检查
# ============================================================

def check_xyz_safe(xyz: Sequence[float], name: str = "target") -> None:
    if len(xyz) != 3:
        raise ValueError(f"{name} 必须是 [x, y, z] 三个数")

    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])

    if z < MIN_SAFE_Z:
        raise ValueError(f"{name} 的 z={z:.3f} m 低于安全高度 {MIN_SAFE_Z:.3f} m，拒绝运动")

    if not (X_MIN <= x <= X_MAX):
        raise ValueError(f"{name} 的 x={x:.3f} m 超出临时安全范围 [{X_MIN}, {X_MAX}]")

    if not (Y_MIN <= y <= Y_MAX):
        raise ValueError(f"{name} 的 y={y:.3f} m 超出临时安全范围 [{Y_MIN}, {Y_MAX}]")


def above(xyz: Sequence[float], dz: float = HOVER_DZ) -> list[float]:
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]) + dz]


# ============================================================
# 第 5 步：机械臂基础动作封装
# ============================================================

def move_xyz(
    controller: CoordinateArmController,
    xyz: Sequence[float],
    *,
    speed: float = TASK_SPEED,
    tool_down: bool = TOOL_DOWN,
    elbow_down=ELBOW_DOWN,
) -> None:
    check_xyz_safe(xyz)
    print(f"\n[MOVE] x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f}")

    controller.move_to_xyz(
        xyz,
        joint_speed=speed,
        tool_down=tool_down,
        require_confirmation=False,
        elbow_down=elbow_down,
    )


def retreat_up(
    controller: CoordinateArmController,
    xyz: Sequence[float],
    *,
    speed: float = TASK_SPEED,
) -> None:
    move_xyz(controller, above(xyz), speed=speed)


# ============================================================
# 第 6 步：抓取试管流程
# ============================================================

def pick_tube(
    controller: CoordinateArmController,
    gripper: GripperInterface,
    pick_xyz: Sequence[float],
    *,
    speed: float = TASK_SPEED,
) -> None:
    check_xyz_safe(pick_xyz, "pick_xyz")

    print("\n========== 抓取试管 ==========")

    print("\n[Pick 1/5] 移动到试管上方")
    move_xyz(controller, above(pick_xyz), speed=speed)

    print("\n[Pick 2/5] 打开夹爪")
    gripper.open()

    print("\n[Pick 3/5] 下降到抓取点")
    move_xyz(controller, pick_xyz, speed=speed)

    print("\n[Pick 4/5] 闭合夹爪")
    gripper.close()
    gripper.hold()

    print("\n[Pick 5/5] 抬起试管")
    retreat_up(controller, pick_xyz, speed=speed)

    print("\n========== 抓取流程结束 ==========")


# ============================================================
# 第 7 步：清洗流程
# ============================================================

def wash_tube(
    controller: CoordinateArmController,
    wash_xyz: Sequence[float],
    *,
    speed: float = TASK_SPEED,
) -> None:
    check_xyz_safe(wash_xyz, "wash_xyz")

    print("\n========== 清洗试管 ==========")

    print("\n[Wash 1/4] 移动到清洗位置上方")
    move_xyz(controller, above(wash_xyz), speed=speed)

    print("\n[Wash 2/4] 下降到清洗位置")
    move_xyz(controller, wash_xyz, speed=speed)

    wash_up = [wash_xyz[0], wash_xyz[1], wash_xyz[2] + WASH_UP_DZ]

    print("\n[Wash 3/4] 执行上下往复清洗")
    for i in range(WASH_CYCLES):
        print(f"  清洗循环 {i + 1}/{WASH_CYCLES}")
        move_xyz(controller, wash_up, speed=speed)
        time.sleep(WASH_DWELL)
        move_xyz(controller, wash_xyz, speed=speed)
        time.sleep(WASH_DWELL)

    print("\n[Wash 4/4] 清洗结束，抬起")
    retreat_up(controller, wash_xyz, speed=speed)

    print("\n========== 清洗流程结束 ==========")


# ============================================================
# 第 8 步：完整任务流程
# ============================================================

def tube_pick_and_wash_task(
    controller: CoordinateArmController,
    gripper: GripperInterface,
    vision: FixedVision,
    wash_xyz: Sequence[float],
    *,
    speed: float = TASK_SPEED,
) -> None:
    print("\n############################################")
    print("        试管抓取 + 清洗任务开始")
    print("############################################")

    pick_xyz = vision.get_tube_position()

    print(f"\n试管抓取点 pick_xyz = {pick_xyz}")
    print(f"清洗点 wash_xyz = {wash_xyz}")

    pick_tube(controller=controller, gripper=gripper, pick_xyz=pick_xyz, speed=speed)
    wash_tube(controller=controller, wash_xyz=wash_xyz, speed=speed)

    print("\n############################################")
    print("        试管抓取 + 清洗任务完成")
    print("############################################")


# ============================================================
# 第 9 步：命令行入口
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="试管抓取/清洗任务系统 v1.0")

    parser.add_argument("--dry-run", action="store_true", help="只计算，不连接真实电机")
    parser.add_argument("--real", action="store_true", help="连接真实机械臂并执行运动")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT, help="串口设备，例如 /dev/ttyACM0")
    parser.add_argument("--speed", type=float, default=TASK_SPEED, help=f"任务速度 rad/s，默认 {TASK_SPEED}")
    parser.add_argument("--task", choices=["full", "pick", "wash"], default="full")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.real and args.dry_run:
        print("错误：--real 和 --dry-run 不能同时使用")
        return 1

    dry_run = not args.real

    if args.speed <= 0 or args.speed > 0.5:
        print("错误：速度建议范围为 (0, 0.5] rad/s")
        return 1

    print("\n========== tube_system_v1_fixed ==========")
    print("模式:", "DRY_RUN 只计算" if dry_run else "REAL 真实控制")
    print("串口:", args.port)
    print("速度:", args.speed)
    print("任务:", args.task)
    print("=========================================")

    if not dry_run:
        print("\n真实运动前必须确认：")
        print("1. 机械臂周围无人")
        print("2. 底座线缆不会被压")
        print("3. 试管点和清洗点坐标已经确认")
        print("4. 机械臂可随时断电")
        print("5. 首次运行建议不放真实试管")
        confirm = input("确认安全后输入 START：").strip()
        if confirm != "START":
            print("未输入 START，取消真实运动")
            return 1

    controller = None

    try:
        # 第 9.1 步：创建机械臂控制器
        controller = CoordinateArmController(
            serial_port=args.port,
            dry_run=dry_run,
            allow_unmeasured_limits=True,
        )

        # 第 9.2 步：当前用虚拟夹爪。
        # 以后真实夹爪做好后，在这里替换：
        #   达妙夹爪：gripper = DMGripper(controller, motor_id=0x??)
        #   舵机夹爪：gripper = ServoGripper()
        #   气动夹爪：gripper = GPIOGripper()
        gripper: GripperInterface = VirtualGripper()

        # 第 9.3 步：当前用固定坐标。
        # 以后摄像头做好后，在这里替换：
        #   vision = CameraVisionPlaceholder()
        # 然后实现 CameraVisionPlaceholder.get_tube_position()
        vision = FixedVision()

        # 第 9.4 步：执行任务
        if args.task == "full":
            tube_pick_and_wash_task(
                controller=controller,
                gripper=gripper,
                vision=vision,
                wash_xyz=WASH_XYZ,
                speed=args.speed,
            )

        elif args.task == "pick":
            pick_xyz = vision.get_tube_position()
            pick_tube(controller=controller, gripper=gripper, pick_xyz=pick_xyz, speed=args.speed)

        elif args.task == "wash":
            wash_tube(controller=controller, wash_xyz=WASH_XYZ, speed=args.speed)

        return 0

    except KeyboardInterrupt:
        print("\n用户中断，准备退出")
        return 130

    except Exception as exc:
        print(f"\n错误：{exc}")
        return 1

    finally:
        if controller is not None:
            controller.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
