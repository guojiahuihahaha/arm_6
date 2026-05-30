#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DM电机基本使用示例
电机ID: 0x01
"""

import time
import sys
import os

# 将当前目录添加到Python路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from DM_Motor_Device import MultiMotorController
from DM_CAN import DM_Motor_Type, Control_Type


def main():
    # ========== 配置参数 ==========
    SERIAL_PORT = 'COM8'      # 串口名称 (Windows: COM8, Linux: /dev/ttyACM0)
    BAUDRATE = 921600         # 波特率
    MOTOR_ID = 0x01           # 电机ID
    MOTOR_TYPE = DM_Motor_Type.DM4340  # 电机型号
    MASTER_ID = 0x11          # 主站ID

    # ========== 创建控制器 ==========
    print("=" * 50)
    print("DM电机基本使用示例")
    print("=" * 50)

    try:
        controller = MultiMotorController(SERIAL_PORT, BAUDRATE)
    except Exception as e:
        print(f"创建控制器失败: {e}")
        return

    # ========== 添加电机 ==========
    print(f"\n1. 添加电机 (ID: 0x{MOTOR_ID:02X}, 型号: {MOTOR_TYPE.name})")
    controller.add_motor(MOTOR_TYPE, MOTOR_ID, MASTER_ID, "test_motor")

    # ========== 使能电机 ==========
    print(f"\n2. 使能电机...")
    controller.enable_motor(MOTOR_ID)
    print(f"电机已使能")

    # ========== 设置控制模式 ==========
    print(f"\n3. 设置控制模式为 POS_VEL...")
    success = controller.setup_motor(MOTOR_ID, Control_Type.POS_VEL, save_to_flash=False)
    if success:
        print("控制模式设置成功")
    else:
        print("控制模式设置失败")

    # 等待电机稳定
    time.sleep(0.5)

    # ========== 读取电机参数（可选） ==========
    print(f"\n4. 读取电机参数...")
    controller.read_motor_params(MOTOR_ID)

    # ========== 电机控制示例 ==========
    print(f"\n5. 开始电机控制演示...")

    try:
        # 示例1: 位置速度控制
        print("\n--- 示例1: 位置速度控制 ---")
        target_position = 3.14   # 目标位置 (rad)
        target_velocity = 2.0   # 目标速度 (rad/s)
        
        print(f"移动到位置: {target_position} rad, 速度: {target_velocity} rad/s")
        for i in range(50):
            controller.control_pos_vel(MOTOR_ID, target_position, target_velocity)
            time.sleep(0.02)
        
        # 获取当前位置
        status = controller.get_motor_status(MOTOR_ID)
        print(f"当前位置: {status['position']:.4f} rad")
        print(f"当前速度: {status['velocity']:.4f} rad/s")
        print(f"当前力矩: {status['torque']:.4f} Nm")

        # 示例2: 速度控制
        print("\n--- 示例2: 速度控制 ---")
        print("设置速度为 5 rad/s")
        for i in range(50):
            controller.control_velocity(MOTOR_ID, 5.0)
            time.sleep(0.02)
        
        status = controller.get_motor_status(MOTOR_ID)
        print(f"当前速度: {status['velocity']:.4f} rad/s")

        # 示例3: MIT模式控制（更精确的位置控制）
        print("\n--- 示例3: MIT模式控制 ---")
        print("切换到MIT模式...")
        controller.setup_motor(MOTOR_ID, Control_Type.MIT, save_to_flash=False)
        time.sleep(0.3)
        
        # MIT模式参数
        kp = 20.0    # 位置增益
        kd = 2.0     # 速度增益
        target_pos = 0.0  # 目标位置
        target_vel = 0.0  # 目标速度
        target_tau = 0.0  # 目标力矩
        
        print(f"MIT模式: kp={kp}, kd={kd}, 目标位置={target_pos} rad")
        for i in range(100):
            controller.control_mit(MOTOR_ID, kp, kd, target_pos, target_vel, target_tau)
            time.sleep(0.01)
        
        status = controller.get_motor_status(MOTOR_ID)
        print(f"当前位置: {status['position']:.4f} rad")
        print(f"当前位置: {status['velocity']:.4f} rad/s")
        print(f"当前力矩: {status['torque']:.4f} Nm")

        # 示例4: 返回零位
        print("\n--- 示例4: 返回零位 ---")
        controller.setup_motor(MOTOR_ID, Control_Type.POS_VEL, save_to_flash=False)
        time.sleep(0.3)
        
        print("返回零位...")
        for i in range(100):
            controller.control_pos_vel(MOTOR_ID, 0.0, 2.0)
            time.sleep(0.02)
        
        status = controller.get_motor_status(MOTOR_ID)
        print(f"当前位置: {status['position']:.4f} rad")

    except KeyboardInterrupt:
        print("\n\n用户中断操作")

    # ========== 打印最终状态 ==========
    print("\n" + "=" * 50)
    print("最终电机状态:")
    controller.print_status()

    # ========== 安全关闭 ==========
    print("\n正在关闭电机控制器...")
    controller.close()
    print("程序结束")


if __name__ == "__main__":
    main()
