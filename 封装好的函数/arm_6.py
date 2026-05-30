#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from DM_Motor_Device import MultiMotorController
from DM_CAN import DM_Motor_Type
from DM_CAN import DM_variable
from DM_CAN import Control_Type

import time

class ArmManager():
    def __init__(self):
        # 创建机械臂对象
        self.arm = MultiMotorController('COM8')
        self.Motortype = DM_Motor_Type.DM4340  # 设定电机类型

        self.Motormode = Control_Type.POS_VEL  # 设置电机工作模式
        # 设定电机id
        self.Motorid = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06]
        # 设定电机主id
        self.Masterid = [0x11, 0x12, 0x13, 0x14, 0x15, 0x16]
        # 设置电机名称 [基础关节, 肩关节, 肘关节]
        self.Motorname = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        print("\033[32m将电机添加到机械臂中\033[0m")
        # 将电机添加到机械臂中
        for i in range( len(self.Motorid) ):
            self.arm.add_motor(motor_type = self.Motortype, motor_id = self.Motorid[i], master_id = self.Masterid[i], motor_name = self.Motorname[i])
        print(" ")

        print("\033[32m设置机械臂各个电机工作模式\033[0m")
        # 设置所有电机控制模式
        for i in range( len(self.Motorid) ):
            self.arm.setup_motor(self.Motorid[i], self.Motormode, save_to_flash=True)
        print(" ")

        # 5. 使能所有电机
        self.arm.enable_all_motors(delay_seconds=0.15)
        print("使能所有电机成功")
        time.sleep(1)

        """ 机械臂开机后, 当前姿态设置为零位 """
        # 6. 如果想要修改机械臂每个电机的初始零位
        # 或者是想要将机械臂开机后, 当前姿态设置为零位, 可以使用下面的函数
        self.arm_set_zero_position()
        #获取机械臂各个电机当前的位置, 以便后续控制使用
        self.get_arm_position(1)
        time.sleep(999999) # 让程序一直运行, 以便观察机械臂状态


    def map_gripper_value(self, value, input_min=0.0, input_max=0.035, output_min=0.0, output_max=-1.5566):
        """
        这个函数用来限制机械手爪的范围, 不要让机械手爪角度超限
        将变量从输入范围映射到输出范围
        
        参数:
        value: 需要映射的原始值
        input_min: 原始范围最小值 (默认0.0)
        input_max: 原始范围最大值 (默认0.035)
        output_min: 目标范围最小值 (默认0.0)
        output_max: 目标范围最大值 (默认-1.5566)
        
        返回:
        映射后的值
        """
        # 防止除以零
        if input_max == input_min:
            return output_min
        
        # 线性映射公式
        normalized_value = (value - input_min) / (input_max - input_min)
        mapped_value = output_min + normalized_value * (output_max - output_min)
        
        return mapped_value


    def control_joint_mit(self, joint_index, kp, kd, position, velocity, torque):
        """
        使用当前模式控制单个关节
        :param joint_index: 关节索引 (0-6)
        :param kp: 比例增益
        :param kd: 微分增益
        :param position: 目标位置(rad)
        :param velocity: 目标速度(rad/s)
        :param torque: 目标力矩(Nm)
        """
        motor_id = self.Motorid[joint_index]
        
        # 首先需要确保电机在MIT模式
        if not hasattr(self, 'motors_mit_mode'):
            self.motors_mit_mode = {}
        
        # 发送MIT控制指令
        self.arm.control_mit(motor_id, kp, kd, position, velocity, torque)

    """ 电机当前模式同时控制多个关节 """
    def control_multiple_joints_mit(self, joint_indices, kp_list, kd_list, 
                                   positions, velocities, torques):
        """
        同时控制多个关节的MIT模式
        :param joint_indices: 关节索引列表
        :param kp_list: 比例增益列表
        :param kd_list: 微分增益列表
        :param positions: 目标位置列表(rad)
        :param velocities: 目标速度列表(rad/s)
        :param torques: 目标力矩列表(Nm)
        """
        for i, joint_idx in enumerate(joint_indices):
            self.control_joint_mit(joint_idx, kp_list[i], kd_list[i], 
                                 positions[i], velocities[i], torques[i])


    """ 机械臂将当前各个电机位置置于归零位置 """
    def arm_set_zero(self):
        self.arm_set_zero_position()
        time.sleep(1)

    """ 机械臂回复到初始位置 """
    def arm_home(self):
        position = [-0.0002, 0.001, -0.0021, -0.0048, -0.0101, -0.0002]
        velocity = 0.524 # Π/6
        for motor_id in self.Motorid:
            self.arm.control_pos_vel(motor_id, position[motor_id - 1], velocity)
        time.sleep(5)


    """ 机械臂失能 """
    def arm_disable(self):
        self.arm.disable_all_motors()


    """ 设置机械臂零位 """
    def arm_set_zero_position(self):
        for motor_id in self.Motorid:
            self.arm.set_zero_position(motor_id)


    """ 获取机械臂各个电机参数 """
    def get_arm_position(self, number):
        pos = [0] * len(self.Motorid)
        # print(pos)
        for i in range(number):  # 尝试5次获取状态
            # print(f"\n第{i+1}次读取状态:")
            for motor_id in self.Motorid:
                status = self.arm.get_motor_status(motor_id, auto_refresh=True)
                if status:
                    pos[motor_id -1] = round(float(status['position']), 4)
                    print(f"  电机 {status['name']}: "
                          f"位置={status['position']:.4f} rad, "
                          f"速度={status['velocity']:.4f} rad/s, "
                          f"力矩={status['torque']:.4f} Nm")
                else:
                    print(f"  无法获取电机 {motor_id} 的状态")
            time.sleep(0.1)  # 等待0.5秒再尝试
        # print(pos)
        return pos


    """ 控制机械臂运动 """
    def arm_move(self, position1, position2, position3, position4, position5, position6, velocity=1):
        pos = [position1, position2, position3, position4, position5, position6]
        vel = velocity
        for motor_id in self.Motorid:
            self.arm.control_pos_vel(motor_id, pos[motor_id-1], vel)

    
    """ 控制机械臂单个关节运动 """
    def single_move(self, input_id, input_pos, input_vel):
        self.arm.control_pos_vel(input_id, input_pos, input_vel)

    def pos_toque_move(self, motor_id, pos, rad, cur):
        self.arm.control_pos_force_normalized(motor_id, pos, rad, cur)


def main():
    arm = ArmManager()
    # print("开启spin")
    try:
        while True:
            # # 机械臂各个电机转动控制示例
            # example_pos = [0.25, -0.5, 0.5, 0.0, 0.0, 0.0]
            # arm.arm_move(example_pos[0], example_pos[1], example_pos[2], example_pos[3], example_pos[4], example_pos[5], velocity=0.5)

            time.sleep(5)
    except KeyboardInterrupt:
        print("\033[33m 程序中断 \033[0m") #黄色字体
    except Exception as e:
        print(f"节点运行出错: {str(e)}")
    finally:
        print("\033[34m 程序准备退出 \033[0m") #蓝色字体
        time.sleep(1)
        print("\033[34m 机械臂回复到零点位置 \033[0m") #蓝色字体
        arm.arm_home()
        print("\033[34m 机械臂准备失能 \033[0m") #蓝色字体
        arm.arm_disable()
        time.sleep(1)
        print("\033[32m 程序退出完成 \033[0m") #绿色字体



if __name__ == '__main__':
    main()
