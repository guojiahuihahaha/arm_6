import math
import time
import serial
# from arm_device.DM_CAN import Motor
# from arm_device.DM_CAN import MotorControl
# from arm_device.DM_CAN import DM_Motor_Type
# from arm_device.DM_CAN import DM_variable
# from arm_device.DM_CAN import Control_Type
from DM_CAN import Motor
from DM_CAN import MotorControl
from DM_CAN import DM_Motor_Type
from DM_CAN import DM_variable
from DM_CAN import Control_Type


class MultiMotorController:
    def __init__(self, serial_port, baudrate=921600):
        """
        初始化多电机控制器
        :param serial_port: 串口设备路径，如 'COM8' 或 '/dev/ttyACM0'
        :param baudrate: 波特率, 默认921600
        """
        self.type = DM_Motor_Type.DM4340
        self.var = DM_variable
        self.con_type = Control_Type.POS_VEL

        self.serial_port = serial_port
        self.baudrate = baudrate
        self.motors = {}  # 存储电机对象，键为电机ID
        self.controller = None
        self.serial_device = None
        
        # 初始化串口和控制器
        self._initialize_serial()
        self._initialize_controller()
        
    def _initialize_serial(self):
        """初始化串口连接"""
        try:
            self.serial_device = serial.Serial(
                self.serial_port, 
                self.baudrate, 
                timeout=0.5
            )
            print(f"串口 {self.serial_port} 已打开")
        except Exception as e:
            print(f"打开串口失败: {e}")
            raise
            
    def _initialize_controller(self):
        """初始化电机控制器"""
        self.controller = MotorControl(self.serial_device)
        
    def add_motor(self, motor_type, motor_id, master_id, motor_name=None):
        """
        添加一个电机
        :param motor_type: 电机类型，如 DM_Motor_Type.DM4340
        :param motor_id: 电机CAN ID
        :param master_id: 主站ID
        :param motor_name: 电机名称（可选），如 "shoulder", "elbow"
        :return: 添加的电机对象
        """
        if motor_id in self.motors:
            print(f"警告: 电机ID {hex(motor_id)} 已存在")
            return self.motors[motor_id]
            
        # 创建电机对象
        motor = Motor(motor_type, motor_id, master_id)
        
        # 添加到控制器
        self.controller.addMotor(motor)
        
        # 存储到字典中
        motor_key = motor_name if motor_name else f"motor_{hex(motor_id)}"
        self.motors[motor_id] = {
            'obj': motor,
            'name': motor_key,
            'type': motor_type,
            'master_id': master_id
        }
        
        # print(f"添加电机: {motor_key} (ID: {hex(motor_id)}, Type: {motor_type.name})")
        return motor


    def control_pos_force(self, motor_id, position, velocity, current):
        """
        EMIT控制模式（力位混合模式）
        :param motor_id: 电机ID
        :param position: 目标位置(rad)
        :param velocity: 目标速度(rad/s)，为放大100倍的值
        :param current: 期望电流标幺值，范围0-10000（实际电流值除以最大电流值，放大10000倍）
        """
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return
            
        motor = self.motors[motor_id]['obj']
        self.controller.control_pos_force(motor, position, velocity, current)

    def control_pos_force_normalized(self, motor_id, position, velocity_rad_s, current_normalized):
        """
        EMIT控制模式（力位混合模式）- 标准化版本
        :param motor_id: 电机ID
        :param position: 目标位置(rad)
        :param velocity_rad_s: 目标速度(rad/s) - 原始值
        :param current_normalized: 期望电流标幺值（0-1之间的浮点数）
        """
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return
            
        # 将速度转换为放大100倍的值
        velocity_scaled = velocity_rad_s * 100
        
        # 将电流标幺值转换为放大10000倍的值
        current_scaled = current_normalized * 10000
        
        motor = self.motors[motor_id]['obj']
        self.controller.control_pos_force(motor, position, velocity_scaled, current_scaled)


    def setup_motor(self, motor_id, control_mode=Control_Type.POS_VEL, save_to_flash=True):
        """
        设置电机控制模式
        :param motor_id: 电机ID
        :param control_mode: 控制模式
        :param save_to_flash: 是否保存到闪存
        :return: 是否成功
        """
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return False
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        # print(f"设置电机 {motor_name} 控制模式为 {control_mode.name}...")
        
        if self.controller.switchControlMode(motor, control_mode):
            # print(f"电机 {motor_name} 切换至 {control_mode.name} 模式成功")
            
            if save_to_flash:
                # print(f"保存电机 {motor_name} 参数到闪存...")
                self.controller.save_motor_param(motor)
                time.sleep(0.1)
                
            # 更新存储的控制模式
            self.motors[motor_id]['control_mode'] = control_mode
            return True
        else:
            print(f"电机 {motor_name} 模式切换失败")
            return False
            
    def read_motor_params(self, motor_id, param_list=None):
        """
        读取电机参数
        :param motor_id: 电机ID
        :param param_list: 要读取的参数列表，默认读取常用参数
        :return: 参数字典
        """
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return {}
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        # 默认参数列表
        if param_list is None:
            param_list = [
                DM_variable.Gr,
                DM_variable.PMAX,
                DM_variable.VMAX,
                DM_variable.TMAX,
                DM_variable.hw_ver,
                DM_variable.sw_ver,
                DM_variable.sub_ver
            ]
            
        params = {}
        print(f"\n读取电机 {motor_name} 参数:")
        for param in param_list:
            try:
                value = self.controller.read_motor_param(motor, param)
                params[param.name] = value
                print(f"  {param.name}: {value}")
            except:
                print(f"  {param.name}: 读取失败")
                
        return params
        
    def enable_motor(self, motor_id, delay_seconds=0.5):
        """
        使能单个电机
        :param motor_id: 电机ID
        :param delay_seconds: 使能后等待时间
        """
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        # print(f"使能电机 {motor_name}...")
        self.controller.enable(motor)
        time.sleep(delay_seconds)
        # print(f"电机 {motor_name} 已使能")
        
    def enable_all_motors(self, delay_seconds=0.5):
        """使能所有电机"""
        # print("\n使能所有电机...")
        # print("\033[32m使能所有电机...\033[0m") #绿色字体
        for motor_id, motor_info in self.motors.items():
            motor = motor_info['obj']
            motor_name = motor_info['name']
            # print(f"  使能 {motor_name}...")
            self.controller.enable(motor)
            time.sleep(0.1)  # 电机之间短暂间隔
            
        time.sleep(delay_seconds)
        # print("所有电机已使能")
        
    def disable_motor(self, motor_id):
        """失能单个电机"""
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        # print(f"失能电机 {motor_name}...")
        self.controller.disable(motor)
        
    def disable_all_motors(self):
        """失能所有电机"""
        # print("\n失能所有电机...")
        # print("\033[32mn失能所有电机\033[0m") #绿色字体
        for motor_id, motor_info in self.motors.items():
            motor = motor_info['obj']
            motor_name = motor_info['name']
            # print(f"  失能 {motor_name}...")
            self.controller.disable(motor)
            time.sleep(0.15)  # 短暂间隔
            
    def set_zero_position(self, motor_id):
        """设置电机零位"""
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        print(f"设置电机 {motor_name} 零位...")
        self.controller.set_zero_position(motor)
        time.sleep(0.5)
        
    def control_pos_vel(self, motor_id, position, velocity):
        """
        控制电机位置和速度
        :param motor_id: 电机ID
        :param position: 目标位置(rad)
        :param velocity: 目标速度(rad/s)
        """
        if motor_id not in self.motors:
            return
            
        motor = self.motors[motor_id]['obj']
        self.controller.control_Pos_Vel(motor, position, velocity)
        
    def control_velocity(self, motor_id, velocity):
        """
        控制电机速度
        :param motor_id: 电机ID
        :param velocity: 目标速度(rad/s)
        """
        if motor_id not in self.motors:
            return
            
        motor = self.motors[motor_id]['obj']
        self.controller.control_Vel(motor, velocity)
        
    def control_torque(self, motor_id, torque):
        """
        控制电机力矩(仅适用于CSP模式)
        :param motor_id: 电机ID
        :param torque: 目标力矩(Nm)
        """
        if motor_id not in self.motors:
            return
            
        motor = self.motors[motor_id]['obj']
        self.controller.control_Tor_CSP(motor, torque)


    def control_mit(self, motor_id, kp, kd, position, velocity, torque):
        """
        MIT控制模式
        :param motor_id: 电机ID
        :param kp: 比例增益
        :param kd: 微分增益
        :param position: 目标位置(rad)
        :param velocity: 目标速度(rad/s)
        :param torque: 目标力矩(Nm)
        """
        if motor_id not in self.motors:
            return
            
        motor = self.motors[motor_id]['obj']
        self.controller.controlMIT(motor, kp, kd, position, velocity, torque)


    # def get_motor_status(self, motor_id):
    #     """
    #     获取电机状态
    #     :return: (position, velocity, torque) 或 None
    #     """
    #     if motor_id not in self.motors:
    #         return None
            
    #     motor = self.motors[motor_id]['obj']
    #     return {
    #         'position': motor.getPosition(),
    #         'velocity': motor.getVelocity(),
    #         'torque': motor.getTorque(),
    #         'name': self.motors[motor_id]['name']
    #     }

    def get_motor_status(self, motor_id, auto_refresh=True):
        """
        获取电机状态
        :param motor_id: 电机ID
        :param auto_refresh: 是否自动刷新状态（默认开启）
        :return: (position, velocity, torque) 或 None
        """
        if motor_id not in self.motors:
            return None
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        # 如果启用了自动刷新，先发送查询指令
        if auto_refresh:
            self.controller.refresh_motor_status(motor)
            # time.sleep(0.01)  # 等待接收数据
        
        return {
            'position': motor.getPosition(),
            'velocity': motor.getVelocity(),
            'torque': motor.getTorque(),
            'name': motor_name
        }

    def refresh_motor_status(self, motor_id):
        """
        主动刷新电机状态（发送查询指令）
        :param motor_id: 电机ID
        """
        if motor_id not in self.motors:
            print(f"错误: 未找到电机ID {hex(motor_id)}")
            return
            
        motor = self.motors[motor_id]['obj']
        motor_name = self.motors[motor_id]['name']
        
        # 调用底层方法发送状态查询指令
        self.controller.refresh_motor_status(motor)
        # time.sleep(0.01)  # 短暂等待接收数据
        
        # 获取更新后的状态
        status = {
            'position': motor.getPosition(),
            'velocity': motor.getVelocity(),
            'torque': motor.getTorque(),
            'name': motor_name
        }
        
        return status
        
    def refresh_all_motors_status(self):
        """刷新所有电机状态"""
        status_dict = {}
        for motor_id in self.motors:
            status = self.refresh_motor_status(motor_id)
            if status:
                status_dict[motor_id] = status
        return status_dict


    def get_all_motors_status(self):
        """获取所有电机状态"""
        status_dict = {}
        for motor_id in self.motors:
            status = self.get_motor_status(motor_id)
            if status:
                status_dict[motor_id] = status
        return status_dict
        
    def print_status(self):
        """打印所有电机状态"""
        print("\n" + "="*50)
        print("电机状态报告:")
        print("="*50)
        
        for motor_id, motor_info in self.motors.items():
            status = self.get_motor_status(motor_id)
            if status:
                print(f"{motor_info['name']} (ID: {hex(motor_id)}):")
                print(f"  位置: {status['position']:7.4f} rad")
                print(f"  速度: {status['velocity']:7.4f} rad/s")
                print(f"  力矩: {status['torque']:7.4f} Nm")
        print("="*50)

        
    def emergency_stop(self):
        """紧急停止所有电机"""
        print("\n!!! 紧急停止所有电机 !!!")
        self.disable_all_motors()
        
    def close(self):
        """安全关闭"""
        print("\n正在关闭电机控制器...")
        
        # 停止所有电机
        for motor_id in self.motors:
            motor = self.motors[motor_id]['obj']
            motor_name = self.motors[motor_id]['name']
            print(f"  停止 {motor_name}...")
            self.control_pos_vel(motor_id, 0, 0)
            
        time.sleep(0.1)
        
        # 失能所有电机
        self.disable_all_motors()
        
        # 关闭串口
        if self.serial_device and self.serial_device.is_open:
            self.serial_device.close()
            print("串口已关闭")
            
        print("电机控制器已安全关闭")
