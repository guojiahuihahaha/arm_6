import math
import time
import numpy as np
from numpy.linalg import norm, solve
import pinocchio

# 导入你之前封装好的达妙机械臂管理类
from arm_6 import ArmManager  

# ==========================================
# 核心函数：使用 Pinocchio 为你的机械臂构建数学模型
# ==========================================
def create_my_robot_model(L1=0.25, L2=0.20, L3=0.10):
    """
    根据你们实物机械臂的结构，动态构建一个 6 轴串联模型
    :param L1: 大臂长度 (米)
    :param L2: 小臂长度 (米)
    :param L3: 手腕到夹爪末端的距离 (米)
    """
    model = pinocchio.Model()
    
    # 1. 关节 1：底座旋转 (绕 Z 轴旋转)
    joint1_id = model.addJoint(0, pinocchio.JointModelRZ(), pinocchio.SE3.Identity(), "joint1")
    
    # 2. 关节 2：肩关节俯仰 (大臂，绕 Y 轴旋转，在 Z 方向高出底座，比如 0.1米)
    placement2 = pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1]))
    joint2_id = model.addJoint(joint1_id, pinocchio.JointModelRY(), placement2, "joint2")
    
    # 3. 关节 3：肘关节俯仰 (小臂，绕 Y 轴旋转，在 Z 方向距离大臂 L1)
    placement3 = pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, L1]))
    joint3_id = model.addJoint(joint2_id, pinocchio.JointModelRY(), placement3, "joint3")
    
    # 4. 关节 4：手腕旋转 1 (绕 X 轴或 Y 轴旋转，距离小臂 L2)
    placement4 = pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, L2]))
    joint4_id = model.addJoint(joint3_id, pinocchio.JointModelRY(), placement4, "joint4")
    
    # 5. 关节 5：手腕侧摆 (绕 Y 轴或 X 轴旋转，这里假设紧挨着关节4)
    placement5 = pinocchio.SE3.Identity()
    joint5_id = model.addJoint(joint4_id, pinocchio.JointModelRX(), placement5, "joint5")
    
    # 6. 关节 6：末端自转 (绕 Z 轴旋转，到夹爪末端距离为 L3)
    placement6 = pinocchio.SE3.Identity()
    joint6_id = model.addJoint(joint5_id, pinocchio.JointModelRZ(), placement6, "joint6")
    
    # 7. 增加末端工具的目标点 (Tool Frame)
    placement_tool = pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, L3]))
    model.addFrame(pinocchio.Frame("tool", joint6_id, 0, placement_tool, pinocchio.FrameType.OP_FRAME))
    
    return model

# ==========================================
# 核心函数：Pinocchio 雅可比阻尼迭代逆解算法
# ==========================================
def pinocchio_solve_ik(model, target_x, target_y, target_z):
    """
    输入目标空间坐标，输出 6 个关节的精确目标弧度
    """
    data = model.createData()
    JOINT_ID = 6  # 最后一轴的ID
    
    # 设定目标位姿：姿态正对前方(单位矩阵)，位置为用户输入的坐标
    oMdes = pinocchio.SE3(np.eye(3), np.array([target_x, target_y, target_z]))
    
    # 初始化搜索起点：默认让 6 个轴都从 0 弧度（或者当前真实角度）开始算
    q = pinocchio.neutral(model)
    
    # 算法参数
    eps = 1e-4       # 精度：误差小于 0.1 毫米时停止
    IT_MAX = 500     # 最大迭代次数
    DT = 1e-1        # 步长系数
    damp = 1e-6      # 阻尼系数（防奇异点崩溃的安全气囊）
    
    i = 0
    success = False
    
    while True:
        pinocchio.forwardKinematics(model, data, q)
        # 计算当前位置和目标的差距
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector  
        
        # 满足精度，提前退出
        if norm(err) < eps:
            success = True
            break
        # 超过迭代次数，判定算不出来
        if i >= IT_MAX:
            success = False
            break
            
        # 核心数学：计算雅可比矩阵并应用阻尼最小二乘法微调 q
        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)  
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * DT)
        
        i += 1
        
    if success:
        # 将求出的关节角限制在 [-pi, pi] 之间，防止电机疯狂旋转过圈
        q_clean = [math.atan2(math.sin(angle), math.cos(angle)) for angle in q.flatten().tolist()]
        return q_clean
    else:
        return None

# ==========================================
# 主程序：下周去实验室直接运行的逻辑
# ==========================================
def main():
    # 1. 第一步：加载你的达妙真实机械臂硬件连接
    # ⚠️ 下周记得到 arm_6.py 里确认 'COM8' 端口是否正确
    print("正在初始化连接达妙电机...")
    arm = ArmManager()
    
    # 2. 第二步：定义你的实物机械臂尺寸（单位：米）
    # ⚠️ 下周到实验室拿尺子精准测量后修改这里！
    L1_LENGTH = 0.25  # 大臂长
    L2_LENGTH = 0.20  # 小臂长
    L3_LENGTH = 0.12  # 手腕到夹爪末梢的长
    
    # 使用刚才写的函数构建模型
    my_model = create_my_robot_model(L1=L1_LENGTH, L2=L2_LENGTH, L3=L3_LENGTH)
    print("✅ Pinocchio 6轴自定义机械臂模型构建成功！")
    
    # 3. 电机上电使能
    print("正在使能 1~6 号真实达妙电机...")
    arm.arm.enable_all_motors()
    time.sleep(1)
    
    try:
        while True:
            print("\n----------------------------------------")
            print("【全六轴灵巧控制】请输入你想让夹爪前往的空间坐标(米)：")
            try:
                user_x = float(input("请输入目标 X (前后距离，如 0.25): "))
                user_y = float(input("请输入目标 Y (左右距离，如 0.0): "))
                user_z = float(input("请输入目标 Z (垂直高度，如 0.2): "))
            except ValueError:
                print("❌ 输入错误，请输入数字！")
                continue
                
            print("🧠 Pinocchio 数值迭代大脑正在全力计算中...")
            t_start = time.time()
            ik_angles = pinocchio_solve_ik(my_model, user_x, user_y, user_z)
            t_end = time.time()
            
            if ik_angles is not None:
                print(f"✅ 逆解计算成功！耗时: {round((t_end - t_start)*1000, 2)} 毫秒")
                print("计算出的 6 个电机弧度分别为：")
                for index, angle in enumerate(ik_angles):
                    print(f"  电机 {index+1}: {round(angle, 3)} 弧度 ({round(math.degrees(angle), 1)} 度)")
                
                # 4. 🚀 见证奇迹的时刻：把 Pinocchio 算出的 6 个角度无缝发给你的达妙机械臂驱动！
                print("🚚 正在向总线发送数据，指挥机械臂移动...")
                arm.arm_move(
                    ik_angles[0], ik_angles[1], ik_angles[2],
                    ik_angles[3], ik_angles[4], ik_angles[5],
                    velocity=0.3  # 初次调试，速度放慢点，安全第一
                )
                print("✨ 机械臂已到达指定坐标！")
            else:
                print("❌ 算法判定：目标坐标在数学上无法收敛（可能太远，或者遇到了物理死区）")
                
    except KeyboardInterrupt:
        print("\n\n[安全退出] 正在紧急切断电机电源...")
    finally:
        # 安全断电，防止电机持续发热或撞击
        arm.arm.disable_all_motors()
        print("所有真实电机已断电，进入放松模式。")

if __name__ == '__main__':
    main()