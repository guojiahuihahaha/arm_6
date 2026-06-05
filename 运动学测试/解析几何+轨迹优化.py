import math
import time
from arm_6 import ArmManager  # 原本机械臂驱动

# ==========================================
# 核心大脑：PDF 公式解析几何 IK
# ==========================================⚠️【底下L1 L2还没改啊】
def pdf_inverse_kinematics(x, y, z, L1=0.25, L2=0.20):
    """
    输入目标坐标，输出 1~4 号关节角度
    """
    try:
        # 1. 转台角
        theta = math.atan2(y, x)

        # 2. 到平面投影距离与三维距离
        OB1 = math.sqrt(x**2 + y**2)
        OB = math.sqrt(x**2 + y**2 + z**2)

        # 越界保护
        if OB > (L1 + L2) or OB < abs(L1 - L2):
            print(f"❌ 坐标 ({x}, {y}, {z}) 超出机械臂范围！")
            return None

        # 3. 大臂角 alpha
        cos_AOB = (L1**2 + OB**2 - L2**2) / (2 * L1 * OB)
        cos_AOB = max(-1.0, min(1.0, cos_AOB))
        angle_AOB = math.acos(cos_AOB)
        angle_z_OB1 = math.atan2(z, OB1)
        alpha = (math.pi / 2) - angle_z_OB1 - angle_AOB

        # 4. 小臂角 beta
        cos_OAB = (L1**2 + L2**2 - OB**2) / (2 * L1 * L2)
        cos_OAB = max(-1.0, min(1.0, cos_OAB))
        angle_OAB = math.acos(cos_OAB)
        beta = (math.pi / 2) - angle_OAB

        # 5. 手腕补偿 gamma
        gamma = - (alpha + beta)

        return theta, alpha, beta, gamma

    except Exception as e:
        print(f"计算出错: {e}")
        return None

# ==========================================
# 插补函数：平滑移动
# ==========================================
def linear_interpolation(q_start, q_end, steps=50):
    """
    生成从 q_start 到 q_end 的平滑插补序列
    """
    trajectory = []
    for i in range(1, steps+1):
        q_step = [q_start[j] + (q_end[j]-q_start[j])*i/steps for j in range(len(q_start))]
        trajectory.append(q_step)
    return trajectory

# ==========================================
# 主程序
# ==========================================
def main():
    arm = ArmManager()
    L1_LENGTH = 0.25
    L2_LENGTH = 0.20

    print("\n正在使能 1~6 号电机...")
    arm.arm.enable_all_motors()
    time.sleep(1)

    try:
        while True:
            try:
                user_x = float(input("目标 X (米): "))
                user_y = float(input("目标 Y (米): "))
                user_z = float(input("目标 Z (米): "))
            except ValueError:
                print("❌ 输入错误，请输入数字！")
                continue

            ik_result = pdf_inverse_kinematics(user_x, user_y, user_z, L1=L1_LENGTH, L2=L2_LENGTH)
            if ik_result is None:
                continue

            q1, q2, q3, q4 = ik_result
            q_target = [q1, q2, q3, q4, 0.0, 0.0]

            # 读取当前角度
            q_start = [arm.arm.get_motor_status(mid)['position'] for mid in arm.arm.Motorid]

            # 插补轨迹
            traj = linear_interpolation(q_start, q_target, steps=50)

            print("🚚 正在平滑移动到目标位置...")
            for q_step in traj:
                arm.arm.arm_move(*q_step, velocity=0.2)
                time.sleep(0.02)

            print("✨ 已到达目标位置")
            # 打印角度（度）
            deg_list = [round(math.degrees(a),1) for a in q_target]
            print(f"关节角度: {deg_list}")

    except KeyboardInterrupt:
        print("\n[安全退出]")
    finally:
        print("正在失能所有电机...")
        arm.arm.disable_all_motors()
        print("机械臂已安全休眠")

if __name__ == '__main__':
    main()