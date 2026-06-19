#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试改进后的逆运动学求解器。
验证在不同目标位置下，特别是大小臂夹角大于90度时的求解精度。
"""

import math
import sys
import numpy as np

# 导入改进的模块
from coordinate_move_dls import (
    build_arm_model, 
    analytic_seed, 
    analytic_seed_dual,
    model_to_motor,
    motor_to_model,
    BASE_TO_SHOULDER_Z,
    L1,
    L2,
    TOOL_LENGTH,
)

def test_forward_kinematics():
    """测试正运动学"""
    print("=" * 60)
    print("测试1: 正运动学验证")
    print("=" * 60)
    
    model = build_arm_model()
    
    # 测试几个已知的关节角度
    test_angles = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 全零位置
        [0.0, 0.3, -0.6, 0.0, 0.0, 0.0],  # 肘部向下
        [0.0, 0.5, -1.0, 0.0, 0.0, 0.0],  # 更大的弯曲
        [0.3, 0.4, -0.8, 0.2, 0.1, -0.1], # 一般位置
    ]
    
    for q in test_angles:
        T, origins, axes = model.forward_kinematics(q)
        p = T[:3, 3]
        print(f"Q = {np.round(q, 3)}")
        print(f"  TCP位置: [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m")
        print()

def test_ik_with_known_position():
    """使用已知关节角度生成目标位置，然后测试IK是否能恢复"""
    print("=" * 60)
    print("测试2: 已知位置的逆运动学测试")
    print("=" * 60)
    
    model = build_arm_model()
    
    # 测试的关节角度（肘部向下配置）
    test_q = [
        [0.0, 0.3, -0.6, 0.3, 0.0, 0.0],
        [0.2, 0.4, -0.8, 0.4, 0.1, -0.1],
        [0.0, 0.5, -1.0, 0.5, 0.0, 0.0],
        [0.0, 0.6, -1.2, 0.6, 0.0, 0.0],  # 接近伸直
    ]
    
    for q_ref in test_q:
        # 计算目标位置
        T_ref, _, _ = model.forward_kinematics(q_ref)
        target_pos = T_ref[:3, 3]
        
        # 计算大小臂夹角
        elbow_angle = abs(q_ref[2])
        elbow_deg = math.degrees(elbow_angle)
        
        print(f"\n参考关节角: {np.round(q_ref, 3)}")
        print(f"目标位置: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}] m")
        print(f"大小臂夹角: {elbow_deg:.1f}°")
        
        # 使用解析几何种子（肘部向下）
        seed = analytic_seed(target_pos, q_ref, elbow_down=True)
        print(f"解析种子(肘部向下): {np.round(seed, 3)}")
        
        # 使用DLS求解
        result = model.solve(
            target_position=target_pos,
            q0=seed,
            mode="position",
            max_iterations=150,
            position_tolerance=0.001,
            damping=0.03,
            max_step=0.08,
            verbose=False,
        )
        
        print(f"IK结果: {np.round(result.q, 3)}")
        print(f"收敛: {result.success}, 位置误差: {result.position_error*1000:.3f} mm")
        
        # 验证结果
        if result.success:
            T_result, _, _ = model.forward_kinematics(result.q)
            p_result = T_result[:3, 3]
            error = np.linalg.norm(p_result - target_pos)
            print(f"验证位置误差: {error*1000:.3f} mm")
        print("-" * 40)

def test_elbow_configurations():
    """测试肘部向上和向下两种配置"""
    print("=" * 60)
    print("测试3: 肘部配置对比测试")
    print("=" * 60)
    
    model = build_arm_model()
    
    # 选择一个可达的目标位置
    target_pos = np.array([0.25, 0.0, 0.25])
    print(f"\n目标位置: {target_pos}")
    print(f"距离肩关节: {np.linalg.norm(target_pos - np.array([0, 0, BASE_TO_SHOULDER_Z])):.3f} m")
    
    # 测试肘部向下配置
    print("\n--- 肘部向下配置 ---")
    seed_down = analytic_seed(target_pos, [0, 0, 0, 0, 0, 0], elbow_down=True)
    print(f"种子: {np.round(seed_down, 3)}")
    
    result_down = model.solve(
        target_position=target_pos,
        q0=seed_down,
        mode="position",
        max_iterations=150,
        position_tolerance=0.001,
        damping=0.03,
        max_step=0.08,
        verbose=False,
    )
    
    print(f"IK结果: {np.round(result_down.q, 3)}")
    print(f"收敛: {result_down.success}, 误差: {result_down.position_error*1000:.3f} mm")
    if result_down.success:
        T_down, _, _ = model.forward_kinematics(result_down.q)
        p_down = T_down[:3, 3]
        print(f"验证位置: [{p_down[0]:.4f}, {p_down[1]:.4f}, {p_down[2]:.4f}]")
        print(f"大小臂夹角: {math.degrees(abs(result_down.q[2] )):.1f}°")
    
    # 测试肘部向上配置
    print("\n--- 肘部向上配置 ---")
    seed_up = analytic_seed(target_pos, [0, 0, 0, 0, 0, 0], elbow_down=False)
    print(f"种子: {np.round(seed_up, 3)}")
    
    result_up = model.solve(
        target_position=target_pos,
        q0=seed_up,
        mode="position",
        max_iterations=150,
        position_tolerance=0.001,
        damping=0.03,
        max_step=0.08,
        verbose=False,
    )
    
    print(f"IK结果: {np.round(result_up.q, 3)}")
    print(f"收敛: {result_up.success}, 误差: {result_up.position_error*1000:.3f} mm")
    if result_up.success:
        T_up, _, _ = model.forward_kinematics(result_up.q)
        p_up = T_up[:3, 3]
        print(f"验证位置: [{p_up[0]:.4f}, {p_up[1]:.4f}, {p_up[2]:.4f}]")
        print(f"大小臂夹角: {math.degrees(abs(result_up.q[2])):.1f}°")

def test_challenging_positions():
    """测试具有挑战性的位置（接近伸直、接近奇异点）"""
    print("=" * 60)
    print("测试4: 挑战性位置测试（大小臂夹角 > 90°）")
    print("=" * 60)
    
    model = build_arm_model()
    
    # 计算一些接近伸直的目标位置
    # 当大小臂几乎伸直时，目标位置应该在以肩关节为中心，半径约为 L1+L2+TOOL_LENGTH 的球面上
    max_reach = L1 + L2 + TOOL_LENGTH
    print(f"最大臂展: {max_reach:.3f} m")
    
    test_positions = [
        # 前方接近伸直
        np.array([0.45, 0.0, 0.15]),
        np.array([0.40, 0.0, 0.20]),
        np.array([0.35, 0.0, 0.25]),
        # 侧方
        np.array([0.30, 0.20, 0.20]),
        np.array([0.25, 0.25, 0.25]),
    ]
    
    for target_pos in test_positions:
        dist = np.linalg.norm(target_pos - np.array([0, 0, BASE_TO_SHOULDER_Z]))
        print(f"\n目标: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"距离肩关节: {dist:.3f} m ({dist/max_reach*100:.1f}% 最大臂展)")
        
        # 使用自动模式（尝试两种配置）
        seed_down = analytic_seed(target_pos, [0, 0, 0, 0, 0, 0], elbow_down=True)
        seed_up = analytic_seed(target_pos, [0, 0, 0, 0, 0, 0], elbow_down=False)
        
        result_down = model.solve(
            target_position=target_pos,
            q0=seed_down,
            mode="position",
            max_iterations=150,
            position_tolerance=0.001,
            damping=0.03,
            max_step=0.08,
            verbose=False,
        )
        
        result_up = model.solve(
            target_position=target_pos,
            q0=seed_up,
            mode="position",
            max_iterations=150,
            position_tolerance=0.001,
            damping=0.03,
            max_step=0.08,
            verbose=False,
        )
        
        print(f"  肘部向下: {'✓' if result_down.success else '✗'} 误差={result_down.position_error*1000:.2f}mm, 夹角={math.degrees(abs(result_down.q[2])):.1f}°")
        print(f"  肘部向上: {'✓' if result_up.success else '✗'} 误差={result_up.position_error*1000:.2f}mm, 夹角={math.degrees(abs(result_up.q[2])):.1f}°")
        
        # 选择最优解
        if result_down.success and result_up.success:
            if result_down.position_error <= result_up.position_error:
                print(f"  → 选择肘部向下配置")
            else:
                print(f"  → 选择肘部向上配置")
        elif result_down.success:
            print(f"  → 选择肘部向下配置（唯一收敛解）")
        elif result_up.success:
            print(f"  → 选择肘部向上配置（唯一收敛解）")
        else:
            print(f"  → 两种配置均未收敛")

def main():
    print("改进的逆运动学求解器测试")
    print(f"机械臂参数: L1={L1}m, L2={L2}m, TOOL={TOOL_LENGTH}m")
    print()
    
    try:
        test_forward_kinematics()
        print()
        test_ik_with_known_position()
        print()
        test_elbow_configurations()
        print()
        test_challenging_positions()
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())