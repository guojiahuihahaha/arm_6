
# -*- coding: utf-8 -*-
"""
轻量级 6 轴机械臂逆运动学：正运动学 + 几何雅可比 + 阻尼最小二乘（DLS）

特点：
1. 不依赖 Pinocchio，只依赖 NumPy；
2. 支持位置、完整位姿、位置+工具轴方向三种任务；
3. 支持关节限位、单步角度限制、阻尼和回溯搜索；
4. 默认仅做数学计算，不连接或控制真实电机。

重要：
- DEFAULT 模型中的关节轴、连杆长度和除 0x02 外的限位只是占位值；
- 在驱动真实机械臂前，必须测量并填写真实参数；
- 机械臂各关节的模型角度正方向必须与电机角度正方向完成标定。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal, Optional

# 限制 NumPy/BLAS 线程，降低嵌入式或低性能 CPU 占用，并避免部分环境初始化卡顿。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

TaskMode = Literal["position", "pose", "position_tool_axis"]


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """向量归一化。"""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("不能归一化零向量")
    return v / n


def skew(v: np.ndarray) -> np.ndarray:
    """返回向量对应的反对称矩阵。"""
    x, y, z = np.asarray(v, dtype=float)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    """使用 Rodrigues 公式计算绕任意单位轴旋转的 3x3 矩阵。"""
    a = normalize(axis)
    K = skew(a)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def transform(rotation: Optional[np.ndarray] = None,
              translation: Optional[Iterable[float]] = None) -> np.ndarray:
    """创建 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=float)
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("rotation 必须为 3x3 矩阵")
        T[:3, :3] = rotation
    if translation is not None:
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError("translation 必须为长度 3 的向量")
        T[:3, 3] = translation
    return T


def orientation_error(R_current: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """
    计算当前姿态到目标姿态的小角度误差向量，结果表达在基坐标系。

    e_R = 0.5 * sum(R_current[:, i] x R_target[:, i])
    """
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )


@dataclass
class IKResult:
    success: bool
    q: np.ndarray
    iterations: int
    position_error: float
    orientation_error: float
    reason: str


class LightweightArmIK:
    """
    串联转动关节机械臂模型。

    建模约定：
    - 每个 joint_axis_local[i] 表示第 i 个关节轴，表达在该关节旋转前的局部坐标系；
    - 第 i 个关节转动后，再沿 link_vectors[i] 平移到下一关节；
    - tool_T 表示最后一个关节到 TCP 的固定变换。
    """

    def __init__(
        self,
        joint_axes_local: Iterable[Iterable[float]],
        link_vectors: Iterable[Iterable[float]],
        q_min: Iterable[float],
        q_max: Iterable[float],
        base_T: Optional[np.ndarray] = None,
        tool_T: Optional[np.ndarray] = None,
    ) -> None:
        self.joint_axes_local = np.asarray(joint_axes_local, dtype=float)
        self.link_vectors = np.asarray(link_vectors, dtype=float)
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.base_T = np.eye(4) if base_T is None else np.asarray(base_T, dtype=float)
        self.tool_T = np.eye(4) if tool_T is None else np.asarray(tool_T, dtype=float)

        if self.joint_axes_local.ndim != 2 or self.joint_axes_local.shape[1] != 3:
            raise ValueError("joint_axes_local 必须为 N x 3")

        self.n = self.joint_axes_local.shape[0]
        if self.link_vectors.shape != (self.n, 3):
            raise ValueError("link_vectors 必须与关节数量一致，为 N x 3")
        if self.q_min.shape != (self.n,) or self.q_max.shape != (self.n,):
            raise ValueError("q_min 和 q_max 必须为长度 N 的向量")
        if np.any(self.q_min >= self.q_max):
            raise ValueError("每个关节必须满足 q_min < q_max")
        if self.base_T.shape != (4, 4) or self.tool_T.shape != (4, 4):
            raise ValueError("base_T 和 tool_T 必须为 4x4")

        self.joint_axes_local = np.vstack([normalize(a) for a in self.joint_axes_local])

    def forward_kinematics(self, q: Iterable[float]):
        """
        正运动学。

        返回：
        - T_tcp：基坐标系到 TCP 的 4x4 变换；
        - origins：每个关节轴上一点在基坐标系下的位置；
        - axes_world：每个关节轴在基坐标系下的方向。
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (self.n,):
            raise ValueError(f"q 必须为长度 {self.n} 的向量")

        T = self.base_T.copy()
        origins = np.zeros((self.n, 3), dtype=float)
        axes_world = np.zeros((self.n, 3), dtype=float)

        for i in range(self.n):
            origins[i] = T[:3, 3]
            axes_world[i] = T[:3, :3] @ self.joint_axes_local[i]

            T_joint = transform(rotation=rodrigues(self.joint_axes_local[i], q[i]))
            T_link = transform(translation=self.link_vectors[i])
            T = T @ T_joint @ T_link

        T_tcp = T @ self.tool_T
        return T_tcp, origins, axes_world

    def geometric_jacobian(self, q: Iterable[float]):
        """计算 TCP 的 6xN 几何雅可比矩阵。"""
        T_tcp, origins, axes_world = self.forward_kinematics(q)
        p_tcp = T_tcp[:3, 3]

        J = np.zeros((6, self.n), dtype=float)
        for i in range(self.n):
            J[:3, i] = np.cross(axes_world[i], p_tcp - origins[i])
            J[3:, i] = axes_world[i]

        return J, T_tcp

    def _task_error_and_jacobian(
        self,
        q: np.ndarray,
        target_position: np.ndarray,
        mode: TaskMode,
        target_rotation: Optional[np.ndarray],
        target_tool_axis: Optional[np.ndarray],
    ):
        J, T_tcp = self.geometric_jacobian(q)
        p_current = T_tcp[:3, 3]
        R_current = T_tcp[:3, :3]
        e_pos = target_position - p_current

        if mode == "position":
            return e_pos, J[:3, :], T_tcp, np.zeros(3)

        if mode == "pose":
            if target_rotation is None:
                raise ValueError("pose 模式必须提供 target_rotation")
            e_rot = orientation_error(R_current, target_rotation)
            return np.concatenate([e_pos, e_rot]), J, T_tcp, e_rot

        if mode == "position_tool_axis":
            if target_tool_axis is None:
                raise ValueError("position_tool_axis 模式必须提供 target_tool_axis")
            z_current = R_current[:, 2]
            z_target = normalize(target_tool_axis)
            # 只要求工具 Z 轴对准目标方向，绕工具轴自身的转角自由。
            e_axis = np.cross(z_current, z_target)
            return np.concatenate([e_pos, e_axis]), J, T_tcp, e_axis

        raise ValueError(f"不支持的任务模式: {mode}")

    def solve(
        self,
        target_position: Iterable[float],
        q0: Iterable[float],
        *,
        mode: TaskMode = "position_tool_axis",
        target_rotation: Optional[np.ndarray] = None,
        target_tool_axis: Optional[Iterable[float]] = None,
        position_weight: float = 1.0,
        orientation_weight: float = 0.25,
        damping: float = 0.03,
        min_damping: float = 1e-4,
        max_damping: float = 1.0,
        max_step: float = 0.08,
        max_iterations: int = 100,
        position_tolerance: float = 1e-3,
        orientation_tolerance: float = np.deg2rad(1.0),
        verbose: bool = False,
        progress_interval: int = 10,
    ) -> IKResult:
        """
        使用阻尼最小二乘法求逆运动学。

        DLS 更新：
            dq = J^T (J J^T + lambda^2 I)^(-1) e

        其中加入任务权重、关节限位、单步限制和简单回溯搜索。
        """
        target_position = np.asarray(target_position, dtype=float)
        if target_position.shape != (3,):
            raise ValueError("target_position 必须为长度 3 的向量")

        q = np.clip(np.asarray(q0, dtype=float), self.q_min, self.q_max)
        if q.shape != (self.n,):
            raise ValueError(f"q0 必须为长度 {self.n} 的向量")

        if target_rotation is not None:
            target_rotation = np.asarray(target_rotation, dtype=float)
            if target_rotation.shape != (3, 3):
                raise ValueError("target_rotation 必须为 3x3")
        if target_tool_axis is not None:
            target_tool_axis = normalize(np.asarray(target_tool_axis, dtype=float))

        if mode == "position":
            weights = np.array([position_weight] * 3, dtype=float)
        else:
            weights = np.array(
                [position_weight] * 3 + [orientation_weight] * 3,
                dtype=float,
            )

        lam = float(damping)
        last_pos_err = float("inf")
        last_rot_err = float("inf")

        for iteration in range(max_iterations + 1):
            e, J_task, _, e_rot = self._task_error_and_jacobian(
                q,
                target_position,
                mode,
                target_rotation,
                target_tool_axis,
            )

            pos_err = float(np.linalg.norm(e[:3]))
            rot_err = 0.0 if mode == "position" else float(np.linalg.norm(e_rot))
            last_pos_err, last_rot_err = pos_err, rot_err

            if verbose and (iteration == 0 or iteration % max(1, progress_interval) == 0):
                print(
                    f"  DLS 迭代 {iteration:03d}: 位置误差={pos_err * 1000.0:.3f} mm, "
                    f"姿态误差={np.degrees(rot_err):.3f} deg, 阻尼={lam:.6f}",
                    flush=True,
                )

            if pos_err <= position_tolerance and rot_err <= orientation_tolerance:
                return IKResult(
                    success=True,
                    q=q.copy(),
                    iterations=iteration,
                    position_error=pos_err,
                    orientation_error=rot_err,
                    reason="收敛",
                )

            if iteration == max_iterations:
                break

            W = np.diag(weights)
            e_w = W @ e
            J_w = W @ J_task
            current_cost = float(np.linalg.norm(e_w))

            accepted = False
            trial_lam = lam

            for _ in range(6):
                A = J_w @ J_w.T + (trial_lam ** 2) * np.eye(J_w.shape[0])
                try:
                    dq = J_w.T @ np.linalg.solve(A, e_w)
                except np.linalg.LinAlgError:
                    trial_lam = min(trial_lam * 2.0, max_damping)
                    continue

                dq = np.clip(dq, -max_step, max_step)
                q_trial = np.clip(q + dq, self.q_min, self.q_max)

                e_trial, _, _, _ = self._task_error_and_jacobian(
                    q_trial,
                    target_position,
                    mode,
                    target_rotation,
                    target_tool_axis,
                )
                trial_cost = float(np.linalg.norm(W @ e_trial))

                if trial_cost < current_cost:
                    q = q_trial
                    lam = max(trial_lam * 0.7, min_damping)
                    accepted = True
                    break

                trial_lam = min(trial_lam * 2.0, max_damping)

            if not accepted:
                return IKResult(
                    success=False,
                    q=q.copy(),
                    iterations=iteration,
                    position_error=last_pos_err,
                    orientation_error=last_rot_err,
                    reason="本轮无法找到降低误差的更新，可能处于奇异位形、关节限位或目标不可达",
                )

        return IKResult(
            success=False,
            q=q.copy(),
            iterations=max_iterations,
            position_error=last_pos_err,
            orientation_error=last_rot_err,
            reason="超过最大迭代次数",
        )


def create_default_arm_model(
    base_height: float = 0.10,
    L1: float = 0.25,
    L2: float = 0.20,
    tool_length: float = 0.12,
) -> LightweightArmIK:
    """
    根据现有 Pinocchio 示例中的假设创建占位模型：
      joint1: RZ
      joint2: RY
      joint3: RY
      joint4: RY
      joint5: RX
      joint6: RZ

    注意：除 0x02 外，其余机械限位尚未实测；下面的 +-pi 仅用于离线计算。
    """
    joint_axes_local = [
        [0, 0, 1],  # 0x01
        [0, 1, 0],  # 0x02
        [0, 1, 0],  # 0x03
        [0, 1, 0],  # 0x04
        [1, 0, 0],  # 0x05
        [0, 0, 1],  # 0x06
    ]

    # 每个关节旋转后，到下一关节的固定平移。
    link_vectors = [
        [0, 0, base_height],
        [0, 0, L1],
        [0, 0, L2],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]

    q_min = [-np.pi, -0.70, -np.pi, -np.pi, -np.pi, -np.pi]
    q_max = [ np.pi,  1.57,  np.pi,  np.pi,  np.pi,  np.pi]
    tool_T = transform(translation=[0, 0, tool_length])

    return LightweightArmIK(
        joint_axes_local=joint_axes_local,
        link_vectors=link_vectors,
        q_min=q_min,
        q_max=q_max,
        tool_T=tool_T,
    )


def demo() -> None:
    """离线演示，不控制电机。"""
    model = create_default_arm_model()

    # 初值应优先替换为真实机械臂当前关节角，能明显减少迭代次数。
    q0 = np.array([0.0, 0.3, -0.8, 0.5, 0.0, 0.0])

    # 生成一个可达的离线测试目标。实际使用时，替换成视觉系统给出的抓取点
    # 以及希望工具轴指向的方向，例如竖直向下 [0, 0, -1]。
    q_reference = np.array([0.30, 0.80, -1.20, 1.00, 0.40, -0.20])
    T_reference, _, _ = model.forward_kinematics(q_reference)

    result = model.solve(
        target_position=T_reference[:3, 3],
        q0=q0,
        mode="position_tool_axis",
        target_tool_axis=T_reference[:3, 2],
        max_iterations=120,
    )

    print("success:", result.success)
    print("reason:", result.reason)
    print("iterations:", result.iterations)
    print("position error (m):", result.position_error)
    print("orientation error (rad):", result.orientation_error)
    print("q (rad):", np.round(result.q, 5))


if __name__ == "__main__":
    demo()
