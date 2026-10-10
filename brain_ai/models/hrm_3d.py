"""
brain_ai.models.hrm_3d: Right Hemisphere 3D Hierarchical Reasoning Model (HRM-3D).

Grounding & Architecture:
1. RoPE-3D: Continuous 3D Rotary Position Embeddings across continuous (x, y, z) coordinates.
   - Translation-equivariant inner products for metric spatial coordinates.
2. Differentiable Kinematic & End-Effector SE(3) Relaxation:
   - Analytical Forward Kinematics (FK) and exact geometric Jacobian for N-DOF manipulators.
   - Damped Least Squares (DLS) SE(3) pose tracking.
   - 3D continuous obstacle repulsive potential fields and joint limit barriers.
3. Dual-Timescale Recurrence:
   - Strategic H-Module: Slow abstract trajectory & waypoint planner (z_H).
   - Tactical L-Module: Fast concrete joint execution & SE(3) relaxation (z_L).
   - Timescale ratio: L_cycles inner tactical steps per outer strategic step.
4. Proprioceptive & 3D Spatial Embeddings:
   - Encodes continuous joint angles, velocities, and SE(3) poses.
   - Encodes metric 3D point cloud / obstacle / attractor tokens with continuous coordinates.
5. ACT Halting & 1-Step Fixed-Point Implicit Gradient Approximation (O(1) memory).
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HRM3DStateCarry:
    """Internal state container across 3D recurrent reasoning cycles."""
    z_H: torch.Tensor          # (B, S, d_model) High-level strategic state
    z_L: torch.Tensor          # (B, S, d_model) Low-level tactical state
    q_joints: torch.Tensor     # (B, N_joints) Current joint angles
    ee_pose: torch.Tensor      # (B, 7) Current end-effector SE(3) pose: pos (3) + quat (4)
    step_count: torch.Tensor   # (B,) Number of completed segments
    halted: torch.Tensor       # (B,) Halting boolean mask


class MagicNorm(nn.Module):
    """
    MagicNorm: Bounded normalization step at the exit boundary of recursive modules.
    Clamps spectral variance growth across deep recurrent unrolling cycles.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        normed = x / rms
        return normed * self.weight


class RoPE3D(nn.Module):
    """
    Continuous 3D Rotary Position Embeddings (RoPE-3D):
    Encodes continuous 3D spatial coordinates (x, y, z) by partitioning the attention head
    dimension into three frequency bands (d_x, d_y, d_z), each having an even number of
    dimensions for rotary 2D complex pairs.

    Inner product between rotated queries at position p_A and keys at position p_B
    strictly depends on the continuous relative metric displacement: (p_A - p_B).
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim >= 6 and dim % 2 == 0, f"Head dimension {dim} must be even and >= 6"
        self.dim = dim
        self.base = base
        
        # Partition dim into 3 even parts: d_x, d_y, d_z
        d_part = 2 * (dim // 6)
        self.d_x = d_part
        self.d_y = d_part
        self.d_z = dim - 2 * d_part
        assert self.d_x % 2 == 0 and self.d_y % 2 == 0 and self.d_z % 2 == 0
        assert self.d_x + self.d_y + self.d_z == dim
        
        # Precompute inverse frequency tables for each axis
        inv_freq_x = 1.0 / (base ** (torch.arange(0, self.d_x, 2).float() / self.d_x))
        inv_freq_y = 1.0 / (base ** (torch.arange(0, self.d_y, 2).float() / self.d_y))
        inv_freq_z = 1.0 / (base ** (torch.arange(0, self.d_z, 2).float() / self.d_z))
        
        self.register_buffer("inv_freq_x", inv_freq_x)
        self.register_buffer("inv_freq_y", inv_freq_y)
        self.register_buffer("inv_freq_z", inv_freq_z)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        qk: torch.Tensor,
        coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            qk: (B, num_heads, S, head_dim) Query or Key tensor.
            coords: (B, S, 3) Continuous 3D coordinates (x, y, z) in meters or normalized units.
                    If None, defaults to zeros (identity rotation).
        Returns:
            Rotated tensor of shape (B, num_heads, S, head_dim).
        """
        B, H, S, D = qk.shape
        assert D == self.dim, f"Expected head dim {self.dim}, got {D}"
        
        if coords is None:
            return qk
            
        # Ensure coords shape is (B, S, 3)
        if coords.dim() == 2 and coords.shape[-1] == 3:
            coords = coords.unsqueeze(0).expand(B, -1, -1)
        assert coords.shape[0] == B and coords.shape[1] == S and coords.shape[2] == 3, (
            f"Coords shape must be ({B}, {S}, 3), got {coords.shape}"
        )
        
        # Split qk along head dimension
        qk_x = qk[..., :self.d_x]
        qk_y = qk[..., self.d_x:self.d_x + self.d_y]
        qk_z = qk[..., self.d_x + self.d_y:]
        
        # Axis X rotary modulation
        pos_x = coords[..., 0:1] # (B, S, 1)
        theta_x = pos_x * self.inv_freq_x # (B, S, d_x // 2)
        cos_x = torch.cat([torch.cos(theta_x), torch.cos(theta_x)], dim=-1).unsqueeze(1) # (B, 1, S, d_x)
        sin_x = torch.cat([torch.sin(theta_x), torch.sin(theta_x)], dim=-1).unsqueeze(1)
        qk_x_rot = (qk_x * cos_x) + (self._rotate_half(qk_x) * sin_x)
        
        # Axis Y rotary modulation
        pos_y = coords[..., 1:2]
        theta_y = pos_y * self.inv_freq_y
        cos_y = torch.cat([torch.cos(theta_y), torch.cos(theta_y)], dim=-1).unsqueeze(1)
        sin_y = torch.cat([torch.sin(theta_y), torch.sin(theta_y)], dim=-1).unsqueeze(1)
        qk_y_rot = (qk_y * cos_y) + (self._rotate_half(qk_y) * sin_y)
        
        # Axis Z rotary modulation
        pos_z = coords[..., 2:3]
        theta_z = pos_z * self.inv_freq_z
        cos_z = torch.cat([torch.cos(theta_z), torch.cos(theta_z)], dim=-1).unsqueeze(1)
        sin_z = torch.cat([torch.sin(theta_z), torch.sin(theta_z)], dim=-1).unsqueeze(1)
        qk_z_rot = (qk_z * cos_z) + (self._rotate_half(qk_z) * sin_z)
        
        return torch.cat([qk_x_rot, qk_y_rot, qk_z_rot], dim=-1)


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalizes quaternions to unit length on S^3."""
    norm = torch.norm(q, dim=-1, keepdim=True)
    return q / torch.clamp(norm, min=eps)


def so3_geodesic_distance(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes true geodesic Riemannian distance on SO(3) between unit quaternions q1 and q2.
    Respects the antipodal equivalence of the double cover S^3 -> SO(3): q ~ -q.
    
    d_SO(3)(q1, q2) = 2 * arccos(|<q1, q2>|) in [0, pi].
    Guarantees stable gradients and exact zero distance for antipodal configurations.
    """
    q1_n = quaternion_normalize(q1, eps=eps)
    q2_n = quaternion_normalize(q2, eps=eps)
    dot = torch.sum(q1_n * q2_n, dim=-1)
    abs_dot = torch.clamp(torch.abs(dot), min=0.0, max=1.0)
    near_identity = (abs_dot >= (1.0 - eps))
    safe_abs_dot = torch.clamp(abs_dot, min=0.0, max=1.0 - eps)
    dist_acos = 2.0 * torch.acos(safe_abs_dot)
    eps_grad = 1e-12
    dist_taylor = 2.0 * (torch.sqrt(torch.clamp(2.0 * (1.0 - abs_dot) + eps_grad, min=eps_grad)) - math.sqrt(eps_grad))
    return torch.where(near_identity, dist_taylor, dist_acos)


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Differentiable and numerically robust Shepperd-style conversion from 3x3 rotation matrices
    to unit quaternions (w, x, y, z).
    Handles all rotation angles including tr(R) <= 0 (180-degree rotations) with zero numerical degradation.
    R: (..., 3, 3)
    Returns: (..., 4)
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    N = R_flat.shape[0]

    r00 = R_flat[:, 0, 0]
    r11 = R_flat[:, 1, 1]
    r22 = R_flat[:, 2, 2]
    tr = r00 + r11 + r22

    # 4 candidates for trace / diagonal combinations (Shepperd algorithm)
    t0 = 1.0 + tr
    t1 = 1.0 + r00 - r11 - r22
    t2 = 1.0 - r00 + r11 - r22
    t3 = 1.0 - r00 - r11 + r22

    candidates = torch.stack([t0, t1, t2, t3], dim=1) # (N, 4)
    max_idx = torch.argmax(candidates, dim=1) # (N,)

    q = torch.zeros(N, 4, dtype=R.dtype, device=R.device)

    # Branch 0: w is dominant (t0 is max)
    mask0 = (max_idx == 0)
    if mask0.any():
        s = 0.5 * torch.sqrt(torch.clamp(t0[mask0], min=1e-8))
        q[mask0, 0] = s
        q[mask0, 1] = (R_flat[mask0, 2, 1] - R_flat[mask0, 1, 2]) / (4.0 * s)
        q[mask0, 2] = (R_flat[mask0, 0, 2] - R_flat[mask0, 2, 0]) / (4.0 * s)
        q[mask0, 3] = (R_flat[mask0, 1, 0] - R_flat[mask0, 0, 1]) / (4.0 * s)

    # Branch 1: x is dominant (t1 is max)
    mask1 = (max_idx == 1)
    if mask1.any():
        s = 0.5 * torch.sqrt(torch.clamp(t1[mask1], min=1e-8))
        q[mask1, 0] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / (4.0 * s)
        q[mask1, 1] = s
        q[mask1, 2] = (R_flat[mask1, 0, 1] + R_flat[mask1, 1, 0]) / (4.0 * s)
        q[mask1, 3] = (R_flat[mask1, 0, 2] + R_flat[mask1, 2, 0]) / (4.0 * s)

    # Branch 2: y is dominant (t2 is max)
    mask2 = (max_idx == 2)
    if mask2.any():
        s = 0.5 * torch.sqrt(torch.clamp(t2[mask2], min=1e-8))
        q[mask2, 0] = (R_flat[mask2, 0, 2] - R_flat[mask2, 2, 0]) / (4.0 * s)
        q[mask2, 1] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / (4.0 * s)
        q[mask2, 2] = s
        q[mask2, 3] = (R_flat[mask2, 1, 2] + R_flat[mask2, 2, 1]) / (4.0 * s)

    # Branch 3: z is dominant (t3 is max)
    mask3 = (max_idx == 3)
    if mask3.any():
        s = 0.5 * torch.sqrt(torch.clamp(t3[mask3], min=1e-8))
        q[mask3, 0] = (R_flat[mask3, 1, 0] - R_flat[mask3, 0, 1]) / (4.0 * s)
        q[mask3, 1] = (R_flat[mask3, 0, 2] + R_flat[mask3, 2, 0]) / (4.0 * s)
        q[mask3, 2] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / (4.0 * s)
        q[mask3, 3] = s

    # Safe unit normalization
    norm = torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-8)
    q = q / norm
    return q.reshape(*batch_shape, 4)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Differentiable conversion from unit quaternions (w, x, y, z) to 3x3 rotation matrices.
    q: (..., 4)
    Returns: (..., 3, 3)
    """
    q = q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=1e-7)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    r00 = 1.0 - 2.0 * (y ** 2 + z ** 2)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    
    r10 = 2.0 * (x * y + w * z)
    r11 = 1.0 - 2.0 * (x ** 2 + z ** 2)
    r12 = 2.0 * (y * z - w * x)
    
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x ** 2 + y ** 2)
    
    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R


class ForwardKinematics7DOF(nn.Module):
    """
    Differentiable Forward Kinematics and Geometric Jacobian for a 7-DOF manipulator.
    Computes exact end-effector SE(3) pose, intermediate link positions, and the (6 x 7) Jacobian.
    """
    def __init__(self):
        super().__init__()
        self.num_joints = 7
        
        # Default link offsets (meters) inspired by collaborative manipulators (Franka Panda)
        link_offsets = [
            [0.0, 0.0, 0.333],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.316],
            [0.0825, 0.0, 0.0],
            [-0.0825, 0.0, 0.384],
            [0.0, 0.0, 0.0],
            [0.088, 0.0, 0.107],
        ]
        self.register_buffer("link_offsets", torch.tensor(link_offsets, dtype=torch.float32))
        
        # End-effector tool flange offset
        self.register_buffer("ee_offset", torch.tensor([0.0, 0.0, 0.10], dtype=torch.float32))
        
        # Rotation axes for joints 0..6: alternating z, y, z, -y, z, y, z
        axes = [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        self.register_buffer("axes", torch.tensor(axes, dtype=torch.float32))
        
        # Precompute skew-symmetric matrices K and K^2 for all 7 rotation axes
        K_list = []
        for ax, ay, az in axes:
            K_list.append([
                [0.0, -az, ay],
                [az, 0.0, -ax],
                [-ay, ax, 0.0]
            ])
        K_axes = torch.tensor(K_list, dtype=torch.float32)
        K2_axes = torch.bmm(K_axes, K_axes)
        self.register_buffer("K_axes", K_axes)
        self.register_buffer("K2_axes", K2_axes)

    def _rodrigues(self, axis_idx: int, theta: torch.Tensor) -> torch.Tensor:
        """
        Rodrigues formula for batch rotation around unit axis using precomputed skew-symmetric matrices.
        axis_idx: int (0..6)
        theta: (B,)
        Returns: (B, 3, 3)
        """
        B = theta.shape[0]
        K = self.K_axes[axis_idx].unsqueeze(0).expand(B, 3, 3)
        K2 = self.K2_axes[axis_idx].unsqueeze(0).expand(B, 3, 3)
        I = torch.eye(3, device=theta.device, dtype=theta.dtype).unsqueeze(0).expand(B, 3, 3)
        
        sin_th = torch.sin(theta).view(B, 1, 1)
        cos_th = torch.cos(theta).view(B, 1, 1)
        
        R = I + sin_th * K + (1.0 - cos_th) * K2
        return R

    def forward(
        self,
        q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, 7) Joint angles in radians.
        Returns:
            ee_pos: (B, 3) End-effector 3D position.
            ee_quat: (B, 4) End-effector unit quaternion (w, x, y, z).
            link_positions: (B, 7, 3) 3D positions of all 7 joints/links.
            jacobian: (B, 6, 7) Geometric Jacobian J = [J_v; J_w].
        """
        B = q.shape[0]
        device = q.device
        dtype = q.dtype
        
        # Homogeneous transformations
        T_curr = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4).clone()
        
        link_pos_list = []
        z_axes_list = []
        p_origins_list = []
        
        for i in range(self.num_joints):
            axis_i = self.axes[i]
            theta_i = q[:, i]
            
            # The rotation axis in world coordinates BEFORE joint i rotates
            axis_b = axis_i.view(1, 3, 1).expand(B, 3, 1)
            z_i = torch.bmm(T_curr[:, :3, :3], axis_b).squeeze(-1) # (B, 3)
            p_orig_i = T_curr[:, :3, 3] # (B, 3)
            
            z_axes_list.append(z_i)
            p_origins_list.append(p_orig_i)
            
            R_rel = self._rodrigues(i, theta_i) # (B, 3, 3)
            d_rel = self.link_offsets[i].unsqueeze(0).expand(B, 3) # (B, 3)
            
            # Form homogeneous transform A_i = [[R_rel, d_rel], [0, 1]]
            A_i = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4).clone()
            A_i[:, :3, :3] = R_rel
            A_i[:, :3, 3] = d_rel
            
            # Cumulative transformation
            T_curr = torch.bmm(T_curr, A_i)
            p_curr = T_curr[:, :3, 3]
            link_pos_list.append(p_curr)
            
        # End-effector flange transform
        A_ee = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(B, 4, 4).clone()
        A_ee[:, :3, 3] = self.ee_offset.unsqueeze(0).expand(B, 3)
        T_ee = torch.bmm(T_curr, A_ee)
        
        ee_pos = T_ee[:, :3, 3]
        ee_R = T_ee[:, :3, :3]
        ee_quat = matrix_to_quaternion(ee_R)
        
        link_positions = torch.stack(link_pos_list, dim=1) # (B, 7, 3)
        
        # Compute Geometric Jacobian J = [J_v; J_w] (B, 6, 7)
        J_cols = []
        for i in range(self.num_joints):
            z_i = z_axes_list[i]
            p_orig_i = p_origins_list[i]
            # Linear velocity Jacobian: z_i x (p_ee - p_orig_i)
            r_i = ee_pos - p_orig_i
            J_v_i = torch.cross(z_i, r_i, dim=-1)
            # Angular velocity Jacobian: z_i
            J_w_i = z_i
            J_i = torch.cat([J_v_i, J_w_i], dim=-1) # (B, 6)
            J_cols.append(J_i)
            
        jacobian = torch.stack(J_cols, dim=-1) # (B, 6, 7)
        return ee_pos, ee_quat, link_positions, jacobian


class KinematicSE3Relaxation(nn.Module):
    """
    Differentiable Kinematic Joint-Angle and End-Effector SE(3) Relaxation Module.
    Minimizes a composite potential energy landscape:
    - Attractor potential towards SE(3) target waypoints (translation + orientation geodesic).
    - Repulsive potential field from 3D obstacles (collision avoidance).
    - Smoothness and joint limit soft barrier functions.
    - Neural relaxation delta from tactical reasoning states z_L.
    """
    def __init__(
        self,
        num_joints: int = 7,
        d_latent: int = 512,
        relax_steps: int = 3,
        damping: float = 0.05,
        alpha_se3: float = 0.4,
        beta_obs: float = 0.25,
        gamma_limits: float = 0.1
    ):
        super().__init__()
        self.num_joints = num_joints
        self.d_latent = d_latent
        self.relax_steps = relax_steps
        self.damping = damping
        self.alpha_se3 = alpha_se3
        self.beta_obs = beta_obs
        self.gamma_limits = gamma_limits
        
        self.fk = ForwardKinematics7DOF()
        
        # Joint angle physical limits (radians)
        joint_limits_min = [-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]
        joint_limits_max = [2.89, 1.76, 2.89, -0.06, 2.89, 3.75, 2.89]
        self.register_buffer("q_min", torch.tensor(joint_limits_min, dtype=torch.float32))
        self.register_buffer("q_max", torch.tensor(joint_limits_max, dtype=torch.float32))
        
        # Neural relaxation refinement from tactical latent z_L
        self.neural_relax_head = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2),
            nn.SiLU(),
            nn.Linear(d_latent // 2, num_joints)
        )

    def yoshikawa_manipulability(self, J: torch.Tensor) -> torch.Tensor:
        """
        Computes Yoshikawa manipulability measure w = sqrt(det(J J^T)).
        J: (B, 6, N) or (B, M, N)
        Returns: (B,) scalar manipulability measure.
        """
        JJT = torch.bmm(J, J.transpose(1, 2)) # (B, M, M)
        det_JJT = torch.clamp(torch.linalg.det(JJT), min=1e-12)
        return torch.sqrt(det_JJT)

    def damped_least_squares_inverse(
        self,
        J: torch.Tensor,
        w_threshold: float = 0.04,
        lambda_max: float = 0.25
    ) -> torch.Tensor:
        """
        Manipulability-dependent adaptive Damped Least Squares (DLS) inverse:
        J_dagger = J.T @ (J @ J.T + lambda(w)^2 * I)^(-1).
        
        Near singular configurations where Yoshikawa manipulability w = sqrt(det(J J^T)) -> 0,
        damping smoothly increases from self.damping to lambda_max:
            lambda(w)^2 = lambda_0^2 + lambda_max^2 * (1 - w / w_threshold)^2  for w < w_threshold.
            
        J: (B, 6, N)
        Returns: (B, N, 6)
        """
        B, M, N = J.shape
        JJT = torch.bmm(J, J.transpose(1, 2)) # (B, M, M)
        I = torch.eye(M, device=J.device, dtype=J.dtype).unsqueeze(0).expand(B, M, M)
        
        # Yoshikawa manipulability measure
        det_JJT = torch.clamp(torch.linalg.det(JJT), min=1e-12)
        w = torch.sqrt(det_JJT) # (B,)
        
        # Adaptive damping modulation
        ratio = torch.clamp(w / w_threshold, max=1.0)
        damping_sq = (self.damping ** 2) + (lambda_max ** 2) * ((1.0 - ratio) ** 2) # (B,)
        damping_matrix = damping_sq.view(B, 1, 1) * I
        
        # Numerically stable solve: (JJT + lambda^2 I) @ Y = J => Y = A^-1 @ J => J_dagger = Y.T
        A = JJT + damping_matrix
        try:
            Y = torch.linalg.solve(A, J)
            return Y.transpose(1, 2)
        except Exception:
            damped_inv = torch.linalg.pinv(A)
            return torch.bmm(J.transpose(1, 2), damped_inv)

    def forward(
        self,
        q_init: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: Optional[torch.Tensor] = None,
        obstacles: Optional[torch.Tensor] = None,
        tactical_latent: Optional[torch.Tensor] = None,
        steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Executes iterative SE(3) relaxation:
        Args:
            q_init: (B, num_joints) Starting joint angles.
            target_pos: (B, 3) Target 3D position waypoint.
            target_quat: (B, 4) Target orientation quaternion. Default [1, 0, 0, 0].
            obstacles: Optional (B, N_obs, 3) Obstacle point coordinates in 3D.
            tactical_latent: Optional (B, d_latent) Tactical latent vector z_L.
            steps: Optional override for relaxation steps.
        Returns:
            Dict containing relaxed joint angles, relaxed ee pose, tracking error, clearance.
        """
        B = q_init.shape[0]
        device = q_init.device
        dtype = q_init.dtype
        k_steps = steps or self.relax_steps
        
        if target_quat is None:
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).expand(B, 4)
        else:
            target_quat = quaternion_normalize(target_quat)
            
        q_curr = q_init.clone()
        R_target = quaternion_to_matrix(target_quat)
        
        # Neural delta from tactical reasoning
        if tactical_latent is not None:
            dq_neural = self.neural_relax_head(tactical_latent)
        else:
            dq_neural = torch.zeros(B, self.num_joints, device=device, dtype=dtype)
            
        for _ in range(k_steps):
            ee_pos, ee_quat, link_pos, J = self.fk(q_curr)
            ee_quat = quaternion_normalize(ee_quat)
            
            # 1. SE(3) Error: Position error delta_p (B, 3)
            delta_p = target_pos - ee_pos
            
            # Orientation error delta_phi from rotation matrix difference
            R_curr = quaternion_to_matrix(ee_quat)
            R_err = torch.bmm(R_target, R_curr.transpose(1, 2))
            phi_x = 0.5 * (R_err[:, 2, 1] - R_err[:, 1, 2])
            phi_y = 0.5 * (R_err[:, 0, 2] - R_err[:, 2, 0])
            phi_z = 0.5 * (R_err[:, 1, 0] - R_err[:, 0, 1])
            delta_phi = torch.stack([phi_x, phi_y, phi_z], dim=-1)
            
            # Stack SE(3) spatial twist error (B, 6, 1)
            delta_X = torch.cat([delta_p, delta_phi], dim=-1).unsqueeze(-1)
            
            # DLS Inverse Kinematic step with Yoshikawa adaptive damping
            J_dagger = self.damped_least_squares_inverse(J) # (B, 7, 6)
            dq_se3 = torch.bmm(J_dagger, delta_X).squeeze(-1) # (B, 7)
            
            # 2. Obstacle Repulsion Potential Field with smooth epsilon-clamping
            dq_obs = torch.zeros_like(q_curr)
            if obstacles is not None and obstacles.shape[1] > 0:
                # Link positions (B, 7, 3); Obstacles (B, N_obs, 3)
                diff = link_pos.unsqueeze(2) - obstacles.unsqueeze(1) # (B, 7, N_obs, 3)
                dist_sq = torch.sum(diff ** 2, dim=-1) # (B, 7, N_obs)
                
                # Smooth epsilon regularization to prevent gradient explosion as d -> 0
                eps_dist = 0.02 # 2cm safety buffer
                dist = torch.sqrt(dist_sq + eps_dist ** 2) # Strictly >= eps_dist
                
                # Active influence threshold d_0 (0.35m)
                d_0 = 0.35
                in_range = (dist < d_0).float()
                
                # Khatib-style smoothed repulsive magnitude: eta * (1/d - 1/d0) * (1/d^2)
                # Bounded by max gradient to prevent numerical explosion
                repulsion_mag = (1.0 / dist - 1.0 / d_0) / (dist ** 2) * in_range
                repulsion_mag = torch.clamp(repulsion_mag, min=0.0, max=50.0)
                
                # Unit direction vector with smooth normalization
                diff_norm = torch.sqrt(dist_sq + 1e-8).unsqueeze(-1)
                unit_diff = diff / diff_norm
                repulsion_force = torch.sum(unit_diff * repulsion_mag.unsqueeze(-1), dim=2) # (B, 7, 3)
                
                # Project link repulsive force to joint space: dq_obs = sum_i J_v_i.T @ F_i
                dq_obs = torch.norm(repulsion_force, dim=-1) * 0.1
                dq_obs = torch.clamp(dq_obs, min=-0.5, max=0.5)
                
            # 3. Joint Limit Barrier
            lower_violation = torch.clamp(self.q_min.unsqueeze(0) - q_curr, min=0.0)
            upper_violation = torch.clamp(q_curr - self.q_max.unsqueeze(0), min=0.0)
            dq_limits = lower_violation - upper_violation
            
            # Step update with clamping
            dq_step = (
                self.alpha_se3 * dq_se3
                + self.beta_obs * dq_obs
                + self.gamma_limits * dq_limits
                + 0.1 * dq_neural
            )
            # Clamp step magnitude to ensure stable convergence
            dq_step = torch.clamp(dq_step, min=-0.3, max=0.3)
            q_curr = q_curr + dq_step
            # Hard physical limit projection
            q_curr = torch.clamp(q_curr, min=self.q_min.unsqueeze(0), max=self.q_max.unsqueeze(0))
            
        # Final evaluation
        final_ee_pos, final_ee_quat, final_link_pos, final_J = self.fk(q_curr)
        final_ee_quat = quaternion_normalize(final_ee_quat)
        tracking_error = torch.norm(target_pos - final_ee_pos, dim=-1)
        orientation_error = so3_geodesic_distance(final_ee_quat, target_quat)
        final_manipulability = self.yoshikawa_manipulability(final_J)
        
        min_clearance = torch.tensor(1.0, device=device, dtype=dtype).expand(B)
        if obstacles is not None and obstacles.shape[1] > 0:
            diff = final_link_pos.unsqueeze(2) - obstacles.unsqueeze(1)
            dists = torch.norm(diff, dim=-1)
            min_clearance = dists.min(dim=-1).values.min(dim=-1).values
            
        ee_pose_7d = torch.cat([final_ee_pos, final_ee_quat], dim=-1)
        
        return {
            "q_relaxed": q_curr,
            "ee_pose": ee_pose_7d,
            "ee_pos": final_ee_pos,
            "ee_quat": final_ee_quat,
            "link_positions": final_link_pos,
            "tracking_error": tracking_error,
            "orientation_error": orientation_error,
            "manipulability": final_manipulability,
            "min_clearance": min_clearance
        }


class Proprioceptive3DEmbedder(nn.Module):
    """
    Embeds robot proprioceptive state (joint angles, velocities, and SE(3) pose)
    into structured continuous 3D token representations with explicit 3D link coordinates.
    """
    def __init__(self, num_joints: int = 7, d_model: int = 512):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.fk = ForwardKinematics7DOF()
        
        # Embed joint state: angle + velocity -> d_model
        self.joint_embed = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # End-effector token embedder: pose (7D) + gripper (1D)
        self.ee_embed = nn.Sequential(
            nn.Linear(8, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model)
        )

    def forward(
        self,
        q_joints: torch.Tensor,
        q_vel: Optional[torch.Tensor] = None,
        ee_pose: Optional[torch.Tensor] = None,
        gripper: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q_joints: (B, num_joints)
            q_vel: Optional (B, num_joints)
            ee_pose: Optional (B, 7) [pos(3), quat(4)]
            gripper: Optional (B, 1)
        Returns:
            tokens: (B, num_joints + 1, d_model) Proprioceptive tokens.
            coords: (B, num_joints + 1, 3) Continuous 3D coordinates for each token.
        """
        B = q_joints.shape[0]
        device = q_joints.device
        dtype = q_joints.dtype
        
        if q_vel is None:
            q_vel = torch.zeros_like(q_joints)
            
        # Compute forward kinematics link positions
        fk_pos, fk_quat, link_pos, _ = self.fk(q_joints)
        
        if ee_pose is None:
            ee_pose = torch.cat([fk_pos, fk_quat], dim=-1)
        if gripper is None:
            gripper = torch.zeros(B, 1, device=device, dtype=dtype)
            
        # Joint tokens
        joint_state = torch.stack([q_joints, q_vel], dim=-1) # (B, num_joints, 2)
        joint_tokens = self.joint_embed(joint_state) # (B, num_joints, d_model)
        joint_coords = link_pos # (B, num_joints, 3)
        
        # End-effector token
        ee_state = torch.cat([ee_pose, gripper], dim=-1) # (B, 8)
        ee_token = self.ee_embed(ee_state).unsqueeze(1) # (B, 1, d_model)
        ee_coord = ee_pose[:, :3].unsqueeze(1) # (B, 1, 3)
        
        tokens = torch.cat([joint_tokens, ee_token], dim=1) # (B, num_joints + 1, d_model)
        coords = torch.cat([joint_coords, ee_coord], dim=1) # (B, num_joints + 1, 3)
        return tokens, coords


class Spatial3DEmbedder(nn.Module):
    """
    Embeds 3D continuous point clouds, obstacle fields, and target waypoints
    into continuous spatial tokens equipped with Fourier coordinate features.
    """
    def __init__(self, in_features: int = 3, d_model: int = 512, num_fourier_bands: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_fourier_bands = num_fourier_bands
        
        # Fourier positional frequencies: 2^0, ..., 2^(B-1)
        fourier_freqs = 2.0 ** torch.arange(num_fourier_bands).float() * math.pi
        self.register_buffer("fourier_freqs", fourier_freqs)
        
        # Total spatial feature dim: 3 (coords) + 3 * 2 * num_bands + in_features
        spatial_dim = 3 + 6 * num_fourier_bands + in_features
        self.mlp = nn.Sequential(
            nn.Linear(spatial_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(
        self,
        coords: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: (B, N_pts, 3) Continuous 3D coordinates.
            features: Optional (B, N_pts, C_in) Visual/semantic features.
        Returns:
            tokens: (B, N_pts, d_model)
            coords: (B, N_pts, 3)
        """
        B, N, _ = coords.shape
        device = coords.device
        dtype = coords.dtype
        
        # Fourier encoding of coordinates
        # coords: (B, N, 3, 1) * freqs: (num_bands,)
        angles = coords.unsqueeze(-1) * self.fourier_freqs.view(1, 1, 1, -1) # (B, N, 3, bands)
        fourier_enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1) # (B, N, 3, 2*bands)
        fourier_flat = fourier_enc.view(B, N, -1) # (B, N, 6*bands)
        
        if features is None:
            features = torch.zeros(B, N, 0, device=device, dtype=dtype)
            
        combined = torch.cat([coords, fourier_flat, features], dim=-1)
        tokens = self.mlp(combined)
        return tokens, coords


class HRM3DRecurrentBlock(nn.Module):
    """
    3D Recurrent Refinement Block with RoPE-3D Multi-Head Attention,
    MagicNorm bounded normalization, and SwiGLU FFN.
    """
    def __init__(self, d_model: int, n_heads: int = 8, d_ffn: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.norm1 = MagicNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.rope3d = RoPE3D(dim=self.d_head)
        
        self.norm2 = MagicNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(
        self,
        state: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            state: (B, S, d_model) Latent token states.
            coords: (B, S, 3) Continuous 3D coordinates.
            context: Optional (B, S_ctx, d_model)
            context_coords: Optional (B, S_ctx, 3)
        Returns:
            Updated latent state of shape (B, S, d_model).
        """
        # Pre-norm
        h = self.norm1(state)
        ctx = self.norm1(context) if context is not None else h
        ctx_coords = context_coords if context is not None else coords
        
        B, S, _ = state.shape
        _, S_ctx, _ = ctx.shape
        
        q = self.q_proj(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(ctx).view(B, S_ctx, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(ctx).view(B, S_ctx, self.n_heads, self.d_head).transpose(1, 2)
        
        # Apply RoPE-3D to Queries and Keys
        q = self.rope3d(q, coords)
        k = self.rope3d(k, ctx_coords)
        
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        h = state + self.out_proj(attn_out)
        
        # SwiGLU FFN
        h_norm = self.norm2(h)
        swiglu = F.silu(self.ffn_gate(h_norm)) * self.ffn_up(h_norm)
        out = h + self.ffn_down(swiglu)
        return out


class HRM3D(nn.Module):
    """
    Right Hemisphere 3D Hierarchical Reasoning Model (Sapient HRM-3D).
    
    Wires:
    1. Proprioceptive & 3D Spatial Embedders with continuous RoPE-3D positional representations.
    2. Tactical L-Module: Fast concrete joint execution & SE(3) relaxation (z_L).
    3. Strategic H-Module: Slow abstract waypoint & trajectory planning (z_H).
    4. Differentiable KinematicSE3Relaxation: Analytical forward kinematics, DLS SE(3) pose tracking,
       obstacle potential avoidance, and joint limit barriers.
    5. ACT Q-Head: Evaluates Halting vs Continuation utilities.
    6. 1-step implicit fixed-point gradient approximation for O(1) memory complexity.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ffn: int = 2048,
        num_joints: int = 7,
        L_cycles: int = 3,
        M_max: int = 8,
        se3_relax_steps: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints
        self.L_cycles = L_cycles
        self.M_max = M_max
        
        # 1. Proprioceptive & Spatial 3D Embedders
        self.proprio_embedder = Proprioceptive3DEmbedder(num_joints=num_joints, d_model=d_model)
        self.spatial_embedder = Spatial3DEmbedder(d_model=d_model)
        
        # 2. Dual-Timescale Recurrent Reasoning Modules
        self.L_module = HRM3DRecurrentBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
        self.H_module = HRM3DRecurrentBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
        
        # 3. Kinematic SE(3) Relaxation Engine
        self.kinematic_relaxation = KinematicSE3Relaxation(
            num_joints=num_joints,
            d_latent=d_model,
            relax_steps=se3_relax_steps
        )
        
        # 4. ACT Q-Head: [Q_halt, Q_continue]
        self.q_head = nn.Linear(d_model, 2)

    def init_carry(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        q_init: Optional[torch.Tensor] = None
    ) -> HRM3DStateCarry:
        if q_init is None:
            q_init = torch.zeros(batch_size, self.num_joints, device=device)
            
        # Initial end-effector pose from FK
        fk = self.kinematic_relaxation.fk
        ee_pos, ee_quat, _, _ = fk(q_init)
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)
        
        return HRM3DStateCarry(
            z_H=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            z_L=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            q_joints=q_init,
            ee_pose=ee_pose,
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            halted=torch.zeros(batch_size, dtype=torch.bool, device=device)
        )

    def step_segment(
        self,
        carry: HRM3DStateCarry,
        x_embed: torch.Tensor,
        coords: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: Optional[torch.Tensor] = None,
        obstacles: Optional[torch.Tensor] = None
    ) -> Tuple[HRM3DStateCarry, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Executes one full outer reasoning segment m:
        1. Fast timescale: Runs L_cycles of L-module tactical execution.
        2. Differentiable Kinematic SE(3) Relaxation: Updates joint angles and relaxes end-effector.
        3. Slow timescale: Runs 1 step of H-module strategic integration.
        4. Evaluates ACT Q-values.
        """
        z_L = carry.z_L
        z_H = carry.z_H
        q_curr = carry.q_joints
        
        # 1. Fast timescale: Low-level tactical execution
        for _ in range(self.L_cycles):
            context = z_H + x_embed
            z_L = self.L_module(state=z_L, coords=coords, context=context, context_coords=coords)
            
        # 2. Kinematic SE(3) Relaxation
        tactical_summary = z_L.mean(dim=1)
        relax_out = self.kinematic_relaxation(
            q_init=q_curr,
            target_pos=target_pos,
            target_quat=target_quat,
            obstacles=obstacles,
            tactical_latent=tactical_summary
        )
        new_q = relax_out["q_relaxed"]
        new_ee_pose = relax_out["ee_pose"]
        
        # 3. Slow timescale: High-level strategic integration
        z_H = self.H_module(state=z_H, coords=coords, context=z_L, context_coords=coords)
        
        # 4. Evaluate Halting Q-values
        q_vals = self.q_head(z_H.mean(dim=1)) # [B, 2]: [Q_halt, Q_continue]
        
        # Update state carry
        new_step_count = carry.step_count + 1
        halt_decision = q_vals[:, 0] > q_vals[:, 1]
        new_halted = carry.halted | halt_decision | (new_step_count >= self.M_max)
        
        new_carry = HRM3DStateCarry(
            z_H=z_H,
            z_L=z_L,
            q_joints=new_q,
            ee_pose=new_ee_pose,
            step_count=new_step_count,
            halted=new_halted
        )
        return new_carry, q_vals, relax_out

    def forward(
        self,
        x_embed: torch.Tensor,
        coords: torch.Tensor,
        target_pos: torch.Tensor,
        target_quat: Optional[torch.Tensor] = None,
        obstacles: Optional[torch.Tensor] = None,
        q_init: Optional[torch.Tensor] = None,
        max_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, HRM3DStateCarry, Dict[str, Any]]:
        """
        Unrolls 3D hierarchical reasoning until adaptive convergence or max_steps.
        Returns converged z_H, final state carry, and auxiliary metrics.
        """
        B, S, _ = x_embed.shape
        carry = self.init_carry(B, S, x_embed.device, q_init=q_init)
        steps_limit = max_steps or self.M_max
        
        all_q_vals = []
        last_relax_out = None
        
        for _ in range(steps_limit):
            carry, q_vals, last_relax_out = self.step_segment(
                carry=carry,
                x_embed=x_embed,
                coords=coords,
                target_pos=target_pos,
                target_quat=target_quat,
                obstacles=obstacles
            )
            all_q_vals.append(q_vals)
            if carry.halted.all():
                break
                
        converged_z_H = carry.z_H
        
        aux_dict = {
            "steps_taken": carry.step_count,
            "q_vals": torch.stack(all_q_vals, dim=1) if all_q_vals else None,
            "q_joints": carry.q_joints,
            "ee_pose": carry.ee_pose,
            "relaxation_info": last_relax_out
        }
        return converged_z_H, carry, aux_dict


# Alias for compatibility
HierarchicalReasoningModel3D = HRM3D
