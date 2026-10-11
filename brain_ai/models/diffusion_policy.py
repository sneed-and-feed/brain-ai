"""
brain_ai.models.diffusion_policy: 1D Temporal CNN Diffusion Policy for Robotics Manipulation.

Implements the 1D Temporal Convolutional Residual Network (U-Net with 1D residual blocks,
dilation, and FiLM conditioning) following Chi et al. (RSS 2023, "Diffusion Policy:
Visuomotor Policy Learning via Action Diffusion") and Hugging Face LeRobot.

Features:
1. Observation Space: 21-dim proprioceptive state:
   - Franka joint positions joint_q (7)
   - Franka joint velocities joint_dq (7)
   - End-effector Cartesian position ee_pos (3)
   - Target goal position target_pos (3)
   - Gripper width gripper_width (1)
   Total: 7 + 7 + 3 + 3 + 1 = 21 dimensions.
2. Action Space: 7-dim Franka joint velocities dq in R^7 clamped to safe kinematic limits.
3. Architecture:
   - 1D Temporal Convolutional Residual Network (ConditionalUnet1D) with dilation and FiLM conditioning.
   - Conditioning: observation embedding via MLP + sinusoidal diffusion timestep embedding.
   - DDPM linear and cosine (squaredcos_cap_v2) beta schedules predicting epsilon (noise).
   - DDIM Fast Sampler (Song et al., 2020) supporting K=8 to 16 denoising steps at inference.
   - Receding Horizon: Prediction horizon T_p = 16 actions, execution horizon T_a = 8 actions.
4. Policy Class:
   - DiffusionRoboticsPolicy(BaseExecutionPolicy) compatible with FrankaKinematicEnv.
   - Maintains an action chunk buffer: replenishes via DDIM denoising when empty or after T_a steps,
     pops actions with low latency between replanning intervals, and logs telemetry.
5. Training & Checkpointing:
   - train_diffusion_policy helper using AdamW and MSE loss.
   - save_diffusion_checkpoint and load_diffusion_checkpoint methods.
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from abc import ABC, abstractmethod

# Franka standard joint velocity limits for action clamping
MAX_JOINT_VELOCITY = torch.tensor(
    [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100],
    dtype=torch.float32
)


class BaseExecutionPolicy(ABC):
    """Abstract base execution policy for robotics manipulation."""

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def compute_action(self, obs: Dict[str, Any], t: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pass


# ==============================================================================
# 1. OBSERVATION PARSER & TENSOR EXTRACTION
# ==============================================================================

def extract_obs_tensor(
    obs: Union[Dict[str, Any], torch.Tensor],
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Extracts and concatenates the unified 21-dimensional proprioceptive observation
    from FrankaKinematicEnv dictionary or raw tensor.

    Observation components (21 dimensions):
      - joint_q (7): Joint angles
      - joint_dq (7): Joint velocities
      - ee_pos (3): End-effector 3D position
      - target_pos (3): Target goal position
      - gripper_width (1): Gripper opening distance

    Args:
        obs: Dict from FrankaKinematicEnv or Tensor of shape [..., 21]
        device: Target torch device (optional)
    Returns:
        Tensor of shape [21] or [B, 21] on specified device.
    """
    if isinstance(obs, torch.Tensor):
        if device is not None:
            return obs.to(device=device, dtype=torch.float32)
        return obs.to(dtype=torch.float32)

    if not isinstance(obs, dict):
        raise TypeError(f"Expected obs to be dict or torch.Tensor, got {type(obs)}")

    # Extract components
    joint_q = obs["joint_q"]
    if not isinstance(joint_q, torch.Tensor):
        joint_q = torch.tensor(joint_q, dtype=torch.float32)

    joint_dq = obs.get("joint_dq")
    if joint_dq is None:
        joint_dq = torch.zeros_like(joint_q)
    elif not isinstance(joint_dq, torch.Tensor):
        joint_dq = torch.tensor(joint_dq, dtype=torch.float32)

    ee_pos = obs.get("ee_pos")
    if ee_pos is None:
        from brain_ai.tasks.robotics_sandbox import FrankaKinematics
        p_ee, _, _ = FrankaKinematics.forward_kinematics(joint_q)
        ee_pos = p_ee
    elif not isinstance(ee_pos, torch.Tensor):
        ee_pos = torch.tensor(ee_pos, dtype=torch.float32)

    target_pos = obs.get("target_pos")
    if target_pos is None:
        target_pos = torch.zeros(3, dtype=torch.float32, device=joint_q.device)
    elif not isinstance(target_pos, torch.Tensor):
        target_pos = torch.tensor(target_pos, dtype=torch.float32)

    gripper_val = obs.get("gripper_width", 0.08)
    if isinstance(gripper_val, (int, float)):
        gripper_w = torch.tensor([gripper_val], dtype=torch.float32, device=joint_q.device)
    elif isinstance(gripper_val, torch.Tensor):
        gripper_w = gripper_val.view(-1).to(device=joint_q.device, dtype=torch.float32)
        if gripper_w.numel() == 0:
            gripper_w = torch.tensor([0.08], dtype=torch.float32, device=joint_q.device)
        elif gripper_w.numel() > 1:
            gripper_w = gripper_w[:1]
    else:
        gripper_w = torch.tensor([0.08], dtype=torch.float32, device=joint_q.device)

    # Handle batched vs unbatched
    if joint_q.dim() == 1:
        res = torch.cat([
            joint_q.view(7),
            joint_dq.view(7),
            ee_pos.view(3),
            target_pos.view(3),
            gripper_w.view(1)
        ], dim=-1)
    else:
        batch_size = joint_q.shape[0]
        if gripper_w.shape[0] != batch_size:
            gripper_w = gripper_w.expand(batch_size, 1)
        elif gripper_w.dim() == 1:
            gripper_w = gripper_w.unsqueeze(-1)

        res = torch.cat([
            joint_q.view(batch_size, 7),
            joint_dq.view(batch_size, 7),
            ee_pos.view(batch_size, 3),
            target_pos.view(batch_size, 3),
            gripper_w.view(batch_size, 1)
        ], dim=-1)

    if device is not None:
        res = res.to(device=device)
    return res


# ==============================================================================
# 2. NEURAL NETWORK LAYERS: SINUSOIDAL EMBEDDING & 1D RESIDUAL BLOCKS
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps (Vaswani et al. / Ho et al. 2020).
    Maps scalar timesteps [B] to dense harmonic features [B, dim].
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def get_group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    """
    Constructs a GroupNorm layer ensuring channels are evenly divisible by number of groups.
    """
    g = preferred_groups
    while g > 1 and channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, channels)


class Conv1dBlock(nn.Module):
    """
    Basic 1D Convolutional block: Conv1d -> GroupNorm -> Mish activation.
    Preserves sequence length via symmetric padding: padding = (kernel_size - 1) * dilation // 2.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        dilation: int = 1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm = get_group_norm(out_channels, n_groups)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ConditionalResidualBlock1D(nn.Module):
    """
    1D Temporal Residual Block with FiLM (Feature-wise Linear Modulation) conditioning
    and configurable temporal dilation following LeRobot and Chi et al. (RSS 2023).

    Architecture:
        x -> Conv1d -> GroupNorm -> FiLM(cond) -> Mish -> Conv1d -> GroupNorm -> Mish + Residual
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        dilation: int = 1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm1 = get_group_norm(out_channels, n_groups)

        # FiLM projection: outputs scale (gamma) and bias (beta) for each channel
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, 2 * out_channels)
        )
        # Small weight initialization for smooth identity-near start
        nn.init.zeros_(self.cond_encoder[-1].weight)
        nn.init.zeros_(self.cond_encoder[-1].bias)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.norm2 = get_group_norm(out_channels, n_groups)
        self.act1 = nn.Mish()
        self.act2 = nn.Mish()

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input feature map [B, in_channels, L]
            cond: Conditioning vector [B, cond_dim]
        Returns:
            Output feature map [B, out_channels, L]
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # FiLM modulation: h = (1 + scale) * h + bias
        embed = self.cond_encoder(cond).unsqueeze(-1)  # [B, 2 * out_channels, 1]
        scale, bias = torch.chunk(embed, 2, dim=1)
        h = (1.0 + scale) * h + bias
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.residual_conv(x)


# ==============================================================================
# 3. 1D TEMPORAL RESIDUAL U-NET (CONDITIONAL UNET 1D)
# ==============================================================================

class ConditionalUnet1D(nn.Module):
    """
    1D Temporal Convolutional Residual U-Net for action trajectory diffusion.

    Takes noisy action trajectory [B, T_p, act_dim], diffusion timestep t [B],
    and 21-dim observation conditioning vector [B, obs_dim], predicting the
    injected noise epsilon [B, T_p, act_dim].

    Architecture:
      - Temporal U-Net with downsampling encoder, bottleneck, and upsampling decoder.
      - Skip connections between symmetrical encoder-decoder stages.
      - Multi-dilation residual blocks to capture multi-scale temporal dynamics.
      - FiLM conditioning at every residual block from joint (time_emb, obs_emb).
    """
    def __init__(
        self,
        act_dim: int = 7,
        obs_dim: int = 21,
        pred_horizon: int = 16,
        down_dims: Tuple[int, ...] = (64, 128, 256),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_dim: int = 128,
        time_dim: int = 64,
        dilations: Tuple[int, ...] = (1, 2, 4)
    ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.pred_horizon = pred_horizon
        self.down_dims = list(down_dims)
        self.cond_dim = cond_dim
        self.dilations = list(dilations)

        # 1. Conditioning Embeddings: Time + Observation
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.obs_emb = nn.Sequential(
            nn.Linear(obs_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.cond_fusion = nn.Sequential(
            nn.Linear(cond_dim * 2, cond_dim),
            nn.Mish()
        )

        # 2. Input Projection
        self.in_conv = nn.Conv1d(act_dim, self.down_dims[0], kernel_size=1)

        # 3. Downsampling Encoder
        num_stages = len(self.down_dims)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(num_stages - 1):
            in_c = self.down_dims[i - 1] if i > 0 else self.down_dims[0]
            out_c = self.down_dims[i]
            d = self.dilations[i] if i < len(self.dilations) else 1
            block1 = ConditionalResidualBlock1D(in_c, out_c, cond_dim, kernel_size, n_groups, dilation=d)
            block2 = ConditionalResidualBlock1D(out_c, out_c, cond_dim, kernel_size, n_groups, dilation=d)
            self.down_blocks.append(nn.ModuleList([block1, block2]))
            self.downsamples.append(nn.Conv1d(out_c, out_c, kernel_size=3, stride=2, padding=1))

        # 4. Bottleneck / Mid Blocks
        mid_in = self.down_dims[-2] if num_stages > 1 else self.down_dims[0]
        mid_out = self.down_dims[-1]
        mid_d = self.dilations[-1] if self.dilations else 1
        self.mid_block1 = ConditionalResidualBlock1D(mid_in, mid_out, cond_dim, kernel_size, n_groups, dilation=mid_d)
        self.mid_block2 = ConditionalResidualBlock1D(mid_out, mid_out, cond_dim, kernel_size, n_groups, dilation=mid_d)

        # 5. Upsampling Decoder
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_stages - 1)):
            in_c = self.down_dims[i + 1]
            out_c = self.down_dims[i]
            d = self.dilations[i] if i < len(self.dilations) else 1
            self.upsamples.append(nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1))
            block1 = ConditionalResidualBlock1D(out_c * 2, out_c, cond_dim, kernel_size, n_groups, dilation=d)
            block2 = ConditionalResidualBlock1D(out_c, out_c, cond_dim, kernel_size, n_groups, dilation=d)
            self.up_blocks.append(nn.ModuleList([block1, block2]))

        # 6. Output Projection
        self.out_conv = nn.Sequential(
            nn.Mish(),
            nn.Conv1d(self.down_dims[0], act_dim, kernel_size=1)
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sample: Action trajectory [B, T_p, act_dim] or [B, act_dim, T_p]
            timestep: Diffusion steps [B] or scalar
            global_cond: Observation conditioning [B, obs_dim]
        Returns:
            Predicted noise [B, T_p, act_dim]
        """
        # Ensure sample is [B, act_dim, T_p] for 1D convolutions
        if sample.shape[-1] == self.act_dim:
            x = sample.transpose(1, 2)
            transposed = True
        else:
            x = sample
            transposed = False

        batch_size = x.shape[0]

        # Ensure timestep is tensor of shape [B]
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=x.device, dtype=torch.long)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size)
        elif timestep.dim() == 1 and timestep.shape[0] != batch_size:
            timestep = timestep.expand(batch_size)

        # Observation embedding
        if global_cond.dim() == 1:
            global_cond = global_cond.unsqueeze(0).expand(batch_size, -1)

        # Compute joint FiLM conditioning vector
        t_emb = self.time_emb(timestep.float())
        o_emb = self.obs_emb(global_cond)
        cond = self.cond_fusion(torch.cat([t_emb, o_emb], dim=-1))

        # Forward through U-Net
        h = self.in_conv(x)
        skips = []

        # Down stages
        for (b1, b2), down in zip(self.down_blocks, self.downsamples):
            h = b1(h, cond)
            h = b2(h, cond)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_block2(h, cond)

        # Up stages
        for up, (b1, b2) in zip(self.upsamples, self.up_blocks):
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = b1(h, cond)
            h = b2(h, cond)

        out = self.out_conv(h)

        if transposed:
            out = out.transpose(1, 2)
        return out


# ==============================================================================
# 4. DIFFUSION MODEL (DDPM NOISE PREDICTION & DDIM FAST SAMPLER)
# ==============================================================================

class DiffusionModel1D(nn.Module):
    """
    1D Temporal Diffusion Model for robotics action sequence generation.
    Supports forward diffusion training (DDPM) and fast reverse sampling (DDIM).
    """
    def __init__(
        self,
        unet: Optional[ConditionalUnet1D] = None,
        obs_dim: int = 21,
        act_dim: int = 7,
        pred_horizon: int = 16,
        num_train_timesteps: int = 100,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        clip_val: float = 2.6,
        **unet_kwargs
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.clip_sample = clip_sample
        self.clip_val = clip_val

        if unet is None:
            self.unet = ConditionalUnet1D(
                act_dim=act_dim,
                obs_dim=obs_dim,
                pred_horizon=pred_horizon,
                **unet_kwargs
            )
        else:
            self.unet = unet

        # Construct beta schedule
        if beta_schedule == "squaredcos_cap_v2":
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = torch.clip(1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)
        elif beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        obs: Union[torch.Tensor, Dict[str, Any]]
    ) -> torch.Tensor:
        """Forward pass predicting noise epsilon."""
        obs_tensor = extract_obs_tensor(obs, device=noisy_actions.device)
        return self.unet(noisy_actions, timesteps, obs_tensor)

    def compute_loss(
        self,
        actions: torch.Tensor,
        obs: Union[torch.Tensor, Dict[str, Any]]
    ) -> torch.Tensor:
        """
        Computes standard DDPM MSE loss on noise prediction.

        Args:
            actions: Clean demonstration actions [B, T_p, act_dim] or [T_p, act_dim]
            obs: Observations [B, obs_dim] or Dict
        Returns:
            Scalar MSE loss.
        """
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)

        device = self.device
        actions = actions.to(device=device, dtype=torch.float32)
        obs_tensor = extract_obs_tensor(obs, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        batch_size = actions.shape[0]

        # Sample uniform random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (batch_size,),
            device=device, dtype=torch.long
        )

        # Sample standard Gaussian noise
        noise = torch.randn_like(actions)

        # Forward diffusion q(x_t | x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(batch_size, 1, 1)
        noisy_actions = sqrt_alpha * actions + sqrt_one_minus_alpha * noise

        # Model predicts noise
        pred_noise = self.unet(noisy_actions, timesteps, obs_tensor)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        num_inference_steps: int = 10,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Fast DDIM reverse diffusion sampler (Song et al., 2020).
        Runs K denoising steps (default K=10, supporting K=8 to 16).

        Args:
            obs: Observation tensor [B, obs_dim] or Dict
            num_inference_steps: Number of DDIM denoising steps K (default 10)
            eta: DDIM stochasticity parameter (0.0 = deterministic)
            generator: Optional torch random generator
        Returns:
            Denoised actions [B, T_p, act_dim]
        """
        device = self.device
        obs_tensor = extract_obs_tensor(obs, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        batch_size = obs_tensor.shape[0]

        # Initial pure Gaussian noise
        shape = (batch_size, self.pred_horizon, self.act_dim)
        x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

        # Sub-sequence of timesteps
        K = max(int(num_inference_steps), 2)
        step_ratio = (self.num_train_timesteps - 1) / (K - 1)
        timesteps = [int(round(i * step_ratio)) for i in range(K)]
        timesteps = list(reversed(timesteps))  # E.g. [99, 88, ..., 0]

        for i in range(K):
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i + 1 < K else -1
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise
            pred_noise = self.unet(x, t_tensor, obs_tensor)

            # Cumulative alpha factors
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

            # Predicted clean sample x_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
            if self.clip_sample:
                pred_x0 = torch.clamp(pred_x0, -self.clip_val, self.clip_val)

            # DDIM update step
            if eta > 0.0 and t_prev >= 0:
                sigma_t = eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
                )
                noise = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            else:
                sigma_t = 0.0
                noise = 0.0

            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * pred_noise

            if t_prev >= 0:
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise
            else:
                x = pred_x0

        return x

    def sample(
        self,
        obs: Union[torch.Tensor, Dict[str, Any]],
        num_inference_steps: int = 10,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Alias to ddim_sample."""
        return self.ddim_sample(obs, num_inference_steps=num_inference_steps, eta=eta, generator=generator)

    def save_checkpoint(self, path: Union[str, Path]) -> str:
        return save_diffusion_checkpoint(self, path)

    def load_checkpoint(self, path: Union[str, Path]):
        return load_diffusion_checkpoint(self, path)


# ==============================================================================
# 5. DIFFUSION ROBOTICS POLICY (RECEDING HORIZON ACTION CHUNKING)
# ==============================================================================

class DiffusionRoboticsPolicy(BaseExecutionPolicy):
    """
    1D Temporal CNN Diffusion Robotics Policy for the Franka Emika Panda.
    Compatible with FrankaKinematicEnv and BaseExecutionPolicy interface.

    Receding Horizon Mechanism:
      - Generates an action sequence of prediction horizon T_p = 16 via DDIM denoising.
      - Maintains an internal action chunk buffer.
      - Executes actions sequentially for execution horizon T_a = 8 steps.
      - When buffer is empty or T_a is reached, re-runs DDIM denoising on the current observation.
      - Simulates and records decision latency (~30-100ms, default 50ms on replanning, <1ms on cache hit).
    """
    def __init__(
        self,
        model: Optional[DiffusionModel1D] = None,
        obs_dim: int = 21,
        act_dim: int = 7,
        pred_horizon: int = 16,
        exec_horizon: int = 8,
        num_ddim_steps: int = 10,
        simulated_latency_ms: float = 50.0,
        device: Union[str, torch.device] = "cpu",
        **model_kwargs
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.pred_horizon = pred_horizon
        self.exec_horizon = exec_horizon
        self.num_ddim_steps = num_ddim_steps
        self.simulated_latency_ms = simulated_latency_ms
        self.device = torch.device(device)

        if model is None:
            self.model = DiffusionModel1D(
                obs_dim=obs_dim,
                act_dim=act_dim,
                pred_horizon=pred_horizon,
                **model_kwargs
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        self.action_buffer: List[torch.Tensor] = []
        self.exec_step_count: int = 0
        self.last_decision_latency_ms: float = 0.0

    def reset(self):
        """Resets the internal receding horizon action buffer."""
        self.action_buffer.clear()
        self.exec_step_count = 0
        self.last_decision_latency_ms = 0.0

    def clear_buffer(self):
        """Forces immediate re-planning on the next step."""
        self.action_buffer.clear()
        self.exec_step_count = 0

    def compute_action(
        self,
        obs: Dict[str, Any],
        t: float
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes joint velocity action command dq [7] and telemetry.

        Args:
            obs: Observation dictionary from FrankaKinematicEnv
            t: Simulation time (s)
        Returns:
            action: Joint velocity command dq [7] in rad/s
            info: Telemetry metadata dictionary
        """
        t0 = time.perf_counter()
        replan_triggered = False

        # Check if action buffer needs replenishment
        if len(self.action_buffer) == 0 or self.exec_step_count >= self.exec_horizon:
            replan_triggered = True
            self.model.eval()

            with torch.no_grad():
                obs_tensor = extract_obs_tensor(obs, device=self.device)
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)  # [1, 21]

                # Run fast DDIM reverse diffusion
                pred_actions = self.model.ddim_sample(
                    obs=obs_tensor,
                    num_inference_steps=self.num_ddim_steps
                )  # [1, 16, 7]

                pred_chunk = pred_actions[0]  # [16, 7]

                # Incorporate replan-time obstacle evasion into the planned chunk
                if isinstance(obs, dict):
                    obstacles = obs.get("obstacles", [])
                    p_ee = obs.get("ee_pos")
                    J = obs.get("jacobian")
                    if p_ee is not None and J is not None and len(obstacles) > 0:
                        from brain_ai.tasks.robotics_sandbox import FrankaKinematics
                        J_pinv = FrankaKinematics.damped_least_squares_pinv(J)
                        composite_dodge = torch.zeros(3, dtype=torch.float32, device=p_ee.device)
                        threat_found = False

                        for o in obstacles:
                            if not o.get("is_active", True):
                                continue
                            o_pos = o["position"]
                            rel_pos = p_ee - o_pos
                            dist = float(torch.norm(rel_pos).item())
                            surface_dist = dist - o.get("radius", 0.05)

                            v_obs = o.get("velocity", torch.zeros(3, device=p_ee.device))
                            v_app = - float(torch.dot(v_obs, rel_pos / (dist + 1e-6)).item()) if dist > 1e-4 else 0.0

                            if surface_dist < 0.22 or (v_app > 0.05 and surface_dist < 0.28):
                                threat_found = True
                                repulse_dir = rel_pos / max(dist, 1e-4)
                                dodge = repulse_dir + torch.tensor([0.0, 0.0, 1.4], dtype=torch.float32, device=p_ee.device)
                                dodge = dodge / (torch.norm(dodge) + 1e-6)
                                w = float(math.exp(-4.0 * max(surface_dist, 0.0)))
                                composite_dodge = composite_dodge + w * dodge

                        if threat_found and float(torch.norm(composite_dodge).item()) > 1e-4:
                            composite_dodge = composite_dodge / (torch.norm(composite_dodge) + 1e-6)
                            v_evade = 1.4 * composite_dodge
                            dq_evade = J_pinv @ v_evade
                            dq_evade_dev = dq_evade.to(pred_chunk.device)
                            for k in range(pred_chunk.shape[0]):
                                decay = 0.95 ** k
                                pred_chunk[k] = pred_chunk[k] + dq_evade_dev * decay

                # Populate action buffer up to execution horizon T_a = 8
                num_to_buffer = min(self.exec_horizon, pred_chunk.shape[0])
                self.action_buffer = [pred_chunk[i].cpu() for i in range(num_to_buffer)]
                self.exec_step_count = 0

            denoise_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            # Simulating / measuring decision latency (~30-100ms, default 50ms)
            if self.simulated_latency_ms > 0:
                self.last_decision_latency_ms = float(self.simulated_latency_ms + denoise_elapsed_ms)
            else:
                self.last_decision_latency_ms = float(denoise_elapsed_ms)
        else:
            # Stored action retrieval overhead (< 1ms)
            self.last_decision_latency_ms = 0.5

        # Pop next action from receding horizon buffer
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop(0)
            self.exec_step_count += 1
        else:
            action = torch.zeros(self.act_dim, dtype=torch.float32)

        # Enforce Franka kinematic velocity limits
        max_vel = MAX_JOINT_VELOCITY.to(action.device)
        if action.shape[0] == 7:
            action = torch.clamp(action, -max_vel, max_vel)
        elif action.shape[0] > 7:
            action = action.clone()
            action[:7] = torch.clamp(action[:7], -max_vel, max_vel)
            action[7:] = torch.clamp(action[7:], 0.0, 0.08)

        telemetry = {
            "policy": "DiffusionRoboticsPolicy",
            "decision_latency_ms": self.last_decision_latency_ms,
            "replan_triggered": replan_triggered,
            "buffer_remaining": len(self.action_buffer),
            "exec_step": self.exec_step_count,
            "exec_horizon": self.exec_horizon,
            "pred_horizon": self.pred_horizon
        }

        return action, telemetry

    def train_policy(
        self,
        dataset: Any,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-4,
        **kwargs
    ) -> List[float]:
        """Trains the underlying diffusion model."""
        return train_diffusion_policy(self, dataset, epochs=epochs, batch_size=batch_size, lr=lr, **kwargs)

    def save_checkpoint(self, path: Union[str, Path]) -> str:
        return save_diffusion_checkpoint(self, path)

    def load_checkpoint(self, path: Union[str, Path]):
        return load_diffusion_checkpoint(self, path)


# ==============================================================================
# 6. TRAINING HELPER & CHECKPOINTING
# ==============================================================================

def train_diffusion_policy(
    policy: Union[DiffusionRoboticsPolicy, DiffusionModel1D],
    dataset: Any,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    verbose: bool = False
) -> List[float]:
    """
    Trains the 1D Temporal CNN Diffusion Policy on demonstration data using AdamW and MSE loss.

    Args:
        policy: DiffusionRoboticsPolicy or DiffusionModel1D instance
        dataset: Dataset containing (obs, actions) pairs, or Dict with 'obs' and 'actions'
        epochs: Number of training epochs
        batch_size: Batch size for DataLoader
        lr: Learning rate for AdamW
        device: Target training device
        weight_decay: L2 regularization factor
        grad_clip: Maximum gradient norm for clipping
        verbose: If True, prints loss at each epoch
    Returns:
        List of average training losses per epoch.
    """
    if isinstance(policy, DiffusionRoboticsPolicy):
        model = policy.model
    elif isinstance(policy, DiffusionModel1D):
        model = policy
    else:
        raise TypeError(f"Unsupported policy type: {type(policy)}")

    if device is not None:
        model = model.to(torch.device(device))
    target_device = model.device

    # Prepare DataLoader
    if isinstance(dataset, dict):
        obs_data = dataset.get("obs", dataset.get("observations", dataset.get("observation", None)))
        act_data = dataset.get("actions", dataset.get("action", None))
        ep_lens = dataset.get("episode_lengths", None)
        if obs_data is None or act_data is None:
            raise ValueError("Dataset dictionary must contain 'obs' and 'actions' keys.")
        if not isinstance(obs_data, torch.Tensor):
            obs_data = torch.tensor(obs_data, dtype=torch.float32)
        if not isinstance(act_data, torch.Tensor):
            act_data = torch.tensor(act_data, dtype=torch.float32)

        # If act_data is [N, act_dim], chunk into [M, pred_horizon, act_dim]
        if act_data.dim() == 2 and act_data.shape[1] == model.act_dim:
            pred_h = model.pred_horizon
            chunk_obs, chunk_acts = [], []
            if ep_lens is not None:
                curr_idx = 0
                for el in ep_lens:
                    for s in range(0, el - pred_h + 1, 2):
                        chunk_obs.append(obs_data[curr_idx + s])
                        chunk_acts.append(act_data[curr_idx + s : curr_idx + s + pred_h])
                    curr_idx += el
            else:
                for s in range(0, len(act_data) - pred_h + 1, 2):
                    chunk_obs.append(obs_data[s])
                    chunk_acts.append(act_data[s : s + pred_h])
            obs_data = torch.stack(chunk_obs)
            act_data = torch.stack(chunk_acts)

        torch_dataset = TensorDataset(obs_data, act_data)
        loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    elif isinstance(dataset, tuple) and len(dataset) == 2:
        d1, d2 = dataset
        if not isinstance(d1, torch.Tensor):
            d1 = torch.tensor(d1, dtype=torch.float32)
        if not isinstance(d2, torch.Tensor):
            d2 = torch.tensor(d2, dtype=torch.float32)
        if d1.shape[-1] == 21 or (d1.dim() > 1 and d1.shape[-1] != 7):
            obs_data, act_data = d1, d2
        else:
            act_data, obs_data = d1, d2
        torch_dataset = TensorDataset(obs_data, act_data)
        loader = DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    elif isinstance(dataset, Dataset):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history: List[float] = []

    model.train()
    for epoch in range(epochs):
        epoch_losses: List[float] = []
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                b1, b2 = batch[0], batch[1]
                # Determine which item is obs and which is action
                if b1.dim() == 3 and b1.shape[-1] == model.act_dim:
                    b_actions, b_obs = b1, b2
                else:
                    b_obs, b_actions = b1, b2
            elif isinstance(batch, dict):
                b_obs = batch.get("obs", batch.get("observations"))
                b_actions = batch.get("actions", batch.get("action"))
            else:
                raise ValueError(f"Unrecognized batch type: {type(batch)}")

            b_obs = extract_obs_tensor(b_obs, device=target_device)
            b_actions = b_actions.to(device=target_device, dtype=torch.float32)

            optimizer.zero_grad()
            loss = model.compute_loss(b_actions, b_obs)
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        loss_history.append(mean_loss)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {mean_loss:.5f}")

    return loss_history


def save_diffusion_checkpoint(
    policy: Union[DiffusionRoboticsPolicy, DiffusionModel1D],
    filepath: Union[str, Path]
) -> str:
    """Saves model weights and configuration to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(policy, DiffusionRoboticsPolicy):
        state = {
            "type": "DiffusionRoboticsPolicy",
            "model_state_dict": policy.model.state_dict(),
            "policy_config": {
                "obs_dim": policy.obs_dim,
                "act_dim": policy.act_dim,
                "pred_horizon": policy.pred_horizon,
                "exec_horizon": policy.exec_horizon,
                "num_ddim_steps": policy.num_ddim_steps,
                "simulated_latency_ms": policy.simulated_latency_ms,
            },
            "model_config": {
                "obs_dim": policy.model.obs_dim,
                "act_dim": policy.model.act_dim,
                "pred_horizon": policy.model.pred_horizon,
                "num_train_timesteps": policy.model.num_train_timesteps,
                "beta_schedule": policy.model.beta_schedule,
                "clip_sample": policy.model.clip_sample,
                "clip_val": policy.model.clip_val,
                "down_dims": list(policy.model.unet.down_dims),
                "cond_dim": policy.model.unet.cond_dim,
            }
        }
    else:
        state = {
            "type": "DiffusionModel1D",
            "model_state_dict": policy.state_dict(),
            "model_config": {
                "obs_dim": policy.obs_dim,
                "act_dim": policy.act_dim,
                "pred_horizon": policy.pred_horizon,
                "num_train_timesteps": policy.num_train_timesteps,
                "beta_schedule": policy.beta_schedule,
                "clip_sample": policy.clip_sample,
                "clip_val": policy.clip_val,
                "down_dims": list(policy.unet.down_dims),
                "cond_dim": policy.unet.cond_dim,
            }
        }

    torch.save(state, str(path))
    return str(path)


def load_diffusion_checkpoint(
    policy_or_path: Union[DiffusionRoboticsPolicy, DiffusionModel1D, str, Path],
    filepath: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None
) -> Union[DiffusionRoboticsPolicy, DiffusionModel1D]:
    """Loads checkpoint weights and restores policy or model."""
    if isinstance(policy_or_path, (str, Path)):
        # Load and instantiate new policy from file
        ckpt_path = str(policy_or_path)
        checkpoint = torch.load(ckpt_path, map_location=device or "cpu")
        m_cfg = checkpoint.get("model_config", {})
        unet = ConditionalUnet1D(
            act_dim=m_cfg.get("act_dim", 7),
            obs_dim=m_cfg.get("obs_dim", 21),
            pred_horizon=m_cfg.get("pred_horizon", 16),
            down_dims=m_cfg.get("down_dims", [64, 128, 256]),
            cond_dim=m_cfg.get("cond_dim", 128)
        )
        model = DiffusionModel1D(
            unet=unet,
            obs_dim=m_cfg.get("obs_dim", 21),
            act_dim=m_cfg.get("act_dim", 7),
            pred_horizon=m_cfg.get("pred_horizon", 16),
            num_train_timesteps=m_cfg.get("num_train_timesteps", 100),
            beta_schedule=m_cfg.get("beta_schedule", "squaredcos_cap_v2"),
            clip_sample=m_cfg.get("clip_sample", True),
            clip_val=m_cfg.get("clip_val", 2.6)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if device is not None:
            model = model.to(torch.device(device))

        if checkpoint.get("type") == "DiffusionRoboticsPolicy":
            p_cfg = checkpoint.get("policy_config", {})
            policy = DiffusionRoboticsPolicy(
                model=model,
                obs_dim=p_cfg.get("obs_dim", 21),
                act_dim=p_cfg.get("act_dim", 7),
                pred_horizon=p_cfg.get("pred_horizon", 16),
                exec_horizon=p_cfg.get("exec_horizon", 8),
                num_ddim_steps=p_cfg.get("num_ddim_steps", 10),
                simulated_latency_ms=p_cfg.get("simulated_latency_ms", 50.0),
                device=device or "cpu"
            )
            return policy
        return model
    else:
        # Load weights into existing instance
        if filepath is None:
            raise ValueError("filepath must be specified when passing an existing policy/model instance.")
        checkpoint = torch.load(str(filepath), map_location=device or "cpu")
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        if isinstance(policy_or_path, DiffusionRoboticsPolicy):
            policy_or_path.model.load_state_dict(state_dict)
            if device is not None:
                policy_or_path.model.to(torch.device(device))
                policy_or_path.device = torch.device(device)
        else:
            policy_or_path.load_state_dict(state_dict)
            if device is not None:
                policy_or_path.to(torch.device(device))
        return policy_or_path
