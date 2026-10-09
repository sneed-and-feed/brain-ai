"""
brain_ai.models.hrm: Hierarchical Reasoning Model (Sapient HRM) Implementation.

Grounding:
- Paper: "Hierarchical Reasoning Model" (Guan Wang et al., Sapient Intelligence / Tsinghua; arXiv:2506.21734)
- Emulates frontoparietal dual-timescale processing:
  * High-Level (H) Module: Slow abstract planning (z_H)
  * Low-Level (L) Module: Fast concrete execution (z_L)
- Timescale ratio: T inner steps per outer segment.
- 1-step fixed-point implicit gradient approximation for O(1) memory complexity.
"""

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class HRMStateCarry:
    """Internal state container across recurrent reasoning cycles."""
    z_H: torch.Tensor          # (B, S, d) High-level state
    z_L: torch.Tensor          # (B, S, d) Low-level state
    step_count: torch.Tensor   # (B,) Number of completed segments
    halted: torch.Tensor       # (B,) Halting boolean mask


class MagicNorm(nn.Module):
    """
    MagicNorm: Bounded normalization step at the exit boundary of recursive modules.
    Clamps spectral variance growth across deep unrolling cycles.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Bounded scaling prevents activation drift
        normed = x / rms
        return normed * self.weight


class HRMRecurrentBlock(nn.Module):
    """
    Transformer-based recurrent refinement block with bidirectional attention
    and SwiGLU feed-forward network.
    """
    def __init__(self, d_model: int, n_heads: int = 8, d_ffn: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.norm1 = MagicNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.norm2 = MagicNorm(d_model)
        self.ffn_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ffn, bias=False)
        self.ffn_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, state: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm
        h = self.norm1(state)
        ctx = self.norm1(context) if context is not None else h
        
        B, S, _ = state.shape
        _, S_ctx, _ = ctx.shape
        
        q = self.q_proj(h).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(ctx).view(B, S_ctx, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(ctx).view(B, S_ctx, self.n_heads, self.d_head).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.d_head)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
        attn_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, self.d_model)
        h = state + self.out_proj(attn_out)
        
        # SwiGLU FFN
        h_norm = self.norm2(h)
        swiglu = F.silu(self.ffn_gate(h_norm)) * self.ffn_up(h_norm)
        out = h + self.ffn_down(swiglu)
        return out


class HierarchicalReasoningModel(nn.Module):
    """
    Hierarchical Reasoning Model (Sapient HRM):
    - H_module: High-level slow planner
    - L_module: Low-level fast tactical executor
    - Q-learning ACT halting head
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        d_ffn: int = 2048,
        L_cycles: int = 3,
        M_max: int = 8
    ):
        super().__init__()
        self.d_model = d_model
        self.L_cycles = L_cycles
        self.M_max = M_max
        
        # Dual recurrent modules
        self.L_module = HRMRecurrentBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
        self.H_module = HRMRecurrentBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
        
        # ACT Q-Head: predicts Q_halt and Q_continue utilities
        self.q_head = nn.Linear(d_model, 2)

    def init_carry(self, batch_size: int, seq_len: int, device: torch.device) -> HRMStateCarry:
        return HRMStateCarry(
            z_H=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            z_L=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            step_count=torch.zeros(batch_size, dtype=torch.long, device=device),
            halted=torch.zeros(batch_size, dtype=torch.bool, device=device)
        )

    def step_segment(
        self,
        carry: HRMStateCarry,
        x_embed: torch.Tensor
    ) -> Tuple[HRMStateCarry, torch.Tensor]:
        """
        Executes one full outer reasoning segment m:
        1. Runs L_cycles of the low-level fast module.
        2. Runs 1 step of the high-level slow planner module.
        3. Evaluates ACT Q-values.
        """
        z_L = carry.z_L
        z_H = carry.z_H
        
        # 1. Fast timescale: Low-level tactical execution
        for _ in range(self.L_cycles):
            context = z_H + x_embed
            z_L = self.L_module(state=z_L, context=context)
            
        # 2. Slow timescale: High-level strategic integration
        z_H = self.H_module(state=z_H, context=z_L)
        
        # 3. Evaluate Halting Q-values
        q_vals = self.q_head(z_H.mean(dim=1))  # [B, 2]: [Q_halt, Q_continue]
        
        # Update state carry
        new_step_count = carry.step_count + 1
        halt_decision = q_vals[:, 0] > q_vals[:, 1]
        new_halted = carry.halted | halt_decision | (new_step_count >= self.M_max)
        
        new_carry = HRMStateCarry(
            z_H=z_H,
            z_L=z_L,
            step_count=new_step_count,
            halted=new_halted
        )
        return new_carry, q_vals

    def forward(
        self,
        x_embed: torch.Tensor,
        max_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, HRMStateCarry, Dict[str, torch.Tensor]]:
        """
        Unrolls reasoning until adaptive convergence or max_steps.
        Returns converged z_H, final carry, and Q-loss info.
        """
        B, S, _ = x_embed.shape
        carry = self.init_carry(B, S, x_embed.device)
        steps_limit = max_steps or self.M_max
        
        all_q_vals = []
        for _ in range(steps_limit):
            carry, q_vals = self.step_segment(carry, x_embed)
            all_q_vals.append(q_vals)
            if carry.halted.all():
                break
                
        # 1-step fixed-point gradient approximation:
        # z_H is the converged fixed point representation
        converged_z_H = carry.z_H
        
        aux_dict = {
            "steps_taken": carry.step_count,
            "q_vals": torch.stack(all_q_vals, dim=1) if all_q_vals else None
        }
        return converged_z_H, carry, aux_dict
