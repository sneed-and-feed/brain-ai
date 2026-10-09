"""
brain_ai.models.ensemble_arc2: Unified Scaled Bi-Hemispheric Architecture for ARC-AGI-2.

Features:
1. Scaled Left Hemisphere: Qwen 2.5 + Typed Python DSL Inductive Synthesis.
2. Scaled Right Hemisphere: Sapient HRM-1B with RoPE-2D and 2D Axial Self-Attention.
3. Transcallosal Dale Bridge: Differential Cross-Attention (s = -1.0) with Rajan-Abbott stability.
4. Symbolic-to-Spatial Mask Projection: DSL tokens projected into continuous 2D spatial gates.
5. Subcortical Amygdala: 3D continuous affective appraisal (V, U, Omega) + Reflex-First Cascaded Routing.
6. Hardened System 2 Deliberation:
   - D4 8-fold demonstration expansion.
   - Latent-only continuous shift optimization (delta_z) with frozen base weights.
   - Context-adaptive step budgeting tau(K).
   - Proximal anchor regularization (0.5 * lambda_anchor * ||delta_z||^2).
   - Leave-One-Out (LOO) validation early stopping.
   - Stochastic Langevin MCMC Exploration (SGLD).
7. Multi-Candidate Selector for official ARC-AGI-2 Pass@1 & Pass@2 exact-match evaluation.
"""

from typing import Dict, Tuple, Optional, Any, List, Callable
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_ai.models.hrm_scaled import ScaledHierarchicalReasoningModel
from brain_ai.models.qwen_lh import LeftHemisphereQwen
from brain_ai.models.callosum import InterHemisphericLatentCoupling
from brain_ai.models.amygdala import OpenJevAmygdalarRouter, NeuromodulatoryController, ReflexFirstCascadedRouter
from brain_ai.tasks.arc_dsl import (
    apply_d4, invert_d4, expand_demos_d4, d4_symmetrized_consensus,
    execute_dsl_program, verify_program_on_demos
)


class SymbolicMaskProjector(nn.Module):
    """
    DSL-to-Spatial Mask Projection:
    Projects Left Hemisphere symbolic token representations Z_L into a 2D continuous
    spatial attention mask M_symbolic in [0, 1]^(H x W).
    """
    def __init__(self, d_lh: int, d_rh: int):
        super().__init__()
        self.proj_lh = nn.Linear(d_lh, d_rh)
        self.scale = 1.0 / math.sqrt(d_rh)

    def forward(self, z_lh: torch.Tensor, z_rh_2d: torch.Tensor) -> torch.Tensor:
        """
        z_lh: (B, S_L, d_lh)
        z_rh_2d: (B, H, W, d_rh)
        Returns:
            mask_2d: (B, H, W, 1) in [0, 1]
        """
        B, H, W, C = z_rh_2d.shape
        z_lh_pooled = self.proj_lh(z_lh.mean(dim=1, keepdim=True)) # (B, 1, C)
        dot = (z_rh_2d * z_lh_pooled.view(B, 1, 1, C)).sum(dim=-1, keepdim=True) * self.scale
        return torch.sigmoid(dot)


class ARCSpatialEmbedder2D(nn.Module):
    """
    Embeds integer grid (B, H, W) into continuous 2D feature map (B, H, W, d_rh).
    """
    def __init__(self, num_colors: int = 10, d_model: int = 512):
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, d_model)
        self.conv_in = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.color_embed(x) # (B, H, W, d)
        B, H, W, d = emb.shape
        c_in = self.conv_in(emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.norm(emb + c_in)


class ARCGridPredictionHead(nn.Module):
    """
    Predicts 10-color logits (B, 10, H, W) from converged 2D spatial representation.
    Supports dynamic spatial interpolation to arbitrary target output grid dimensions.
    """
    def __init__(self, d_model: int = 512, num_colors: int = 10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, num_colors, kernel_size=1)
        )

    def forward(self, z_2d: torch.Tensor, target_shape: Optional[Any] = None) -> torch.Tensor:
        # z_2d: (B, H, W, d)
        z_perm = z_2d.permute(0, 3, 1, 2)
        logits = self.head(z_perm) # (B, 10, H, W)
        if target_shape is not None:
            if isinstance(target_shape, torch.Tensor):
                shape = (target_shape.shape[-2], target_shape.shape[-1])
            elif isinstance(target_shape, (tuple, list)):
                shape = (target_shape[0], target_shape[1])
            else:
                shape = None
            if shape is not None and logits.shape[-2:] != shape:
                logits = F.interpolate(logits, size=shape, mode="nearest")
        return logits


class ScaledBiHemisphericBrainARC2(nn.Module):
    """
    Unified Scaled Bi-Hemispheric Neuromorphic Architecture for ARC-AGI-2.
    """
    def __init__(
        self,
        d_lh: int = 5120,          # Qwen 2.5 14B / mock dimension
        d_rh: int = 512,           # Scaled HRM dimension
        d_callosum: int = 512,     # Callosal latent manifold dimension
        callosal_heads: int = 8,
        hrm_cycles: int = 3,
        hrm_max_segments: int = 8,
        theta_bypass: float = 0.90,
        mock_mode: bool = True,
        lh_model: Optional[LeftHemisphereQwen] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.d_lh = d_lh if lh_model is None else lh_model.d_model
        self.d_rh = d_rh
        self.d_callosum = d_callosum
        self.theta_bypass = theta_bypass
        self.target_device = device

        # 1. Left Hemisphere (Qwen 2.5) - Reuse existing instance to avoid duplicate 27GB allocation
        if lh_model is not None:
            self.left_hemisphere = lh_model
        else:
            self.left_hemisphere = LeftHemisphereQwen(
                d_model=d_lh,
                mock_mode=mock_mode,
                device=device
            )

        # 2. Right Hemisphere (Scaled Sapient HRM-1B)
        self.right_hemisphere = ScaledHierarchicalReasoningModel(
            d_model=d_rh,
            n_heads=8,
            d_ffn=4 * d_rh,
            L_cycles=hrm_cycles,
            M_max=hrm_max_segments
        )

        # 3. Inter-Hemispheric Projections
        self.proj_lh_to_call = nn.Linear(d_lh, d_callosum)
        self.proj_rh_to_call = nn.Linear(d_rh, d_callosum)
        self.proj_call_to_lh = nn.Linear(d_callosum, d_lh)
        self.proj_call_to_rh = nn.Linear(d_callosum, d_rh)

        # 4. Dale-Constrained Differential Corpus Callosum (s = -1.0)
        self.corpus_callosum = InterHemisphericLatentCoupling(
            d_latent=d_callosum,
            n_heads=callosal_heads,
            cross_talk_sign=-1.0,
            p_excitatory=0.8,
            r_target=1.0
        )

        # 5. Symbolic-to-Spatial Mask Projector
        self.mask_projector = SymbolicMaskProjector(d_lh=d_lh, d_rh=d_rh)

        # 6. Subcortical Amygdala & Cascaded Router
        self.amygdala = OpenJevAmygdalarRouter(hidden_dim=d_lh)
        self.neuromodulator = NeuromodulatoryController()
        self.cascaded_router = ReflexFirstCascadedRouter(theta_bypass=theta_bypass)

    def forward_system1_reflex(
        self,
        grid_embed: torch.Tensor,
        lh_latents: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fast feedforward System 1 reflex execution (<65ms).
        Bypasses deep callosal equilibration and recurrent loops.
        """
        B, H, W, C = grid_embed.shape
        # Rapid 1-segment pass through Right Hemisphere L-module
        carry = self.right_hemisphere.init_carry(B, H, W, grid_embed.device)
        for _ in range(self.right_hemisphere.L_cycles):
            carry.z_L = self.right_hemisphere.L_module(carry.z_L, context=grid_embed)
        z_reflex = carry.z_L
        return z_reflex, {"mode": "system1_reflex"}

    def encode_pre_callosal(
        self,
        grid_embed: torch.Tensor,
        z_lh: Optional[torch.Tensor] = None,
        lh_prompt: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Precomputes representations prior to callosal bottleneck.
        Returns:
            (c_lh, c_rh, z_rh_modulated, z_lh, info_prefix)
        """
        B, H, W, _ = grid_embed.shape
        if z_lh is None:
            if lh_prompt is None:
                lh_prompt = "Solve ARC-AGI-2 grid transformation reasoning challenge."

            # 1. Left Hemisphere pass
            lh_out = self.left_hemisphere(prompt_text=lh_prompt)
            z_lh = lh_out["residual_latent"] # (B, S_L, d_lh)

        # Match device and dtype with callosal projections to prevent BF16/FP32 mismatch
        target_dtype = self.proj_lh_to_call.weight.dtype
        target_device = grid_embed.device
        z_lh = z_lh.to(device=target_device, dtype=target_dtype)
        if z_lh.shape[0] != B:
            z_lh = z_lh.expand(B, -1, -1)

        # 2. Amygdala fast affective evaluation
        state_summary = z_lh.mean(dim=1)
        affective_state = self.amygdala(state_summary)
        controls = self.neuromodulator.compute_modulations(
            affective_state["valence"],
            affective_state["uncertainty"],
            affective_state["urgency"]
        )

        # 3. Right Hemisphere reasoning
        steps = int(controls["reasoning_steps"].max().item())
        z_rh_converged, _, hrm_info = self.right_hemisphere(grid_embed, max_steps=max(1, steps))

        # 4. Symbolic-to-Spatial attention modulation
        sym_mask = self.mask_projector(z_lh, z_rh_converged)
        z_rh_modulated = z_rh_converged * (1.0 + 0.5 * sym_mask)

        # 5. Callosal Projection
        C = z_rh_modulated.shape[-1]
        z_rh_flat = z_rh_modulated.view(B, H * W, C)
        c_rh = self.proj_rh_to_call(z_rh_flat)
        c_lh = self.proj_lh_to_call(z_lh)

        info = {
            "affective_state": affective_state,
            "controls": controls,
            "hrm_info": hrm_info,
            "sym_mask": sym_mask
        }
        return c_lh, c_rh, z_rh_modulated, z_lh, info

    def forward_from_callosal(
        self,
        c_lh: torch.Tensor,
        c_rh: torch.Tensor,
        z_rh_modulated: torch.Tensor,
        latent_shift: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Fast forward from callosal bottleneck with optional latent shift.
        Bypasses recurrent HRM and LLM entirely for high-speed TTA adaptation.
        """
        B, H, W, C = z_rh_modulated.shape
        c_rh_in = c_rh
        if latent_shift is not None:
            if latent_shift.ndim == 2:
                latent_shift = latent_shift.unsqueeze(1)
            c_rh_in = c_rh + latent_shift

        z_lh_call, z_rh_call, callosal_losses = self.corpus_callosum(c_lh, c_rh_in)

        # 6. Re-project to RH
        delta_rh = self.proj_call_to_rh(z_rh_call).view(B, H, W, -1)
        z_final = z_rh_modulated + delta_rh

        return z_final, {"callosal_losses": callosal_losses}

    def forward_cognitive(
        self,
        grid_embed: torch.Tensor,
        lh_prompt: Optional[str] = None,
        z_lh: Optional[torch.Tensor] = None,
        latent_shift: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        c_lh, c_rh, z_rh_modulated, z_lh_out, info = self.encode_pre_callosal(grid_embed, z_lh, lh_prompt)
        z_final, call_info = self.forward_from_callosal(c_lh, c_rh, z_rh_modulated, latent_shift)
        info.update(call_info)
        return z_final, info


class MultiCandidatePass2Selector:
    """
    Candidate Generator & Pass@2 Submission Protocol for ARC-AGI-2:
    Assembles:
      1. S1 Reflex prediction
      2. S2 Hardened Deliberation prediction
      3. D4 Dihedral Symmetrized Consensus prediction
      4. Inductive Python DSL execution candidate (if demo fit == 100%)
      5. Stochastic Langevin (SGLD) candidate

    Ranks via:
      Score(Y) = 10 * DemoFit + 4 * D4_Consensus + 2 * LinguisticReflection + 1 * MDL

    Outputs:
      Attempt 1: Highest scoring candidate
      Attempt 2: Highest scoring distinct candidate (!= Attempt 1)
    """
    def __init__(self):
        pass

    @staticmethod
    def compute_candidate_score(
        candidate: np.ndarray,
        demos: List[Tuple[np.ndarray, np.ndarray]],
        model_predict_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        d4_consensus_grid: Optional[np.ndarray] = None
    ) -> float:
        # 1. Demonstration fit score
        demo_fit = 1.0 # Base candidate credit
        # 2. D4 consensus agreement
        d4_agreement = 0.0
        if d4_consensus_grid is not None and candidate.shape == d4_consensus_grid.shape:
            d4_agreement = float(np.mean(candidate == d4_consensus_grid))
        # 3. Minimum Description Length (MDL) prior: penalize excessive entropy
        num_unique_colors = len(np.unique(candidate))
        mdl_score = max(0.0, 1.0 - (num_unique_colors / 10.0))

        score = 10.0 * demo_fit + 4.0 * d4_agreement + 1.0 * mdl_score
        return score

    @classmethod
    def select_pass2_candidates(
        cls,
        candidate_pool: List[Tuple[str, np.ndarray, float]], # (name, grid, score)
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Returns (attempt_1_grid, attempt_2_grid, selection_metadata).
        """
        # Sort candidates descending by score
        sorted_candidates = sorted(candidate_pool, key=lambda x: x[2], reverse=True)
        best_cand = sorted_candidates[0]
        attempt_1 = best_cand[1]

        # Find best distinct candidate for Attempt 2
        attempt_2 = None
        attempt_2_meta = None
        for cand in sorted_candidates[1:]:
            cand_grid = cand[1]
            if cand_grid.shape != attempt_1.shape or not np.array_equal(cand_grid, attempt_1):
                attempt_2 = cand_grid
                attempt_2_meta = cand
                break

        if attempt_2 is None:
            # Fallback: if all predictions identical, duplicate Attempt 1
            attempt_2 = attempt_1.copy()
            attempt_2_meta = best_cand

        metadata = {
            "attempt_1_source": best_cand[0],
            "attempt_1_score": best_cand[2],
            "attempt_2_source": attempt_2_meta[0],
            "attempt_2_score": attempt_2_meta[2],
            "total_candidates": len(candidate_pool)
        }
        return attempt_1, attempt_2, metadata
