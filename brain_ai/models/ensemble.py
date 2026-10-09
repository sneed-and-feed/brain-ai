"""
brain_ai.models.ensemble: Unified Bi-Hemispheric Brain Architecture.

Wires:
1. Left Hemisphere (LH): Language / Symbolic synthesis (Qwen backbone with forward hooks)
2. Right Hemisphere (RH): Recurrent multi-timescale constraint engine (Sapient HRM)
3. Corpus Callosum: Dale-constrained Differential Cross-Attention bridge (s = -1)
4. Computational Amygdala: Open-Jev System 1 salience router & neuromodulatory controller
"""

from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_ai.models.callosum import InterHemisphericLatentCoupling
from brain_ai.models.amygdala import OpenJevAmygdalarRouter, NeuromodulatoryController, ReflexFirstCascadedRouter
from brain_ai.models.hrm import HierarchicalReasoningModel


class BiHemisphericBrain(nn.Module):
    """
    Unified Bi-Hemispheric Neuromorphic Architecture:
    - Left: Autoregressive linguistic / symbolic model
    - Right: Recurrent hierarchical reasoning model
    - Bridge: Dale-constrained Corpus Callosum with inhibitory cross-talk
    - Router: Fast System 1 Amygdala modulating dynamics & Reflex-First Cascade
    """
    def __init__(
        self,
        d_lh: int = 5120,      # Qwen 14B hidden dimension (or test dimension)
        d_rh: int = 512,       # HRM base hidden dimension
        d_callosum: int = 512, # Callosal latent manifold dimension
        callosal_heads: int = 4,
        hrm_cycles: int = 3,
        hrm_max_segments: int = 8,
        theta_bypass: float = 0.90
    ):
        super().__init__()
        self.d_lh = d_lh
        self.d_rh = d_rh
        self.d_callosum = d_callosum
        
        # 1. Right Hemisphere: Hierarchical Reasoning Engine
        self.right_hemisphere = HierarchicalReasoningModel(
            d_model=d_rh,
            n_heads=8,
            d_ffn=4 * d_rh,
            L_cycles=hrm_cycles,
            M_max=hrm_max_segments
        )
        
        # 2. Computational Amygdala, Neuromodulator & Performance-Gated Router
        self.amygdala = OpenJevAmygdalarRouter(hidden_dim=d_lh)
        self.neuromodulator = NeuromodulatoryController()
        self.cascaded_router = ReflexFirstCascadedRouter(theta_bypass=theta_bypass)
        
        # 3. Inter-Hemispheric Projections
        # Projects LH residual stream down to callosal manifold
        self.proj_lh_to_call = nn.Linear(d_lh, d_callosum)
        # Projects RH latent state to callosal manifold
        self.proj_rh_to_call = nn.Linear(d_rh, d_callosum)
        
        # Projects callosal output back into LH residual stream
        self.proj_call_to_lh = nn.Linear(d_callosum, d_lh)
        # Projects callosal output back into RH latent state
        self.proj_call_to_rh = nn.Linear(d_callosum, d_rh)
        
        # 4. Corpus Callosum E-I Bridge
        self.corpus_callosum = InterHemisphericLatentCoupling(
            d_latent=d_callosum,
            n_heads=callosal_heads,
            cross_talk_sign=-1.0, # Net transcallosal inhibition (Jeong 2026)
            p_excitatory=0.8,
            r_target=1.0
        )
        
        # Cross-hemispheric conflict detector
        self.conflict_head = nn.Sequential(
            nn.Linear(2 * d_callosum, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        lh_latents: torch.Tensor,        # (B, S_L, d_lh) From Left Hemisphere intermediate layer
        rh_inputs: torch.Tensor,         # (B, S_R, d_rh) Sensory grid / problem embeddings for RH
        candidate_logits: Optional[torch.Tensor] = None # For Amygdala decision entropy
    ) -> Dict[str, Any]:
        """
        Executes a synchronized bi-hemispheric cognitive pass:
        1. Amygdalar fast appraisal (sub-25ms affective evaluation).
        2. Neuromodulatory gain calculation.
        3. Right Hemisphere recurrent convergence (budgeted by Amygdala).
        4. Transcallosal E-I exchange.
        5. Modulated updates back to both hemispheres.
        """
        # Step 1: Subcortical Amygdala Affective Estimation
        state_summary = lh_latents.mean(dim=1)
        affective_state = self.amygdala(state_summary, candidate_logits=candidate_logits)
        
        # Step 2: Neuromodulatory Control Parameters
        controls = self.neuromodulator.compute_modulations(
            affective_state["valence"],
            affective_state["uncertainty"],
            affective_state["urgency"]
        )
        
        # Step 3: Right Hemisphere Reasoning
        # If System 1 reflex bypass is active, skip RH deep reasoning
        if controls["bypass_active"].all():
            z_rh_converged = rh_inputs
            hrm_info = {"bypassed": True}
        else:
            allocated_steps = int(controls["reasoning_steps"].max().item())
            z_rh_converged, _, hrm_info = self.right_hemisphere(
                rh_inputs, 
                max_steps=max(1, allocated_steps)
            )
            hrm_info["bypassed"] = False

        # Step 4: Callosal Projections
        z_lh_call = self.proj_lh_to_call(lh_latents)
        z_rh_call = self.proj_rh_to_call(z_rh_converged)
        
        # Step 5: Transcallosal E-I Differential Cross-Talk
        gamma_e = controls["gamma_E"].mean().item()
        gamma_i = controls["gamma_I"].mean().item()
        
        z_lh_coupled, z_rh_coupled, callosum_losses = self.corpus_callosum(
            z_left=z_lh_call,
            z_right=z_rh_call,
            gamma_e_mod=gamma_e,
            gamma_i_mod=gamma_i
        )
        
        # Step 6: Modulated Contralateral Injections
        delta_lh = self.proj_call_to_lh(z_lh_coupled - z_lh_call)
        delta_rh = self.proj_call_to_rh(z_rh_coupled - z_rh_call)
        
        lh_updated = lh_latents + delta_lh
        rh_updated = z_rh_converged + delta_rh
        
        # Step 7: Epistemic Conflict Detection
        # Pooled latents comparison
        lh_pool = z_lh_call.mean(dim=1)
        rh_pool = z_rh_call.mean(dim=1)
        conflict_score = self.conflict_head(torch.cat([lh_pool, rh_pool], dim=-1)).squeeze(-1)
        
        return {
            "lh_latents_updated": lh_updated,
            "rh_latents_updated": rh_updated,
            "affective_state": affective_state,
            "neuromodulatory_controls": controls,
            "conflict_score": conflict_score,
            "callosum_losses": callosum_losses,
            "hrm_info": hrm_info
        }
