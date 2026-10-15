"""
brain_ai.models.callosum: Corpus Callosum Inter-Hemispheric E-I Bridge.

Implements:
1. DaleLinear: Weight parameterization W = Softplus(V) * D strictly enforcing Dale's Principle.
   Initialized via Rajan-Abbott (PRL 2006) balanced condition (f_E * mu_E = f_I * mu_I, lambda_outlier = 0).
2. HomeostaticSynapticScaling: Turrigiano synaptic scaling & RMSNorm to prevent runaway/collapse.
3. DaleDifferentialCrossAttention: Differential attention (Ye et al., ICLR 2025) with Dale constraints.
4. InterHemisphericLatentCoupling: Transcallosal inhibitory coupling (Hong Jeong 2026; s = -1).
"""

import copy
import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DaleLinear(nn.Module):
    """
    Linear layer strictly enforcing Dale's Principle:
        W = W_{nonneg} @ diag(D)
    Presynaptic columns are partitioned into Excitatory (+1) and Inhibitory (-1).
    Initialized using Rajan & Abbott (PRL 2006) random matrix balance:
        f_E * mu_E = f_I * mu_I  =>  lambda_outlier = 0
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        p_excitatory: float = 0.8,
        mode: str = "softplus",  # 'softplus', 'abs', or 'clamp'
        beta: float = 1.0,
        spectral_radius: float = 1.0,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.beta = beta
        
        # Partition presynaptic inputs into E and I populations
        n_excitatory = int(round(in_features * p_excitatory))
        n_inhibitory = in_features - n_excitatory
        self.n_excitatory = n_excitatory
        self.n_inhibitory = n_inhibitory
        
        # Fixed Dale diagonal signature: +1 for E, -1 for I
        d_sign = torch.ones(in_features)
        d_sign[n_excitatory:] = -1.0
        self.register_buffer("d_sign", d_sign)
        
        # Unconstrained latent parameter V
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
            
        self._init_rajan_abbott(n_excitatory, n_inhibitory, spectral_radius)

    def _init_rajan_abbott(self, n_e: int, n_i: int, target_radius: float):
        """
        Rajan & Abbott (PRL 2006) balanced initialization:
        Eliminates the outlier eigenvalue by enforcing: f_E * mu_E = f_I * mu_I
        Scales variance so bulk radius R = sqrt(N * (f_E * var_E + f_I * var_I)) approx target_radius.
        """
        f_e = n_e / self.in_features
        f_i = n_i / self.in_features
        
        # Target means and variances
        mu_e = 1.0 / math.sqrt(self.in_features)
        mu_i = (f_e / f_i) * mu_e  # Perfect mean cancellation eliminates outlier eigenvalue
        
        var_target = (target_radius ** 2) / (self.in_features * (f_e + f_i * ((f_e / f_i) ** 2)))
        std_target = math.sqrt(max(var_target, 1e-6))
        
        with torch.no_grad():
            self.V[:, :n_e] = torch.randn(self.out_features, n_e) * std_target + mu_e
            self.V[:, n_e:] = torch.randn(self.out_features, n_i) * std_target + mu_i
            # If using softplus, inverse map to latent V space
            if self.mode == "softplus":
                self.V.data = torch.log(torch.exp(self.beta * self.V.data.clamp(min=1e-4)) - 1.0) / self.beta

    def get_effective_weight(self) -> torch.Tensor:
        """Returns effective weight matrix W satisfying Dale's Principle."""
        if self.mode == "softplus":
            w_nonneg = F.softplus(self.V, beta=self.beta)
        elif self.mode == "abs":
            w_nonneg = torch.abs(self.V)
        elif self.mode == "clamp":
            w_nonneg = torch.clamp(self.V, min=0.0)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
            
        return w_nonneg * self.d_sign

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.get_effective_weight()
        return F.linear(x, W, self.bias)


class HomeostaticSynapticScaling(nn.Module):
    """
    Implements Turrigiano (2008) synaptic scaling and RMSNorm normalization.
    Locks latent activations to a target energy level r_target, preventing
    runaway explosion (||h|| -> inf) or coma collapse (||h|| -> 0).
    """
    def __init__(self, d_model: int, r_target: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.r_target = r_target
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Root-mean-square energy per token
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Scale to target activity level
        x_normed = (x / rms) * self.scale * self.r_target
        
        # Auxiliary homeostatic penalty: (rms - r_target)^2
        homeo_loss = torch.mean((rms - self.r_target) ** 2)
        return x_normed, homeo_loss


class DaleDifferentialCrossAttention(nn.Module):
    """
    Dale-Constrained Differential Cross-Attention:
    - Projections obey Dale's Principle (W_Q, W_K, W_V >= 0 via DaleLinear).
    - Attention scores computed via differential subtraction (Ye et al., ICLR 2025):
        Attn = [Softmax(Q1 @ K1.T / sqrt(d)) - lambda * Softmax(Q2 @ K2.T / sqrt(d))] @ V
      where the second term acts as an active inhibitory filter.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        p_excitatory: float = 0.8,
        lambda_init: float = 0.5
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Dual projections for differential attention (E: channel 1, I: channel 2)
        self.q_proj = DaleLinear(d_model, 2 * d_model, p_excitatory=p_excitatory)
        self.k_proj = DaleLinear(d_model, 2 * d_model, p_excitatory=p_excitatory)
        self.v_proj = DaleLinear(d_model, d_model, p_excitatory=p_excitatory)
        self.out_proj = DaleLinear(d_model, d_model, p_excitatory=p_excitatory)
        
        # Learnable inhibitory damping coefficient lambda in (0, 1)
        self.raw_lambda = nn.Parameter(torch.tensor(math.log(lambda_init / (1.0 - lambda_init))))

    def forward(
        self,
        query_latents: torch.Tensor,       # (B, T_q, D)
        key_value_latents: torch.Tensor,   # (B, T_kv, D)
    ) -> torch.Tensor:
        B, T_q, _ = query_latents.shape
        _, T_kv, _ = key_value_latents.shape
        
        # Project queries and keys into dual streams: (B, T, 2, n_heads, d_head)
        q = self.q_proj(query_latents).view(B, T_q, 2, self.n_heads, self.d_head)
        k = self.k_proj(key_value_latents).view(B, T_kv, 2, self.n_heads, self.d_head)
        v = self.v_proj(key_value_latents).view(B, T_kv, self.n_heads, self.d_head)
        
        # Split into Excitatory (1) and Inhibitory (2) attention queries/keys
        q_e, q_i = q[:, :, 0], q[:, :, 1]
        k_e, k_i = k[:, :, 0], k[:, :, 1]
        
        # Transpose to (B, n_heads, T, d_head)
        q_e = q_e.transpose(1, 2)
        q_i = q_i.transpose(1, 2)
        k_e = k_e.transpose(1, 2)
        k_i = k_i.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute Scaled Dot-Product Attention maps
        scale = 1.0 / math.sqrt(self.d_head)
        attn_e = F.softmax(torch.matmul(q_e, k_e.transpose(-2, -1)) * scale, dim=-1)
        attn_i = F.softmax(torch.matmul(q_i, k_i.transpose(-2, -1)) * scale, dim=-1)
        
        # Differential Attention: Excitatory map minus scaled Inhibitory map
        lambd = torch.sigmoid(self.raw_lambda)
        diff_attn = attn_e - lambd * attn_i
        
        # Apply attention to values
        out = torch.matmul(diff_attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        
        return self.out_proj(out)


class InterHemisphericLatentCoupling(nn.Module):
    """
    Bio-Plausible Corpus Callosum Inter-Hemispheric Coupling Module.
    Couples two lateralized latent vector spaces Z_L (Left) and Z_R (Right).
    
    Grounding (Hong Jeong 2026; arXiv:2603.03355):
    - Excitatory cross-talk (s = +1) triggers bank-dominance collapse.
    - Inhibitory cross-talk (s = -1) actively suppresses contralateral
      monopolization, inducing stable functional lateralization.
    """
    def __init__(
        self,
        d_latent: int,
        n_heads: int = 4,
        cross_talk_sign: float = -1.0,  # -1.0 enforces net transcallosal inhibition
        p_excitatory: float = 0.8,
        r_target: float = 1.0
    ):
        super().__init__()
        self.d_latent = d_latent
        self.cross_talk_sign = cross_talk_sign
        
        # Cross-Attention Bridges (Corpus Callosum projections)
        self.callosum_r_to_l = DaleDifferentialCrossAttention(
            d_latent, n_heads=n_heads, p_excitatory=p_excitatory
        )
        self.callosum_l_to_r = DaleDifferentialCrossAttention(
            d_latent, n_heads=n_heads, p_excitatory=p_excitatory
        )
        
        # Synaptic Scaling & Homeostatic Normalization
        self.scaling_left = HomeostaticSynapticScaling(d_latent, r_target=r_target)
        self.scaling_right = HomeostaticSynapticScaling(d_latent, r_target=r_target)
        
        # Base coupling gain parameter
        self.callosal_coupling_gain = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        z_left: torch.Tensor,               # (B, T_L, D)
        z_right: torch.Tensor,              # (B, T_R, D)
        gamma_e_mod: Optional[float] = None, # Neuromodulatory gain override
        gamma_i_mod: Optional[float] = None  # Neuromodulatory gain override
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute Transcallosal Contralateral Messages
        delta_l = self.callosum_r_to_l(query_latents=z_left, key_value_latents=z_right)
        delta_r = self.callosum_l_to_r(query_latents=z_right, key_value_latents=z_left)
        
        # Base coupling
        base_coupling = torch.tanh(self.callosal_coupling_gain) * self.cross_talk_sign
        
        if gamma_i_mod is not None:
            effective_coupling = base_coupling * gamma_i_mod
        else:
            effective_coupling = base_coupling
            
        z_left_coupled = z_left + effective_coupling * delta_l
        z_right_coupled = z_right + effective_coupling * delta_r
        
        # Apply Homeostatic Synaptic Scaling
        z_left_out, homeo_l = self.scaling_left(z_left_coupled)
        z_right_out, homeo_r = self.scaling_right(z_right_coupled)
        
        # Compute E-I Balance Flux Disparity (token-averaged flux in each direction)
        mean_flux_l = torch.norm(delta_l, p=1, dim=-1).mean(dim=-1)
        mean_flux_r = torch.norm(delta_r, p=1, dim=-1).mean(dim=-1)
        flux_disparity = torch.mean((mean_flux_l - mean_flux_r) ** 2)
        total_homeo_loss = homeo_l + homeo_r + 0.05 * flux_disparity
        
        loss_dict = {
            "loss_homeostatic": total_homeo_loss,
            "homeo_left": homeo_l,
            "homeo_right": homeo_r,
            "callosal_flux_disparity": flux_disparity
        }
        
        return z_left_out, z_right_out, loss_dict


class CallosalConditioning(nn.Module):
    """
    Corpus Callosum Conditioning Module for Bi-Hemispheric Architecture.
    Connects frozen Left Hemisphere (LH) latents to the recursive reasoner's
    inner state and conditioning tokens.

    Supports:
    1. Dale's law constraints:
       - 'dale_inhibitory' (s = -1.0): DaleLinear + DaleDifferentialCrossAttention + negative cross-talk.
       - 'dale_excitatory' (s = +1.0): DaleLinear + DaleDifferentialCrossAttention + positive cross-talk.
       - 'unconstrained' (s = +1.0): Standard Linear + Multihead Cross-Attention.
    2. Homeostatic Synaptic Scaling (Turrigiano 2008 RMS normalization).
    3. Task conditioning token synthesis from LH latents (for TRM / HRM).
    4. Direct recurrent reasoner state conditioning.
    """
    def __init__(
        self,
        d_lh: int = 4096,
        d_reasoner: int = 512,
        d_callosum: int = 512,
        num_query_tokens: int = 16,
        n_heads: int = 4,
        p_excitatory: float = 0.8,
        cross_talk_sign: float = -1.0,
        constraint_type: str = "dale_inhibitory",
        r_target: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_lh = d_lh
        self.d_reasoner = d_reasoner
        self.d_callosum = d_callosum
        self.num_query_tokens = num_query_tokens
        self.n_heads = n_heads
        self.p_excitatory = p_excitatory
        self.constraint_type = constraint_type
        self.r_target = r_target

        if constraint_type == "dale_inhibitory":
            self.cross_talk_sign = -1.0
        elif constraint_type == "dale_excitatory":
            self.cross_talk_sign = 1.0
        elif constraint_type == "unconstrained":
            self.cross_talk_sign = float(cross_talk_sign)
        else:
            raise ValueError(f"Unknown constraint_type: {constraint_type}")

        # Projection from LH latent space to Callosal manifold
        if self.constraint_type in ("dale_inhibitory", "dale_excitatory"):
            self.proj_lh_to_call = DaleLinear(d_lh, d_callosum, p_excitatory=p_excitatory)
            self.cross_attn = DaleDifferentialCrossAttention(
                d_callosum, n_heads=n_heads, p_excitatory=p_excitatory
            )
        else:
            self.proj_lh_to_call = nn.Linear(d_lh, d_callosum)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_callosum, num_heads=n_heads, batch_first=True, dropout=dropout
            )

        # Learnable task query tokens (used to synthesize task conditioning representations)
        self.task_queries = nn.Parameter(
            torch.randn(1, num_query_tokens, d_callosum) * (1.0 / math.sqrt(d_callosum))
        )

        # Projections for direct reasoner state conditioning
        if d_reasoner != d_callosum:
            self.proj_reasoner_to_call = nn.Linear(d_reasoner, d_callosum)
            self.proj_call_to_reasoner = nn.Linear(d_callosum, d_reasoner)
        else:
            self.proj_reasoner_to_call = nn.Identity()
            self.proj_call_to_reasoner = nn.Identity()

        # Synaptic scaling & Homeostatic normalization
        self.scaling = HomeostaticSynapticScaling(d_callosum, r_target=r_target)

        # Base coupling gain parameter (learnable)
        self.coupling_gain = nn.Parameter(torch.tensor(0.2))

    def _cross_attend(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Helper to run cross-attention whether Dale or unconstrained."""
        if self.constraint_type in ("dale_inhibitory", "dale_excitatory"):
            return self.cross_attn(query_latents=query, key_value_latents=key_value)
        else:
            out, _ = self.cross_attn(query=query, key=key_value, value=key_value)
            return out

    def condition_task(self, lh_latents: torch.Tensor, delta_z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Synthesizes task conditioning tokens from LH latents.
        lh_latents: (B, S_L, d_lh) or (S_L, d_lh)
        delta_z: optional continuous latent offset (num_query_tokens, d_reasoner),
                 (B, num_query_tokens, d_reasoner), or flattened
        Returns: (B, num_query_tokens, d_reasoner)
        """
        if lh_latents.dim() == 2:
            lh_latents = lh_latents.unsqueeze(0)
        B = lh_latents.size(0)

        # Ensure input dtype matches internal module parameter dtype (e.g. bfloat16 in TRM)
        target_dtype = self.task_queries.dtype
        if lh_latents.dtype != target_dtype:
            lh_latents = lh_latents.to(dtype=target_dtype)

        # Project LH to callosal manifold
        z_lh_call = self.proj_lh_to_call(lh_latents)  # (B, S_L, d_callosum)

        # Expand learnable query tokens for batch
        q = self.task_queries.expand(B, -1, -1)  # (B, num_query_tokens, d_callosum)

        # Cross-attention delta
        delta = self._cross_attend(query=q, key_value=z_lh_call)

        # Apply coupling gain and Dale cross-talk sign
        eff_coupling = torch.tanh(self.coupling_gain) * self.cross_talk_sign
        q_coupled = q + eff_coupling * delta

        # Apply Homeostatic Synaptic Scaling
        q_normed, _ = self.scaling(q_coupled)

        # Project to reasoner dimension
        out = self.proj_call_to_reasoner(q_normed)  # (B, num_query_tokens, d_reasoner)

        if delta_z is not None:
            if delta_z.dtype != out.dtype:
                delta_z = delta_z.to(dtype=out.dtype, device=out.device)
            if delta_z.dim() == 2:
                if delta_z.size(-1) == self.d_reasoner and delta_z.size(0) == self.num_query_tokens:
                    out = out + delta_z.unsqueeze(0)
                elif delta_z.size(-1) == self.num_query_tokens * self.d_reasoner:
                    out = out + delta_z.view(-1, self.num_query_tokens, self.d_reasoner)
                else:
                    out = out + delta_z.unsqueeze(0)
            elif delta_z.dim() == 3:
                out = out + delta_z
            elif delta_z.dim() == 1 and delta_z.numel() == self.num_query_tokens * self.d_reasoner:
                out = out + delta_z.view(1, self.num_query_tokens, self.d_reasoner)

        return out

    def condition_state(self, reasoner_state: torch.Tensor, lh_latents: torch.Tensor) -> torch.Tensor:
        """
        Directly conditions the recursive reasoner's inner state with LH latents.
        reasoner_state: (B, S_R, d_reasoner)
        lh_latents: (B, S_L, d_lh)
        Returns: (B, S_R, d_reasoner)
        """
        if lh_latents.dim() == 2:
            lh_latents = lh_latents.unsqueeze(0)

        target_dtype = self.task_queries.dtype
        if lh_latents.dtype != target_dtype:
            lh_latents = lh_latents.to(dtype=target_dtype)
        if reasoner_state.dtype != target_dtype:
            reasoner_state = reasoner_state.to(dtype=target_dtype)

        # Project both to callosal space
        q = self.proj_reasoner_to_call(reasoner_state)  # (B, S_R, d_callosum)
        z_lh_call = self.proj_lh_to_call(lh_latents)     # (B, S_L, d_callosum)

        # Cross-attend
        delta = self._cross_attend(query=q, key_value=z_lh_call)

        eff_coupling = torch.tanh(self.coupling_gain) * self.cross_talk_sign
        q_coupled = q + eff_coupling * delta

        q_normed, _ = self.scaling(q_coupled)
        delta_reasoner = self.proj_call_to_reasoner(q_normed - q)
        return reasoner_state + delta_reasoner

    def forward(
        self,
        query_latents: torch.Tensor,
        lh_latents: torch.Tensor
    ) -> torch.Tensor:
        """Default forward: conditions query_latents with lh_latents."""
        return self.condition_state(query_latents, lh_latents)


class CallosalPuzzleEmbedding(nn.Module):
    """
    Drop-in replacement for TRM's self.puzzle_emb embedding table.
    Instead of an unconditioned learned lookup table for puzzle IDs,
    it maps puzzle_identifiers to cached LH latents and passes them
    through the CallosalConditioning module.

    Supports System 2 Test-Time Latent Adaptation (TTA):
    - active_delta_z: temporary continuous latent offset during forward pass
    - puzzle_delta_z: dictionary of adapted latent offsets per puzzle ID
    """
    def __init__(
        self,
        cached_lh_latents: torch.Tensor,
        callosal_conditioning: CallosalConditioning,
        num_query_tokens: int = 16,
        d_reasoner: int = 512,
        flatten_output: bool = True,
        puzzle_to_base_id: Optional[torch.Tensor] = None
    ):
        super().__init__()
        # persistent=False avoids serializing static cached latents in every resume checkpoint
        self.register_buffer("cached_lh_latents", cached_lh_latents, persistent=False)
        if puzzle_to_base_id is not None:
            self.register_buffer("puzzle_to_base_id", puzzle_to_base_id.long(), persistent=False)
        else:
            self.puzzle_to_base_id = None
        self.callosum = callosal_conditioning
        self.num_query_tokens = num_query_tokens
        self.d_reasoner = d_reasoner
        self.flatten_output = flatten_output
        self.__dict__["_active_delta_z"] = None
        self.__dict__["puzzle_delta_z"] = {}

    @property
    def active_delta_z(self) -> Optional[torch.Tensor]:
        return self.__dict__.get("_active_delta_z", None)

    @active_delta_z.setter
    def active_delta_z(self, delta_z: Optional[torch.Tensor]) -> None:
        self.__dict__["_active_delta_z"] = delta_z

    def set_active_delta_z(self, delta_z: Optional[torch.Tensor]) -> None:
        """Sets an active delta_z applied to all items in current forward pass."""
        self.__dict__["_active_delta_z"] = delta_z

    def clear_active_delta_z(self) -> None:
        self.__dict__["_active_delta_z"] = None

    @property
    def puzzle_delta_z(self) -> Dict[int, torch.Tensor]:
        return self.__dict__.setdefault("puzzle_delta_z", {})

    @puzzle_delta_z.setter
    def puzzle_delta_z(self, val: Dict[int, torch.Tensor]) -> None:
        self.__dict__["puzzle_delta_z"] = val

    def set_puzzle_delta_z(self, puzzle_id: int, delta_z: torch.Tensor) -> None:
        """Sets adapted delta_z for a specific puzzle ID (and its base ID if mapped)."""
        pid = int(puzzle_id)
        self.puzzle_delta_z[pid] = delta_z
        if self.puzzle_to_base_id is not None and 0 <= pid < self.puzzle_to_base_id.size(0):
            base_id = int(self.puzzle_to_base_id[pid].item())
            self.puzzle_delta_z[base_id] = delta_z

    def clear_puzzle_delta_z(self) -> None:
        self.__dict__["puzzle_delta_z"].clear()

    def get_unadapted_task_tokens(self, puzzle_identifier: int) -> torch.Tensor:
        """Returns Z_0 for a puzzle identifier without any delta_z applied."""
        dev = self.cached_lh_latents.device
        pid = torch.tensor([puzzle_identifier], device=dev)
        if self.puzzle_to_base_id is not None and self.puzzle_to_base_id.numel() > 0:
            safe_pids = torch.clamp(pid, 0, self.puzzle_to_base_id.size(0) - 1)
            base_pids = self.puzzle_to_base_id[safe_pids]
        else:
            base_pids = torch.clamp(pid, 0, self.cached_lh_latents.size(0) - 1)
        batch_lh = self.cached_lh_latents[base_pids]
        with torch.no_grad():
            task_tokens = self.callosum.condition_task(batch_lh)
        return task_tokens  # (1, num_query_tokens, d_reasoner)

    def forward(self, puzzle_identifiers: torch.Tensor) -> torch.Tensor:
        # puzzle_identifiers: (B,) integer tensor of puzzle IDs
        dev = self.cached_lh_latents.device
        pids = puzzle_identifiers.to(dev)
        if self.cached_lh_latents.size(0) > 0:
            if self.puzzle_to_base_id is not None and self.puzzle_to_base_id.numel() > 0:
                safe_pids = torch.clamp(pids, 0, self.puzzle_to_base_id.size(0) - 1)
                base_pids = self.puzzle_to_base_id[safe_pids]
            else:
                base_pids = torch.clamp(pids, 0, self.cached_lh_latents.size(0) - 1)
            batch_lh = self.cached_lh_latents[base_pids]  # (B, S_L, d_lh)
        else:
            batch_lh = torch.zeros(pids.size(0), 32, self.callosum.d_lh, device=dev)

        task_tokens = self.callosum.condition_task(batch_lh)  # (B, num_query_tokens, d_reasoner)

        # Apply active_delta_z if present
        if self.active_delta_z is not None:
            d_z = self.active_delta_z.to(dtype=task_tokens.dtype, device=task_tokens.device)
            if d_z.dim() == 2 and d_z.shape == (self.num_query_tokens, self.d_reasoner):
                task_tokens = task_tokens + d_z.unsqueeze(0)
            elif d_z.dim() == 3 and d_z.shape[-2:] == (self.num_query_tokens, self.d_reasoner):
                task_tokens = task_tokens + d_z
            elif d_z.numel() == self.num_query_tokens * self.d_reasoner:
                task_tokens = task_tokens + d_z.view(1, self.num_query_tokens, self.d_reasoner)
            elif d_z.dim() == 2 and d_z.size(-1) == self.num_query_tokens * self.d_reasoner and d_z.size(0) == pids.size(0):
                task_tokens = task_tokens + d_z.view(pids.size(0), self.num_query_tokens, self.d_reasoner)
            else:
                raise ValueError(f"Shape mismatch for active_delta_z: expected ({self.num_query_tokens}, {self.d_reasoner}) but got {d_z.shape}")
        elif self.puzzle_delta_z:
            offsets = []
            has_offset = False
            for i in range(pids.size(0)):
                pid = int(pids[i].item())
                base_pid = int(base_pids[i].item()) if 'base_pids' in locals() else pid
                d_z = self.puzzle_delta_z.get(pid, self.puzzle_delta_z.get(base_pid, None))
                if d_z is not None:
                    offsets.append(d_z.view(self.num_query_tokens, self.d_reasoner).to(dtype=task_tokens.dtype, device=task_tokens.device))
                    has_offset = True
                else:
                    offsets.append(torch.zeros(self.num_query_tokens, self.d_reasoner, dtype=task_tokens.dtype, device=task_tokens.device))
            if has_offset:
                task_tokens = task_tokens + torch.stack(offsets, dim=0)

        if self.flatten_output:
            return task_tokens.contiguous().view(puzzle_identifiers.size(0), -1)
        return task_tokens


# ---------------------------------------------------------------------------
# Grid Sequence Helpers
# ---------------------------------------------------------------------------

def encode_grid_to_trm_seq(grid: Sequence[Sequence[int]], pad_r: int = 0, pad_c: int = 0, side: int = 30) -> np.ndarray:
    """Encodes a 2D ARC grid to TRM canonical sequence (PAD=0, EOS=1, colors 2..11)."""
    if not grid or len(grid) == 0:
        return np.zeros(side * side, dtype=np.uint8)
    g = np.asarray(grid, dtype=np.int32)
    if g.ndim == 1:
        g = g.reshape(1, -1)
    if g.size == 0 or g.shape[0] == 0 or g.shape[1] == 0:
        return np.zeros(side * side, dtype=np.uint8)

    # Crop if grid exceeds canonical canvas side
    if g.shape[0] > side or g.shape[1] > side:
        g = g[:side, :side]

    nrow, ncol = g.shape
    pad_r = max(0, min(pad_r, side - nrow))
    pad_c = max(0, min(pad_c, side - ncol))

    # Clip ARC colors to 0..9 and map to tokens 2..11
    g_tok = np.clip(g, 0, 9) + 2

    pad_after_r = max(0, side - pad_r - nrow)
    pad_after_c = max(0, side - pad_c - ncol)
    out = np.pad(g_tok, ((pad_r, pad_after_r), (pad_c, pad_after_c)), constant_values=0)
    if pad_r + nrow < side:
        out[pad_r + nrow, pad_c:pad_c + ncol] = 1
    if pad_c + ncol < side:
        out[pad_r:pad_r + nrow, pad_c + ncol] = 1
    return out.astype(np.uint8).flatten()


def decode_trm_seq_to_grid(seq: Any, side: int = 30) -> Optional[List[List[int]]]:
    """Inverts TRM canonical sequence encoding into a 2D grid."""
    if seq is None:
        return None
    arr = np.asarray(seq)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size != side * side:
            s = int(round(math.sqrt(arr.size)))
            if s * s == arr.size:
                side = s
                grid = arr.reshape(side, side)
            else:
                return None
        else:
            grid = arr.reshape(side, side)
    elif arr.ndim == 2:
        grid = arr
    else:
        return None

    mask = grid >= 2
    if not mask.any():
        return None
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    sub = grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].astype(np.int32)
    # Clip to valid ARC colors 0..9; tokens 0 (PAD) and 1 (EOS) inside grid -> color 0
    g = np.clip(sub - 2, 0, 9).astype(np.uint8)
    return g.tolist()


def infer_arc_output_shape(
    demos: Sequence[ Mapping[str, Any] ],
    test_input_grid: Sequence[Sequence[int]],
    side: int = 30
) -> Tuple[int, int]:
    """
    Infers the target output grid dimensions (H_out, W_out) strictly from task
    demonstration pairs ('train') and test_input_grid. Never accesses test outputs.
    """
    if not test_input_grid or not test_input_grid[0]:
        return 1, 1
    in_h = min(side, len(test_input_grid))
    in_w = min(side, len(test_input_grid[0]))
    if not demos:
        return in_h, in_w

    in_shapes = [(len(d["input"]), len(d["input"][0])) for d in demos if d.get("input") and d["input"][0]]
    out_shapes = [(len(d["output"]), len(d["output"][0])) for d in demos if d.get("output") and d["output"][0]]
    if len(in_shapes) != len(out_shapes) or not in_shapes:
        return in_h, in_w

    # 1. Exact input == output shape match across all demos
    if all(ish == osh for ish, osh in zip(in_shapes, out_shapes)):
        return in_h, in_w

    # 2. Constant output shape across all demos
    if len(set(out_shapes)) == 1:
        return min(side, max(1, out_shapes[0][0])), min(side, max(1, out_shapes[0][1]))

    # 3. Transposed shape across all demos (H_out == W_in and W_out == H_in)
    if all(osh[0] == ish[1] and osh[1] == ish[0] for ish, osh in zip(in_shapes, out_shapes)):
        return in_w, in_h

    # 4. Constant multiplicative scale factor (H_out = k_h * H_in, W_out = k_w * W_in)
    ratios_h = [osh[0] / max(1, ish[0]) for ish, osh in zip(in_shapes, out_shapes)]
    ratios_w = [osh[1] / max(1, ish[1]) for ish, osh in zip(in_shapes, out_shapes)]
    if max(ratios_h) - min(ratios_h) < 1e-4 and max(ratios_w) - min(ratios_w) < 1e-4:
        pred_h = max(1, min(side, int(round(in_h * ratios_h[0]))))
        pred_w = max(1, min(side, int(round(in_w * ratios_w[0]))))
        return pred_h, pred_w

    # 5. Constant additive offset (H_out = H_in + d_h, W_out = W_in + d_w)
    diffs_h = [osh[0] - ish[0] for ish, osh in zip(in_shapes, out_shapes)]
    diffs_w = [osh[1] - ish[1] for ish, osh in zip(in_shapes, out_shapes)]
    if len(set(diffs_h)) == 1 and len(set(diffs_w)) == 1:
        pred_h = max(1, min(side, in_h + diffs_h[0]))
        pred_w = max(1, min(side, in_w + diffs_w[0]))
        return pred_h, pred_w

    # 6. Dual-panel split with 1px separator line (H_in == 2*H_out + 1 or W_in == 2*W_out + 1)
    if all(ish[0] == 2 * osh[0] + 1 and ish[1] == osh[1] for ish, osh in zip(in_shapes, out_shapes)):
        if (in_h - 1) % 2 == 0 and in_h > 1:
            return max(1, min(side, (in_h - 1) // 2)), in_w
    if all(ish[1] == 2 * osh[1] + 1 and ish[0] == osh[0] for ish, osh in zip(in_shapes, out_shapes)):
        if (in_w - 1) % 2 == 0 and in_w > 1:
            return in_h, max(1, min(side, (in_w - 1) // 2))

    # 7. Non-zero foreground bounding box shape match across all demos
    def _fg_bbox_shape(g: Sequence[Sequence[int]]) -> Tuple[int, int]:
        arr = np.asarray(g)
        mask = (arr != 0)
        if not mask.any():
            return int(arr.shape[0]), int(arr.shape[1])
        rs, cs = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        return int(rs[-1] - rs[0] + 1), int(cs[-1] - cs[0] + 1)

    if all(_fg_bbox_shape(d["input"]) == osh for d, osh in zip(demos, out_shapes)):
        bh, bw = _fg_bbox_shape(test_input_grid)
        return max(1, min(side, bh)), max(1, min(side, bw))

    try:
        from darksnail.shell import infer_arc_output_shape_beltrami

        return infer_arc_output_shape_beltrami(demos, test_input_grid, side=side)
    except (ImportError, Exception):
        pass

    return in_h, in_w


class BatchDict(dict):
    """Dictionary supporting both mapping and attribute-style access for model inputs."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'BatchDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwraps DDP/DataParallel (.module) and torch.compile (._orig_mod) wrappers."""
    curr = model
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        if hasattr(curr, "module") and getattr(curr, "module") is not None:
            curr = getattr(curr, "module")
        elif hasattr(curr, "_orig_mod") and getattr(curr, "_orig_mod") is not None:
            curr = getattr(curr, "_orig_mod")
        else:
            break
    return curr


def wire_callosal_embedding(root: nn.Module, emb: nn.Module) -> None:
    """
    Universally wires a CallosalPuzzleEmbedding into any model hierarchy
    (ACTLossHead -> TRM -> TRM_Inner -> puzzle_emb), ensuring that all
    wrappers, submodules, and inner reasoning layers use the callosal embedding.
    """
    root.puzzle_emb = emb
    for attr in ("model", "module", "_orig_mod"):
        sub = getattr(root, attr, None)
        if sub is not None:
            sub.puzzle_emb = emb
            if hasattr(sub, "inner"):
                sub.inner.puzzle_emb = emb
            for sub_attr in ("model", "module", "_orig_mod"):
                sub_sub = getattr(sub, sub_attr, None)
                if sub_sub is not None:
                    sub_sub.puzzle_emb = emb
                    if hasattr(sub_sub, "inner"):
                        sub_sub.inner.puzzle_emb = emb
    if hasattr(root, "inner"):
        root.inner.puzzle_emb = emb
    for _, mod in root.named_modules():
        if hasattr(mod, "inner") and hasattr(mod.inner, "puzzle_emb"):
            mod.inner.puzzle_emb = emb


def find_callosal_embedding(model: nn.Module) -> Optional[Any]:
    """
    Recursively and systematically resolves CallosalPuzzleEmbedding across wrappers
    (e.g., ACTLossHead model.model, model.inner, DDP/DataParallel model.module, torch.compile _orig_mod).
    """
    candidates = [model]
    for attr in ("model", "inner", "module", "_orig_mod"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append(sub)
            for sub_attr in ("model", "inner", "module", "_orig_mod"):
                sub_sub = getattr(sub, sub_attr, None)
                if sub_sub is not None:
                    candidates.append(sub_sub)
                    for sub_sub_attr in ("model", "inner", "module", "_orig_mod"):
                        leaf = getattr(sub_sub, sub_sub_attr, None)
                        if leaf is not None:
                            candidates.append(leaf)

    for cand in candidates:
        if cand is None:
            continue
        if hasattr(cand, "puzzle_emb"):
            emb = getattr(cand, "puzzle_emb")
            if hasattr(emb, "set_active_delta_z") and hasattr(emb, "set_puzzle_delta_z"):
                return emb
        if hasattr(cand, "inner") and hasattr(getattr(cand, "inner"), "puzzle_emb"):
            emb = getattr(cand.inner, "puzzle_emb")
            if hasattr(emb, "set_active_delta_z") and hasattr(emb, "set_puzzle_delta_z"):
                return emb

    # Deep fallback: traverse named modules
    for _, m in model.named_modules():
        if hasattr(m, "set_active_delta_z") and hasattr(m, "set_puzzle_delta_z"):
            return m

    return None


# ---------------------------------------------------------------------------
# System 2 Test-Time Latent Adaptation (TTA)
# ---------------------------------------------------------------------------

def _ensure_carry_device(carry: Any, device: torch.device) -> Any:
    """Recursively walks dataclasses, mappings, and sequences to ensure all tensors match device."""
    if isinstance(carry, torch.Tensor):
        return carry.to(device)
    if hasattr(carry, "__dataclass_fields__"):
        for f in carry.__dataclass_fields__:
            val = getattr(carry, f)
            setattr(carry, f, _ensure_carry_device(val, device))
        return carry
    if isinstance(carry, dict):
        return {k: _ensure_carry_device(v, device) for k, v in carry.items()}
    if isinstance(carry, list):
        return [_ensure_carry_device(v, device) for v in carry]
    if isinstance(carry, tuple):
        return tuple(_ensure_carry_device(v, device) for v in carry)
    return carry


def _clone_carry(carry: Any) -> Any:
    """Recursively detaches and clones tensors inside a carry dataclass, dict, list, or tuple."""
    if isinstance(carry, torch.Tensor):
        return carry.detach().clone()
    if hasattr(carry, "__dataclass_fields__"):
        kwargs = {f: _clone_carry(getattr(carry, f)) for f in carry.__dataclass_fields__}
        return type(carry)(**kwargs)
    if isinstance(carry, dict):
        return {k: _clone_carry(v) for k, v in carry.items()}
    if isinstance(carry, list):
        return [_clone_carry(v) for v in carry]
    if isinstance(carry, tuple):
        return tuple(_clone_carry(v) for v in carry)
    return carry


class System2LatentAdaptor:
    """
    System 2 Test-Time Latent Adaptation (TTA) with Continuous Latent Relaxation (delta_z).

    Formalized in docs/RESEARCH_PLAN.md §4.2 and ARCHITECTURE_SPEC.md §3:
    1. Takes the initial callosal conditioning state Z_0 = Callosum(Z_L) for a task.
    2. Freezes underlying model weights (reasoner + callosum).
    3. Optimizes a continuous float32 master latent offset delta_z in R^(num_query_tokens x d_reasoner)
       for N_TTA steps (default 30) using Adam strictly on the task's demonstration
       input/output pairs (L_CE) at the model's terminal recurrent ACT step.
    4. Includes mean-normalized proximal quadratic regularization (0.5 * lambda_anchor * mean(delta_z^2))
       to bound latent drift without overwhelming token-mean cross-entropy gradients.
    5. Bridges delta_z across the spatial canvas via transcallosal readout through the frozen
       lm_head (identically zero at delta_z = 0, preserving exact Step 0 equivalence).
    6. Evaluates the adapted state Z* = Z_0 + delta_z on the held-out test challenge grid.
    7. Implements Step 0 baseline comparison: if post-adaptation demonstration loss
       does not improve over Step 0, falls back to Z_0 (weak Pareto monotonicity).
    8. Strict leakage guarantee: test challenge outputs are NEVER accessed during TTA.
    """
    def __init__(
        self,
        n_steps: int = 30,
        lr: float = 0.05,
        lambda_anchor: float = 0.1,
        max_act_steps: int = 16,
        readout_scale: float = 6.0,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_anchor = lambda_anchor
        self.max_act_steps = max_act_steps
        self.readout_scale = readout_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._basis_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pinv_cache: Dict[Any, torch.Tensor] = {}
        self._task_demos_by_pid: Dict[int, list] = {}
        self._active_target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None
        self._active_color_preserving: bool = False
        self._active_unroll_steps: int = 1
        self._cached_carry_trajectory: List[Any] = []
        self._full_carry_history: List[Any] = []
        self.respect_controller_schedule: bool = False
        self._active_ray_quantiles: Optional[List[float]] = None

    def _get_pre_terminal_carry(
        self,
        model: nn.Module,
        batch: BatchDict,
        model_device: Optional[Union[torch.device, str, int]] = None,
        unroll_steps: Optional[int] = None
    ) -> Tuple[Optional[Any], int]:
        """
        Rolls an ACT recurrent model forward under torch.no_grad() (with delta_z=None)
        and records carry_history across steps t = 0 .. T - 1.
        Returns (carry_history[-k_unroll], total_act_steps) and stores
        self._cached_carry_trajectory = carry_history[-k_unroll:].
        """
        if isinstance(model_device, int) and not isinstance(model_device, bool):
            unroll_steps = model_device
            model_device = None
        if model_device is None:
            if isinstance(batch, Mapping) and "inputs" in batch and isinstance(batch["inputs"], torch.Tensor):
                model_device = batch["inputs"].device
            else:
                try:
                    model_device = next(model.parameters()).device
                except StopIteration:
                    model_device = torch.device(self.device)
        elif isinstance(model_device, str):
            model_device = torch.device(model_device)

        m_inner = _unwrap_model(model)
        m_act = _unwrap_model(getattr(m_inner, "model", m_inner))

        carry_provider = None
        if hasattr(m_act, "initial_carry"):
            carry_provider = m_act
        elif hasattr(m_inner, "initial_carry"):
            carry_provider = m_inner

        if carry_provider is None:
            self._cached_carry_trajectory = []
            self._full_carry_history = []
            return None, 1

        k_unroll = max(
            1,
            min(
                int(unroll_steps if unroll_steps is not None else getattr(self, "_active_unroll_steps", 1)),
                self.max_act_steps,
            ),
        )

        callosal_emb = find_callosal_embedding(model)
        saved_dz = getattr(callosal_emb, "active_delta_z", None) if callosal_emb is not None else None
        if callosal_emb is not None and hasattr(callosal_emb, "clear_active_delta_z"):
            callosal_emb.clear_active_delta_z()

        try:
            with torch.no_grad():
                with torch.device(model_device):
                    carry = carry_provider.initial_carry(batch)
                carry = _ensure_carry_device(carry, model_device)
                carry_history: List[Any] = []
                steps = 0
                while True:
                    carry_history.append(_clone_carry(carry))
                    if hasattr(m_act, "forward") and m_act is not m_inner:
                        carry, _ = m_act(carry, batch)
                        all_finish = bool(getattr(carry, "halted", torch.tensor(True)).all().item())
                    else:
                        try:
                            res = m_inner(["logits"], carry=carry, batch=batch)
                        except TypeError:
                            try:
                                res = m_inner(carry=carry, batch=batch, return_keys=["logits"])
                            except TypeError:
                                res = m_inner(carry, batch)
                        if isinstance(res, tuple) and len(res) >= 1:
                            carry = res[0]
                            last = res[-1]
                            if len(res) >= 3 and isinstance(last, torch.Tensor):
                                all_finish = bool(last.all().item())
                            elif len(res) >= 3 and isinstance(last, bool):
                                all_finish = last
                            else:
                                halted_attr = getattr(carry, "halted", carry.get("halted", None) if isinstance(carry, dict) else None)
                                all_finish = bool(halted_attr.all().item()) if isinstance(halted_attr, torch.Tensor) else True
                        else:
                            all_finish = True
                    steps += 1
                    if all_finish or (steps >= self.max_act_steps):
                        break
                eff_k = max(1, min(k_unroll, len(carry_history)))
                self._full_carry_history = carry_history
                self._cached_carry_trajectory = carry_history[-eff_k:]
                return _clone_carry(carry_history[-eff_k]), max(1, steps)
        except Exception:
            self._cached_carry_trajectory = []
            self._full_carry_history = []
            return None, 1
        finally:
            if callosal_emb is not None and hasattr(callosal_emb, "set_active_delta_z"):
                callosal_emb.set_active_delta_z(saved_dz)

    def _compute_2d_dct_basis(self, inputs: torch.Tensor, num_q: int) -> torch.Tensor:
        """
        Computes a scale-invariant 2D Discrete Cosine Transform (DCT) spatial basis
        over each grid's active bounding box (H_b, W_b), returning (B, L, num_q).
        """
        B, L = inputs.shape[0], inputs.shape[1]
        device = inputs.device
        side = int(round(math.sqrt(L)))
        if side * side == L:
            idx = torch.arange(L, device=device, dtype=torch.float32)
            r = idx // side
            c = idx % side
            active = (inputs >= 2)
            if active.any():
                r_max = torch.where(active, r.unsqueeze(0), torch.zeros(1, device=device)).amax(dim=1, keepdim=True).clamp(min=0.0) + 1.0
                c_max = torch.where(active, c.unsqueeze(0), torch.zeros(1, device=device)).amax(dim=1, keepdim=True).clamp(min=0.0) + 1.0
            else:
                r_max = torch.full((B, 1), float(side), device=device)
                c_max = torch.full((B, 1), float(side), device=device)
            r_norm = ((r.unsqueeze(0) + 0.5) / r_max).clamp(0.0, 1.0)
            c_norm = ((c.unsqueeze(0) + 0.5) / c_max).clamp(0.0, 1.0)
            q_side = max(1, int(round(math.sqrt(num_q))))
            q_idx = torch.arange(num_q, device=device, dtype=torch.float32)
            u = (q_idx // q_side).view(1, 1, num_q)
            v = (q_idx % q_side).view(1, 1, num_q)
            basis = torch.cos(math.pi * r_norm.unsqueeze(-1) * u) * torch.cos(math.pi * c_norm.unsqueeze(-1) * v)
            return basis / math.sqrt(float(num_q))
        else:
            pos = (torch.arange(L, device=device, dtype=torch.float32) + 0.5) / max(1, L)
            q_idx = torch.arange(num_q, device=device, dtype=torch.float32).view(1, 1, num_q)
            basis = torch.cos(math.pi * pos.view(1, L, 1) * q_idx) / math.sqrt(float(num_q))
            return basis.expand(B, L, num_q)

    def _get_spatial_relational_tensors(
        self,
        inputs: torch.Tensor,
        target_hw: Optional[Union[Tuple[int, int], Sequence[Tuple[int, int]]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds (and caches per input tensor) the color-equivariant basis E (B, L, M_eq, 12)
        and color-relational spatial feature basis Phi (B, L, K_rel) on the 2D canvas.
        """
        content_hash = hashlib.blake2b(
            inputs.detach().cpu().numpy().tobytes(), digest_size=16
        ).hexdigest()
        hw_key = (
            tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in target_hw)
            if isinstance(target_hw, (list, tuple))
            else target_hw
        )
        cache_key = (content_hash, tuple(inputs.shape), hw_key, str(inputs.device))
        if cache_key in self._basis_cache:
            cached_E, cached_Phi = self._basis_cache[cache_key]
            return cached_E, cached_Phi

        B, L = inputs.shape
        device = inputs.device
        side = int(round(math.sqrt(L)))
        g = inputs.long().clamp(0, 11).view(B, side, side)
        oh = F.one_hot(g, num_classes=12).float()  # (B, side, side, 12)

        fg_mask = torch.zeros(1, 1, 1, 12, device=device)
        fg_mask[:, :, :, 3:] = 1.0
        oh_fg = oh * fg_mask

        pad_oh = F.pad(oh.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="constant", value=0.0).permute(0, 2, 3, 1)
        n_up = pad_oh[:, 0:side, 1:1 + side, :]
        n_down = pad_oh[:, 2:2 + side, 1:1 + side, :]
        n_left = pad_oh[:, 1:1 + side, 0:side, :]
        n_right = pad_oh[:, 1:1 + side, 2:2 + side, :]
        n_ul = pad_oh[:, 0:side, 0:side, :]
        n_ur = pad_oh[:, 0:side, 2:2 + side, :]
        n_dl = pad_oh[:, 2:2 + side, 0:side, :]
        n_dr = pad_oh[:, 2:2 + side, 2:2 + side, :]

        card_fg = (n_up + n_down + n_left + n_right) * fg_mask
        diag_fg = (n_ul + n_ur + n_dl + n_dr) * fg_mask
        horiz_fg = (n_left + n_right) * fg_mask
        vert_fg = (n_up + n_down) * fg_mask

        row_fg_max = oh_fg.amax(dim=2, keepdim=True).expand(B, side, side, 12)
        col_fg_max = oh_fg.amax(dim=1, keepdim=True).expand(B, side, side, 12)
        row_fg_sum = oh_fg.sum(dim=2, keepdim=True).expand(B, side, side, 12) / 10.0
        col_fg_sum = oh_fg.sum(dim=1, keepdim=True).expand(B, side, side, 12) / 10.0
        crosshair_fg = row_fg_max * col_fg_max

        idx_1d = torch.arange(side, device=device)
        r_grid, c_grid = torch.meshgrid(idx_1d, idx_1d, indexing="ij")
        active = (g >= 2)

        eq_per_sample = []
        rel_per_sample = []

        for b in range(B):
            act_b = active[b]
            if act_b.any():
                hb = int(r_grid[act_b].max().item()) + 1
                wb = int(c_grid[act_b].max().item()) + 1
            else:
                hb, wb = side, side

            if target_hw is None:
                th, tw = hb, wb
            elif (
                isinstance(target_hw, (list, tuple))
                and len(target_hw) > 0
                and isinstance(target_hw[0], (list, tuple))
            ):
                th, tw = target_hw[min(b, len(target_hw) - 1)]
            else:
                th, tw = target_hw  # type: ignore[misc]
            oh_b = oh[b]
            oh_fg_b = oh_fg[b]
            gb = g[b]

            rh_flip = (hb - 1 - r_grid).clamp(0, side - 1)
            cw_flip = (wb - 1 - c_grid).clamp(0, side - 1)

            s_fh = oh_b[r_grid, cw_flip]
            s_fv = oh_b[rh_flip, c_grid]
            s_r180 = oh_b[rh_flip, cw_flip]
            s_tr = oh_b[c_grid, r_grid]
            s_r90 = oh_b[(wb - 1 - c_grid).clamp(0, side - 1), r_grid]
            s_r270 = oh_b[c_grid, (hb - 1 - r_grid).clamp(0, side - 1)]
            s_atr = oh_b[(wb - 1 - c_grid).clamp(0, side - 1), (hb - 1 - r_grid).clamp(0, side - 1)]

            s_fh_fg = s_fh * fg_mask[0]
            s_fv_fg = s_fv * fg_mask[0]
            s_r180_fg = s_r180 * fg_mask[0]
            s_tr_fg = s_tr * fg_mask[0]

            is_upscale = float(th > hb or tw > wb)
            r_mod = r_grid % max(1, hb)
            c_mod = c_grid % max(1, wb)
            r_mod_f = (hb - 1 - r_mod).clamp(0, side - 1)
            c_mod_f = (wb - 1 - c_mod).clamp(0, side - 1)
            qr = (r_grid // max(1, hb)) % 2
            qc = (c_grid // max(1, wb)) % 2

            t_base = oh_b[r_mod, c_mod] * is_upscale
            t_fh = torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod, c_mod_f], oh_b[r_mod, c_mod]) * is_upscale
            t_fv = torch.where((qr == 1).unsqueeze(-1), oh_b[r_mod_f, c_mod], oh_b[r_mod, c_mod]) * is_upscale
            t_fh_alt_row = torch.where((qr == 1).unsqueeze(-1), oh_b[r_mod, c_mod_f], oh_b[r_mod, c_mod]) * is_upscale
            t_fv_alt_col = torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod_f, c_mod], oh_b[r_mod, c_mod]) * is_upscale
            t_fhv = torch.where(
                (qr == 1).unsqueeze(-1),
                torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod_f, c_mod_f], oh_b[r_mod_f, c_mod]),
                torch.where((qc == 1).unsqueeze(-1), oh_b[r_mod, c_mod_f], oh_b[r_mod, c_mod])
            ) * is_upscale

            r_div = (r_grid // max(1, hb)).clamp(0, side - 1)
            c_div = (c_grid // max(1, wb)).clamp(0, side - 1)
            macro_fg = (gb[r_div, c_div] >= 3).float().unsqueeze(-1) * is_upscale
            t_kron = macro_fg * oh_b[r_mod, c_mod]
            micro_fg = (gb[r_mod, c_mod] >= 3).float().unsqueeze(-1) * is_upscale
            t_kron_inv = micro_fg * oh_b[r_div, c_div]

            is_scaled = float(th != hb or tw != wb)
            r_sc = ((r_grid * hb) // max(1, th)).clamp(0, side - 1)
            c_sc = ((c_grid * wb) // max(1, tw)).clamp(0, side - 1)
            t_scale = oh_b[r_sc, c_sc] * is_scaled

            has_v_panel = float(hb >= 2 * th)
            has_h_panel = float(wb >= 2 * tw)
            v_off = (th + 1) if (hb == 2 * th + 1) else th
            h_off = (tw + 1) if (wb == 2 * tw + 1) else tw
            p_down = oh_b[(r_grid + v_off).clamp(0, side - 1), c_grid] * has_v_panel
            p_right = oh_b[r_grid, (c_grid + h_off).clamp(0, side - 1)] * has_h_panel

            fg_b = (gb >= 3)
            if fg_b.any():
                r0 = int(r_grid[fg_b].min().item())
                c0 = int(c_grid[fg_b].min().item())
            else:
                r0, c0 = 0, 0
            crop_fg = oh_b[(r_grid + r0).clamp(0, side - 1), (c_grid + c0).clamp(0, side - 1)]

            eq_b = torch.stack([
                oh_b, oh_fg_b,
                n_up[b], n_down[b], n_left[b], n_right[b],
                card_fg[b], diag_fg[b], horiz_fg[b], vert_fg[b],
                row_fg_max[b], col_fg_max[b], row_fg_sum[b], col_fg_sum[b], crosshair_fg[b],
                s_fh, s_fv, s_r180, s_tr, s_r90, s_r270, s_atr,
                s_fh_fg, s_fv_fg, s_r180_fg, s_tr_fg,
                t_base, t_fh, t_fv, t_fh_alt_row, t_fv_alt_col, t_fhv, t_kron, t_kron_inv, t_scale,
                p_down, p_right, crop_fg
            ], dim=-2)
            eq_per_sample.append(eq_b)

            p1_fg = (gb >= 3).float()
            p2_v_fg = (gb[(r_grid + v_off).clamp(0, side - 1), c_grid] >= 3).float() * has_v_panel
            p2_h_fg = (gb[r_grid, (c_grid + h_off).clamp(0, side - 1)] >= 3).float() * has_h_panel

            v_and = p1_fg * p2_v_fg
            v_or = ((p1_fg + p2_v_fg) >= 1.0).float() * has_v_panel
            v_xor = (p1_fg != p2_v_fg).float() * has_v_panel
            v_nor = ((p1_fg == 0) & (p2_v_fg == 0)).float() * has_v_panel

            h_and = p1_fg * p2_h_fg
            h_or = ((p1_fg + p2_h_fg) >= 1.0).float() * has_h_panel
            h_xor = (p1_fg != p2_h_fg).float() * has_h_panel
            h_nor = ((p1_fg == 0) & (p2_h_fg == 0)).float() * has_h_panel

            in_box = ((r_grid < th) & (c_grid < tw)).float()
            eos_border = (((r_grid == th) & (c_grid <= tw)) | ((c_grid == tw) & (r_grid <= th))).float()
            pad_ext = (1.0 - in_box) * (1.0 - eos_border)

            scalar_b = torch.stack([
                p1_fg, 1.0 - p1_fg,
                v_and, v_or, v_xor, v_nor,
                h_and, h_or, h_xor, h_nor,
                in_box, eos_border, pad_ext
            ], dim=-1)

            rel_b = torch.cat([
                oh_b,
                card_fg[b], diag_fg[b],
                row_fg_max[b], col_fg_max[b], crosshair_fg[b],
                s_fh, s_fv, s_r180, s_tr,
                t_base, t_fh_alt_row, t_kron, t_scale,
                p_down, p_right,
                scalar_b
            ], dim=-1)
            rel_per_sample.append(rel_b)

        E_t = torch.stack(eq_per_sample, dim=0).view(B, L, -1, 12).permute(0, 1, 3, 2).contiguous()
        Phi = torch.stack(rel_per_sample, dim=0).view(B, L, -1).contiguous()
        if len(self._basis_cache) >= 8:
            self._basis_cache.clear()
        self._basis_cache[cache_key] = (E_t, Phi)
        return E_t, Phi

    @staticmethod
    def _is_demo_color_preserving(demos: Optional[Sequence[Mapping[str, Any]]]) -> bool:
        """
        Determines strictly from task demonstrations whether every demo output only uses
        foreground colors present in its own demo input across multiple distinct demo palettes.
        """
        if not demos or len(demos) < 2:
            return False
        in_palettes = []
        for d in demos:
            inp_c = set(np.unique(d["input"]).tolist()) - {0}
            out_c = set(np.unique(d["output"]).tolist()) - {0}
            if not out_c or not out_c.issubset(inp_c):
                return False
            in_palettes.append(frozenset(inp_c))
        return len(set(in_palettes)) > 1

    def _get_lm_pinv(self, lm_weight: torch.Tensor) -> torch.Tensor:
        """Computes and caches the right-pseudoinverse of the frozen readout head weight."""
        w_hash = hashlib.blake2b(
            lm_weight.detach().float().cpu().numpy().tobytes(), digest_size=16
        ).hexdigest()
        key = (w_hash, tuple(lm_weight.shape), str(lm_weight.device))
        if key not in self._pinv_cache:
            if len(self._pinv_cache) >= 4:
                self._pinv_cache.clear()
            self._pinv_cache[key] = torch.linalg.pinv(lm_weight.detach().float())
        return self._pinv_cache[key]

    @staticmethod
    def _set_carry_zh_zl(carry: Any, z_H: torch.Tensor, z_L: torch.Tensor) -> Any:
        """Updates (z_H, z_L) on carry or carry.inner_carry without mutating cached tensors."""
        ic = getattr(carry, "inner_carry", None)
        if ic is not None and hasattr(ic, "z_H") and hasattr(ic, "z_L"):
            if hasattr(ic, "__dataclass_fields__"):
                ic_kw = {f: getattr(ic, f) for f in ic.__dataclass_fields__}
                ic_kw["z_H"] = z_H
                ic_kw["z_L"] = z_L
                new_ic = type(ic)(**ic_kw)
            else:
                new_ic = copy.copy(ic)
                new_ic.z_H = z_H
                new_ic.z_L = z_L
            if hasattr(carry, "__dataclass_fields__"):
                c_kw = {f: getattr(carry, f) for f in carry.__dataclass_fields__}
                c_kw["inner_carry"] = new_ic
                return type(carry)(**c_kw)
            new_c = copy.copy(carry)
            new_c.inner_carry = new_ic
            return new_c

        if hasattr(carry, "z_H") and hasattr(carry, "z_L"):
            if hasattr(carry, "__dataclass_fields__"):
                c_kw = {f: getattr(carry, f) for f in carry.__dataclass_fields__}
                c_kw["z_H"] = z_H
                c_kw["z_L"] = z_L
                return type(carry)(**c_kw)
            new_c = copy.copy(carry)
            new_c.z_H = z_H
            new_c.z_L = z_L
            return new_c

        if isinstance(carry, dict):
            new_c = dict(carry)
            if "z_H" in new_c and "z_L" in new_c:
                new_c["z_H"] = z_H
                new_c["z_L"] = z_L
            elif "inner_carry" in new_c and isinstance(new_c["inner_carry"], dict):
                new_ic = dict(new_c["inner_carry"])
                new_ic["z_H"] = z_H
                new_ic["z_L"] = z_L
                new_c["inner_carry"] = new_ic
            return new_c

        return carry

    @staticmethod
    def _get_carry_zh_zl(carry: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extracts (z_H, z_L) from carry or carry.inner_carry if present."""
        if carry is None:
            return None, None
        ic = getattr(carry, "inner_carry", None)
        if ic is not None and hasattr(ic, "z_H") and hasattr(ic, "z_L"):
            return getattr(ic, "z_H", None), getattr(ic, "z_L", None)
        if hasattr(carry, "z_H") and hasattr(carry, "z_L"):
            return getattr(carry, "z_H", None), getattr(carry, "z_L", None)
        if isinstance(carry, dict):
            if "z_H" in carry and "z_L" in carry:
                return carry["z_H"], carry["z_L"]
            if "inner_carry" in carry and isinstance(carry["inner_carry"], dict):
                ic_d = carry["inner_carry"]
                return ic_d.get("z_H", None), ic_d.get("z_L", None)
        return None, None

    @staticmethod
    def _clear_carry_halted(carry: Any) -> Any:
        """Sets halted mask to False on intermediate unroll steps so step s+1 does not reset."""
        if hasattr(carry, "halted") and isinstance(carry.halted, torch.Tensor):
            if bool(carry.halted.any().item()):
                if hasattr(carry, "__dataclass_fields__"):
                    c_kw = {f: getattr(carry, f) for f in carry.__dataclass_fields__}
                    c_kw["halted"] = torch.zeros_like(carry.halted)
                    return type(carry)(**c_kw)
                new_c = copy.copy(carry)
                new_c.halted = torch.zeros_like(carry.halted)
                return new_c
        elif isinstance(carry, dict) and "halted" in carry and isinstance(carry["halted"], torch.Tensor):
            if bool(carry["halted"].any().item()):
                new_c = dict(carry)
                new_c["halted"] = torch.zeros_like(carry["halted"])
                return new_c
        return carry

    def _maybe_apply_cascade_to_carry(
        self,
        carry: Any,
        m_act: nn.Module,
        batch: BatchDict,
        delta_z: Optional[torch.Tensor],
        step_idx: int,
        unroll_steps: int,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Applies self._apply_slow_fast_cascade(z_H, z_L, delta_z, step_idx=step_idx, unroll_steps=unroll_steps)
        to carry before an ACT recurrent step when delta_z is not None.
        """
        if delta_z is None or not hasattr(self, "_apply_slow_fast_cascade"):
            return carry, None

        # If carry is at step 0 with halted=True and m_act.inner has reset_carry, initialize H_init/L_init first
        if (
            hasattr(carry, "halted")
            and isinstance(carry.halted, torch.Tensor)
            and bool(carry.halted.all().item())
            and hasattr(m_act, "inner")
            and hasattr(m_act.inner, "reset_carry")
            and hasattr(carry, "inner_carry")
        ):
            try:
                reset_inner = m_act.inner.reset_carry(carry.halted, carry.inner_carry)
                reset_steps = (
                    torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
                    if hasattr(carry, "steps") and isinstance(carry.steps, torch.Tensor)
                    else getattr(carry, "steps", None)
                )
                reset_data = (
                    {
                        k: torch.where(
                            carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                            batch[k],
                            v,
                        )
                        for k, v in carry.current_data.items()
                        if k in batch
                    }
                    if hasattr(carry, "current_data") and isinstance(carry.current_data, dict)
                    else getattr(carry, "current_data", None)
                )
                reset_halted = torch.zeros_like(carry.halted)
                if hasattr(carry, "__dataclass_fields__"):
                    c_kw = {f: getattr(carry, f) for f in carry.__dataclass_fields__}
                    c_kw["inner_carry"] = reset_inner
                    if "steps" in c_kw and reset_steps is not None:
                        c_kw["steps"] = reset_steps
                    if "current_data" in c_kw and reset_data is not None:
                        c_kw["current_data"] = reset_data
                    c_kw["halted"] = reset_halted
                    carry = type(carry)(**c_kw)
            except Exception:
                pass

        z_H, z_L = self._get_carry_zh_zl(carry)
        if (
            isinstance(z_H, torch.Tensor)
            and isinstance(z_L, torch.Tensor)
            and bool(torch.isfinite(z_H).all().item())
            and bool(torch.isfinite(z_L).all().item())
        ):
            z_L_new = self._apply_slow_fast_cascade(
                z_H, z_L, delta_z, step_idx=step_idx, unroll_steps=unroll_steps
            )
            dz_L = z_L_new - z_L.detach()
            if dz_L.requires_grad and z_H.shape == dz_L.shape:
                z_H_mod = z_H + 0.1 * (dz_L - dz_L.detach())
            else:
                z_H_mod = z_H
            carry = self._set_carry_zh_zl(carry, z_H_mod, z_L_new)
            return carry, dz_L
        return carry, None

    def _apply_callosal_canvas_readout(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        inputs: torch.Tensor,
        delta_z: Optional[torch.Tensor],
        carry: Any = None,
        callosal_emb: Any = None
    ) -> torch.Tensor:
        """
        Projects delta_z across the spatial canvas through the frozen readout head (lm_head).
        Because every term is strictly linear in delta_z, this contribution is identically
        zero at delta_z = 0 (preserving 100% Step 0 System 1 equivalence) while eliminating
        the prefix-slicing, stablemax logit scale, and bfloat16 quantization choke points.
        """
        logits_f32 = logits.float()
        if logits_f32.dim() == 3:
            logit_std = logits_f32.std(dim=-1, keepdim=True)
            if bool((logit_std > 2.5).any().item()):
                scale = torch.clamp(logit_std / 1.5, min=1.0)
                logits_f32 = logits_f32 / scale

        if delta_z is None or logits_f32.dim() != 3 or logits_f32.size(1) != inputs.size(1):
            return logits_f32

        if callosal_emb is not None and hasattr(callosal_emb, "num_query_tokens"):
            num_q = callosal_emb.num_query_tokens
            d_r = callosal_emb.d_reasoner
        elif delta_z.dim() == 2:
            num_q, d_r = delta_z.shape
        else:
            return logits_f32

        if delta_z.numel() != num_q * d_r:
            return logits_f32

        # Locate frozen readout head weight (vocab_size, d_reasoner)
        lm_weight = None
        m_inner = _unwrap_model(model)
        m_act = _unwrap_model(getattr(m_inner, "model", m_inner))
        for cand in (getattr(m_act, "inner", None), m_act, m_inner):
            if cand is None:
                continue
            for attr in ("lm_head", "head"):
                head_mod = getattr(cand, attr, None)
                if head_mod is not None and hasattr(head_mod, "weight") and isinstance(head_mod.weight, torch.Tensor):
                    if head_mod.weight.shape == (logits_f32.size(-1), d_r):
                        lm_weight = head_mod.weight.detach().float()
                        break
            if lm_weight is not None:
                break

        if lm_weight is None:
            return logits_f32

        B, L = inputs.shape[0], inputs.shape[1]
        d_z = delta_z.float().view(1, num_q, d_r).to(logits_f32.device)
        d_z_batch = d_z.expand(B, num_q, d_r)

        side = int(round(math.sqrt(L)))
        if side * side == L and logits_f32.size(-1) == 12 and delta_z.numel() >= 4096:
            # Full-rank color-equivariant & spatial-relational canvas readout for 2D ARC grids
            E_t, Phi = self._get_spatial_relational_tensors(inputs, target_hw=self._active_target_hw)
            M_eq = E_t.size(-1)
            K_rel = Phi.size(-1)
            flat_dz = d_z.reshape(-1)
            eq_gain = 4.0 if self._active_color_preserving else 2.0
            rel_gain = 0.25 if self._active_color_preserving else 1.0
            w_eq = flat_dz[:M_eq].view(M_eq, 1) * eq_gain
            needed = K_rel * 12
            rem = flat_dz[M_eq : M_eq + needed].view(K_rel, 12)
            rel_scale = torch.full((K_rel, 1), 0.6 * rel_gain, device=rem.device, dtype=rem.dtype)
            rel_scale[:12, 0] = 2.0 * rel_gain
            u_rel = rem * rel_scale
            shift_eq = torch.matmul(E_t, w_eq).squeeze(-1)
            shift_rel = torch.matmul(Phi, u_rel)
            raw_shift = shift_eq + shift_rel
            W_pinv = self._get_lm_pinv(lm_weight)
            dz_canvas = self.readout_scale * torch.matmul(raw_shift, W_pinv.t())

            z_H = getattr(getattr(carry, "inner_carry", None), "z_H", None)
            if isinstance(z_H, torch.Tensor) and z_H.dim() == 3 and z_H.size(1) == L + num_q and z_H.size(2) == d_r:
                z_prefix = F.normalize(z_H[:, :num_q, :].detach().float(), dim=-1)
                z_canvas = F.normalize(z_H[:, num_q:, :].detach().float(), dim=-1)
                attn = F.softmax(torch.matmul(z_canvas, z_prefix.transpose(-2, -1)) * math.sqrt(float(num_q)), dim=-1)
                dz_canvas = dz_canvas + (0.25 * self.readout_scale) * torch.matmul(attn, d_z_batch)

            delta_logits = F.linear(dz_canvas, lm_weight)
            return logits_f32 + delta_logits

        # Compact fallback for low-dimensional unit test models (delta_z.numel() < 512)
        if num_q > 12:
            tok_idx = inputs.long().clamp(0, 11).unsqueeze(-1).expand(B, L, d_r)
            dz_tok = torch.gather(d_z_batch[:, :12, :], 1, tok_idx)
            dct_basis = self._compute_2d_dct_basis(inputs, num_q - 12)
            dz_dct = 0.25 * torch.bmm(dct_basis, d_z_batch[:, 12:, :])
        else:
            tok_idx = (inputs.long().clamp(min=0) % num_q).unsqueeze(-1).expand(B, L, d_r)
            dz_tok = torch.gather(d_z_batch, 1, tok_idx)
            dct_basis = self._compute_2d_dct_basis(inputs, num_q)
            dz_dct = 0.25 * torch.bmm(dct_basis, d_z_batch)

        z_H = getattr(getattr(carry, "inner_carry", None), "z_H", None)
        if isinstance(z_H, torch.Tensor) and z_H.dim() == 3 and z_H.size(1) == L + num_q and z_H.size(2) == d_r:
            z_prefix = F.normalize(z_H[:, :num_q, :].detach().float(), dim=-1)
            z_canvas = F.normalize(z_H[:, num_q:, :].detach().float(), dim=-1)
            attn = F.softmax(torch.matmul(z_canvas, z_prefix.transpose(-2, -1)) * math.sqrt(float(num_q)), dim=-1)
            dz_spatial = 0.25 * torch.matmul(attn, d_z_batch)
            dz_canvas = self.readout_scale * (dz_tok + dz_dct + dz_spatial)
        else:
            dz_canvas = self.readout_scale * (dz_tok + dz_dct)

        delta_logits = F.linear(dz_canvas, lm_weight)

        has_active_cells = bool((inputs >= 2).any().item())
        if has_active_cells and logits_f32.size(-1) >= 12:
            active_mask = (inputs >= 2).unsqueeze(-1).float()
            color_channel_mask = torch.ones(1, 1, logits_f32.size(-1), device=logits_f32.device, dtype=torch.float32)
            color_channel_mask[:, :, :2] = 0.0
            delta_logits = delta_logits * active_mask * color_channel_mask

        return logits_f32 + delta_logits

    def _extract_terminal_logits(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        pids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        delta_z: Optional[torch.Tensor] = None,
        pre_terminal_carry: Optional[Any] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Runs the terminal recurrent step (or K-step trajectory unroll from pre_terminal_carry)
        or feedforward pass and returns (logits_f32, direct_loss).
        """
        callosal_emb = find_callosal_embedding(model)
        if callosal_emb is not None and hasattr(callosal_emb, "set_active_delta_z"):
            callosal_emb.set_active_delta_z(delta_z)

        model_device = inputs.device
        batch_dict = {"inputs": inputs, "puzzle_identifiers": pids}
        if labels is not None:
            batch_dict["labels"] = labels
        else:
            batch_dict["labels"] = torch.zeros_like(inputs)
        batch = BatchDict(batch_dict)

        m_inner = _unwrap_model(model)
        m_act = _unwrap_model(getattr(m_inner, "model", m_inner))

        carry_provider = None
        if hasattr(m_act, "initial_carry"):
            carry_provider = m_act
        elif hasattr(m_inner, "initial_carry"):
            carry_provider = m_inner

        logits = None
        direct_loss = None

        if carry_provider is not None:
            try:
                k_unroll = max(1, int(getattr(self, "_active_unroll_steps", 1)))
                if pre_terminal_carry is not None:
                    carry = _clone_carry(pre_terminal_carry)
                    eff_k = (
                        len(self._cached_carry_trajectory)
                        if getattr(self, "_cached_carry_trajectory", None)
                        else k_unroll
                    )
                    total_act_steps = (
                        len(self._full_carry_history)
                        if getattr(self, "_full_carry_history", None)
                        else max(eff_k, self.max_act_steps)
                    )
                else:
                    carry, total_act_steps = self._get_pre_terminal_carry(
                        model, batch, model_device, unroll_steps=k_unroll
                    )
                    if callosal_emb is not None and hasattr(callosal_emb, "set_active_delta_z"):
                        callosal_emb.set_active_delta_z(delta_z)
                    if carry is None:
                        with torch.device(model_device):
                            carry = carry_provider.initial_carry(batch)
                        carry = _ensure_carry_device(carry, model_device)
                        eff_k = 1
                    else:
                        eff_k = (
                            len(self._cached_carry_trajectory)
                            if getattr(self, "_cached_carry_trajectory", None)
                            else 1
                        )

                # Route A: Call underlying recurrent reasoner m_act(carry, batch) directly across eff_k steps
                if hasattr(m_act, "forward") and m_act is not m_inner:
                    try:
                        carry_a = carry
                        new_carry = carry_a
                        outputs = None
                        for s in range(eff_k):
                            t_step = max(0, total_act_steps - eff_k + s)
                            carry_a, dz_L = self._maybe_apply_cascade_to_carry(
                                carry_a, m_act, batch, delta_z, step_idx=t_step, unroll_steps=eff_k
                            )
                            res = m_act(carry_a, batch)
                            if isinstance(res, tuple) and len(res) >= 2:
                                new_carry, outputs = res[0], res[1]
                                if dz_L is not None and dz_L.requires_grad:
                                    nh, nl = self._get_carry_zh_zl(new_carry)
                                    if isinstance(nh, torch.Tensor) and isinstance(nl, torch.Tensor):
                                        st = dz_L - dz_L.detach()
                                        nl_b = nl + st if (not nl.requires_grad and nl.shape == st.shape) else nl
                                        nh_b = nh + 0.1 * st if (not nh.requires_grad and nh.shape == st.shape) else nh
                                        new_carry = self._set_carry_zh_zl(new_carry, nh_b, nl_b)
                                if s < eff_k - 1:
                                    carry_a = self._clear_carry_halted(new_carry)
                            else:
                                outputs = None
                                break
                        if isinstance(outputs, dict) and "logits" in outputs:
                            logits = self._apply_callosal_canvas_readout(
                                model, outputs["logits"], inputs, delta_z, carry=new_carry, callosal_emb=callosal_emb
                            )
                    except Exception:
                        logits = None

                # Route B: Call m_inner (ACTLossHead or direct ACT model) across eff_k steps
                if logits is None and direct_loss is None:
                    req_keys = ["logits"] if labels is None else []
                    carry_b = carry
                    res = None
                    for s in range(eff_k):
                        t_step = max(0, total_act_steps - eff_k + s)
                        carry_b, dz_L = self._maybe_apply_cascade_to_carry(
                            carry_b, m_inner, batch, delta_z, step_idx=t_step, unroll_steps=eff_k
                        )
                        try:
                            res = m_inner(req_keys, carry=carry_b, batch=batch)
                        except TypeError:
                            try:
                                res = m_inner(carry=carry_b, batch=batch, return_keys=req_keys)
                            except TypeError:
                                res = m_inner(carry_b, batch)
                        if isinstance(res, tuple) and len(res) >= 2:
                            new_carry = res[0]
                            if dz_L is not None and dz_L.requires_grad:
                                nh, nl = self._get_carry_zh_zl(new_carry)
                                if isinstance(nh, torch.Tensor) and isinstance(nl, torch.Tensor):
                                    st = dz_L - dz_L.detach()
                                    nl_b = nl + st if (not nl.requires_grad and nl.shape == st.shape) else nl
                                    nh_b = nh + 0.1 * st if (not nh.requires_grad and nh.shape == st.shape) else nh
                                    new_carry = self._set_carry_zh_zl(new_carry, nh_b, nl_b)
                            if s < eff_k - 1:
                                carry_b = self._clear_carry_halted(new_carry)
                        else:
                            break

                    if isinstance(res, tuple) and len(res) >= 2:
                        new_carry = res[0]
                        if len(res) >= 4 and isinstance(res[3], dict) and "logits" in res[3] and not res[3]["logits"].requires_grad and labels is None:
                            logits = self._apply_callosal_canvas_readout(
                                model, res[3]["logits"], inputs, delta_z, carry=new_carry, callosal_emb=callosal_emb
                            )
                        else:
                            cand = res[1]
                            if isinstance(cand, dict) and "logits" in cand:
                                logits = self._apply_callosal_canvas_readout(
                                    model, cand["logits"], inputs, delta_z, carry=new_carry, callosal_emb=callosal_emb
                                )
                            elif isinstance(cand, torch.Tensor):
                                if cand.dim() >= 2:
                                    logits = self._apply_callosal_canvas_readout(
                                        model, cand, inputs, delta_z, carry=new_carry, callosal_emb=callosal_emb
                                    )
                                else:
                                    direct_loss = cand.float()
            except Exception:
                logits = None
                direct_loss = None

        if logits is None and direct_loss is None:
            # Fallback for feedforward or non-ACT architectures
            raw_batch = BatchDict({"inputs": inputs, "puzzle_identifiers": pids})
            if labels is not None:
                raw_batch["labels"] = labels
            try:
                out = model(raw_batch)
            except (TypeError, ValueError, AttributeError):
                try:
                    out = model(inputs, puzzle_identifiers=pids)
                except (TypeError, ValueError, AttributeError):
                    out = model(inputs)

            if isinstance(out, (dict, BatchDict)):
                if "logits" in out and out["logits"] is not None:
                    logits = self._apply_callosal_canvas_readout(
                        model, out["logits"], inputs, delta_z, carry=None, callosal_emb=callosal_emb
                    )
                elif "loss" in out and out["loss"] is not None:
                    direct_loss = out["loss"].float()
            elif hasattr(out, "logits") and getattr(out, "logits") is not None:
                logits = self._apply_callosal_canvas_readout(
                    model, getattr(out, "logits"), inputs, delta_z, carry=None, callosal_emb=callosal_emb
                )
            elif hasattr(out, "loss") and getattr(out, "loss") is not None:
                direct_loss = getattr(out, "loss").float()
            elif isinstance(out, tuple):
                for elem in out:
                    if isinstance(elem, dict) and "logits" in elem:
                        logits = self._apply_callosal_canvas_readout(
                            model, elem["logits"], inputs, delta_z, carry=None, callosal_emb=callosal_emb
                        )
                        break
                    elif hasattr(elem, "logits") and getattr(elem, "logits") is not None:
                        logits = self._apply_callosal_canvas_readout(
                            model, getattr(elem, "logits"), inputs, delta_z, carry=None, callosal_emb=callosal_emb
                        )
                        break
                    elif isinstance(elem, torch.Tensor) and elem.dim() >= 2:
                        logits = self._apply_callosal_canvas_readout(
                            model, elem, inputs, delta_z, carry=None, callosal_emb=callosal_emb
                        )
                        break
                if logits is None and len(out) > 0 and isinstance(out[0], torch.Tensor) and out[0].dim() == 0:
                    direct_loss = out[0].float()
            elif isinstance(out, torch.Tensor):
                if out.dim() == 0:
                    direct_loss = out.float()
                else:
                    logits = self._apply_callosal_canvas_readout(
                        model, out, inputs, delta_z, carry=None, callosal_emb=callosal_emb
                    )

        return logits, direct_loss

    def _compute_weighted_ce(self, logits_f32: torch.Tensor, demo_targets: torch.Tensor) -> torch.Tensor:
        """Computes token-weighted cross-entropy supervising colors (2..11), EOS (1), and exterior PAD (0)."""
        has_non_pad = bool((demo_targets != 0).any().item())
        if not has_non_pad:
            return F.cross_entropy(logits_f32.reshape(-1, logits_f32.size(-1)), demo_targets.reshape(-1))
        per_tok = F.cross_entropy(
            logits_f32.reshape(-1, logits_f32.size(-1)),
            demo_targets.reshape(-1),
            reduction="none"
        ).view_as(demo_targets)
        weights = torch.where(demo_targets >= 1, 1.0, 0.15)
        return (per_tok * weights).sum() / weights.sum().clamp(min=1.0)

    def compute_demo_loss(
        self,
        model: nn.Module,
        demo_inputs: torch.Tensor,
        demo_targets: torch.Tensor,
        puzzle_id: int,
        delta_z: Optional[torch.Tensor] = None,
        pre_terminal_carry: Optional[Any] = None
    ) -> torch.Tensor:
        """Runs a forward pass on demonstration pairs and returns float32 cross-entropy loss."""
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)

        demo_inputs = demo_inputs.to(model_device)
        demo_targets = demo_targets.to(model_device)
        pids = torch.full((demo_inputs.size(0),), puzzle_id, dtype=torch.long, device=model_device)

        logits, direct_loss = self._extract_terminal_logits(
            model=model,
            inputs=demo_inputs,
            pids=pids,
            labels=demo_targets,
            delta_z=delta_z,
            pre_terminal_carry=pre_terminal_carry
        )

        loss = None
        if logits is not None:
            loss = self._compute_weighted_ce(logits.float(), demo_targets)
        elif direct_loss is not None:
            loss = direct_loss.float()

        if loss is None:
            raise ValueError(f"Unrecognized model output type: {type(model)}")

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(100.0, device=demo_inputs.device, dtype=torch.float32, requires_grad=True)

        return loss

    def _decode_logits_with_shape(
        self,
        logits_2d: torch.Tensor,
        test_input_grid: Sequence[Sequence[int]],
        target_hw: Optional[Tuple[int, int]] = None,
        temperature: float = 0.0
    ) -> List[List[int]]:
        """
        Decodes a single sample's (L, V) logits into a 2D ARC color grid (values 0..9).
        If target_hw = (H_out, W_out) is known or raw decoding yields an uncropped 30x30 canvas,
        crops to (H_out, W_out) and decodes over valid ARC color channels (2..11).
        """
        L, V = logits_2d.shape
        side = int(round(math.sqrt(L)))
        if temperature > 0:
            probs = F.softmax(logits_2d / temperature, dim=-1)
            pred_seq = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
        else:
            pred_seq = torch.argmax(logits_2d, dim=-1).cpu().numpy()

        raw_grid = decode_trm_seq_to_grid(pred_seq, side=side)
        in_h = len(test_input_grid) if test_input_grid else 1
        in_w = len(test_input_grid[0]) if (test_input_grid and test_input_grid[0]) else 1

        if side * side == L and V >= 12:
            if target_hw is not None:
                th, tw = target_hw
            elif raw_grid is None or (len(raw_grid) == side and len(raw_grid[0]) == side and (in_h < side or in_w < side)):
                th, tw = in_h, in_w
            else:
                return raw_grid

            th = max(1, min(side, th))
            tw = max(1, min(side, tw))
            color_logits = logits_2d.view(side, side, V)[:th, :tw, 2:12]
            if temperature > 0:
                c_probs = F.softmax(color_logits.reshape(-1, 10) / temperature, dim=-1)
                c_idx = torch.multinomial(c_probs, num_samples=1).view(th, tw).cpu().numpy()
            else:
                c_idx = torch.argmax(color_logits, dim=-1).cpu().numpy()
            return c_idx.astype(int).tolist()

        if raw_grid is None:
            return [list(row) for row in test_input_grid]
        return raw_grid

    def adapt_task(
        self,
        model: nn.Module,
        task_data: dict,
        puzzle_id: int = 0,
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_anchor: Optional[float] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Adapts continuous latent offset delta_z strictly on task demonstration pairs.
        task_data: dict containing "train" list of {"input": ..., "output": ...} pairs.
        """
        steps = n_steps if n_steps is not None else self.n_steps
        step_lr = lr if lr is not None else self.lr
        anchor_wt = lambda_anchor if lambda_anchor is not None else self.lambda_anchor

        demos = task_data.get("train", [])
        if not demos:
            raise ValueError("Task has no demonstration ('train') pairs for TTA.")

        self._task_demos_by_pid[puzzle_id] = demos
        out_shapes = [(len(d["output"]), len(d["output"][0])) for d in demos if d.get("output") and d["output"][0]]
        if len(set(out_shapes)) == 1:
            self._active_target_hw = out_shapes[0]
        elif len(out_shapes) == len(demos) and len(out_shapes) > 0:
            self._active_target_hw = out_shapes
        else:
            self._active_target_hw = None
        self._active_color_preserving = self._is_demo_color_preserving(demos)

        # Strict leakage guard: ensure test outputs are NEVER passed or referenced here
        demo_inps = [encode_grid_to_trm_seq(d["input"]) for d in demos]
        demo_outs = [encode_grid_to_trm_seq(d["output"]) for d in demos]

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)

        demo_inputs = torch.tensor(np.array(demo_inps), dtype=torch.long, device=model_device)
        demo_targets = torch.tensor(np.array(demo_outs), dtype=torch.long, device=model_device)
        pids = torch.full((demo_inputs.size(0),), puzzle_id, dtype=torch.long, device=model_device)

        callosal_emb = find_callosal_embedding(model)

        if callosal_emb is not None and hasattr(callosal_emb, "num_query_tokens"):
            num_query_tokens = callosal_emb.num_query_tokens
            d_reasoner = callosal_emb.d_reasoner
        else:
            num_query_tokens = 16
            d_reasoner = 512

        if not getattr(self, "respect_controller_schedule", False):
            if d_reasoner >= 256 and demo_inputs.size(1) == 900:
                step_lr = max(step_lr, 0.05)
                anchor_wt = min(anchor_wt, 0.1)

        # Freeze all model parameters
        orig_requires_grad = {}
        for name, p in model.named_parameters():
            orig_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        was_training = model.training
        model.eval()

        try:
            # Pre-roll recurrent carry once under no_grad
            demo_batch = BatchDict({"inputs": demo_inputs, "labels": demo_targets, "puzzle_identifiers": pids})
            pre_terminal_carry, act_steps_unrolled = self._get_pre_terminal_carry(model, demo_batch, model_device)

            # Step 0 Baseline Demonstration loss & predictions evaluation (delta_z = 0)
            with torch.no_grad():
                logits_step0, _ = self._extract_terminal_logits(
                    model, demo_inputs, pids, labels=demo_targets, delta_z=None, pre_terminal_carry=pre_terminal_carry
                )
                if logits_step0 is not None:
                    loss_step0 = float(self._compute_weighted_ce(logits_step0.float(), demo_targets).item())
                    preds_step0 = torch.argmax(logits_step0, dim=-1)
                else:
                    loss_step0 = float(self.compute_demo_loss(model, demo_inputs, demo_targets, puzzle_id, delta_z=None, pre_terminal_carry=pre_terminal_carry).item())
                    preds_step0 = None

            # Optional Step-0 schedule refinement hook (when ZenoAmygdalaController is active)
            s0_cb = getattr(self, "_step0_schedule_callback", None)
            if callable(s0_cb):
                s1_fit_est = max(0.0, min(1.0, 1.0 - loss_step0 / 8.0))
                s0_sched = s0_cb(loss_step0, s1_fit_est)
                if s0_sched is not None:
                    steps = int(s0_sched.num_steps)
                    step_lr = float(s0_sched.lr)
                    anchor_wt = float(s0_sched.anchor_weight)
                    if getattr(self, "_full_carry_history", None):
                        eff_k = max(1, min(int(s0_sched.unroll_steps), len(self._full_carry_history)))
                        self._cached_carry_trajectory = self._full_carry_history[-eff_k:]
                        pre_terminal_carry = _clone_carry(self._full_carry_history[-eff_k])

            # Always maintain delta_z in float32 to prevent bfloat16/float16 underflow during Adam updates
            delta_z = nn.Parameter(torch.zeros(num_query_tokens, d_reasoner, device=model_device, dtype=torch.float32))
            optimizer = torch.optim.Adam([delta_z], lr=step_lr)

            best_demo_loss = loss_step0
            best_delta_z = torch.zeros_like(delta_z.data)
            step_losses = [loss_step0]
            grad_norm_step0 = 0.0
            grad_norm_final = 0.0

            for step in range(steps):
                optimizer.zero_grad()
                ce_loss = self.compute_demo_loss(
                    model, demo_inputs, demo_targets, puzzle_id, delta_z=delta_z, pre_terminal_carry=pre_terminal_carry
                )
                # Normalize proximal quadratic anchor by numel() (mean) so it matches token-mean CE scale
                anchor_loss = 0.5 * anchor_wt * torch.mean(delta_z ** 2)
                total_loss = ce_loss + anchor_loss
                total_loss.backward()

                if delta_z.grad is not None:
                    gnorm = float(torch.norm(delta_z.grad).item())
                    if step == 0:
                        grad_norm_step0 = gnorm
                    grad_norm_final = gnorm

                optimizer.step()

                cur_demo_loss = float(ce_loss.item())
                step_losses.append(cur_demo_loss)
                if cur_demo_loss < best_demo_loss:
                    best_demo_loss = cur_demo_loss
                    best_delta_z = delta_z.data.clone()

            # Final check at terminal adapted delta_z
            with torch.no_grad():
                terminal_loss = float(self.compute_demo_loss(
                    model, demo_inputs, demo_targets, puzzle_id, delta_z=delta_z, pre_terminal_carry=pre_terminal_carry
                ).item())
                if terminal_loss < best_demo_loss:
                    best_demo_loss = terminal_loss
                    best_delta_z = delta_z.data.clone()

            # Step 0 Baseline Comparison & Weak Pareto Monotonicity Fallback
            if best_demo_loss < loss_step0 - 1e-7:
                final_delta_z = best_delta_z
                accepted = True
            else:
                final_delta_z = torch.zeros_like(delta_z.data)
                accepted = False

            # Count discrete cell flips on demonstration pairs
            demo_cells_flipped = 0
            if accepted and preds_step0 is not None:
                with torch.no_grad():
                    logits_final, _ = self._extract_terminal_logits(
                        model, demo_inputs, pids, labels=demo_targets, delta_z=final_delta_z, pre_terminal_carry=pre_terminal_carry
                    )
                    if logits_final is not None:
                        preds_final = torch.argmax(logits_final, dim=-1)
                        active_demo_mask = (demo_targets != 0) if (demo_targets != 0).any() else torch.ones_like(demo_targets, dtype=torch.bool)
                        demo_cells_flipped = int(((preds_final != preds_step0) & active_demo_mask).sum().item())

            diagnostics = {
                "puzzle_id": puzzle_id,
                "step0_loss": loss_step0,
                "final_demo_loss": best_demo_loss if accepted else loss_step0,
                "delta_loss": (best_demo_loss - loss_step0) if accepted else 0.0,
                "accepted": accepted,
                "norm_delta_z": float(torch.norm(final_delta_z).item()),
                "rms_delta_z": float(torch.sqrt(torch.mean(final_delta_z ** 2)).item()),
                "grad_norm_step0": grad_norm_step0,
                "grad_norm_final": grad_norm_final,
                "recurrent_steps_unrolled": act_steps_unrolled,
                "demo_cells_flipped": demo_cells_flipped,
                "step_losses": step_losses
            }

            return final_delta_z, diagnostics

        finally:
            self._active_target_hw = None
            self._active_color_preserving = False
            for name, p in model.named_parameters():
                if name in orig_requires_grad:
                    p.requires_grad = orig_requires_grad[name]
            if was_training:
                model.train()
            if callosal_emb is not None and hasattr(callosal_emb, "clear_active_delta_z"):
                callosal_emb.clear_active_delta_z()

    def predict_test_pair(
        self,
        model: nn.Module,
        test_input_grid: List[List[int]],
        puzzle_id: int,
        delta_z: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        demos: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> List[List[int]]:
        """Runs model inference on a single test input grid with the given delta_z."""
        callosal_emb = find_callosal_embedding(model)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)

        task_demos = demos if demos is not None else self._task_demos_by_pid.get(puzzle_id)
        target_hw = infer_arc_output_shape(task_demos, test_input_grid) if task_demos else None
        self._active_target_hw = target_hw
        self._active_color_preserving = self._is_demo_color_preserving(task_demos)

        inp_seq = encode_grid_to_trm_seq(test_input_grid)
        inp_tensor = torch.tensor(inp_seq, dtype=torch.long, device=model_device).unsqueeze(0)
        pid_tensor = torch.tensor([puzzle_id], dtype=torch.long, device=model_device)

        model.eval()
        try:
            with torch.no_grad():
                logits, _ = self._extract_terminal_logits(
                    model=model,
                    inputs=inp_tensor,
                    pids=pid_tensor,
                    labels=None,
                    delta_z=delta_z,
                    pre_terminal_carry=None
                )
                if logits is None:
                    raise ValueError(f"Cannot extract logits from model output: {type(model)}")

                pred_grid = self._decode_logits_with_shape(
                    logits[0], test_input_grid, target_hw=target_hw, temperature=temperature
                )
        finally:
            self._active_target_hw = None
            self._active_color_preserving = False
            if callosal_emb is not None and hasattr(callosal_emb, "clear_active_delta_z"):
                callosal_emb.clear_active_delta_z()

        return pred_grid

    def predict_s1_and_s2_pair(
        self,
        model: nn.Module,
        test_input_grid: List[List[int]],
        puzzle_id: int,
        delta_z: Optional[torch.Tensor],
        accepted: bool = True,
        demos: Optional[Sequence[Mapping[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Jointly evaluates System 1 (delta_z=None) and System 2 (delta_z) on a test input grid,
        sharing the pre-terminal recurrent rollout and computing logit shift & cell flip diagnostics.
        When System 2 is accepted:
          - If pred_s2 != pred_s1: attempt_1 = pred_s2, attempt_2 = pred_s1.
          - If pred_s2 == pred_s1: attempt_1 = pred_s1, and attempt_2 uses ray extrapolation
            along the TTA logit shift direction (l0 + alpha * (l1 - l0)) to flip the highest-velocity
            cells while guaranteeing zero regression via attempt_1 = pred_s1.
        """
        callosal_emb = find_callosal_embedding(model)
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)

        task_demos = demos if demos is not None else self._task_demos_by_pid.get(puzzle_id)
        target_hw = infer_arc_output_shape(task_demos, test_input_grid) if task_demos else None
        self._active_target_hw = target_hw
        self._active_color_preserving = self._is_demo_color_preserving(task_demos)

        inp_seq = encode_grid_to_trm_seq(test_input_grid)
        inp_tensor = torch.tensor(inp_seq, dtype=torch.long, device=model_device).unsqueeze(0)
        pid_tensor = torch.tensor([puzzle_id], dtype=torch.long, device=model_device)
        carry_batch = BatchDict({
            "inputs": inp_tensor,
            "labels": torch.zeros_like(inp_tensor),
            "puzzle_identifiers": pid_tensor
        })

        model.eval()
        try:
            with torch.no_grad():
                pre_carry, _ = self._get_pre_terminal_carry(model, carry_batch, model_device)

                logits_s1, _ = self._extract_terminal_logits(
                    model=model,
                    inputs=inp_tensor,
                    pids=pid_tensor,
                    labels=None,
                    delta_z=None,
                    pre_terminal_carry=pre_carry
                )
                if logits_s1 is None:
                    raise ValueError(f"Cannot extract logits from model output: {type(model)}")

                pred_s1 = self._decode_logits_with_shape(logits_s1[0], test_input_grid, target_hw=target_hw)

                if not accepted or delta_z is None or float(torch.norm(delta_z).item()) == 0.0:
                    return {
                        "pred_s1": pred_s1,
                        "pred_s2": pred_s1,
                        "attempt_1": pred_s1,
                        "attempt_2": pred_s1,
                        "changed_attempt_1": False,
                        "changed_attempt_2": False,
                        "cells_changed_attempt1": 0,
                        "cells_changed_attempt2": 0,
                        "mean_abs_logit_shift": 0.0,
                        "max_abs_logit_shift": 0.0
                    }

                logits_s2, _ = self._extract_terminal_logits(
                    model=model,
                    inputs=inp_tensor,
                    pids=pid_tensor,
                    labels=None,
                    delta_z=delta_z,
                    pre_terminal_carry=pre_carry
                )
                pred_s2 = self._decode_logits_with_shape(logits_s2[0], test_input_grid, target_hw=target_hw)

                l0 = logits_s1[0].float()
                l1 = logits_s2[0].float()
                dl = l1 - l0
                active_mask = (inp_tensor[0] >= 2)
                if active_mask.any() and l0.size(-1) >= 12:
                    dl_active = dl[active_mask, 2:]
                else:
                    dl_active = dl

                mean_shift = float(dl_active.abs().mean().item()) if dl_active.numel() > 0 else 0.0
                max_shift = float(dl_active.abs().max().item()) if dl_active.numel() > 0 else 0.0
                arr_s1 = np.asarray(pred_s1)
                arr_s2 = np.asarray(pred_s2)
                cells_changed_1 = int((arr_s1 != arr_s2).sum()) if arr_s1.shape == arr_s2.shape else int(arr_s2.size)

                if pred_s2 != pred_s1:
                    attempt_1 = pred_s2
                    attempt_2 = pred_s1
                    cells_changed_2 = 0
                else:
                    # pred_s2 == pred_s1: attempt_1 already holds pred_s1 (0 regression guarantee).
                    # Use ray extrapolation l0 + alpha*dl for attempt_2 to cross argmax ties along TTA velocity.
                    attempt_1 = pred_s1
                    attempt_2 = pred_s1
                    cells_changed_2 = 0
                    if max_shift > 1e-6:
                        c0 = torch.argmax(l0, dim=-1)
                        active_indices = torch.where(active_mask)[0] if active_mask.any() else torch.arange(l0.size(0), device=l0.device)
                        thresholds = []
                        c_start = 2 if (active_mask.any() and l0.size(-1) >= 12) else 0
                        for idx in active_indices.tolist():
                            cur_c = int(c0[idx].item())
                            m_row = l0[idx, cur_c] - l0[idx, c_start:]
                            v_row = dl[idx, c_start:] - dl[idx, cur_c]
                            pos = (v_row > 1e-6) & (m_row > 0.0)
                            if pos.any():
                                min_alpha = float((m_row[pos] / v_row[pos]).min().item())
                                if min_alpha > 0.0:
                                    thresholds.append(min_alpha)
                        if thresholds:
                            thresholds.sort()
                            ray_q_list = getattr(self, "_active_ray_quantiles", None)
                            q_val = (
                                float(ray_q_list[0])
                                if (isinstance(ray_q_list, (list, tuple)) and len(ray_q_list) > 0)
                                else 0.20
                            )
                            q_idx = min(len(thresholds) - 1, max(0, int(q_val * len(thresholds))))
                            alpha_star = min(max(thresholds[q_idx] * 1.05, 1.05), 100.0)
                            l_extrap = l0 + alpha_star * dl
                            pred_extrap = self._decode_logits_with_shape(l_extrap, test_input_grid, target_hw=target_hw)
                            if pred_extrap is not None and pred_extrap != pred_s1:
                                attempt_2 = pred_extrap
                                arr_ext = np.asarray(pred_extrap)
                                cells_changed_2 = int((arr_ext != arr_s1).sum()) if arr_ext.shape == arr_s1.shape else int(arr_ext.size)

                return {
                    "pred_s1": pred_s1,
                    "pred_s2": pred_s2,
                    "attempt_1": attempt_1,
                    "attempt_2": attempt_2,
                    "changed_attempt_1": (attempt_1 != pred_s1),
                    "changed_attempt_2": (attempt_2 != pred_s1),
                    "cells_changed_attempt1": cells_changed_1,
                    "cells_changed_attempt2": cells_changed_2,
                    "mean_abs_logit_shift": mean_shift,
                    "max_abs_logit_shift": max_shift
                }
        finally:
            self._active_target_hw = None
            self._active_color_preserving = False
            if callosal_emb is not None and hasattr(callosal_emb, "clear_active_delta_z"):
                callosal_emb.clear_active_delta_z()

    def adapt_and_evaluate_submission(
        self,
        model: nn.Module,
        challenges: Dict[str, dict],
        identifiers_map: Optional[Union[List[str], Dict[str, int]]] = None,
        n_steps: Optional[int] = None,
        lr: Optional[float] = None,
        lambda_anchor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Runs System 2 TTA across tasks in challenges and produces Kaggle-format submissions.
        Returns:
          - submission_s2: adapted predictions (attempt_1=adapted, attempt_2=S1 fallback or ray-extrapolated)
          - submission_s1: unadapted predictions (attempt_1=S1, attempt_2=S1)
          - diagnostics: per-task adaptation diagnostics
        """
        sub_s2: Dict[str, list] = {}
        sub_s1: Dict[str, list] = {}
        diagnostics: Dict[str, dict] = {}
        task_demos_list = list(challenges.get("train", [])) if "train" in challenges else None

        for tid, task in challenges.items():
            if isinstance(identifiers_map, list):
                pid = identifiers_map.index(tid) if tid in identifiers_map else 0
            elif isinstance(identifiers_map, dict):
                pid = identifiers_map.get(tid, 0)
            else:
                pid = 0

            # 1. Adapt delta_z strictly on demonstration pairs
            delta_z, diag = self.adapt_task(
                model=model,
                task_data=task,
                puzzle_id=pid,
                n_steps=n_steps,
                lr=lr,
                lambda_anchor=lambda_anchor
            )

            # 2. Predict test challenge grids
            sub_s2[tid] = []
            sub_s1[tid] = []
            pair_diags = []
            demos = task.get("train", task_demos_list)
            for test_pair in task.get("test", []):
                inp = test_pair["input"]
                pair_res = self.predict_s1_and_s2_pair(
                    model=model,
                    test_input_grid=inp,
                    puzzle_id=pid,
                    delta_z=delta_z if diag["accepted"] else None,
                    accepted=diag["accepted"],
                    demos=demos
                )
                sub_s1[tid].append({"attempt_1": pair_res["pred_s1"], "attempt_2": pair_res["pred_s1"]})
                sub_s2[tid].append({"attempt_1": pair_res["attempt_1"], "attempt_2": pair_res["attempt_2"]})
                pair_diags.append({
                    "changed_attempt_1": pair_res["changed_attempt_1"],
                    "changed_attempt_2": pair_res["changed_attempt_2"],
                    "cells_changed_attempt1": pair_res["cells_changed_attempt1"],
                    "cells_changed_attempt2": pair_res["cells_changed_attempt2"],
                    "mean_abs_logit_shift": pair_res["mean_abs_logit_shift"],
                    "max_abs_logit_shift": pair_res["max_abs_logit_shift"]
                })

            diag["test_pairs"] = pair_diags
            diagnostics[tid] = diag

        return {
            "submission_s2": sub_s2,
            "submission_s1": sub_s1,
            "diagnostics": diagnostics
        }


