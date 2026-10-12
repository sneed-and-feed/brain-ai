"""
brain_ai.models.callosum: Corpus Callosum Inter-Hemispheric E-I Bridge.

Implements:
1. DaleLinear: Weight parameterization W = Softplus(V) * D strictly enforcing Dale's Principle.
   Initialized via Rajan-Abbott (PRL 2006) balanced condition (f_E * mu_E = f_I * mu_I, lambda_outlier = 0).
2. HomeostaticSynapticScaling: Turrigiano synaptic scaling & RMSNorm to prevent runaway/collapse.
3. DaleDifferentialCrossAttention: Differential attention (Ye et al., ICLR 2025) with Dale constraints.
4. InterHemisphericLatentCoupling: Transcallosal inhibitory coupling (Hong Jeong 2026; s = -1).
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
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


class BatchDict(dict):
    """Dictionary supporting both mapping and attribute-style access for model inputs."""
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'BatchDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def find_callosal_embedding(model: nn.Module) -> Optional[Any]:
    """
    Recursively and systematically resolves CallosalPuzzleEmbedding across wrappers
    (e.g., standard model, model.inner, DDP/DataParallel model.module, EMA copies).
    """
    candidates = [model]
    for attr in ("inner", "module"):
        sub = getattr(model, attr, None)
        if sub is not None:
            candidates.append(sub)
            for sub_attr in ("inner", "module"):
                sub_sub = getattr(sub, sub_attr, None)
                if sub_sub is not None:
                    candidates.append(sub_sub)

    for cand in candidates:
        if cand is None:
            continue
        if hasattr(cand, "puzzle_emb") and hasattr(getattr(cand, "puzzle_emb"), "set_active_delta_z"):
            return cand.puzzle_emb
        if hasattr(cand, "inner") and hasattr(getattr(cand, "inner"), "puzzle_emb"):
            emb = getattr(cand.inner, "puzzle_emb")
            if hasattr(emb, "set_active_delta_z"):
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


class System2LatentAdaptor:
    """
    System 2 Test-Time Latent Adaptation (TTA) with Continuous Latent Relaxation (delta_z).

    Formalized in docs/RESEARCH_PLAN.md §4.2 and ARCHITECTURE_SPEC.md §3:
    1. Takes the initial callosal conditioning state Z_0 = Callosum(Z_L) for a task.
    2. Freezes underlying model weights (reasoner + callosum).
    3. Optimizes a continuous latent offset delta_z in R^(num_query_tokens x d_reasoner)
       for N_TTA steps (default 30) using Adam strictly on the task's demonstration
       input/output pairs (L_CE).
    4. Includes proximal quadratic regularization (0.5 * lambda_anchor * ||delta_z||_2^2)
       to bound latent drift.
    5. Evaluates the adapted state Z* = Z_0 + delta_z on the held-out test challenge grid.
    6. Implements Step 0 baseline comparison: if post-adaptation demonstration loss
       does not improve over Step 0, falls back to Z_0 (weak Pareto monotonicity).
    7. Strict leakage guarantee: test challenge outputs are NEVER accessed during TTA.
    """
    def __init__(
        self,
        n_steps: int = 30,
        lr: float = 0.01,
        lambda_anchor: float = 1.0,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.n_steps = n_steps
        self.lr = lr
        self.lambda_anchor = lambda_anchor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute_demo_loss(
        self,
        model: nn.Module,
        demo_inputs: torch.Tensor,
        demo_targets: torch.Tensor,
        puzzle_id: int,
        delta_z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Runs a forward pass on demonstration pairs and returns cross-entropy loss."""
        callosal_emb = find_callosal_embedding(model)
        if callosal_emb is not None and hasattr(callosal_emb, "set_active_delta_z"):
            callosal_emb.set_active_delta_z(delta_z)

        # Ensure all input tensors strictly align with the model parameter device
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)

        demo_inputs = demo_inputs.to(model_device)
        demo_targets = demo_targets.to(model_device)
        pids = torch.full((demo_inputs.size(0),), puzzle_id, dtype=torch.long, device=model_device)
        batch = BatchDict({"inputs": demo_inputs, "labels": demo_targets, "puzzle_identifiers": pids})

        loss = None
        m_inner = model.module if hasattr(model, "module") else model
        m_act = getattr(m_inner, "model", m_inner)

        # 1. Primary path for ACT recurrent models (TinyRecursiveReasoningModel_ACTV1)
        carry_provider = None
        if hasattr(m_act, "initial_carry"):
            carry_provider = m_act
        elif hasattr(m_inner, "initial_carry"):
            carry_provider = m_inner

        if carry_provider is not None:
            try:
                # Set default device context so torch.empty/zeros/ones inside initial_carry allocate on model_device
                with torch.device(model_device):
                    carry = carry_provider.initial_carry(batch)
                carry = _ensure_carry_device(carry, model_device)

                # Route A: Call underlying recurrent reasoner m_act(carry, batch) directly
                if hasattr(m_act, "forward") and m_act is not m_inner:
                    try:
                        res = m_act(carry, batch)
                        if isinstance(res, tuple) and len(res) >= 2:
                            outputs = res[1]
                            if isinstance(outputs, dict) and "logits" in outputs:
                                logits = outputs["logits"]
                                valid_mask = (demo_targets != 0)
                                if valid_mask.any():
                                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), demo_targets.reshape(-1), ignore_index=0)
                                else:
                                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), demo_targets.reshape(-1))
                    except Exception:
                        loss = None

                # Route B: Call m_inner (ACTLossHead) with return_keys=[]
                if loss is None:
                    try:
                        res = m_inner([], carry=carry, batch=batch)
                    except TypeError:
                        res = m_inner(carry=carry, batch=batch, return_keys=[])

                    if isinstance(res, tuple) and len(res) >= 2:
                        cand = res[1]
                        if isinstance(cand, torch.Tensor):
                            loss = cand
                        elif isinstance(cand, dict) and "logits" in cand:
                            logits = cand["logits"]
                            valid_mask = (demo_targets != 0)
                            if valid_mask.any():
                                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), demo_targets.reshape(-1), ignore_index=0)
                            else:
                                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), demo_targets.reshape(-1))
            except Exception:
                loss = None

        if loss is None:
            # Fallback for feedforward or non-ACT architectures
            try:
                out = model(batch)
            except (TypeError, ValueError, AttributeError):
                try:
                    out = model(demo_inputs, puzzle_identifiers=pids)
                except (TypeError, ValueError, AttributeError):
                    out = model(demo_inputs)

            if isinstance(out, dict) or isinstance(out, BatchDict):
                if "loss" in out and out["loss"] is not None:
                    loss = out["loss"]
                elif "logits" in out:
                    logits = out["logits"]
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), demo_targets.reshape(-1), ignore_index=0)
            elif hasattr(out, "loss") and getattr(out, "loss") is not None:
                loss = getattr(out, "loss")
            elif hasattr(out, "logits") and getattr(out, "logits") is not None:
                logits = getattr(out, "logits")
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), demo_targets.reshape(-1), ignore_index=0)
            elif isinstance(out, tuple):
                first = out[0]
                if isinstance(first, torch.Tensor):
                    if first.dim() == 0:
                        loss = first
                    elif first.dim() >= 2:
                        loss = F.cross_entropy(first.reshape(-1, first.size(-1)), demo_targets.reshape(-1), ignore_index=0)
                if loss is None:
                    for elem in out[1:]:
                        if isinstance(elem, torch.Tensor) and elem.dim() >= 2:
                            loss = F.cross_entropy(elem.reshape(-1, elem.size(-1)), demo_targets.reshape(-1), ignore_index=0)
                            break
            elif isinstance(out, torch.Tensor):
                if out.dim() == 0:
                    loss = out
                else:
                    loss = F.cross_entropy(out.reshape(-1, out.size(-1)), demo_targets.reshape(-1), ignore_index=0)

        if loss is None:
            raise ValueError(f"Unrecognized model output type: {type(model)}")

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(100.0, device=demo_inputs.device, requires_grad=True)

        return loss

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

        # Strict leakage guard: ensure test outputs are NEVER passed or referenced here
        demo_inps = [encode_grid_to_trm_seq(d["input"]) for d in demos]
        demo_outs = [encode_grid_to_trm_seq(d["output"]) for d in demos]

        device = self.device
        demo_inputs = torch.tensor(np.array(demo_inps), dtype=torch.long, device=device)
        demo_targets = torch.tensor(np.array(demo_outs), dtype=torch.long, device=device)

        callosal_emb = find_callosal_embedding(model)

        if callosal_emb is not None and hasattr(callosal_emb, "num_query_tokens"):
            num_query_tokens = callosal_emb.num_query_tokens
            d_reasoner = callosal_emb.d_reasoner
            param_dtype = getattr(callosal_emb.callosum.task_queries, "dtype", torch.float32)
        else:
            num_query_tokens = 16
            d_reasoner = 512
            param_dtype = torch.float32

        # Freeze all model parameters
        orig_requires_grad = {}
        for name, p in model.named_parameters():
            orig_requires_grad[name] = p.requires_grad
            p.requires_grad = False

        was_training = model.training
        model.eval()

        try:
            # Step 0 Baseline Demonstration loss evaluation (delta_z = 0)
            with torch.no_grad():
                loss_step0 = float(self.compute_demo_loss(model, demo_inputs, demo_targets, puzzle_id, delta_z=None).item())

            delta_z = nn.Parameter(torch.zeros(num_query_tokens, d_reasoner, device=device, dtype=param_dtype))
            optimizer = torch.optim.Adam([delta_z], lr=step_lr)

            best_demo_loss = loss_step0
            best_delta_z = torch.zeros_like(delta_z.data)
            step_losses = [loss_step0]

            for step in range(steps):
                optimizer.zero_grad()
                ce_loss = self.compute_demo_loss(model, demo_inputs, demo_targets, puzzle_id, delta_z=delta_z)
                anchor_loss = 0.5 * anchor_wt * torch.sum(delta_z ** 2)
                total_loss = ce_loss + anchor_loss
                total_loss.backward()
                optimizer.step()

                cur_demo_loss = float(ce_loss.item())
                step_losses.append(cur_demo_loss)
                if cur_demo_loss < best_demo_loss:
                    best_demo_loss = cur_demo_loss
                    best_delta_z = delta_z.data.clone()

            # Final check at terminal adapted delta_z
            with torch.no_grad():
                terminal_loss = float(self.compute_demo_loss(model, demo_inputs, demo_targets, puzzle_id, delta_z=delta_z).item())
                if terminal_loss < best_demo_loss:
                    best_demo_loss = terminal_loss
                    best_delta_z = delta_z.data.clone()

            # Step 0 Baseline Comparison & Weak Pareto Monotonicity Fallback
            if best_demo_loss < loss_step0:
                final_delta_z = best_delta_z
                accepted = True
            else:
                final_delta_z = torch.zeros_like(delta_z.data)
                accepted = False

            diagnostics = {
                "puzzle_id": puzzle_id,
                "step0_loss": loss_step0,
                "final_demo_loss": best_demo_loss if accepted else loss_step0,
                "delta_loss": (best_demo_loss - loss_step0) if accepted else 0.0,
                "accepted": accepted,
                "norm_delta_z": float(torch.norm(final_delta_z).item()),
                "step_losses": step_losses
            }

            return final_delta_z, diagnostics

        finally:
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
        temperature: float = 0.0
    ) -> List[List[int]]:
        """Runs model inference on a single test input grid with the given delta_z."""
        callosal_emb = find_callosal_embedding(model)
        if callosal_emb is not None and hasattr(callosal_emb, "set_active_delta_z"):
            callosal_emb.set_active_delta_z(delta_z)

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device(self.device)

        inp_seq = encode_grid_to_trm_seq(test_input_grid)
        inp_tensor = torch.tensor(inp_seq, dtype=torch.long, device=model_device).unsqueeze(0)
        pid_tensor = torch.tensor([puzzle_id], dtype=torch.long, device=model_device)
        batch = BatchDict({"inputs": inp_tensor, "puzzle_identifiers": pid_tensor})

        model.eval()
        with torch.no_grad():
            m_inner = model.module if hasattr(model, "module") else model
            m_act = getattr(m_inner, "model", m_inner)
            logits = None

            carry_provider = None
            if hasattr(m_act, "initial_carry"):
                carry_provider = m_act
            elif hasattr(m_inner, "initial_carry"):
                carry_provider = m_inner

            if carry_provider is not None:
                try:
                    carry_batch = BatchDict({"inputs": inp_tensor, "labels": torch.zeros_like(inp_tensor), "puzzle_identifiers": pid_tensor})
                    with torch.device(model_device):
                        carry = carry_provider.initial_carry(carry_batch)
                    carry = _ensure_carry_device(carry, model_device)
                    steps = 0
                    while True:
                        if hasattr(m_act, "forward") and m_act is not m_inner:
                            carry, outputs = m_act(carry, carry_batch)
                            if isinstance(outputs, dict) and "logits" in outputs:
                                logits = outputs["logits"]
                            all_finish = getattr(carry, "halted", torch.tensor(True)).all().item()
                        else:
                            try:
                                res = m_inner(["logits"], carry=carry, batch=carry_batch)
                            except TypeError:
                                res = m_inner(carry=carry, batch=carry_batch, return_keys=["logits"])
                            if isinstance(res, tuple):
                                carry = res[0]
                                all_finish = res[-1]
                                if len(res) >= 4 and isinstance(res[3], dict) and "logits" in res[3]:
                                    logits = res[3]["logits"]
                                elif len(res) >= 2 and isinstance(res[1], dict) and "logits" in res[1]:
                                    logits = res[1]["logits"]
                        steps += 1
                        if (all_finish is True) or (steps >= 16):
                            break
                except Exception:
                    logits = None

            if logits is None:
                try:
                    out = model(batch)
                except (TypeError, ValueError, AttributeError):
                    try:
                        out = model(inp_tensor, puzzle_identifiers=pid_tensor)
                    except (TypeError, ValueError, AttributeError):
                        out = model(inp_tensor)

                if (isinstance(out, dict) or isinstance(out, BatchDict)) and "logits" in out:
                    logits = out["logits"]
                elif hasattr(out, "logits") and getattr(out, "logits") is not None:
                    logits = getattr(out, "logits")
                elif isinstance(out, tuple):
                    for elem in out:
                        if isinstance(elem, dict) and "logits" in elem:
                            logits = elem["logits"]
                            break
                        elif hasattr(elem, "logits") and getattr(elem, "logits") is not None:
                            logits = getattr(elem, "logits")
                            break
                        elif isinstance(elem, torch.Tensor) and elem.dim() >= 2:
                            logits = elem
                            break
                elif isinstance(out, torch.Tensor) and out.dim() >= 2:
                    logits = out

            if logits is None:
                raise ValueError(f"Cannot extract logits from model output: {type(out) if 'out' in locals() else type(model)}")

            if temperature > 0:
                probs = F.softmax(logits[0] / temperature, dim=-1)
                pred_seq = torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()
            else:
                pred_seq = torch.argmax(logits[0], dim=-1).cpu().numpy()

        pred_grid = decode_trm_seq_to_grid(pred_seq)
        if pred_grid is None:
            pred_grid = [list(row) for row in test_input_grid]
        return pred_grid

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
          - submission_s2: adapted predictions (attempt_1=adapted, attempt_2=S1 fallback)
          - submission_s1: unadapted predictions (attempt_1=S1, attempt_2=S1)
          - diagnostics: per-task adaptation diagnostics
        """
        sub_s2: Dict[str, list] = {}
        sub_s1: Dict[str, list] = {}
        diagnostics: Dict[str, dict] = {}

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
            diagnostics[tid] = diag

            # 2. Predict test challenge grids
            sub_s2[tid] = []
            sub_s1[tid] = []
            for test_pair in task.get("test", []):
                inp = test_pair["input"]
                pred_s1 = self.predict_test_pair(model, inp, puzzle_id=pid, delta_z=None)
                pred_s2 = self.predict_test_pair(model, inp, puzzle_id=pid, delta_z=delta_z if diag["accepted"] else None)

                sub_s1[tid].append({"attempt_1": pred_s1, "attempt_2": pred_s1})
                sub_s2[tid].append({"attempt_1": pred_s2, "attempt_2": pred_s1})

        return {
            "submission_s2": sub_s2,
            "submission_s1": sub_s1,
            "diagnostics": diagnostics
        }

