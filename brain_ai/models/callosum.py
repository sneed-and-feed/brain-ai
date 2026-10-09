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
from typing import Dict, Tuple, Optional
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
