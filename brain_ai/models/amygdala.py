"""
brain_ai.models.amygdala: Fast System 1 Affective & Salience Router based on Open-Jev (Zefan Cai).

Provides sub-30ms continuous affective state estimation and neuromodulatory control:
1. OpenJevAmygdalarRouter: Extracts 3D continuous Affective State vector a = [V, U, Omega]^\top
   - Valence: Bipolar [-1, +1]
   - Threat / Epistemic Uncertainty: Fused hazard risk and decision entropy [0, 1]
   - Urgency / Cognitive Complexity: Ordinal expectation [0, 1]
2. NeuromodulatoryController: Translates affective state into downstream controls:
   - Callosal gains gamma_E, gamma_I
   - Generative LLM temperature scaling T_gen
   - Recurrent step budget N_steps and System 1 Reflex Bypass
"""

import math
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OpenJevScalarDecisionHead(nn.Module):
    """
    Open-Jev Scalar Readout Head for Dynamic Candidate Scoring.
    Maps terminal candidate embeddings to scalar decision energies.
    """
    def __init__(self, hidden_dim: int = 3584, mlp_dim: int = 1024, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_dim, 1, bias=True)
        )

    def forward(self, candidate_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            candidate_hidden_states: [batch_size, num_candidates, hidden_dim]
        Returns:
            probabilities: [batch_size, num_candidates]
            logits: [batch_size, num_candidates]
        """
        logits = self.mlp(candidate_hidden_states).squeeze(-1)
        scaled_logits = logits / torch.clamp(self.temperature, min=1e-3)
        probabilities = F.softmax(scaled_logits, dim=-1)
        return probabilities, logits


class OpenJevAmygdalarRouter(nn.Module):
    """
    Open-Jev based Amygdalar Affective/Salience Router.
    Attaches continuous affective readout heads onto a Transformer hidden state.
    """
    def __init__(self, hidden_dim: int = 3584, mlp_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Valence Head: Bipolar [-1, +1]
        self.valence_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
            nn.Tanh()
        )
        
        # 2. Threat / Risk Head: Probability of hazard [0, 1]
        self.threat_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid()
        )
        
        # 3. Urgency / Complexity Head: Ordinal distribution over 4 tiers [0, 1]
        self.urgency_head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 4)
        )
        self.register_buffer("complexity_weights", torch.tensor([0.0, 0.333, 0.666, 1.0]))

    def forward(
        self, 
        state_repr: torch.Tensor, 
        candidate_logits: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_repr: Terminal token representation [B, hidden_dim]
            candidate_logits: Optional logits over task options to compute decision entropy [B, K]
        Returns:
            Dictionary containing continuous (Valence, Threat/Uncertainty, Urgency)
        """
        # 1. Valence V in [-1, +1]
        valence = self.valence_head(state_repr).squeeze(-1)
        
        # 2. Intrinsic Threat Risk in [0, 1]
        r_threat = self.threat_head(state_repr).squeeze(-1)
        
        # Epistemic Decision Entropy H_norm in [0, 1]
        if candidate_logits is not None and candidate_logits.shape[-1] > 1:
            K = candidate_logits.shape[-1]
            probs = F.softmax(candidate_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            h_norm = entropy / math.log(K)
            uncertainty = r_threat + h_norm - (r_threat * h_norm)
        else:
            uncertainty = r_threat
            
        # 3. Urgency / Complexity Omega in [0, 1]
        urgency_logits = self.urgency_head(state_repr)
        urgency_probs = F.softmax(urgency_logits, dim=-1)
        urgency = torch.sum(urgency_probs * self.complexity_weights, dim=-1)
        
        return {
            "valence": valence,          # V in [-1, 1]
            "uncertainty": uncertainty,  # U in [0, 1]
            "urgency": urgency           # Omega in [0, 1]
        }


class NeuromodulatoryController(nn.Module):
    """
    Translates Amygdalar (Valence, Uncertainty, Urgency) into downstream controls:
    - gamma_E, gamma_I: Corpus callosum interhemispheric coupling gains
    - T: Generative LLM temperature scaling
    - N_steps: Recurrent step budget for reasoning engines
    """
    def __init__(
        self,
        gamma_E0: float = 1.0,
        gamma_I0: float = 0.5,
        T_min: float = 0.05,
        T_max: float = 1.0,
        N_min: int = 1,
        N_max: int = 16,
        theta_U: float = 0.15,
        theta_Omega: float = 0.20
    ):
        super().__init__()
        self.gamma_E0 = gamma_E0
        self.gamma_I0 = gamma_I0
        self.T_min = T_min
        self.T_max = T_max
        self.N_min = N_min
        self.N_max = N_max
        self.theta_U = theta_U
        self.theta_Omega = theta_Omega

    def compute_modulations(
        self, 
        V: torch.Tensor, 
        U: torch.Tensor, 
        Omega: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates all downstream gain controls and budgets.
        """
        # 1. Corpus Callosum Interhemispheric Gains
        gamma_E = self.gamma_E0 * (1.0 + 0.5 * torch.tanh(V)) * torch.pow(1.0 - torch.clamp(U, 0.0, 0.99), 1.5) * (1.0 - 0.7 * Omega)
        gamma_E = torch.clamp(gamma_E, min=0.0, max=2.0)
        
        gamma_I = self.gamma_I0 * (1.0 + 2.0 * torch.pow(U, 2) + 1.5 * Omega - 0.3 * torch.clamp(V, min=0.0))
        gamma_I = torch.clamp(gamma_I, min=0.1, max=5.0)

        # 2. Downstream LLM Temperature Scaling
        temp_scale = torch.sigmoid((V - 0.0) / 0.5) * torch.exp(-1.5 * U - 1.2 * Omega)
        temperature = self.T_min + (self.T_max - self.T_min) * temp_scale
        temperature = torch.clamp(temperature, min=self.T_min, max=self.T_max)

        # 3. Recurrent Step Budget (System 1 Reflex Bypass vs System 2 Reasoning)
        bypass_mask = (U < self.theta_U) & (Omega < self.theta_Omega) & (V >= 0.0)
        
        negative_val_threat = torch.clamp(-V, min=0.0) * U
        phi = 0.40 * Omega + 0.35 * U + 0.25 * negative_val_threat
        phi = torch.pow(torch.clamp(phi, min=0.0, max=1.0), 1.8)
        
        continuous_steps = self.N_min + (self.N_max - self.N_min) * phi
        n_steps = torch.round(continuous_steps).to(torch.long)
        
        n_steps = torch.where(bypass_mask, torch.zeros_like(n_steps), n_steps)

        return {
            "gamma_E": gamma_E,
            "gamma_I": gamma_I,
            "temperature": temperature,
            "reasoning_steps": n_steps,
            "bypass_active": bypass_mask
        }
