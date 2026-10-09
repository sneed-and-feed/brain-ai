# Bi-Hemispheric Neuromorphic AI (`brain-ai`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)

A biologically inspired AI architecture coupling an open-weight Large Language Model (Left Hemisphere) with a multi-timescale Hierarchical Reasoning Model (Right Hemisphere), unified via a Dale-constrained Excitatory-Inhibitory Corpus Callosum and gated by a sub-25ms Computational Amygdala.

---

## Architecture Overview

```
                     [ Input State / Query x ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼ (Sub-25ms)                                    ▼
┌─────────────────────────┐                   ┌───────────────────┐
│  Computational Amygdala │                   │  Sensory Latent   │
│   (Open-Jev 2B Head)    │                   │    Embeddings     │
└───────────┬─────────────┘                   └─────────┬─────────┘
            │                                           │
  Affective State a = [V, U, Ω]                         │
  - Valence V in [-1, +1]                               │
  - Threat/Uncertainty U in [0, 1]                      │
  - Urgency/Load Ω in [0, 1]                            │
            │                                           │
            ▼                                           │
┌─────────────────────────┐                             │
│ Neuromodulatory Control │                             │
│ (Adaptive Gain LC-NE)   │                             │
└───────────┬─────────────┘                             │
            │                                           │
   ┌────────┴───────────────────┐                       │
   │ Gains: γ_E, γ_I            │                       │
   │ Temp: T_gen                │                       │
   │ Budget: N_steps            │                       │
   ▼                            ▼                       ▼
┌─────────────────────┐   ┌──────────────┐    ┌───────────────────┐
│   Left Hemisphere   │   │    Corpus    │    │  Right Hemisphere │
│  (Qwen 14B BF16)    │◄─►│   Callosum   │◄──►│    (Sapient HRM   │
│ Linguistic / Syntax │   │ (Dale E-I)   │    │  Multi-Timescale) │
└─────────────────────┘   └──────────────┘    └───────────────────┘
```

### Core Components
1. **Left Hemisphere ($\mathcal{H}_L$)**:
   - **Backbone**: Qwen 2.5 / 3 14B in native BF16.
   - **Role**: Sequential symbolic manipulation, linguistic synthesis, and syntax.
   - **Interface**: Intermediate residual stream hook points ($l \in \{16, 32, 48\}$) with LoRA rank 64 adapters.
2. **Right Hemisphere ($\mathcal{H}_R$)**:
   - **Backbone**: Sapient HRM (27M reasoning module or 1B text-aligned module; Guan Wang et al., 2025/2026, [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)).
   - **Role**: Non-autoregressive spatial topology, cellular constraint satisfaction, and multi-timescale recurrence ($H_{\text{slow}}$ and $L_{\text{fast}}$).
   - **Memory**: 1-step fixed-point equilibrium implicit gradient approximation ($O(1)$ memory).
3. **Corpus Callosum ($\mathcal{C}_{LR}$)**:
   - **Dale's Principle**: Synaptic projections enforce $W = \operatorname{Softplus}(V) \cdot D$ with strict column sign segregation ($80\%$ excitatory, $20\%$ inhibitory).
   - **Rajan-Abbott Balance**: Balanced initialization ($f_E \mu_E = f_I \mu_I$) eliminates the outlier eigenvalue ($\lambda_{\text{outlier}} = 0, R \le 1.0$).
   - **Differential Cross-Attention**: Employs biologically inspired differential attention ($A_E - \lambda A_I$) with net **transcallosal inhibition** ($s = -1.0$; Hong Jeong 2026, [arXiv:2603.03355](https://arxiv.org/abs/2603.03355)) to prevent bank-dominance collapse.
   - **Homeostasis**: Turrigiano synaptic scaling and RMSNorm to prevent epileptic explosion ($\|h\| \to \infty$) and coma collapse ($\|h\| \to 0$).
4. **Computational Amygdala ($\mathcal{A}$)**:
   - **Backbone**: Open-Jev (Zefan Cai, 2026) scalar readout heads.
   - **Outputs**: 3D Affective State $\mathbf{a} = [V, U, \Omega]^\top$ (Valence, Threat/Uncertainty, Urgency).
   - **Modulation**: Dynamically controls excitatory gain $\gamma_E$, inhibitory damping $\gamma_I$, generation temperature $T_{\text{gen}}$, and triggers a sub-20ms System 1 reflex bypass on routine, low-risk inputs.

---

## Hardware Envelope (80GB A100 VRAM Allocation)

| Component | Architecture | Precision | Static VRAM | Dynamic / Cache | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Left Hemisphere | Qwen 2.5 / 3 14B (Frozen Base) | BF16 | 29.4 GB | 6.0 GB | 35.4 GB |
| LH Adapters | LoRA ($r=64$) on Q, V, Gate, Down | BF16 | 0.4 GB | 4.5 GB | 4.9 GB |
| Right Hemisphere | Sapient HRM (27M / 1B) | BF16 | 0.06 / 2.0 GB | 1.5 GB | 1.6 / 3.5 GB |
| Corpus Callosum | 3x Dale Differential Cross-Attn | BF16 | 0.3 GB | 1.2 GB | 1.5 GB |
| Amygdala | Open-Jev 2B (Frozen) + Heads | FP8 / BF16 | 2.2 GB | 0.5 GB | 2.7 GB |
| Runtime & CUDA | FlashAttention-2, PyTorch overhead | — | 3.5 GB | 4.5 GB | 8.0 GB |
| **Total Ensemble** | **Full Bi-Hemispheric System** | **BF16 / FP8** | **35.9 GB** | **18.2 GB** | **54.1 GB / 80 GB** |

**Net Headroom**: **~25.9 GB safety margin** on a single 80GB A100.

---

## Directory Structure

```
brain-ai/
├── brain_ai/
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       ├── callosum.py       # Dale's Principle, Differential Attn, Synaptic Scaling
│       ├── amygdala.py       # Open-Jev System 1 Router & Neuromodulator
│       ├── hrm.py            # Sapient HRM, Dual Recurrence, MagicNorm, ACT
│       └── ensemble.py       # BiHemisphericBrain wiring LH, RH, Callosum, Amygdala
├── docs/
│   ├── ARCHITECTURE_SPEC.md  # Formal mathematical specification
│   └── RESEARCH_PLAN.md      # E2E research plan, benchmarks, and ablation matrix
├── tests/
│   └── test_callosum.py      # Module unit tests and mathematical compliance
├── pyproject.toml
└── README.md
```

---

## Quickstart

### Installation
```bash
git clone https://github.com/sneed-and-feed/brain-ai.git
cd brain-ai
uv sync
```

### Running Tests
```bash
uv run pytest
```

### Basic Forward Pass
```python
import torch
from brain_ai.models.ensemble import BiHemisphericBrain

brain = BiHemisphericBrain(
    d_lh=5120,          # Left hemisphere dimension (e.g. Qwen 14B)
    d_rh=512,           # Right hemisphere dimension (HRM)
    d_callosum=512,      # Callosal manifold dimension
    callosal_heads=4,
    hrm_cycles=3
)

# Simulated input latents
lh_latents = torch.randn(2, 64, 5120)  # [Batch, SeqLen, D_LH]
rh_inputs = torch.randn(2, 16, 512)    # [Batch, GridTokens, D_RH]

outputs = brain(lh_latents, rh_inputs)

print("Updated LH Latents:", outputs["lh_latents_updated"].shape)
print("Updated RH Latents:", outputs["rh_latents_updated"].shape)
print("Conflict Score:", outputs["conflict_score"])
print("Affective State:", outputs["affective_state"])
print("Neuromodulatory Controls:", outputs["neuromodulatory_controls"])
```

---

## Primary References

- **Sapient HRM**: Wang, G., Li, J. et al. (2025). *Hierarchical Reasoning Model.* [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).
- **Inhibitory Callosal Cross-Talk**: Jeong, H. (2026). *Inhibitory Cross-Talk Enables Functional Lateralization in Attention-Coupled Latent Memory.* [arXiv:2603.03355](https://arxiv.org/abs/2603.03355).
- **Differential Transformer**: Ye, T., Dong, L. et al. (ICLR 2025 Oral). *Differential Transformer.* [arXiv:2410.05258](https://arxiv.org/abs/2410.05258).
- **Rajan-Abbott Random Matrix Balance**: Rajan, K., & Abbott, L. F. (2006). *Eigenvalue spectra of random matrices for neural networks with Dale's law.* Physical Review Letters, 97(18), 188104.
- **Open-Jev**: Cai, Z. (2026). *Open-Jev: Open-Weight System One Models for Typed Decisions.* `github.com/Zefan-Cai/Open-Jev`.
- **Adaptive Gain Theory**: Aston-Jones, G., & Cohen, J. D. (2005). *An integrative theory of locus coeruleus-norepinephrine function.* Annual Review of Neuroscience, 28, 403-450.
- **The Split Brain & Interpreter**: Gazzaniga, M. S. (2000). *Cerebral specialization and interhemispheric communication.* Brain, 123(7), 1293-1326.
