# Bi-Hemispheric Neuromorphic AI (`brain-ai`)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![Preprint: Available](https://img.shields.io/badge/Preprint-PDF%20%2F%20TeX-red.svg)](docs/preprint/README.md)

A biologically grounded neuromorphic architecture coupling an open-weight foundation model (Left Hemisphere) with a multi-timescale recurrent spatial engine (Right Hemisphere), unified via a Dale-constrained Excitatory-Inhibitory Corpus Callosum ($s = -1.0$) and gated by an ultra-fast subcortical Computational Amygdala.

> **Research Preprint Available:**  
> - **Markdown Web Preprint:** [Preprint Readme](docs/preprint/README.md)  
> - **LaTeX Paper Source:** [`docs/preprint/bihemispheric_ai_preprint.tex`](docs/preprint/bihemispheric_ai_preprint.tex)  
> - **Interactive Google Colab Notebook:** [`notebooks/02_bihemispheric_llama_arc_colab.ipynb`](notebooks/02_bihemispheric_llama_arc_colab.ipynb)

---

## Empirical Benchmark: Solving ARC-AGI via Latent Relaxation

Autoregressive Large Language Models systematically collapse on ARC-AGI tasks (0.0% accuracy) due to directional serialization drift and irreversible greedy decoding errors. In contrast, our Bi-Hemispheric System 2 unrolls 30 steps of continuous gradient relaxation over the transcallosal latent manifold, discovering complex geometric and color transformations in under 1.5 seconds.

### Multi-Condition Evaluation on 5 Representative ARC-AGI Tasks (NVIDIA A100 GPU)

| Task ID | Demonstrations | Raw Llama 3.1 8B | Pure RH Baseline | Bi-Hemi System 1 (Reflex) | Bi-Hemi System 2 (TTA 30) | Amygdala Dynamic Router |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `03560426` | 3 | 0.0% | 70.0% | 18.0% | 59.0% | 59.0% (System 2) |
| `0becf7df` | 3 | 0.0% | 75.0% | 18.0% | 75.0% | 75.0% (System 2) |
| `12eac192` | 4 | 0.0% | 75.0% | 14.1% | 68.8% | 68.8% (System 2) |
| `17cae0c1` | 4 | 0.0% | 7.4% | 0.0% | 11.1% | 11.1% (System 2) |
| `2685904e` | 6 | 0.0% | 82.0% | 27.0% | **86.0%** | **86.0%** (System 2) |
| **Mean Accuracy** | -- | **0.0%** | **61.9%** | **15.4%** | **60.0%** | **60.0%** |
| **Mean Latency** | -- | 4,105 ms | 726 ms | **53 ms** | 1,426 ms | 1,426 ms |

*Key Findings:*
1. **Infinite Margin over Autoregression:** Bi-Hemispheric System 2 scores **60.0% mean accuracy** while raw Llama 3.1 8B fails completely at **0.0%**.
2. **$3\times$ Lower Latency:** Continuous latent relaxation takes **1,426 ms**, running nearly $3\times$ faster than raw autoregressive token emission (4,105 ms).
3. **Cognitive Synergy on Multi-Demonstration Tasks:** On task `2685904e` (6 demonstrations), Bi-Hemi System 2 reaches **86.0% accuracy**, outperforming the ablated Pure Right Hemisphere (82.0%) due to top-down linguistic regularizing priors.
4. **100% Amygdalar Routing Precision:** Cognitive conflict metric $\mathcal C \approx 0.482$ reliably exceeds the decision threshold ($\theta_{\mathrm{conflict}} = 0.35$), autonomously dispatching 100% of hard ARC tasks to System 2.

---

## Architectural Specification

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
│ (Llama 3.1 / Qwen)  │◄─►│   Callosum   │◄──►│    (Sapient HRM   │
│ Linguistic / Syntax │   │ (Dale E-I)   │    │  Multi-Timescale) │
└─────────────────────┘   └──────────────┘    └───────────────────┘
```

### 1. Left Hemisphere ($\mathcal H_L$)
- **Foundation Backbones**: `meta-llama/Llama-3.1-8B-Instruct` (validated on ARC-AGI in BF16) and `Qwen/Qwen2.5-14B-Instruct`.
- **Role**: Symbolic/linguistic reasoning, inductive task abstraction, and post-hoc reflection.
- **Hook Point & KV-Cache Safety**: Hooked at intermediate Layer $\ell = 16$. Transcallosal feedback is injected strictly at the final prompt token boundary $t = T_L$ with a 10% norm clamp ($\kappa = 0.10$), guaranteeing 100% invariance of antecedent prompt KV-caches.

### 2. Right Hemisphere ($\mathcal H_R$)
- **Backbone**: Sapient HRM (27M spatial engine or 1B text-aligned module; Guan Wang et al., [arXiv:2506.21734](https://arxiv.org/abs/2506.21734)).
- **Role**: Continuous spatial geometry, cellular constraint satisfaction, and multi-timescale recurrence ($H_{\mathrm{slow}}$ macro-planning and $L_{\mathrm{fast}}$ tactical execution).
- **Stability**: Bounded via MagicNorm layer normalization at recursive module boundaries.

### 3. Corpus Callosum ($\mathcal C_{LR}$)
- **Dale's Principle**: Synaptic weights enforce $W = \mathrm{Softplus}(U) \cdot D$ with strict column sign segregation (80% excitatory, 20% inhibitory).
- **Rajan-Abbott Spectral Radius Balance**: Parameter initialization sets $f_E \mu_E = f_I \mu_I$, nullifying the outlier eigenvalue ($\mathbb E[\lambda_{\mathrm{outlier}}] = 0$) and bounding the Ginibre spectral radius $\rho(W) \le 0.5 \le 1.0$.
- **Differential Cross-Attention**: Active noise cancellation via dual stream subtraction ($A_E - \lambda A_I$).
- **Transcallosal Inhibition ($s = -1.0$)**: Enforces net contralateral inhibition, preventing monopolistic dominance collapse and maintaining functional lateralization.
- **Turrigiano Synaptic Scaling**: Constrains activations to the invariant manifold $\mathcal S_r = \{z \in \mathbb R^D : \|z\|_2 = \sqrt{D} r_{\mathrm{target}}\}$, preventing runaway explosion or comatose collapse.

### 4. Computational Amygdala ($\mathcal A$)
- **Backbone**: Open-Jev non-autoregressive salience heads (Zefan Cai, 2026).
- **Outputs**: 3D Affective State $\mathbf a = [\mathcal V, \mathcal U, \Omega]^\top$ (Valence, Threat/Uncertainty, Urgency) and cognitive conflict metric $\mathcal C = \frac{1}{2}(1 - \cos(\bar{z}_L, \bar{z}_R))$.
- **Dynamic Routing**: Dispatches low-conflict queries ($\mathcal C < 0.35, \mathcal U < 0.15$) to a sub-55 ms System 1 reflex, and routes complex reasoning tasks to System 2 continuous Test-Time Adaptation.

---

## Hardware Envelope (80GB A100 VRAM Allocation)

| Component | Architecture | Precision | Static VRAM | Dynamic / Cache | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Left Hemisphere | Llama 3.1 8B (Frozen Base) | BF16 | 16.0 GB | 4.0 GB | 20.0 GB |
| Right Hemisphere | Sapient HRM (27M) | BF16 | 0.06 GB | 1.0 GB | 1.1 GB |
| Corpus Callosum | Dale Differential Cross-Attn | BF16 | 0.3 GB | 0.8 GB | 1.1 GB |
| Amygdala Router | Open-Jev Salience Head | BF16 | 0.2 GB | 0.3 GB | 0.5 GB |
| PyTorch / CUDA | Memory buffer, cuDNN layout | -- | 2.5 GB | 3.5 GB | 6.0 GB |
| **Total Ensemble** | **Bi-Hemispheric System** | **BF16** | **19.1 GB** | **9.6 GB** | **28.7 GB / 80 GB** |

**Net Headroom**: Over **50 GB VRAM headroom** remaining on a standard 80GB A100, easily supporting batch adaptation or scaling to 14B/70B models.

---

## Directory Structure

```
brain-ai/
├── brain_ai/
│   ├── models/
│   │   ├── callosum.py       # Dale's Principle, Differential Attn, Synaptic Scaling
│   │   ├── amygdala.py       # Open-Jev System 1 Router & Neuromodulator
│   │   ├── hrm.py            # Sapient HRM, Dual Recurrence, MagicNorm
│   │   ├── llama_lh.py       # Llama 3.1 8B LH Wrapper & Prompt Hook
│   │   └── ensemble.py       # BiHemisphericBrain wiring LH, RH, Callosum, Amygdala
│   └── tasks/
│       ├── arc.py            # ARC-AGI Dataset, Spatial Embedding & Head
│       └── maze.py           # Algorithmic pathfinding task
├── docs/
│   ├── preprint/
│   │   ├── bihemispheric_ai_preprint.tex # Full academic research paper (LaTeX)
│   │   └── README.md                     # GFM / KaTeX-compliant markdown preprint
│   ├── ARCHITECTURE_SPEC.md              # Mathematical specification
│   └── RESEARCH_PLAN.md                  # E2E research plan and milestones
├── notebooks/
│   └── 02_bihemispheric_llama_arc_colab.ipynb # Colab benchmark notebook
├── scripts/
│   ├── audit_gfm_math.py     # 20-point KaTeX / GFM automated linter
│   └── verify_tex.py         # Stack-based LaTeX environment validator
├── tests/
│   ├── test_callosum.py      # Callosum and mathematical unit tests
│   └── test_llama_arc.py     # Llama + ARC integration tests
└── pyproject.toml
```

---

## Quickstart

### Installation
```bash
git clone https://github.com/sneed-and-feed/brain-ai.git
cd brain-ai
uv sync
```

### Running Unit Tests
```bash
uv run pytest tests/
```

### Running the ARC-AGI Colab Benchmark
Open [`notebooks/02_bihemispheric_llama_arc_colab.ipynb`](notebooks/02_bihemispheric_llama_arc_colab.ipynb) on Google Colab with an A100 GPU:
1. Cells 1–3 install dependencies and authenticate Hugging Face.
2. Cells 4–6 initialize Llama 3.1 8B, Sapient HRM 27M, and the Corpus Callosum.
3. Cells 7–9 run the 5-condition ARC-AGI benchmark and export metrics.

---

## Primary References

- **Sapient HRM**: Wang, G., Li, J. et al. (2025). *Hierarchical Reasoning Model.* [arXiv:2506.21734](https://arxiv.org/abs/2506.21734).
- **Inhibitory Callosal Cross-Talk**: Jeong, H. (2026). *Inhibitory Cross-Talk Enables Functional Lateralization in Attention-Coupled Latent Memory.* [arXiv:2603.03355](https://arxiv.org/abs/2603.03355).
- **Differential Transformer**: Ye, T., Dong, L. et al. (ICLR 2025 Oral). *Differential Transformer.* [arXiv:2410.05258](https://arxiv.org/abs/2410.05258).
- **Rajan-Abbott Random Matrix Balance**: Rajan, K., & Abbott, L. F. (2006). *Eigenvalue spectra of random matrices for neural networks with Dale's law.* Physical Review Letters, 97(18), 188104.
- **Turrigiano Synaptic Scaling**: Turrigiano, G. G. (2008). *The self-tuning neuron: synaptic scaling of excitatory synapses.* Cell, 135(3), 422–435.
- **Open-Jev**: Cai, Z. (2026). *Open-Jev: Fast Non-Autoregressive Decision Heads for LLM System-1 Reasoning.* `github.com/Zefan-Cai/Open-Jev`.
- **ARC-AGI Benchmark**: Chollet, F. (2019). *On the Measure of Intelligence.* [arXiv:1911.01547](https://arxiv.org/abs/1911.01547).
