# Architectural Specification: Bi-Hemispheric Neuromorphic AI

## 1. System Overview & Tensor Dimensions

```
   Left Hemisphere (LH)                 Right Hemisphere (RH)
    [Qwen 14B / BF16]                    [Sapient HRM / BF16]
     d_lh = 5120                          d_rh = 512 / 1536
          │                                      │
          ▼ (Linear Projection)                  ▼ (Linear Projection)
   Z_L in R^(B x S_L x 512)              Z_R in R^(B x S_R x 512)
          \                                      /
           \──────► [ Corpus Callosum ] ◄───────/
                    - Dale-Constrained Differential Cross-Attn
                    - Inhibitory Cross-Talk (s = -1.0)
                    - Rajan-Abbott Balanced Initialization
                    - Turrigiano Synaptic Scaling + RMSNorm
```

### Module Specifications
- **Left Hemisphere**:
  - Backbone: `Qwen/Qwen2.5-14B-Instruct` (or `Qwen3-14B`) in BF16
  - Hidden Dimension: $d_{\text{LH}} = 5,120$, Layers $L = 48$, Heads $Q=40, KV=8$
  - Hook points: Layers 16, 32, 48
  - Trainable parameters: LoRA rank 64 on $W_q, W_v, W_{\text{gate}}, W_{\text{down}}$ (~50M params)
- **Right Hemisphere**:
  - Backbone: `sapientinc/HRM` (27M) or scaled HRM (~10.5M parameters)
  - Hidden Dimension: $d_{\text{RH}} = 512$
  - Modules: $L_\theta$ (Fast low-level Transformer), $H_\phi$ (Slow high-level Transformer)
  - Timescale Ratio: $T : 1$ (default $T = 3$)
  - Optimization: 1-step fixed-point implicit function gradient approximation ($O(1)$ memory)
- **Corpus Callosum Bridge**:
  - Manifold Dimension: $D_{\text{call}} = 512$
  - Projection Heads: $N_{\text{heads}} = 4$, $D_{\text{head}} = 128$
  - Dale Partition: $80\%$ Excitatory ($+1$), $20\%$ Inhibitory ($-1$)
  - Cross-talk sign: $s = -1.0$ (net transcallosal inhibition)
  - Normalization: Turrigiano synaptic scaling with target energy $r^* = 1.0$
- **Computational Amygdala**:
  - Backbone: Independent small MLP router (inspired by Open-Jev)
  - Latency: $< 25\text{ ms}$ (non-autoregressive scalar heads)
  - Continuous 3D Outputs:
    - Valence $V \in [-1, 1]$
    - Threat/Uncertainty $U \in [0, 1]$
    - Urgency/Complexity $\Omega \in [0, 1]$

---

## 2. Mathematical Formulations

### Dale's Law Reparameterization
```math
W = \mathrm{Softplus}(V, \beta=1.0) \cdot \mathrm{diag}(s_1, \dots, s_{d_{\text{in}}})
```
where $s_j \in \{+1, -1\}$ with $f_E = 0.8$ and $f_I = 0.2$.

### Rajan-Abbott Spectral Matrix Balance
To ensure $\lambda_{\text{outlier}} = 0$ and $R \le 1.0$:
```math
\mu_E = \frac{1}{\sqrt{d_{\text{in}}}}, \quad \mu_I = \frac{f_E}{f_I} \mu_E = 4.0 \cdot \mu_E
```
```math
\sigma_E = \sigma_I = \frac{R_{\text{target}}}{\sqrt{d_{\text{in}} \left(f_E + f_I \left(\frac{f_E}{f_I}\right)^2\right)}}
```

### Dale-Constrained Differential Attention
```math
\mathrm{DiffAttn}(Q_1, Q_2, K_1, K_2, V) = \left[ \mathrm{softmax}\left(\frac{Q_1 K_1^\top}{\sqrt{d_k}}\right) - \lambda \cdot \mathrm{softmax}\left(\frac{Q_2 K_2^\top}{\sqrt{d_k}}\right) \right] V
```

### Transcallosal Gated Coupling
```math
Z_L^{\text{coupled}} = Z_L - \tanh(\gamma_{\text{call}}) \cdot \Delta Z_{R \to L}
```
```math
Z_R^{\text{coupled}} = Z_R - \tanh(\gamma_{\text{call}}) \cdot \Delta Z_{L \to R}
```

### Neuromodulatory Gain Control Laws
```math
\gamma_E(V, U, \Omega) = \gamma_{E0} \cdot (1 + 0.5 \tanh(V)) \cdot (1 - U)^{1.5} \cdot (1 - 0.7 \Omega)
```
```math
\gamma_I(V, U, \Omega) = \gamma_{I0} \cdot (1 + 2.0 U^2 + 1.5 \Omega - 0.3 \max(0, V))
```
```math
T_{\text{gen}}(V, U, \Omega) = T_{\min} + (T_{\max} - T_{\min}) \cdot \sigma\left(\frac{V}{0.5}\right) \cdot \exp(-1.5 U - 1.2 \Omega)
```
```math
N_{\text{steps}} = \mathrm{clip}\left(\left\lfloor N_{\min} + (N_{\max} - N_{\min}) \cdot \left[0.4 \Omega + 0.35 U + 0.25 \max(0, -V) U\right]^{1.8}\right\rceil, 1, 16\right)
```
If $(U < 0.15) \wedge (\Omega < 0.20) \wedge (V \ge 0.0)$, activate System 1 Bypass ($N_{\text{steps}} = 0$).
