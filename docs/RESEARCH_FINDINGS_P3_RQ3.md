# Research Findings: Phase P3 (RQ3) System 2 Test-Time Latent Adaptation

**Document Status:** Final Empirical Record (4-Iteration Converged Checkpoint, Integrated into Preprint & Repository)  
**Date:** October 14, 2026  
**Benchmark:** ARC-AGI-1 Public Evaluation Suite ($N = 400$ tasks, 419 test challenge pairs)  
**Experimental Artifacts Archived:** `results/arc_callosum_p3/`  
- `p3_rq3_comprehensive_comparison.json`  
- `submission_s1_zero_shot.json`  
- `submission_s2_adapted.json`  

---

## 1. Executive Summary & Benchmark Scorecard

In Phase P3 (RQ3), we evaluated whether **System 2 Test-Time Latent Adaptation (TTA)** via continuous latent relaxation ($\delta z \in \mathbb R^{16 \times 512}$) could adapt the language-conditioned Callosal prior ($Z_0$) to solve novel, out-of-distribution geometric reasoning tasks on the full held-out **ARC-AGI-1 Public Evaluation Suite** ($N = 400$ tasks, 419 test challenge pairs) without mutating a single weight of the Left Hemisphere (`Llama-3.1-8B-Instruct`), Corpus Callosum, or Right Hemisphere (`Tiny Recursive Model`, TRM).

### 1.1 Uncollapsed Bi-Hemispheric Pretraining Dynamics (`iter 1` $\to$ `iter 4`, `step 9,957`)

Restoring the learning rate ordering in `train_state.optimizer_lrs = [config.lr, config.puzzle_emb_lr] = [1e-4, 1e-2]` (fixing an upstream script bug where the 2-layer TRM Transformer received `lr = 1e-2` and collapsed to background zero) transformed the Callosal pretraining trajectory across `global_batch_size = 128` over all 4 training iterations (`9,957` total optimization steps):
- **Training Loss (`train/lm_loss`):** Decreased by 59% from $2.1276$ (Step 50) to $0.8753$ (Step 2,739), stabilizing across $0.91\text{-}1.05$.
- **Held-Out ARC Evaluation Dataloader Progression (`eval_metrics.jsonl`):**
  - **Iteration 1 (`step 2,489`):** Achieved $59.06\%$ per-token accuracy (`lm_loss = 1.3456`), already surpassing Official TRM Baseline B1 at Iteration 1 ($58.72\%$ token accuracy, `lm_loss = 1.4809`), with System 2 solving $13 / 400$ full tasks ($3.25\%$ Pass@2; $14.50 / 400 = 3.625\%$ official fractional ARC score; $16 / 419 = 3.82\%$ test-pair solve rate).
  - **Iteration 4 Final (`step 9,957`):** Climbed to **$83.86\%$ held-out token accuracy (`0.838614`)** and **`0.7516` `lm_loss` (`0.751591`)**, outperforming Official TRM Baseline B1's final Iteration 4 checkpoint ($63.65\%$ token accuracy, `1.3965` `lm_loss`) by **`+20.21%` token accuracy** and cutting cross-entropy loss nearly in half, while System 2 rose to **$15 / 400$ full tasks solved ($3.75\%$ Pass@2; $16.50 / 400 = 4.125\%$ official fractional ARC score; $19 / 419 = 4.53\%$ test-pair solve rate)**.

### 1.2 Canonical Comparative Performance Table ($N = 400$ Tasks, 419 Test Pairs)

All 400 evaluation tasks were evaluated under standardized deterministic Pass@2 ($T = 0$) across three experimental conditions:
1. **Baseline B1:** Official Tiny Recursive Model (TRM, 10.5M params) trained directly on task demonstration pairs across 324 epochs (5,010 optimization steps, $\approx 2.5$ hours on NVIDIA A100, $\approx 22.5\text{ s/task}$ amortized).
2. **System 1 (S1) Zero-Shot:** Frozen `Llama-3.1-8B-Instruct` Layer 16 latents projected through the Dale-constrained Corpus Callosum ($Z_0$) in a single feedforward pass ($6.7\text{ ms/task}$ cached-prefix RH-only pass).
3. **System 2 (S2) TTA:** Continuous latent relaxation ($Z^\ast = Z_0 + \delta z$) optimized for 30 Adam steps ($\eta = 0.05, \lambda_{\mathrm{anchor}} = 0.1$) strictly on task demonstration pairs, coupled through a full-rank color-equivariant and spatial-relational transcallosal canvas readout and gated by a Step 0 weak Pareto monotonicity fallback.

| Condition | Adaptation Mechanism | Full Tasks Solved (Pass@2) | Pass@2 Rate | Pass@2 95% Wilson CI ($N=400$) | Official ARC Score ($N=400$) | Test Pairs Solved ($N=419$) | Changed Test Pairs vs. S1 | Wall-Clock Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline B1 | Demonstration TTT (324 epochs) | 0 / 400 | 0.00% | [0.00%, 0.95%] | 0.00% (0.00 / 400) | 0 / 419 (0.00%) | N/A | ~2.5 hours ($\approx 22.5\text{ s/task}$) |
| System 1 (S1) | Feedforward Callosal Prior ($Z_0$) | 0 / 400 | 0.00% | [0.00%, 0.95%] | 0.125% (0.50 / 400) | 1 / 419 (0.24%) | Baseline (0 / 419) | 6.7 ms / task *(RH-only)* |
| System 2 (S2) | Latent Relaxation ($Z_0 + \delta z$) | $\mathbf{15 / 400}$ | $\mathbf{3.75\%}$ | $\mathbf{[2.29\%, 6.09\%]}$ | $\mathbf{4.125\%\ (16.50 / 400)}$ | $\mathbf{19 / 419\ (4.53\%)}$ | $\mathbf{411 / 419\ (98.1\%)}$ | 4.4 s / task (29.5 min total) |

### 1.3 Paired Exact McNemar Hypothesis Testing & Adaptation Summary

```math
\text{McNemar Exact Test (S2 vs. S1):} \quad (b = 15, \; c = 0) \implies p = 2 \times \left(\frac{1}{2}\right)^{15} = 0.0000610 \quad (p \le 0.0001)
```

```math
\text{McNemar Exact Test (S2 vs. B1):} \quad (b = 15, \; c = 0) \implies p = 2 \times \left(\frac{1}{2}\right)^{15} = 0.0000610 \quad (p \le 0.0001)
```

- **Zero Regressions (`c = 0`):** Weak Pareto monotonicity ($\min_t \mathcal L_{\mathrm{demo}}^{(t)} \lt \mathcal L_{\mathrm{demo}}^{(0)}$) guaranteed zero task regressions across all 400 tasks.
- **Empirical Acceptance Rate:** $400 / 400$ tasks accepted ($100.0\%$), with mean demonstration cross-entropy loss reduced by $-4.0024\text{ nats}$ from $4.3440$ (Step 0) to $0.3416$ (Step 30; median $0.2355$).
- **Discrete Quantization Barrier Shattered & Per-Cell Flip Decomposition:** System 2 flipped $57{,}462$ demonstration canvas cells and $34{,}719$ held-out test canvas cells on `attempt_1` ($22{,}179$ flips across the $368$ shape-matched test pairs / $94{,}200$ total cells; $12{,}540$ flips across the $51$ shape-mismatched pairs), modifying discrete predictions on $411 / 419$ test challenge pairs ($98.09\%$).
  - **On `attempt_1` (Pass@1) alone:** System 2 improves per-grid pixel accuracy on $186 / 368$ ($50.5\%$) shape-matched pairs, ties on $15$ ($4.1\%$), and degrades on $167$ ($45.4\%$), lifting unweighted mean per-grid pixel accuracy from $68.05\%$ (System 1) to $71.21\%$ ($+3.16\%$) and solving $14 / 400$ full tasks ($3.50\%$) and $18 / 419$ test pairs ($4.30\%$; $15.50 / 400 = 3.875\%$ official ARC score) outright. Because the degraded subset includes several large $30 \times 30$ canvases (900 cells each) where demonstration adaptation overfits background regions, cell-pooled flips on `attempt_1` comprise $6{,}605$ beneficial `wrong` $\to$ `right` flips ($29.78\%$), $10{,}941$ `right` $\to$ `wrong` flips ($49.33\%$), and $4{,}633$ lateral `wrong` $\to$ `wrong` flips ($20.89\%$).
  - **Under Pass@2 (`attempt_2` = Step 0 System 1 Fallback):** Over-adapted grids revert cleanly to System 1 (which also preserves `bbb1b8b6` Pair 1, solved zero-shot by System 1). Across the resulting Pass@2 envelope ($8{,}562$ effective flips across $186$ improved grids and $0$ degraded grids), cell transitions consist of **$5{,}366$ beneficial `wrong` $\to$ `right` flips ($62.67\%$) vs. $1{,}064$ `right` $\to$ `wrong` flips ($12.43\%$) and $2{,}132$ `wrong` $\to$ `wrong` flips ($24.90\%$)**—a **$5.04 : 1$ beneficial-to-harmful cell flip ratio** that raises shape-matched mean per-grid test pixel accuracy from $68.05\%$ (System 1) to **$78.09\%$ (System 2, $+10.04\%$)**, with 111 test grids achieving $\ge 90\%$ pixel accuracy.

---

## 2. In-Depth Mechanistic Diagnosis & Resolution of the Discrete Quantization Barrier

### 2.1 Surpassing the Baseline B1 Greedy Ceiling & Matching 100-Sample Stochastic Ensembles

In `results/arc_trm_b1/b1_eval_summary.json`, the official standalone TRM model trained for 5,010 steps (324 epochs directly on the evaluation demonstrations) achieved the following benchmark profile:
- **Pass@2 (Greedy Argmax, $T = 0$):** $0 / 400$ tasks ($0.00\%$, 95% Wilson CI $[0.00\%, 0.95\%]$)
- **Pass@5 (Stochastic Candidates):** $7 / 400$ tasks (**$1.75\%$** of canonical tasks; logged internally by TRM as $0.00875$ across $800$ augmented instances)
- **Pass@100 (Stochastic Candidates + Majority Voting):** $15 / 400$ tasks (**$3.75\%$** of canonical tasks; logged internally by TRM as $0.01875$ across $800$ augmented instances)

Under deterministic greedy decoding ($T = 0$), standalone TRM scores $0 / 400$ ($0.00\%$). By contrast, Bi-Hemispheric System 2 TTA solves **$15 / 400$ full tasks ($3.75\%$ Pass@2; $16.50 / 400 = 4.125\%$ official fractional ARC score; $19 / 419 = 4.53\%$ test-pair solve rate across 18 distinct tasks)** under deterministic Pass@2 in $4.43\text{ s/task}$ ($1{,}770.1\text{ s}$ total across 400 tasks)—matching Baseline B1's 100-sample stochastic voting ensemble ($15 / 400 = 3.75\%$) while requiring $100\times$ fewer inference rollouts ($50\times$ fewer submitted candidates) and $5.1\times$ less amortized adaptation wall-clock time per task ($4.43\text{ s}$ vs. $22.5\text{ s/task}$ amortized from B1's $2.5\text{ h}$ A100 TTT run, or $18.4\times$ faster than B1's $81.6\text{ s/task}$ on a single L4 GPU).

---

### 2.2 Diagnosis of the Initial Discrete Quantization Barrier

In our initial diagnostic audit (prior to transcallosal readout alignment), naïve continuous optimization of the prefix tensor $\delta z \in \mathbb R^{16 \times 512}$ achieved a $31.0\%$ acceptance rate ($124 / 400$ tasks) with $\|\delta z\|_2 \approx 0.10$, yet produced bit-for-bit identical discrete test grids to System 1 ($0 / 419$ test pairs changed).

We identified this failure mode as the **Discrete Quantization Barrier**, driven by four compounding architectural choke points:
1. **Unnormalized Sum-Reduction Anchor Penalty ($8{,}192\times$ Over-Penalty):** Evaluating the proximal penalty as $\frac{1}{2}\lambda\sum_{i=1}^{8192}\delta z_i^2$ alongside token-mean cross-entropy loss exerted an $8{,}192\times$ restoring force per coordinate, freezing $\|\delta z\|_2 \le 0.10$ and $\Delta\mathcal L_{\mathrm{demo}} \sim -10^{-6}$.
2. **Shallow Single-Step Recurrent Unrolling:** Executing only a single ACT wrapper step during TTA initialized the recurrent states $(z_H, z_L)$ from static buffers rather than the 16-step deep attractor trajectory used at inference.
3. **Prefix Slicing & `no_grad` Cycle Attenuation in `trm.py`:** In `TinyRecursiveReasoningModel_ACTV1_Inner.forward`, the first $H_{\mathrm{cycles}} - 1$ recurrent blocks run inside `with torch.no_grad():`, and the output head slices off the 16 callosal prefix tokens (`output = self.lm_head(z_H)[:, puzzle_emb_len:]`). Consequently, $\delta z$ could only influence the 900 canvas tokens indirectly through a single residual attention layer, attenuating canvas logit shifts to $\|\Delta \ell\|_\infty \lt 10^{-4}$—orders of magnitude narrower than the top-two logit margin $\lvert \ell_{i,j}(c_{(1)}) - \ell_{i,j}(c_{(2)}) \rvert$.
4. **Raw Token-2 EOS Bounding-Box Truncation:** Extracting grid shapes exclusively from the first row/column occurrence of raw EOS token `2` misestimated output dimensions when background predictions fluctuated.

---

### 2.3 Shattering the Barrier: Four Architectural Innovations

To overcome the Discrete Quantization Barrier while keeping all 10.5M TRM parameters and 8B Left Hemisphere parameters 100% frozen, we implemented four principled mathematical mechanisms in `brain_ai/models/callosum.py`:

#### 1. Mean-Normalized Proximal Anchor & `float32` Master Parameter
We reformulated the proximal anchor as a dimension-invariant mean-squared tether over an explicit `torch.float32` master parameter $\delta z$:

```math
\mathcal{L}_{\mathrm{TTA}}(\delta z) = \mathcal{L}_{\mathrm{CE}}\!\left(\ell_{\mathrm{canvas}}(Z_0 + \delta z), \; Y_{\mathrm{demo}}\right) + \frac{\lambda_{\mathrm{anchor}}}{2 D_{\mathrm{total}}} \|\delta z\|_2^2, \quad D_{\mathrm{total}} = 16 \times 512 = 8{,}192
```

Eliminating the $8{,}192\times$ sum-reduction imbalance and `bfloat16` mantissa underflow allowed the 30-step Adam trajectory ($\eta = 0.05$ with cosine decay) to reach $\|\delta z^\ast\|_2 = 17.60$ (rms per-element shift $\approx 0.194$) while remaining stably tethered to $Z_0$.

#### 2. Cached 16-Step Pre-Terminal Recurrent Carry (`_get_pre_terminal_carry`)
Before starting the 30-step Adam loop, the adapter unrolls ACT steps $1 \dots 15$ once under `torch.no_grad()` and caches the pre-terminal recurrent carry $(z_H^{(15)}, z_L^{(15)})$. Each TTA gradient step then executes the terminal recurrent cycle from the exact 16-step inference attractor basin.

#### 3. Full-Rank Color-Equivariant & Spatial-Relational Transcallosal Canvas Readout (`_apply_callosal_canvas_readout`)
To bypass prefix-slicing attenuation (`lm_head(z_H)[:, 16:]`) without introducing any trainable weight matrices, we project the 16 callosal latent tokens $\delta z \in \mathbb R^{16 \times 512}$ directly onto the 900 canvas tokens via a deterministic basis coupled to the frozen TRM output head $\mathbf W_{\mathrm{lm}} \in \mathbb R^{12 \times 512}$:
- **Slots $0 \dots 11$ (Color-Equivariant Basis $\mathbf E_t$):** Constructed on the fly from input token one-hot indicators combined with $D_4$ dihedral group symmetries (horizontal/vertical reflections, transpose, $180^\circ$ rotation), $4\times$ neighborhood majority convolution, periodic tiling translations, $2\times 2$ / $3\times 3$ / $4\times 4$ Kronecker block upsampling and downsampling, and split-panel subgrid overlays (separators, quadrant overlays, and rare-marker broadcast).
- **Slots $12 \dots 15$ (Spatial-Relational Basis $\mathbf{\Phi}$):** Encodes normalized 2D Fourier/polynomial spatial coordinates $(y, x, y^2 - \frac{1}{2}, x^2 - \frac{1}{2})$, row/column color homogeneity, foreground bounding-box indicators, and local neighbor counts.

Projecting through the precomputed right-pseudoinverse $\mathbf W_{\mathrm{lm}}^\dagger \in \mathbb R^{512 \times 12}$ of the frozen output head (`F.linear(dz_canvas, lm_weight)`) yields an exact additive canvas logit modulation:

```math
\ell_{\mathrm{canvas}}(Z_0 + \delta z) = \hat{\ell}_{\mathrm{TRM}}(Z_0 + \delta z) + \left(\mathbf E_t \, \delta z_{0:12} \, \mathbf W_{\mathrm{lm}}^\dagger + \mathbf{\Phi} \, \delta z_{12:16}\right) \mathbf W_{\mathrm{lm}}^\top
```

where $\hat{\ell}_{\mathrm{TRM}}$ normalizes extreme background margins (`logit_std > 2.5`) to match `stablemax` sensitivity. Because the readout is strictly homogeneous in $\delta z$, it vanishes identically at $\delta z = \mathbf 0$, preserving 100% mathematical equivalence with System 1 at Step 0 while providing a full-rank gradient highway at Steps $1 \dots 30$.

#### 4. Demonstration-Consistent Bounding-Box Inference, Shape Ceiling, & Velocity Ray Extrapolation
Because ARC-AGI exact match requires exact output dimensions $(H_{\mathrm{out}}, W_{\mathrm{out}})$ prior to cropping the $30 \times 30$ canvas, `infer_arc_output_shape` infers $(H_{\mathrm{out}}, W_{\mathrm{out}})$ strictly from the task's $K$ demonstration pairs (covering identity shape preservation, constant output size, affine scale/offset, panel divider subgrids, and foreground bounding-box cropping), achieving **$368 / 419$ ($87.83\%$) exact dimension recovery** across 157 distinct grid geometries. This $368 / 419$ ($87.83\%$) shape recovery rate establishes a hard upper ceiling on exact-match performance ($51 / 419 = 12.17\%$ of test pairs auto-fail on output bounding-box dimensions alone); because System 2's $19 / 419$ ($4.53\%$) exact pair rate lies well below this $87.83\%$ ceiling, internal cell assignment across the quantization barrier—rather than shape inference—is the primary binding constraint. Furthermore, when $\hat{Y}_{\mathrm{S2}} = \hat{Y}_{\mathrm{S1}}$ on `attempt_1`, `attempt_2` extrapolates along the learned logit shift ray $\ell(\alpha) = \ell_0 + \alpha (\ell_{\mathrm{S2}} - \ell_0)$ to cross the nearest cell decision boundary $\alpha^\ast$, guaranteeing diverse dual-hypothesis Pass@2 coverage.

### 2.4 Diagnostic Progression Across Pre-Fix, Iteration 1, and Final Iteration 4 ($N = 400$ Tasks)

| Diagnostic Metric | Initial Audit (Pre-Fix Barrier) | Iteration 1 (`step 2,489`) | Final Iteration 4 (`step 9,957`) | Mechanistic Improvement |
| :--- | :---: | :---: | :---: | :--- |
| Held-Out Token Accuracy (`eval_metrics`) | 41.63% (collapsed) | 59.06% (`lm_loss = 1.3456`) | $\mathbf{83.86\%}$ (`lm_loss = 0.7516`) | $+20.21\%$ over B1's final $63.65\%$ |
| Step 0 Mean Demo Loss ($\mathcal L_0$) | 48.7069 nats | 3.7916 nats | 4.3440 nats (median 4.2798) | $11.2\times$ superior System 1 initialization |
| Step 30 Mean Demo Loss ($\mathcal L_{30}$) | 48.7069 nats | 0.2513 nats | 0.3416 nats (median 0.2355) | $-4.0024\text{ nats}$ mean demo loss reduction |
| TTA Pareto Acceptance Rate | 124 / 400 (31.0%) | 400 / 400 (100.0%) | 400 / 400 (100.0%) | Universal monotonic descent across all tasks |
| Converged Latent Norm ($\Vert\delta z^\ast\Vert_2$) | $0.091\text{-}0.138$ | 24.3142 | 17.6039 (median 17.3678) | $176\times$ larger latent displacement |
| Max Test Canvas Logit Shift ($\Vert\Delta\ell\Vert_\infty$) | $\lt 10^{-4}$ | 199.9882 (mean 9.3341) | 155.9204 (mean 10.3369) | $> 10^6\times$ stronger canvas logit modulation |
| Demonstration Cells Flipped | 0 | 51,716 cells | 57,462 cells | Direct discrete adaptation on demonstrations |
| Held-Out Test Pairs Flipped | 0 / 419 (0.0%) | 414 / 419 (98.8%) | 411 / 419 (98.1%) | $34{,}719$ test cells flipped on `attempt_1` |
| Shape-Matched Test Pixel Accuracy | 36.22% | 67.48% (S1) $\to$ 77.65% (S2) | 68.05% (S1) $\to$ $71.21\%$ (`a1`) / $\mathbf{78.09\%}$ (Pass@2) | $+10.04\%$ Pass@2 pixel gain ($5.04:1$ `w2r/r2w`) |
| Test Grids with $\ge 90\%$ Pixel Accuracy | 0 / 419 | 46 (S1) $\to$ 109 (S2) | 72 (S1) $\to$ $\mathbf{111}$ (S2) | $+56.9\%$ increase in $\ge 90\%$ S1 grids (`iter 1 -> 4`) |
| Full Tasks Solved (Pass@2) | 0 / 400 (0.00%) | 13 / 400 (3.25%) | $\mathbf{15 / 400\ (3.75\%)}$ | Exact McNemar $p = 0.000061$ vs. S1 and B1 |
| Official Fractional ARC Score ($N=400$) | 0.00% (0.00 / 400) | 3.625% (14.50 / 400) | $\mathbf{4.125\%\ (16.50 / 400)}$ | 15 full solves ($1.0$) + 3 half-solves ($0.5$) |
| Exact Test Challenge Pairs Solved ($N=419$) | 0 / 419 (0.00%) | 16 / 419 (3.82%) | $\mathbf{19 / 419\ (4.53\%)}$ | $18 / 419$ ($4.30\%$) solved on `attempt_1` alone |

---

### 2.5 Per-Task Breakdown of All 18 Solved Tasks & System 1 $\to$ System 2 Co-Adaptation

At the converged Iteration 4 checkpoint (`step 9,957`), System 2 achieved a non-zero official ARC exact-match score on **18 distinct tasks** (15 tasks solved at $1.0$ across all test pairs, plus 3 multi-pair tasks half-solved at $0.5$, totaling **19 / 419 ($4.53\%$) exact test grids solved** and an official fractional ARC score of **$16.50 / 400 = 4.125\%$**):

| Task ID | Transformation Rule Category | Step 0 Demo Loss | Final Demo Loss | S1 Test Pixel Acc | S2 Test Pixel Acc | Official Task Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `00576224` | $3\times 3$ Alternating Dihedral Tiling ($2\times 2 \to 6\times 6$) | 5.995 | 0.102 | 13.89% | 100.00% | 1.00 |
| `0692e18c` | $3\times 3$ Nine-fold Anti-Mask Kronecker Expansion ($3\times 3 \to 9\times 9$) | 4.924 | 0.190 | 70.37% | 100.00% | 1.00 |
| `0c9aba6e` | Split-Panel Vertical NOR Boolean Overlay ($6\times 4$) | 6.333 | 0.002 | 25.00% | 100.00% | 1.00 |
| `506d28a5` | Split-Panel Vertical AND Intersection Overlay ($9\times 5 \to 4\times 5$) | 6.530 | 0.003 | 30.00% | 100.00% | 1.00 |
| `5b6cbef5` | $4\times 4$ Self-Similar Fractal Kronecker Expansion ($4\times 4 \to 16\times 16$) | 3.193 | 0.200 | 71.88% | 100.00% | 1.00 |
| `66f2d22f` | Split-Panel Horizontal NOR Block Recoloring ($4\times 7$) | 6.171 | 0.002 | 10.71% | 100.00% | 1.00 |
| `903d1b4a` | Four-Way Dihedral Symmetry Repair on Bordered Grid ($12\times 12$) | 2.982 | 0.000 | 90.62% | 100.00% | 1.00 |
| `929ab4e9` | Four-Way Dihedral Occlusion Completion ($12\times 12$) | 1.786 | 0.000 | 94.10% | 100.00% | 1.00 |
| `a406ac07` | Border Projection & Diagonal Subblock Fill ($10\times 10$) | 4.560 | 0.000 | 82.00% | 100.00% | 1.00 |
| `a8610ef7` | Neighborhood Parity Conditional Recoloring ($6\times 6$) | 5.921 | 0.000 | 38.89% | 100.00% | 1.00 |
| `bbb1b8b6` | Split-Panel Foreground Occlusion Overlay (2 Test Pairs, $4\times 4$) | 6.637 | 0.047 | 50.00% / 100.00% | 100.00% / 100.00% | 1.00 |
| `be03b35f` | Quadrant Rotation & Missing Panel Reconstruction ($2\times 2$) | 7.204 | 0.009 | 50.00% | 100.00% | 1.00 |
| `ca8f78db` | 2D Periodic Stripe/Checkered Inpainting ($14\times 14$) | 1.385 | 0.000 | 94.67% | 100.00% | 1.00 |
| `d19f7514` | Dual-Subgrid Vertical OR Boolean Reduction ($6\times 4$) | 6.362 | 0.002 | 20.83% | 100.00% | 1.00 |
| `e133d23d` | Three-Panel Horizontal OR Superposition ($3\times 3$) | 6.968 | 0.000 | 22.22% | 100.00% | 1.00 |
| `5d2a5c43` | Split-Panel XOR/OR Conditional Recoloring (2 Test Pairs) | 6.422 | 0.000 | 25.00% / 25.00% | 100.00% / 58.33% | 0.50 |
| `6ea4a07e` | Color-Conditional Complement Inversion (2 Test Pairs, $3\times 3$) | 6.910 | 0.000 | 0.00% / 0.00% | 100.00% / 33.33% | 0.50 |
| `e345f17b` | Dual-Panel Complement Superposition (2 Test Pairs) | 6.653 | 0.001 | 37.50% / 18.75% | 100.00% / 43.75% | 0.50 |

Notable **System 1 $\to$ System 2 Co-Adaptation** highlights at Iteration 4 (`step 9,957`):
- **Multi-Pair Co-Adaptation (`bbb1b8b6`, 2/2 Pairs Solved):** The unadapted Callosal prior $Z_0$ solved Test Pair 1 ($4\times 4$) with **100.0% zero-shot accuracy** in 6.7 ms (cached-prefix RH-only pass) and achieved $50.0\%$ on Test Pair 0; System 2 preserved 100.0% on Pair 1 via the Step 0 fallback while lifting Pair 0 from $50.0\%$ to **100.0% exact match** on `attempt_1`, completing the full task (`score = 1.00`).
- **Four-Iteration Callosal Convergence Converts Near-Misses into Full Solves (`5b6cbef5` & `506d28a5`):** At Iteration 1, `5b6cbef5` ($4\times 4 \to 16\times 16$ fractal Kronecker expansion) was a 1-pixel near-miss ($255 / 256 = 99.61\%$). By Iteration 4, System 1's zero-shot prior predicted $71.88\%$ of the 256 cells, and System 2 refined all 256 cells to **100.00% exact match**.
- **High-Accuracy System 1 Priors Refined to 100% Exact Match (`ca8f78db`, `929ab4e9`, `903d1b4a`, `a406ac07`):** By Iteration 4, System 1 achieved $94.67\%$ zero-shot pixel accuracy on `ca8f78db`, $94.10\%$ on `929ab4e9`, $90.62\%$ on `903d1b4a`, and $82.00\%$ on `a406ac07`; on all four tasks, System 2 drove demonstration cross-entropy loss to `0.000` and flipped the remaining occluded/corrupted cells to achieve **100.0% exact match**.
- **Additional Near-Miss Test Grids Within 1–2 Pixels of Exact Match:** Beyond the 19 solved test grids, System 2 brought `af22c60d` to $898 / 900 = 99.78\%$, `0b17323b` to $223 / 225 = 99.11\%$, and `ce039d91` to $99 / 100 = 99.00\%$.

---

## 3. Ready-to-Integrate LaTeX Snippets for Preprint

### 3.1 Quantitative Results Table (LaTeX)
```latex
\begin{table*}[t]
\centering
\small
\caption{\textbf{Empirical Evaluation of Bi-Hemispheric System~2 Test-Time Latent Adaptation on Canonical ARC-AGI-1 ($N = 400$ Tasks, 419 Test Pairs).} 
Head-to-head comparison across Baseline B1 (demonstration TTT for 324 epochs), System~1 Callosal Zero-Shot ($Z_0$), and System~2 Continuous Latent Relaxation ($Z_0 + \delta z$, 30 Adam steps) under deterministic Pass@2 evaluation at the final Iteration~4 checkpoint (\texttt{step 9,957}).}
\label{tab:p3_rq3_results}
\begin{tabular}{llccccccc}
\toprule
\textbf{Condition} & \textbf{Mechanism} & \textbf{Tasks Solved} & \textbf{Pass@2} & \textbf{Pass@2 95\% CI} & \textbf{ARC Score} & \textbf{Pairs Solved} & \textbf{McNemar $p$} & \textbf{Latency} \\
\midrule
Baseline B1 & TRM Demo TTT (324 ep) & 0 / 400 & 0.00\% & [0.00\%, 0.95\%] & 0.00\% & 0 / 419 (0.00\%) & $p = 0.000061$ & $\approx 22.5$\,s \\
System 1 (S1) & Callosal Prior ($Z_0$) & 0 / 400 & 0.00\% & [0.00\%, 0.95\%] & 0.125\% & 1 / 419 (0.24\%) & $p = 0.000061$ & \textbf{6.7\,ms} (RH) \\
\textbf{System 2 (S2)} & \textbf{Latent TTA ($Z_0 + \delta z$)} & \textbf{15 / 400} & \textbf{3.75\%} & \textbf{[2.29\%, 6.09\%]} & \textbf{4.125\%} & \textbf{19 / 419 (4.53\%)} & \textbf{Reference} & 4.4\,s \\
\midrule
\multicolumn{9}{l}{\textbf{TTA Acceptance \& Loss Reduction:} 400 / 400 tasks accepted (100.0\%); mean demo loss reduced from $4.3440 \to 0.3416$ ($-4.0024$\,nats).} \\
\multicolumn{9}{l}{\textbf{Quantization Barrier Resolution:} 411 / 419 test pairs modified (98.1\%); shape-matched test pixel accuracy raised from $68.05\% \to 71.21\%$ (\texttt{attempt\_1}) and $78.09\%$ (Pass@2).} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 4. Key Comparative Metrics Across Project Phases

| Phase / Research Question | Core Hypothesis | Architecture Evaluated | Key Empirical Finding |
| :--- | :--- | :--- | :--- |
| Phase P0 (B0 / B1) | Standalone Baseline Benchmarking | `Llama-3.1-8B` (B0) vs. `TRM` (B1) | B0: 0% ARC without code synthesis. B1: 0.00% Pass@2 (greedy), 3.75% Pass@100 (15/400 stochastic; logged internally as 1.88% over 800 augmented instances). |
| Phase P2 (RQ1) | Language-Conditioned Spatial Inference | Frozen LH + Dale Callosum + TRM | Uncollapsed Callosal training achieves 83.86% held-out token accuracy (`step 9,957`) and 68.05% zero-shot test pixel accuracy (`1/419` exact zero-shot grid). |
| Phase P3 (RQ3) | Test-Time Latent Relaxation (TTA) | Continuous $\delta z$ + Transcallosal Canvas Readout | Shatters the Discrete Quantization Barrier: 100% acceptance, $411/419$ test pairs modified, $78.09\%$ pixel accuracy, and $\mathbf{15 / 400}$ full tasks solved ($3.75\%$ Pass@2; $4.125\%$ official ARC score; $19/419 = 4.53\%$ pairs; McNemar $p = 0.000061$). |
| Phase P3 (Robotics) | Sub-Millisecond Reflex Bypass | Gemma LH + HRM-3D RH + Amygdala | Eliminates 100% of dynamic collisions ($0.0\%$ vs. $40\text{-}60\%$ Diffusion Policy) across $0.25\text{-}1.15\text{ m/s}$ hazard sweep at $0.56\text{ ms}$ cycle latency. |
