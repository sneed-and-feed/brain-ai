# Research Plan: Bi-Hemispheric Neuromorphic Architecture

**Revision:** 2026-10-10 (supersedes the 2026-10-09 plan and [`IMPLEMENTATION_PLAN_TO_100.md`](IMPLEMENTATION_PLAN_TO_100.md))  
**Companion documents:** [`preprint/README.md`](preprint/README.md), [`ARCHITECTURE_SPEC.md`](ARCHITECTURE_SPEC.md), [`EMBODIED_ROBOTICS_REPORT.md`](EMBODIED_ROBOTICS_REPORT.md)  
**Compute:** Google Colab Pro+ (single-accelerator sessions; see §7)

Source legend: ✅ checked against a primary source on 2026-10-10 (arcprize.org, arXiv, official code, Hugging Face). ⚠️ secondary source or unverified; confirm before citing.

---

## 0. Summary

1. **What exists.** A working codebase with these parts:
   - a Dale-constrained differential cross-attention "callosum";
   - a from-scratch HRM-style recurrent reasoner;
   - latent test-time adaptation with a proximal anchor and a no-regression floor;
   - D4 consensus and a Pass@2 selector;
   - Colab notebooks for ARC;
   - a CPU kinematic robotics sandbox.
2. **What the evidence supports.** An audit of the code and the Colab logs (§2) found problems with every reported ARC number:
   - Every reported ARC number is a per-cell (pixel) accuracy computed with the ground-truth output shape supplied. None is an exact-match task score.
   - The "ARC-AGI-2" evaluation actually ran on the ARC-AGI-1 public evaluation set.
   - Phase 1 trained on the same test pairs it was evaluated on.
   - The Phase 1 "cascaded router" chose S1 or S2 per task by looking at test accuracy, so it is an oracle.
   - The Phase 2 "System 1" path does not include the language model.

   **Consequently, no current result supports any claim about ARC task-solving ability, about the language model's contribution, or about any benefit from the Dale callosum.** Exact-match scores have not been measured.
3. **Field context.** Frontier models now score 91–95% on ARC-AGI-2 (semi-private). Under offline, small-compute rules, the best result is NVARC at 24.0% (private) / 27.6% (semi-private). Tiny recursive reasoners such as HRM and TRM score 2–8% on ARC-AGI-2. The ARC Prize analysis attributes most of HRM's performance to its outer refinement loop and to per-task embeddings trained on the evaluation demonstrations (§3).
4. **Revised direction.** The plan now asks two falsifiable questions under official scoring:
   - Can a frozen LLM's encoding of the demonstrations replace the per-task learned puzzle embedding of an HRM/TRM-class reasoner, so that it generalizes to unseen tasks? (RQ1)
   - Is inhibitory, E-I-constrained coupling a better conditioning pathway than standard alternatives at matched parameters? (RQ2)

   Both fit a Colab Pro+ budget if LLM latents are cached offline and ARC training runs at reduced, disclosed scale.

---

## 1. Current State (as of 2026-10-10)

| Component | Implementation | Notes |
| :--- | :--- | :--- |
| Left Hemisphere (LH) | Frozen `Llama-3.1-8B-Instruct` (Phase 1), `Qwen2.5-14B-Instruct` (Phase 2); mean-pooled mid-layer residual stream | The Phase 2 prompt is `"Solve ARC-AGI-2 challenge task {task_id}."` and contains **no grid content**. No log confirms whether the real Qwen weights or the mock loaded in the final runs. |
| Right Hemisphere (RH) | From-scratch HRM-style reimplementation ([`hrm.py`](../brain_ai/models/hrm.py), [`hrm_scaled.py`](../brain_ai/models/hrm_scaled.py)) | No Sapient checkpoint is loaded. The Phase 2 "HRM-1B" printed 10,498,052 parameters, so the 1B label is incorrect. |
| Corpus Callosum | `DaleLinear` + differential cross-attention + RMS (Turrigiano-style) scaling ([`callosum.py`](../brain_ai/models/callosum.py)) | Unit-tested; numerically stable at initialization. |
| Amygdala | Small MLP router (`hidden_dim=32`) named after Open-Jev ([`amygdala.py`](../brain_ai/models/amygdala.py)) | Does not load Open-Jev weights. Routing is driven by thresholds on demonstration pixel-fit. |
| Inference | S1 reflex; latent δz TTA with proximal anchor (λ = 2.0) and step-0 floor; D4 consensus; 6-primitive D4 "DSL"; Pass@2 selector | The "DSL" covers only identity, rotations and flips. |
| Robotics | CPU kinematic Franka sandbox ([`robotics_sandbox.py`](../brain_ai/tasks/robotics_sandbox.py)) | Policies are hand-coded controllers; see §2.4. |

---

## 2. Evidence Audit

### 2.1 Measurement and protocol defects

| ID | Defect | Location | Effect |
| :--- | :--- | :--- | :--- |
| M1 | Per-cell accuracy is reported as "exact-match accuracy". | Phase 1: `(pred == gt).mean()` in notebook 02. Phase 2: `safe_pixel_acc` in [`generate_phase2_notebook.py`](../scripts/generate_phase2_notebook.py). The `match_*` fields are computed but never reported. | Every ARC number in the preprint. |
| M2 | The ground-truth output shape is given to the model: the head receives `target_shape`, and predictions are cropped or padded to that shape. | Phase 1 crop `[:H_out, :W_out]`; Phase 2 `target_shape = test_target.shape` | Hides output-size errors. ARC requires predicting the output dimensions. |
| M3 | "Pass@2 accuracy" is the larger of the two attempts' pixel scores. | `pix_pass2 = max(...)` | Inflates Pass@2 and its significance test against S1. |
| M4 | The "ARC-AGI-2 held-out N=400" set is the **ARC-AGI-1** public evaluation set loaded from `fchollet/ARC-AGI`. | `data/arc/arc_master.zip` = `ARC-AGI-master` | Mislabelled benchmark. ARC-AGI-2 public eval has 120 tasks. |
| M5 | Phase 1 test-label leakage: pretraining samples `test_pairs[0]` input→output from a merged training+evaluation pool (≤15×15), and the benchmark scores `test_pairs[0]` of tasks drawn from that same pool. | `ARCDataset._load_from_zip`, `get_batch` | Phase 1 accuracies are contaminated. |
| M6 | The Phase 2 N=25 runs used ARC-AGI-1 **training** tasks, which are also the alignment-training pool. | Colab logs (tasks `007bbfb7`, `1f642eb9`, …) | The abstract's Phase 2 figures are in-distribution. |
| M7 | The Phase 1 "Reflex-First Cascaded Router, 66.25%" equals the per-task maximum of S1 and S2 **test** accuracy. | Re-score cell; reproduces 66.25 exactly from the N=25 table | Oracle selection, not routing. |
| M8 | The raw-LLM baseline used `max_new_tokens=128`, which truncates most grids, and scored 0 unless the shape matched. It was then tested against a constant zero vector. | Notebook 02, Condition 1 | The 0.0% result and the p = 2×10⁻⁸ are artifacts. |
| M9 | The Phase 2 S1 path is a single RH L-module segment, with no LLM and no callosum. | `forward_system1_reflex` | 6.7 ms is RH-only latency. It is not comparable with Phase 1's 67 ms, which included an 8B forward pass. |
| M10 | Simulated and hard-coded quantities: "naive S2" is `0.85 × S1` with a constant 1,450 ms, and the DSL fit has a 0.50 floor. | Phase 2 benchmark cell | Must be removed or disclosed. |
| M11 | The committed `checkpoints/phase2/benchmark_metrics_phase2.json` comes from a 2-task local CPU dry run (0/2 exact). The Colab N=400 per-task file (`eval400_progress.json`) was never committed. | Git history | The headline results have no reproducible artifact. |
| M12 | Single seed; no run-to-run variance; no multiple-comparison control. | All runs | Significance claims are unsupported. |

### 2.2 Claims ledger

| # | Claim (preprint) | Status | Required action |
| :--- | :--- | :--- | :--- |
| 1 | ARC-AGI-1 S1 59.4% "exact-match" at 67 ms, at parity with HRM | Restate | Report as a pixel-accuracy pilot with oracle shape and contamination (M1, M2, M5). Not evidence of task solving. |
| 2 | Cascaded router 66.3%, Wilcoxon p = 0.042 | **Retract** | Oracle selection (M7). |
| 3 | 23× and 62× speedups "at parity" | Retract 62×; restate 23× | 62× rests on M8. The 23× compares a forward pass with a 30-step optimization, which is expected by construction, so it is not "distillation" unless S1 is shown to reproduce what TTA computes. |
| 4 | "ARC-AGI-2, N=400": S1 50.04%, Pass@2 67.67%, t(399) = 14.7 | **Retract** benchmark name; restate metric | M1–M4. The t-test compares a max over candidates with a single candidate, so it is inflated by construction. |
| 5 | Abstract's Phase 2 figures (58.13%, 64.37%, N=25) | Remove | Training-set tasks (M6); inconsistent with §5 of the preprint. |
| 6 | 67 → 6.7 ms reflex speed-up attributed to CoordConv | **Retract** attribution | The LLM was removed from the path (M9). |
| 7 | Hardened S2 "11-fold recovery" (5.61% → 58.76%) | Restate | The floor makes S2 fall back to S1. At N=400, S2 − S1 = +0.15 pp, so there is no evidence that adaptation helps. |
| 8 | Amygdala allocates 64% / 82.5% of tasks to S2 | Restate | Rule-based thresholds on demonstration pixel-fit. No calibration evidence. |
| 9 | Bidirectional linguistic reflection (task `332efdb3`) | Restate as anecdote | The explanation was never checked against the task's actual rule; faithfulness is untested. |
| 10 | Theorem 2.1 "guarantees" spectral radius ≤ 0.5 and "precludes explosion" | Narrow | Holds in expectation **at initialization** for i.i.d. weights. Nothing is guaranteed during or after training. |
| 11 | Lemma 2.2 "Lyapunov energy conservation" | Narrow | This is the norm identity of RMS normalization. Drop the dynamical-systems language. |
| 12 | Proposition 2.1(2): the injection stays "strictly within the local linear basin" | Downgrade | Design heuristic; measure the perplexity shift empirically. |
| 13 | Conflict-metric statistics, p < 10⁻¹⁵ | Fix | §2.4 reports a range of 0.441–0.449 while §4.3.4 reports 0.530–0.551. Remove the p-value. |
| 14 | "First to hybridize discrete LLMs with continuous recurrent engines" | Remove | Prior art includes Coconut, recurrent-depth models and Energy-Based Transformers (§3.4). |
| 15 | Robotics: 100% collision elimination | Restate | See §2.4. |

### 2.3 Citation corrections

| Preprint reference | Problem | Correct citation |
| :--- | :--- | :--- |
| [3] HRM, "G. Wang, J. Li, and T. Zhang" | Wrong authors | G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, S. Song, Y. Abbasi Yadkori, "Hierarchical Reasoning Model," arXiv:2506.21734, 2025 ✅ |
| [9] "S. Hong and H. Jeong, Functional lateralization in deep neural networks…" | Wrong author and title. Single author. | H. Jeong, "Inhibitory Cross-Talk Enables Functional Lateralization in Attention-Coupled Latent Memory," arXiv:2603.03355, 2026 ✅ |
| [12] "Z. Cai, Open-Jev…, arXiv:2502.08912" | That arXiv ID is an unrelated mathematics paper. No Open-Jev paper exists. | Cite as software: Z. Cai, *Open-Jev* (GitHub `Zefan-Cai/Open-Jev`; HF `ZefanCai/Open-Jev-2B`, a LoRA + scalar head on Qwen3.5-2B), 2026 ✅. Also state that the repository's router is an independent small MLP. |
| [13] Differential Transformer, "T. Ye, J. Dong, and H. Sun" | Wrong authors | T. Ye, L. Dong, Y. Xia, Y. Sun, Y. Zhu, G. Huang, F. Wei, ICLR 2025, arXiv:2410.05258 ✅ |
| [17] Tree of Thoughts, NeurIPS 2024 | Wrong year | NeurIPS 2023 |
| [18] Song, Yang & Wang, PLOS Comp. Biol. 2023, "Reward-based training, bursty activity…" | Wrong title and year | H. F. Song, G. R. Yang, X.-J. Wang, "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework," *PLOS Comput. Biol.* 12(2): e1004792, 2016 ✅ |
| [14] Rajan & Abbott 2006 | Title abbreviated incorrectly | "Eigenvalue Spectra of Random Matrices for Neural Networks," *Phys. Rev. Lett.* 97, 188104 ✅ |
| (missing) | Directly relevant prior work is not cited | TRM (arXiv:2510.04871), ARC Prize HRM analysis, Cornford et al. ICLR 2021, Li et al. NeurIPS 2023, CompressARC (arXiv:2512.06104), Akyürek et al. (arXiv:2411.07279). See §11. |

### 2.4 Robotics status

The sandbox kinematics are carefully implemented (DLS, Shepperd quaternions, capsule collision checks), but the benchmark does not test the architecture:

- **Bi-hemispheric policy.** It is a hand-coded time-to-contact and repulsive-field controller that reads ground-truth obstacle position, velocity and radius. The amygdala module is constructed but never called. No LLM, HRM or callosum runs in the control loop. `reflex_latency_ms = 2.5` is a constructor constant.
- **"Monolithic VLA".** It contains no model. It is a proportional controller whose command is held for 500 ms; that latency is assigned, not measured.
- **Diffusion Policy.** This is a real 2.75M-parameter model, but it receives no obstacle input, and its avoidance is a hand-coded vector. Interfaces are therefore not matched, despite the report's claim.
- **Velocity confound.** The confound raised in review was never re-run with a corrected design.

**Status:** deferred (§4.3). The report should be retitled as a control-rate study (100 Hz reactive control vs 2 Hz open-loop replanning), with the limitations above stated.

### 2.5 Re-score of the recovered N=400 artifact

The Colab per-task file was recovered as [`data/eval400_progress.json`](../data/eval400_progress.json) (400 unique ARC-AGI-1 evaluation tasks, `00576224` … `ff72ca3e`). Re-scoring it, with the true output shape still supplied (M2), gives:

| Condition | Pixel accuracy (as reported) | Exact match (test pair 0) |
| :--- | :---: | :---: |
| System 1 reflex | 50.04% | 1 / 400 (0.25%) |
| Hardened System 2 | 50.19% | not recorded |
| D4 consensus | 49.73% | not recorded |
| Cascaded Pass@1 | 62.57% | 2 / 400 (0.50%) |
| Cascaded Pass@2 | 67.67% | 2 / 400 (0.50%) |

The same pixel metric, applied to baselines that do no learning:

| Trivial baseline (true output shape given) | Pixel accuracy | Exact match |
| :--- | :---: | :---: |
| Predict all zeros | 48.95% | 1 / 400 |
| Most frequent colour in the demonstration outputs | 55.49% | 2 / 400 |
| Copy the test input | 64.18% | 1 / 400 |
| Best of 6 D4 primitives, chosen by demonstration fit (no model) | 65.31% | 2 / 400 |

Findings:
- System 1, System 2 and D4 consensus all score below simply copying the test input.
- The Pass@2 gain over System 1 comes from the non-learned D4 primitive candidate and from taking the larger of two pixel scores (M3).
- The two exact solves need no model:
  - `e872b94a` has an all-zero 3×1 output, which a zero prediction matches once the shape is supplied.
  - `be03b35f` was solved by the D4-primitive candidate.
- 330 of 400 tasks have System 2 pixel accuracy identical to System 1: the no-regression floor reverted the adaptation.
- 19 tasks have more than one test pair, but only `test[0]` was scored.

**Conclusion:** the artifact contains no learned signal above trivial baselines, so there is no result to recover. Further work must start from the P0/P1 protocol (§8) rather than from a re-run of notebook 03.

**Reusable code:** the callosum, D4 utilities, Pass@2 selector, incremental checkpointing and notebook infrastructure can be reused unchanged.

---

## 3. External Landscape (October 2026)

### 3.1 Benchmarks and scoring

| Benchmark | Public train | Public eval | Semi-private / private | Notes |
| :--- | :---: | :---: | :---: | :--- |
| ARC-AGI-1 | 400 | 400 | 100 / 100 | Public eval and semi-private are **not** difficulty-calibrated: HRM dropped about 9 pp ✅ |
| ARC-AGI-2 | 1,000 | 120 | 120 / 120 | The eval sets are calibrated to each other ⚠️ (split sizes come from secondary sources). The public training set reportedly reuses ARC-AGI-1 tasks ⚠️, so deduplicate by grid hash before training. |
| ARC-AGI-3 | — | — | — | Interactive game environments scored on action efficiency; live ✅ |

**Official scoring** ✅ (from `arcprize/arc-agi-benchmarking`, `scoring.py`):
- Each test pair gets 2 attempts.
- A pair counts as solved only if one attempt matches the entire output grid exactly, including its dimensions.
- Task score = solved test pairs ÷ test pairs.
- Benchmark score = mean task score.

### 3.2 Performance regimes (ARC-AGI-2 semi-private unless noted) ✅

| System class | Example | Score | Cost per task |
| :--- | :--- | :---: | :---: |
| Frontier model | GPT-6 Astra (Max) / Claude Opus 5.5 (High) | 95.0% / 93.3% | USD 1.12 / 0.41 |
| Refinement scaffold | GPT-5.2 (Refine.) / Gemini 3 Pro + Poetiq | 72.9% / 54.0% | USD 38.99 / 30.57 |
| Offline Kaggle (2025 winner) | NVARC (≈4B Qwen TTT + synthetic data + TRM components) | 27.6% (24.0% private) | USD 0.20 |
| Tiny recursive reasoner | TRM (7M) / HRM (27M) | 6.3% / 2.0% | USD 2.10 / 1.68 |

ARC Prize 2025 (Kaggle, private) ✅:
- Ranking: NVARC 24.0%, the ARChitects 16.5%, MindsAI 12.6%.
- Paper awards went to TRM, to Pourcel et al., and to Liao & Gu (CompressARC).

ARC Prize 2026 ✅:
- Submissions are due **2 Nov 2026**, papers 8 Nov.
- Kaggle runs offline, so API-based systems are excluded.
- Winners must open-source their code.

### 3.3 Recursive reasoners: what is known

The ARC Prize analysis of HRM ("The Hidden Drivers of HRM's Performance on ARC-AGI", 15 Aug 2025) ✅ found:
- The H/L hierarchy contributes little: a same-size transformer comes within about 5 pp.
- The **outer refinement loop** is the main driver. Going from 1 to 2 loops adds 13 pp, and training with refinement matters more than refining at inference.
- Cross-task transfer is limited. Training only on the 400 eval tasks' demonstrations still reaches 31%, compared with 41% overall.
- 30 augmentations come within about 4 pp of 1,000.
- HRM conditions on a learned **puzzle-ID embedding** trained on the evaluation tasks' demonstrations, which makes it test-time training in effect. It cannot handle a task ID it never saw in training.

TRM (Jolicoeur-Martineau, arXiv:2510.04871) ✅ is a 7M-parameter, 2-layer model with these paper-reported results:

| Benchmark | TRM score |
| :--- | :---: |
| ARC-AGI-1 | 44.6% |
| ARC-AGI-2 | 7.8% |
| Sudoku-Extreme | 87.4% |
| Maze-Hard | 85.3% |

Follow-ups worth reading first:
- arXiv:2512.11847, on identity conditioning in TRM. ⚠️ A search summary reports that accuracy collapses without task IDs.
- arXiv:2511.02886, test-time adaptation of TRMs.
- arXiv:2601.10679, "reasoning or guessing" in HRMs.
- arXiv:2609.39967, "What Limits Recursive Reasoning Models".

### 3.4 Prior work for each component

| Component | Closest prior work | Implication |
| :--- | :--- | :--- |
| Inhibitory callosal coupling | Jeong 2026 (arXiv:2603.03355): inhibitory cross-talk yields specialization; excitatory coupling lets one bank take over the other. | Replicate this in the conditioning setting (RQ2) rather than claim novelty for it. |
| Dale's law in ANNs | Cornford et al., ICLR 2021 (DANNs); Li et al., NeurIPS 2023: the singular-value spectrum at initialization matters more than the sign constraints. | Separate the effect of sign constraints from the effect of spectral initialization (RQ2b). |
| Latent reasoning coupled to LLMs | Coconut (arXiv:2412.06769); recurrent depth (arXiv:2502.05171); Energy-Based Transformers (arXiv:2507.02092) | Remove the "first" claim and position the work against these. |
| Test-time adaptation on ARC | Akyürek et al. (arXiv:2411.07279); Product of Experts (arXiv:2505.07859); CompressARC (arXiv:2512.06104) | Compare latent-only δz adaptation against weight-space test-time training at matched compute (RQ3). |
| Lateralized networks | arXiv:2608.19514; arXiv:2209.06862; arXiv:2407.11456 | Cite in related work. |

### 3.5 Implications

1. Claims of competitiveness with frontier systems are out of scope. The appropriate comparison class is offline, small-compute recursive reasoners: TRM, HRM and CompressARC.
2. The clearest open problem in that class is **task conditioning without per-task embedding training**. The architecture can test this directly: the LH reads the demonstrations, and the callosum conditions the reasoner on them.
3. Any study of a recursive reasoner must control the number of outer refinement loops; otherwise loop count confounds every other effect.

---

## 4. Research Questions and Hypotheses

### 4.1 Primary

**RQ1: Language-conditioned task inference.** Can a frozen LLM's encoding of serialized demonstrations replace the learned puzzle-ID embedding of a TRM-class reasoner?

- **H1.** At matched reasoner size and outer-loop count, conditioning on LH latents through the callosum yields higher official Pass@2 on **unseen** tasks than either control:
  - (a) no task conditioning;
  - (b) a learned demonstration encoder with matched trainable parameters.
- **Reference point.** (c) The puzzle-ID embedding trained on evaluation demonstrations (HRM/TRM protocol) serves as the test-time-training reference.
- **Falsifier.** If (b) matches or beats the LH variant, the LLM adds nothing beyond a generic encoder.

**RQ2: Excitatory-inhibitory coupling.** Is an inhibitory, Dale-constrained callosum a better conditioning pathway than standard alternatives?

- **H2a.** At matched parameter count, inhibitory coupling (s = −1) matches or exceeds excitatory coupling (s = +1), unconstrained cross-attention and FiLM in Pass@2 and training stability. This replicates Jeong (2026) in this setting.
- **H2b.** Most of any Dale effect comes from spectral initialization rather than the sign constraint. Tested with a 2×2 factorial: Dale sign × spectral initialization.

### 4.2 Secondary

**RQ3: Test-time adaptation.**
- **H3.** Latent-only δz adaptation with leave-one-out model selection on the demonstrations improves Pass@2 over no adaptation.
- It should do so at lower wall-clock cost than weight-space test-time training (puzzle embedding or LoRA) at matched compute.

**RQ4: Metacognitive routing.**
- **H4.** A router that uses only demonstration-derived signals predicts test exact-match well enough to cut cost at a fixed Pass@2. Candidate signals:
  - leave-one-out exact-match on the demonstrations;
  - D4 agreement;
  - candidate entropy.
- **Reporting.** Risk–coverage curves and AUROC, with the threshold pre-registered.

### 4.3 Removed or deferred

| Item | Decision | Reason |
| :--- | :--- | :--- |
| Hallucination suppression (SimpleQA, TruthfulQA, HaluEval) | Removed | No tested pathway by which the architecture changes LLM outputs. Revisit only if the R→L injection shows a measurable effect. |
| Robotics | Deferred | Requires the fixes listed in §2.4 first. |
| "HRM-Text-1B" | Removed | Unverified artifact. |
| Bidirectional verbalization | Deferred | Requires a faithfulness metric, e.g. executing a program derived from the explanation. |

---

## 5. Evaluation Protocol (binding for all future results)

1. **Scoring.** Port the official ARC scorer: per test pair, two attempts, whole-grid exact match including dimensions, and fractional task scores. Pixel accuracy may appear only as a labelled diagnostic.
2. **No oracle information.** The model must predict output dimensions. Remove `target_shape` from inference paths.
3. **Splits.**

| Use | Data |
| :--- | :--- |
| Training | ARC-AGI-1 training (400). Optionally ARC-AGI-2 training, after deduplicating against the eval set by grid hash. |
| Development | 10% of training tasks, held out |
| Primary report | ARC-AGI-1 public eval (400) |
| Final report | ARC-AGI-2 public eval (120) |
| Algorithmic controls | Sudoku-Extreme and Maze-Hard (HRM/TRM releases) |

4. **Leakage guard.** Assert at run start that no evaluation test output appears anywhere in the training stream (hash check), and fail the run if it does.
5. **Disclosure.** State explicitly whether evaluation-task demonstrations are used for any training (puzzle embeddings, TTA, LoRA).
6. **Statistics.**
   - Run at least 3 seeds and report mean ± SD across seeds.
   - Compare systems with McNemar's exact test (binary task outcomes) or a paired bootstrap over tasks (fractional scores).
   - Apply Holm correction within each ablation family.
   - Keep sampling error in mind: at N = 120, one task is 0.83 pp, and a 95% binomial interval at 20% accuracy is about ±7 pp.
7. **Cost and latency.**
   - Report end-to-end wall-clock per task (including any LH forward pass, with `torch.cuda.synchronize()`).
   - Report accelerator type and compute units per task.
8. **Provenance.**
   - Commit per-task JSON results and a run manifest from Colab. The manifest records git commit, GPU, model IDs, a mock/real flag, seed and wall-clock.
   - Runs abort if any model falls back to a mock.
9. **Pre-registration.** Commit each hypothesis, its primary metric and its analysis to `docs/prereg/` before running.

---

## 6. Baselines and Ablations

All conditions use the same reasoner scale, number of outer refinement loops, augmentation budget and training steps.

| ID | Condition | Purpose |
| :--- | :--- | :--- |
| B0 | Trivial floors: all zeros, copy test input, most-frequent colour, best D4 primitive by demonstration fit | Floor. Reported in every table; the recovered N=400 artifact did not beat them (§2.5). |
| B1 | TRM, official code, with puzzle-ID embedding (trained on eval demonstrations) | Test-time-training reference (RQ1c) |
| B2 | TRM without task conditioning | RQ1a |
| B3 | TRM + learned demonstration encoder (parameter-matched) | RQ1b, the key control |
| B4 | CompressARC (optional) | Zero-pretraining reference |
| B5 | Direct LLM prompting with a sufficient token budget, exact match only | Sanity check; no significance testing against it |
| A1 | LH latent from a task-ID-only prompt (the current setup) | Shows the current LH conditioning carries no task information |
| A2 | LH layer sweep (¼, ½, ¾ depth) | Where task information lives in the LH |
| A3 | LH scale: Qwen2.5 0.5B / 1.5B / 7B / 14B (Qwen3 equivalents as a replication) | Scaling curve; cheap with cached latents |
| A4 | Conditioning pathway: FiLM / concatenation / cross-attention / differential attention / Dale differential attention | RQ2a |
| A5 | Coupling sign: +1 / −1 / learned | RQ2a, Jeong (2026) replication |
| A6 | Dale sign × spectral initialization (2×2) | RQ2b |
| A7 | TTA: none / δz / δz + leave-one-out / puzzle-embedding TTT / LoRA | RQ3 |
| A8 | Outer refinement loops: 1 / 4 / 16 | Controls the known main driver |
| A9 | Test-time D4 consensus on/off | Isolates the augmentation contribution |

---

## 7. Compute Plan (Colab Pro+)

### 7.1 Platform constraints

| Item | Value | Source |
| :--- | :--- | :--- |
| Accelerators | Vary over time and are not guaranteed. Seen in practice: T4, L4 (24 GB), A100 (40 GB, sometimes 80 GB), H100, TPU v5e/v6e. | FAQ ✅; specific types ⚠️ |
| Background execution | Up to 24 h per session while compute units remain | FAQ ✅ |
| Monthly allotment | About 600 compute units for about USD 50 | ⚠️ |
| Burn rate | A100 about 10–15 units/h, so roughly 40–60 A100-hours per month; H100 higher; T4 about 2 units/h | ⚠️ Check *Runtime → View resources* |
| Exhausted units | Account falls back to free-tier limits | FAQ ✅ |

### 7.2 Memory fit

| Model | BF16 weights | 4-bit (NF4/AWQ) | Fits |
| :--- | :---: | :---: | :--- |
| Qwen2.5-14B / Qwen3-14B | ≈ 29.5 GB | ≈ 10 GB | A100-80 in BF16; A100-40 or L4 only in 4-bit |
| Llama-3.1-8B | ≈ 16 GB | ≈ 5.5 GB | Any A100; L4 in BF16 at short context |
| TRM / HRM-class reasoner (7–27M) | < 0.2 GB | — | Any GPU. Training throughput is the binding limit, not memory. |

### 7.3 Design decisions forced by the budget

1. **Cache LH latents offline.**
   - Run one forward pass per task × D4 view × selected layers and save the latents to Drive.
   - This takes the LLM out of the training loop, so reasoner training fits on an L4 or A100-40 and every LH-scale ablation (A3) costs only one caching pass.
   - The LH is frozen throughout, so caching loses nothing.
2. **Reduced augmentation budget.**
   - Use 30–300 augmentations per task instead of 1,000, justified by the HRM analysis above.
   - All conditions share the same budget, and the reduced scale is disclosed as a limitation.
3. **Development order.**
   - Debug on Sudoku-Extreme and Maze-Hard subsets first: single GPU, fast iteration, and a known TRM reference.
   - Then ARC-AGI-1.
   - ARC-AGI-2 is used only for the final report.
4. **Session resilience.**
   - Checkpoint to Drive at least every 30 minutes, and make training resumable from any checkpoint.
   - Write per-task evaluation results incrementally and commit them at the end of the run.
5. **Software.**
   - Use PyTorch with BF16, `torch.compile` and SDPA/FlashAttention on A100/H100.
   - Avoid TPUs unless the reasoner is ported to JAX, because PyTorch/XLA overhead is not justified at this scale.
   - Use T4 sessions only for smoke tests.

### 7.4 Budget estimate (to be replaced by pilot measurements)

| Work package | Accelerator | Estimated A100-hours |
| :--- | :--- | :---: |
| P0 integrity re-runs and scorer validation | A100-40 / L4 | 5 |
| LH latent caching (4 model sizes × 2 datasets × 8 views) | A100-80 | 6–10 |
| P1 TRM reproduction: Sudoku-Extreme (reduced) + ARC-AGI-1 B1–B3 (reduced) | A100-40 | 55–85 |
| P2 RQ1 main + A1–A3, 3 seeds | A100-40 | 40 |
| P3 RQ2 ablations A4–A6, 3 seeds | A100-40 / L4 | 40 |
| P4 RQ3/RQ4 (A7, router) | A100-40 | 20 |
| P5 ARC-AGI-2 final, 3 seeds | A100-40 | 20 |
| **Total** | | **≈ 190–220** |

At 40–60 A100-hours per month, the programme needs about 3–5 months of Pro+ allotment, or extra purchased compute units.

> [!WARNING] Compute Validation
> TRM reported ~18h on 1× L40S for Sudoku, and ~3 days on 4× H100 for ARC. These figures have been verified from the official repository README. Notebook 04 runs a throughput probe to dynamically size the ARC batch constraint for the Colab 40GB/80GB A100. Absolute scores will fall below published TRM numbers, and only within-study comparisons are valid.

---

## 8. Roadmap and Decision Gates

| Phase | Window | Deliverables | Gate (go / no-go) |
| :--- | :--- | :--- | :--- |
| P0 Integrity | 12–25 Oct 2026 | Notebook 04; `brain_ai.eval`; preprint corrections (§9) | Exact-match scorer matches official; leakage guard passes; B1 baseline establishes TRM anchor. |
| P1 Baselines | 26 Oct – 22 Nov 2026 | B0–B3 on Sudoku-Extreme (subset) and ARC-AGI-1 at reduced scale, 3 seeds | B1 > B2 on ARC-AGI-1. This reproduces the known value of task conditioning; otherwise debug the pipeline before going further. |
| P2 RQ1 | 23 Nov – 20 Dec 2026 | LH-conditioned reasoner; A1–A3 | The LH variant beats B3, paired test at p < 0.05 after Holm correction, consistent across 3 seeds. **If not:** report the negative result and continue RQ2 with B3 as the conditioning source. |
| P3 RQ2 | Jan 2027 | A4–A6 | None; results are reported whatever their sign. |
| P4 RQ3 / RQ4 | Feb 2027 | A7; router risk–coverage analysis | Adopt TTA only if it improves Pass@2 at matched compute. |
| P5 Final | Mar 2027 | ARC-AGI-2 public eval; revised preprint; code and per-task results released | All claims trace to committed artifacts. |

**ARC Prize 2026** (deadline 2 Nov 2026) is **not** targeted, because no validated exact-match result exists. Revisit for ARC Prize 2027 (not yet announced ⚠️) or a workshop submission after P2.

### P0 code fixes
1. Added `brain_ai/eval/arc_scorer.py`: an official-equivalent scorer with unit tests against reference cases.
2. Added `brain_ai/eval/leakage.py` and `legacy.py` to lock semantics and formalize the prior null result.
3. Created `scripts/trm_colab_runner.py` adding resumable checkpointing, dynamic batch sizing, and decoupled evaluation to the official TRM codebase. *Deviation*: TRM's ARC evaluator defaults to `aggregated_voting=True`, pooling votes across eval epochs; our runner sets `aggregated_voting=False` to ensure reproducing under resume is idempotent.
4. Created `notebooks/04_integrity_and_trm_baseline_colab.ipynb` to formalize the B0 floors and TRM B1 baselines without modifying the official repository.

---

## 9. Preprint Correction Checklist

1. Retitle and rewrite the abstract as an architecture description with pilot diagnostics. Remove all exact-match, ARC-AGI-2 and significance claims until P0–P2 results exist.
2. Relabel every accuracy as "per-cell accuracy, oracle output shape", or replace it with official exact-match scores once re-scored.
3. Retract the Phase 1 cascaded-router result (M7) and the raw-LLM significance tests (M8).
4. Rename "ARC-AGI-2 (N=400)" to "ARC-AGI-1 public evaluation (N=400)".
5. Remove the CoordConv latency attribution (M9) and disclose that the S1 path excludes the LH.
6. Narrow Theorem 2.1, Lemma 2.2 and Proposition 2.1 as listed in §2.2, items 10–12.
7. Fix the conflict-metric inconsistency and the citation errors (§2.3). Add the missing related work (§3.4).
8. Update [`ARCHITECTURE_SPEC.md`](ARCHITECTURE_SPEC.md): the Open-Jev router is an independent MLP, and the scaled HRM has about 10.5M parameters.
9. Mark [`IMPLEMENTATION_PLAN_TO_100.md`](IMPLEMENTATION_PLAN_TO_100.md) as superseded. Its projected accuracy figures have no empirical basis.
10. Restate the robotics report as described in §2.4.

---

## 10. Risks

| Risk | Likelihood | Mitigation |
| :--- | :--- | :--- |
| The LH adds nothing beyond a learned encoder (H1 false) | Moderate to high | Control B3 makes this a clean negative result, which is still publishable; RQ2 continues regardless. |
| Reduced-scale baselines are far below published numbers | High | Disclose; restrict claims to within-study comparisons. |
| Compute shortfall or accelerator unavailability | Moderate | Cached latents; L4 fallback; shrink to Sudoku/Maze plus an ARC subset. |
| Effects are smaller than sampling error at N = 120 | High | Treat ARC-AGI-1 eval (N = 400) as the primary report; use 3 seeds and paired tests. |
| Puzzle-embedding protocol confuses readers about leakage | Moderate | Explicit disclosure (§5 item 5); report B1 separately from unseen-task conditions. |

---

## 11. References

Verified ✅ on 2026-10-10 unless marked.

1. F. Chollet, "On the Measure of Intelligence," arXiv:1911.01547, 2019.
2. ARC Prize Foundation, ARC-AGI-1/2/3 benchmark pages, leaderboard and competition pages: https://arcprize.org (accessed 2026-10-10).
3. ARC Prize Foundation, official scorer: https://github.com/arcprize/arc-agi-benchmarking.
4. ARC Prize 2025 technical report, arXiv:2601.10904, 2026.
5. ARC Prize Foundation, "The Hidden Drivers of HRM's Performance on ARC-AGI," https://arcprize.org/blog/hrm-analysis, 15 Aug 2025.
6. G. Wang, J. Li, Y. Sun, X. Chen, C. Liu, Y. Wu, M. Lu, S. Song, Y. Abbasi Yadkori, "Hierarchical Reasoning Model," arXiv:2506.21734, 2025.
7. A. Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks," arXiv:2510.04871, 2025.
8. "TRM on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute," arXiv:2512.11847 (findings ⚠️).
9. "What Limits Recursive Reasoning Models: Optimization, Architecture and Test-Time Scaling," arXiv:2609.39967, 2026.
10. I. Liao, A. Gu, "ARC-AGI Without Pretraining," arXiv:2512.06104.
11. E. Akyürek et al., "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning," arXiv:2411.07279.
12. D. Franzen, J. Disselhoff, D. Hartmann, "Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective," arXiv:2505.07859, 2025.
13. H. Jeong, "Inhibitory Cross-Talk Enables Functional Lateralization in Attention-Coupled Latent Memory," arXiv:2603.03355, 2026.
14. J. Cornford, D. Kalajdzievski, M. Leite, A. Lamarquette, D. M. Kullmann, B. Richards, "Learning to Live with Dale's Principle: ANNs with Separate Excitatory and Inhibitory Units," ICLR 2021 ⚠️ (venue ID).
15. P. Li, J. Cornford, A. Ghosh, B. Richards, "Learning Better with Dale's Law: A Spectral Perspective," NeurIPS 2023 ⚠️ (venue).
16. K. Rajan, L. F. Abbott, "Eigenvalue Spectra of Random Matrices for Neural Networks," *Phys. Rev. Lett.* 97, 188104, 2006.
17. H. F. Song, G. R. Yang, X.-J. Wang, "Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework," *PLOS Comput. Biol.* 12(2): e1004792, 2016.
18. T. Ye, L. Dong, Y. Xia, Y. Sun, Y. Zhu, G. Huang, F. Wei, "Differential Transformer," ICLR 2025, arXiv:2410.05258.
19. S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, Y. Tian, "Training Large Language Models to Reason in a Continuous Latent Space," arXiv:2412.06769.
20. J. Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach," arXiv:2502.05171, 2025.
21. A. Gladstone et al., "Energy-Based Transformers are Scalable Learners and Thinkers," arXiv:2507.02092, 2025.
22. Z. Cai, *Open-Jev* (software), https://github.com/Zefan-Cai/Open-Jev; https://huggingface.co/ZefanCai/Open-Jev-2B, 2026.
23. E. Miller, "Adding Error Bars to Evals," arXiv:2411.00640, 2024.
24. T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," *Neural Computation* 10(7), 1998 ⚠️.
25. Google Colab FAQ, https://research.google.com/colaboratory/faq.html (accessed 2026-10-10).
