# End-to-End Research Plan: Bi-Hemispheric Neuromorphic AI

## 1. Research Objectives & Hypotheses

1. **Hypothesis 1 (Sample Efficiency & Spatial Generalization)**:
   A lateralized architecture coupling a continuous recurrent spatial engine (HRM) to a linguistic interpreter (Qwen) solves ARC-AGI and complex constraint satisfaction (Sudoku, graph pathfinding) with 10x fewer training demonstrations than a monolithic LLM.
2. **Hypothesis 2 (Hallucination Suppression via Callosal Inhibition)**:
   Net inhibitory cross-talk ($s = -1.0$) combined with Amygdala-driven gain escalation ($\gamma_I \uparrow$ under high uncertainty $U \to 1$) physically suppresses token hallucination and confabulation under epistemic ambiguity.
3. **Hypothesis 3 (Inference Latency Advantage)**:
   Non-autoregressive candidate scoring (Open-Jev) and continuous latent recurrent cycles achieve Pareto-dominant accuracy vs. wall-clock latency compared to multi-thousand-token Chain-of-Thought decoding loops.

---

## 2. Benchmark Suite

| Benchmark | Domain | Primary Metrics | Purpose |
| :--- | :--- | :--- | :--- |
| **ARC-AGI-1 / ARC-AGI-2** | Abstraction & Reasoning | Pass@1, Pass@2 Exact Match (%) | Spatial and topological generalization |
| **Sudoku-Extreme** | Constraint Satisfaction | Exact Completion Accuracy (%) | Error propagation and back-tracking |
| **CLRS-30 / NLGraph** | Algorithmic Graph Theory | Valid Path Rate, Violation Count | Relational graph reasoning |
| **SimpleQA & TruthfulQA** | Factual Precision | ECE, Brier Score, Refusal Rate | Calibration under epistemic ambiguity |
| **HaluEval** | Hallucination Probing | AUROC, AURAC | Verification of callosal conflict gating |
| **Pareto Scaling Analysis** | Inference Efficiency | TTFT, Latency (s), GFLOPs / task | Latent recurrence vs. CoT token scaling |

## 2. Model Progression Strategy

```
Phase 1: Fast Prototyping & Dynamic Stability (Current)
  Left Hemisphere:  Llama 3.1 8B (BF16, Frozen Base + LoRA r=64)
  Right Hemisphere: Sapient HRM 27M (Recurrent spatial grid engine)
  Amygdala:         Open-Jev 2B (Sub-20ms scalar affective router)
  Primary Focus:    Rapid iteration, callosal E-I balance, Sudoku + ARC-AGI-1

Phase 2: Scale-Up & Full Symbolic-Linguistic Integration
  Left Hemisphere:  Qwen 2.5 14B / Qwen 3 14B (BF16, LoRA r=64)
  Right Hemisphere: Sapient HRM-Text-1B (PrefixLM text-token reasoning)
  Amygdala:         Open-Jev 9B (Nuanced policy & risk router)
  Primary Focus:    Full ARC Prize, ConceptARC, SimpleQA, and Hallucination suppression
```

---

## 3. Benchmark Suite & Evaluation Protocol

| Benchmark | Domain | Primary Metrics | Purpose |
| :--- | :--- | :--- | :--- |
| **ARC-AGI-1** | Abstraction & Reasoning | Pass@1, Pass@2 Exact Match (%) | Gold standard for inductive spatial generalization |
| **ConceptARC** | Diagnostic Spatial Primitives | 16 Concept Sub-scores (%) | Isolates specific spatial capabilities (Topology, Objectness, Center) |
| **Sudoku-Extreme / Mazes** | Algorithmic Constraint Satisfaction | Exact Completion Accuracy (%) | Programmatic control task to verify non-divergence without CoT |
| **SimpleQA & TruthfulQA** | Factual Calibration (Phase 2) | ECE, Brier Score, Refusal Rate | Measures callosal epistemic uncertainty braking |
| **HaluEval** | Hallucination Probing (Phase 2) | AUROC, AURAC | Verification of callosal inhibitory conflict suppression |
| **Pareto Scaling Analysis** | Inference Efficiency | TTFT, Latency (s), GFLOPs / task | Latent recurrence vs. token-burning Chain-of-Thought |

---

## 4. Baseline Suite & Systematic Ablation Matrix

```
Baselines:
- B1: Standalone LLM (Llama 3.1 8B / Qwen 14B) - Direct Prompting
- B2: Standalone LLM + Chain-of-Thought (1024 tokens)
- B3: Standalone HRM (27M)
- B4: Unconstrained Dense MLP Bridge (No Dale, No E-I)
- B5: Standard Dense Bidirectional Cross-Attention (Softmax only)

Ablations:
- A1: Unidirectional Left -> Right only
- A2: Unidirectional Right -> Left only
- A3: Pure Excitatory Callosum (W_E >= 0, W_I = 0)
- A4: Pure Inhibitory Callosum (W_I >= 0, W_E = 0)
- A5: Bi-Hemispheric System without Amygdala Gating
- Full System: Llama/Qwen + Sapient HRM + Dale E-I Callosum + Amygdala Router
```

---

## 5. Execution Roadmap & Milestones

### Phase 0: Scaffolding & Setup (Days 1–2)
- [x] Cloned repository `sneed-and-feed/brain-ai`.
- [x] Initialized Python package structure and `pyproject.toml`.
- [x] Implemented core modules: `callosum.py`, `amygdala.py`, `hrm.py`, `ensemble.py`.
- [x] Created unit tests in `tests/test_callosum.py`.

### Phase 1: Callosal E-I Stability Verification (Days 3–5)
- [ ] Run synthetic stress testing script (`verify_stability.py`) over 500 recurrent steps.
- [ ] Confirm absence of runaway explosion ($\|h\| \to \infty$) and coma collapse ($\|h\| \to 0$).
- [ ] Verify Rajan-Abbott random matrix eigenvalue spectrum and singular value density.
- [ ] Tune homeostatic loss weighting ($\lambda_{\text{homeo}}, \lambda_{\text{flux}}$).

### Phase 2: Amygdalar Gating & Reflex Bypass Calibration (Days 6–8)
- [ ] Hook Open-Jev backbone with 3D continuous affective heads ($V, U, \Omega$).
- [ ] Calibrate gain curves ($\gamma_E, \gamma_I$) under simulated adversarial inputs.
- [ ] Measure System 1 reflex bypass latency on Colab Pro+ A100 (<25ms target).

### Phase 3: Benchmark Runs & Publication Baseline (Days 9–14)
- [ ] Evaluate Standalone HRM and Qwen 14B on ARC-AGI-1 and Sudoku-Extreme.
- [ ] Train callosal bridge using staged curriculum (Stage 1: Bridge pretraining; Stage 2: Joint LoRA finetuning).
- [ ] Evaluate full system against Baselines B1–B5 and Ablations A1–A5.
- [ ] Generate latency-accuracy Pareto plots and prepare technical report.
