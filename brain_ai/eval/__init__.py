"""
brain_ai.eval: evaluation utilities that follow the official ARC Prize protocol.

Modules:
- arc_scorer: official-equivalent ARC scoring (per test pair, 2 attempts, whole-grid exact match),
  plus non-learned floor baselines that never see the ground-truth output shape.
- stats: paired significance tests and confidence intervals for small-N benchmark comparisons.
- leakage: split-hygiene checks between training data and evaluation test pairs.
- legacy: re-scoring of artifacts produced by the pre-audit (notebook 01-03) pipeline.
"""
