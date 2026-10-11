#!/usr/bin/env python
"""
scripts/system2_tta_evaluator.py

Evaluator for System 2 Test-Time Latent Adaptation (TTA) with Continuous Latent Relaxation (delta_z).
Academic implementation for Phase P3 / RQ3 (RESEARCH_PLAN.md §4.2, ARCHITECTURE_SPEC.md §3).

Workflow:
1. Takes the Callosal conditioning state Z_0 = Callosum(Z_L) for each evaluation task.
2. Freezes the underlying model weights.
3. Optimizes continuous latent offset delta_z in R^(num_query_tokens x d_reasoner)
   for N_TTA = 30 Adam steps strictly on task demonstration input/output pairs (L_CE).
4. Proximal quadratic regularization (0.5 * lambda_anchor * ||delta_z||_2^2) to bound drift.
5. Implements Step 0 baseline comparison: if post-adaptation demonstration loss
   does not improve over Step 0, falls back to Z_0 (weak Pareto monotonicity).
6. Evaluates test challenge grids on held-out test inputs (test outputs are NEVER accessed).
7. Outputs official Kaggle submission and scores against ground truth with Wilson 95% CIs
   and paired McNemar tests against Baseline B1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Ensure brain-ai is on PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from brain_ai.models.callosum import (
    CallosalConditioning,
    CallosalPuzzleEmbedding,
    System2LatentAdaptor,
    encode_grid_to_trm_seq,
    decode_trm_seq_to_grid
)
from brain_ai.models.lh_cache import load_lh_cache
from brain_ai.eval.arc_scorer import (
    load_kaggle_split,
    score_submission,
    ScoreReport
)
from brain_ai.eval.stats import wilson_interval, mcnemar_exact


def run_tta_evaluation(
    model: nn.Module,
    challenges: Dict[str, dict],
    solutions: Optional[Dict[str, list]] = None,
    identifiers_map: Optional[Any] = None,
    n_steps: int = 30,
    lr: float = 0.01,
    lambda_anchor: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    b1_submission_path: Optional[str] = None,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes System 2 TTA across tasks in challenges and scores against solutions if provided.
    """
    adaptor = System2LatentAdaptor(
        n_steps=n_steps,
        lr=lr,
        lambda_anchor=lambda_anchor,
        device=device
    )

    print(f"\n[System 2 TTA] Beginning adaptation on {len(challenges)} tasks ({n_steps} Adam steps, lambda_anchor={lambda_anchor}, lr={lr})...")
    t0 = time.time()
    results = adaptor.adapt_and_evaluate_submission(
        model=model,
        challenges=challenges,
        identifiers_map=identifiers_map,
        n_steps=n_steps,
        lr=lr,
        lambda_anchor=lambda_anchor
    )
    duration = time.time() - t0

    sub_s2 = results["submission_s2"]
    sub_s1 = results["submission_s1"]
    diags = results["diagnostics"]

    accepted_tasks = [tid for tid, d in diags.items() if d.get("accepted", False)]
    acceptance_rate = len(accepted_tasks) / max(1, len(challenges))
    avg_delta_z_norm = float(sum(d.get("norm_delta_z", 0.0) for d in diags.values()) / max(1, len(diags)))
    avg_loss_reduction = float(sum(d.get("delta_loss", 0.0) for d in diags.values()) / max(1, len(diags)))

    print(f"[System 2 TTA] Adaptation complete in {duration:.1f}s ({duration/len(challenges):.3f}s/task).")
    print(f"  Accepted tasks: {len(accepted_tasks)}/{len(challenges)} ({100*acceptance_rate:.1f}%)")
    print(f"  Mean delta_z L2 norm: {avg_delta_z_norm:.4f}")
    print(f"  Mean demonstration loss reduction: {avg_loss_reduction:.4f}")

    eval_summary: Dict[str, Any] = {
        "n_tasks": len(challenges),
        "duration_seconds": duration,
        "n_steps": n_steps,
        "lr": lr,
        "lambda_anchor": lambda_anchor,
        "accepted_tasks_count": len(accepted_tasks),
        "acceptance_rate": acceptance_rate,
        "avg_delta_z_norm": avg_delta_z_norm,
        "avg_loss_reduction": avg_loss_reduction,
        "diagnostics": diags
    }

    if solutions is not None:
        print("\n=== SYSTEM 1 (ZERO-SHOT FEEDFORWARD) SCORING ===")
        rep_s1 = score_submission(sub_s1, solutions)
        lo_s1, hi_s1 = wilson_interval(rep_s1.tasks_fully_solved, rep_s1.n_tasks)
        print(rep_s1.summary())
        print(f"Exact match (fully solved): {rep_s1.tasks_fully_solved}/{rep_s1.n_tasks} [{100*lo_s1:.2f}%, {100*hi_s1:.2f}%] 95% Wilson CI")

        print("\n=== SYSTEM 2 (TEST-TIME LATENT ADAPTATION) SCORING ===")
        rep_s2 = score_submission(sub_s2, solutions)
        lo_s2, hi_s2 = wilson_interval(rep_s2.tasks_fully_solved, rep_s2.n_tasks)
        print(rep_s2.summary())
        print(f"Exact match (fully solved): {rep_s2.tasks_fully_solved}/{rep_s2.n_tasks} [{100*lo_s2:.2f}%, {100*hi_s2:.2f}%] 95% Wilson CI")

        eval_summary["system1"] = {
            "score": rep_s1.score,
            "tasks_fully_solved": rep_s1.tasks_fully_solved,
            "wilson_95_ci": [lo_s1, hi_s1]
        }
        eval_summary["system2"] = {
            "score": rep_s2.score,
            "tasks_fully_solved": rep_s2.tasks_fully_solved,
            "wilson_95_ci": [lo_s2, hi_s2]
        }

        # McNemar test vs Baseline B1 if B1 submission available
        if b1_submission_path and os.path.exists(b1_submission_path):
            with open(b1_submission_path) as f:
                b1_sub = json.load(f)
            rep_b1 = score_submission(b1_sub, solutions)
            task_ids = sorted(solutions.keys())
            s2_correct = [rep_s2.per_task.get(tid, 0.0) == 1.0 for tid in task_ids]
            b1_correct = [rep_b1.per_task.get(tid, 0.0) == 1.0 for tid in task_ids]
            mc_result = mcnemar_exact(s2_correct, b1_correct)
            print("\n=== HEAD-TO-HEAD McNEMAR TEST vs BASELINE B1 ===")
            print(f"Baseline B1 Solved: {rep_b1.tasks_fully_solved}/{rep_b1.n_tasks}")
            print(f"System 2 TTA Solved: {rep_s2.tasks_fully_solved}/{rep_s2.n_tasks}")
            print(f"S2 only: {mc_result['only_a']}, B1 only: {mc_result['only_b']}, two-sided p-value: {mc_result['p_value']:.4f}")
            eval_summary["mcnemar_vs_b1"] = mc_result

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "submission_s1_zero_shot.json"), "w") as f:
            json.dump(sub_s1, f)
        with open(os.path.join(out_dir, "submission_s2_adapted.json"), "w") as f:
            json.dump(sub_s2, f)
        with open(os.path.join(out_dir, "system2_tta_summary.json"), "w") as f:
            json.dump(eval_summary, f, indent=2)
        print(f"\nArtifacts saved to: {out_dir}")

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="System 2 TTA Evaluator on ARC")
    parser.add_argument("--challenges", required=True, help="Path to challenges JSON file")
    parser.add_argument("--solutions", default=None, help="Optional path to solutions JSON file")
    parser.add_argument("--lh-cache-path", required=True, help="Path to cached LH latents .pt")
    parser.add_argument("--weights-path", default=None, help="Path to model weights checkpoint")
    parser.add_argument("--b1-submission", default=None, help="Path to Baseline B1 submission for McNemar test")
    parser.add_argument("--out-dir", required=True, help="Directory to save submissions and summary")
    parser.add_argument("--n-steps", type=int, default=30, help="Number of TTA Adam steps")
    parser.add_argument("--lr", type=float, default=0.01, help="TTA Adam learning rate")
    parser.add_argument("--lambda-anchor", type=float, default=1.0, help="Proximal anchor weight")
    parser.add_argument("--identifiers-map", default=None, help="Path to identifiers.json")
    args = parser.parse_args()

    with open(args.challenges) as f:
        challenges = json.load(f)

    solutions = None
    if args.solutions and os.path.exists(args.solutions):
        with open(args.solutions) as f:
            solutions = json.load(f)

    identifiers = None
    if args.identifiers_map and os.path.exists(args.identifiers_map):
        with open(args.identifiers_map) as f:
            identifiers = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LH cache from {args.lh_cache_path}...")
    cached_latents, meta = load_lh_cache(args.lh_cache_path, device=device)

    # Wire minimal Callosal reasoning model
    d_lh = cached_latents.shape[-1]
    callosum_cond = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=512,
        d_callosum=512,
        num_query_tokens=16,
        n_heads=4,
        constraint_type="dale_inhibitory"
    ).to(device)

    p2b = meta.get("puzzle_to_base_id") if isinstance(meta, dict) else None
    callosal_emb = CallosalPuzzleEmbedding(
        cached_lh_latents=cached_latents,
        callosal_conditioning=callosum_cond,
        num_query_tokens=16,
        d_reasoner=512,
        flatten_output=True,
        puzzle_to_base_id=p2b
    ).to(device)

    # If full TRM model weights are supplied
    # In standalone mode, if model is not full TRM, test/evaluate with mock or loaded wrapper
    from brain_ai.models.hrm import HierarchicalReasoningModel
    class StandaloneBiHemisphericModel(nn.Module):
        def __init__(self, puzzle_emb):
            super().__init__()
            self.puzzle_emb = puzzle_emb
            self.token_emb = nn.Embedding(12, 512)
            self.head = nn.Linear(512, 12)

        def forward(self, batch):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_emb = self.puzzle_emb(pids).view(-1, 16, 512)
            x_emb = self.token_emb(inps)
            # Combine puzzle embedding with token embeddings
            h = x_emb + p_emb.mean(dim=1, keepdim=True)
            logits = self.head(h)
            return {"logits": logits}

    model = StandaloneBiHemisphericModel(callosal_emb).to(device)
    if args.weights_path and os.path.exists(args.weights_path):
        print(f"Loading weights from {args.weights_path}...")
        raw_sd = torch.load(args.weights_path, map_location=device, weights_only=False)
        sd = raw_sd.get("model", raw_sd) if isinstance(raw_sd, dict) else raw_sd
        model.load_state_dict(sd, strict=False)

    run_tta_evaluation(
        model=model,
        challenges=challenges,
        solutions=solutions,
        identifiers_map=identifiers,
        n_steps=args.n_steps,
        lr=args.lr,
        lambda_anchor=args.lambda_anchor,
        device=device,
        b1_submission_path=args.b1_submission,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
