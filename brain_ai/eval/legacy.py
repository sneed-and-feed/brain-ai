"""
Re-scoring of artifacts produced by the pre-audit pipeline (notebooks 01-03).

The legacy per-task records store per-cell ("pixel") accuracies computed with the ground-truth output
shape supplied, plus boolean exact-match flags for the first test pair only. This module reports the
exact-match counts, which are the only quantities comparable with official ARC scoring (and even these
are upper bounds, because the output shape was given to the model).
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np


def rescore_legacy_progress(path: str) -> Dict[str, object]:
    with open(path) as f:
        records = json.load(f)
    n = len(records)
    out: Dict[str, object] = {"n_tasks": n, "note": "first test pair only; ground-truth output shape supplied (upper bound)"}
    for key in ("match_s1", "match_pass1", "match_pass2"):
        if key in records[0]:
            k = int(sum(bool(r[key]) for r in records))
            out[key] = {"solved": k, "rate": k / n}
    for key in ("pix_s1", "pix_s2_hard", "pix_d4", "pix_pass1", "pix_pass2"):
        if key in records[0]:
            out[f"reported_{key}_mean"] = float(np.mean([r[key] for r in records]))
    out["solved_task_ids_pass2"] = [r["task_id"] for r in records if r.get("match_pass2")]
    return out
