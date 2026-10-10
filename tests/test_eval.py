"""
Tests for brain_ai.eval: official-equivalent ARC scoring, floor baselines, paired statistics,
leakage checks (including TRM's sequence encoding) and the legacy re-scorer.
"""

import json
import os
import zipfile

import numpy as np
import pytest

from brain_ai.eval.arc_scorer import (
    FLOOR_BASELINES,
    floor_submission,
    grids_equal,
    score_floors,
    score_submission,
    score_task,
)
from brain_ai.eval.leakage import (
    _decode_trm_arc_seq,
    assert_no_leakage,
    raw_arc_overlap,
)
from brain_ai.eval.legacy import rescore_legacy_progress
from brain_ai.eval.stats import holm, mcnemar_exact, paired_bootstrap, wilson_interval

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Scoring semantics
# ---------------------------------------------------------------------------

def test_grids_equal_requires_identical_dimensions():
    assert grids_equal([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    assert not grids_equal([[1, 2]], [[1, 2], [0, 0]])          # different height
    assert not grids_equal([[1, 2, 0]], [[1, 2]])               # different width
    assert not grids_equal(None, [[1]])
    assert not grids_equal([[1], [2, 3]], [[1], [2, 3]])        # ragged attempt is invalid


def test_score_task_two_attempts_and_fractional_credit():
    sols = [[[1]], [[2]]]
    assert score_task([{"attempt_1": [[0]], "attempt_2": [[1]]}, {"attempt_1": [[2]], "attempt_2": [[0]]}], sols) == 1.0
    assert score_task([{"attempt_1": [[1]], "attempt_2": [[1]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}], sols) == 0.5
    assert score_task(None, sols) == 0.0
    assert score_task([{"attempt_1": [[1]]}], sols) == 0.5     # missing second test pair scores 0


def test_score_submission_missing_tasks_count_as_zero():
    sols = {"a": [[[1]]], "b": [[[2]]]}
    rep = score_submission({"a": [{"attempt_1": [[1]], "attempt_2": [[9]]}]}, sols)
    assert rep.score == pytest.approx(0.5)
    assert rep.n_pairs_solved == 1 and rep.n_test_pairs == 2 and rep.tasks_fully_solved == 1


def test_floors_never_use_output_shape():
    task = {"train": [{"input": [[1, 2], [3, 4]], "output": [[5]]}], "test": [{"input": [[1, 2], [3, 4]]}]}
    for name in FLOOR_BASELINES:
        pred = floor_submission({"t": task}, name)["t"][0]["attempt_1"]
        assert np.asarray(pred).shape == (2, 2), name   # input shape, not the 1x1 output shape


def test_best_d4_floor_selects_correct_primitive():
    g = [[1, 2], [3, 4]]
    task = {"train": [{"input": g, "output": np.rot90(np.array(g)).tolist()}], "test": [{"input": [[5, 6], [7, 8]]}]}
    pred = floor_submission({"t": task}, "best_d4_primitive_by_demo_fit")["t"][0]["attempt_1"]
    assert pred == np.rot90(np.array([[5, 6], [7, 8]])).tolist()


def _load_arc1_eval_from_repo_zip():
    path = os.path.join(REPO, "data", "arc", "arc_master.zip")
    if not os.path.exists(path):
        pytest.skip("data/arc/arc_master.zip not available")
    challenges, solutions = {}, {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if "/data/evaluation/" in n and n.endswith(".json"):
                t = json.loads(z.read(n))
                tid = os.path.basename(n)[:-5]
                challenges[tid] = {"train": t["train"], "test": [{"input": p["input"]} for p in t["test"]]}
                solutions[tid] = [p["output"] for p in t["test"]]
    return challenges, solutions


def test_floor_scores_on_arc_agi_1_eval_are_near_zero():
    challenges, solutions = _load_arc1_eval_from_repo_zip()
    assert len(challenges) == 400
    reports = score_floors(challenges, solutions)
    for name, rep in reports.items():
        assert rep.n_tasks == 400
        assert rep.score < 0.05, f"{name}: {rep.summary()}"
    # The copy-input floor solves no task in ARC-AGI-1 eval except by coincidence.
    assert reports["copy_test_input"].n_pairs_solved <= 2


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_mcnemar_exact_known_values():
    a = [True] * 10 + [False] * 10
    b = [False] * 10 + [False] * 10
    r = mcnemar_exact(a, b)
    assert r["only_a"] == 10 and r["only_b"] == 0
    assert r["p_value"] == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(0)
    base = rng.random(200)
    shifted = paired_bootstrap(base + 0.1, base, n_boot=2000, seed=1)
    assert shifted["ci_low"] > 0.0 and shifted["mean_diff"] == pytest.approx(0.1)
    null = paired_bootstrap(base, base, n_boot=2000, seed=1)
    assert null["ci_low"] <= 0.0 <= null["ci_high"]


def test_wilson_and_holm():
    lo, hi = wilson_interval(24, 120)
    assert 0.13 < lo < 0.2 < hi < 0.29
    adj = holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.04) and adj["c"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def _trm_encode(grid, pad_r, pad_c, side=30):
    """Re-implementation of TRM's np_grid_to_seq_translational_augment for a single grid."""
    g = np.asarray(grid, dtype=np.uint8)
    nrow, ncol = g.shape
    out = np.pad(g + 2, ((pad_r, side - pad_r - nrow), (pad_c, side - pad_c - ncol)), constant_values=0)
    if pad_r + nrow < side:
        out[pad_r + nrow, pad_c:pad_c + ncol] = 1
    if pad_c + ncol < side:
        out[pad_r:pad_r + nrow, pad_c + ncol] = 1
    return out.flatten()


@pytest.mark.parametrize("pad", [(0, 0), (3, 7), (25, 26)])
def test_decode_trm_sequence_roundtrip(pad):
    g = np.array([[0, 1, 2], [3, 4, 9]], dtype=np.uint8)
    dec = _decode_trm_arc_seq(_trm_encode(g, *pad))
    assert dec is not None and np.array_equal(dec, g)


def test_raw_overlap_flags_leaked_pair_and_duplicate_task():
    shared = {"input": [[1, 2]], "output": [[2, 1]]}
    eval_chal = {"e1": {"train": [{"input": [[3]], "output": [[4]]}], "test": [{"input": shared["input"]}]},
                 "e2": {"train": [{"input": [[5]], "output": [[6]]}], "test": [{"input": [[7]]}]}}
    eval_sol = {"e1": [shared["output"]], "e2": [[[8]]]}
    training = {"train": ({"t1": {"train": [shared], "test": []},
                           "t2": {"train": [{"input": [[5]], "output": [[6]]}], "test": []}}, None)}
    rep = raw_arc_overlap(eval_chal, eval_sol, training)
    assert rep.leaked_test_pairs == [("e1", 0)]
    assert ("e2", "train/t2") in rep.duplicate_tasks
    with pytest.raises(RuntimeError):
        assert_no_leakage(rep, "unit test")


# ---------------------------------------------------------------------------
# Legacy artifact
# ---------------------------------------------------------------------------

def test_legacy_rescore_matches_audit():
    path = os.path.join(REPO, "data", "eval400_progress.json")
    if not os.path.exists(path):
        pytest.skip("legacy artifact not present")
    r = rescore_legacy_progress(path)
    assert r["n_tasks"] == 400
    assert r["match_s1"]["solved"] == 1
    assert r["match_pass2"]["solved"] == 2
    assert r["reported_pix_pass2_mean"] == pytest.approx(0.6767, abs=1e-3)
