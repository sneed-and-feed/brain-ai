"""
Official-equivalent ARC-AGI scoring and non-learned floor baselines.

Scoring semantics (mirrors arcprize/arc-agi-benchmarking, src/arc_agi_benchmarking/scoring/scoring.py):
- Each test pair of a task receives up to two attempts ("attempt_1", "attempt_2").
- A test pair is solved iff at least one attempt equals the ground-truth output grid exactly,
  including its dimensions.
- Task score = solved test pairs / number of test pairs (fractional for multi-output tasks).
- Benchmark score = mean task score over all tasks in the solutions file.
  Tasks missing from the submission score 0.

Floor baselines never receive the ground-truth output shape. They are the minimum bar that any
learned system must clear (RESEARCH_PLAN.md, baseline B0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Grid = List[List[int]]
Submission = Dict[str, List[Dict[str, Optional[Grid]]]]
Solutions = Dict[str, List[Grid]]
Challenges = Dict[str, dict]

ATTEMPT_KEYS = ("attempt_1", "attempt_2")


# ---------------------------------------------------------------------------
# I/O helpers (Kaggle-format files: <prefix>_<subset>_challenges.json / _solutions.json)
# ---------------------------------------------------------------------------

def load_kaggle_split(prefix: str, subset: str) -> tuple[Challenges, Solutions]:
    """Load ARC challenges and solutions in Kaggle format, e.g. prefix='kaggle/combined/arc-agi'."""
    with open(f"{prefix}_{subset}_challenges.json", "r") as f:
        challenges = json.load(f)
    with open(f"{prefix}_{subset}_solutions.json", "r") as f:
        solutions = json.load(f)
    missing = set(challenges) ^ set(solutions)
    if missing:
        raise ValueError(f"Challenge/solution task IDs differ for subset '{subset}': {sorted(missing)[:5]} ...")
    for tid, sols in solutions.items():
        if len(sols) != len(challenges[tid]["test"]):
            raise ValueError(f"Task {tid}: {len(sols)} solutions for {len(challenges[tid]['test'])} test inputs")
    return challenges, solutions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _is_valid_grid(g) -> bool:
    if not isinstance(g, list) or len(g) == 0:
        return False
    if not all(isinstance(row, list) and len(row) > 0 for row in g):
        return False
    width = len(g[0])
    return all(len(row) == width for row in g)


def grids_equal(a, b) -> bool:
    """Whole-grid exact match including dimensions. Malformed attempts never match."""
    if a is None or b is None or not _is_valid_grid(a) or not _is_valid_grid(b):
        return False
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        return False
    return all(int(x) == int(y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def score_task(task_attempts: Optional[Sequence[Dict[str, Optional[Grid]]]], task_solutions: Sequence[Grid]) -> float:
    """Fraction of test pairs solved by at least one of the (up to two) attempts."""
    if not task_solutions:
        raise ValueError("Task has no test outputs")
    solved = 0
    for idx, truth in enumerate(task_solutions):
        attempts = task_attempts[idx] if task_attempts is not None and idx < len(task_attempts) else None
        if attempts is None:
            continue
        if any(grids_equal(attempts.get(k), truth) for k in ATTEMPT_KEYS):
            solved += 1
    return solved / len(task_solutions)


@dataclass
class ScoreReport:
    score: float                      # mean task score in [0, 1]
    n_tasks: int
    n_test_pairs: int
    n_pairs_solved: int
    per_task: Dict[str, float] = field(default_factory=dict)

    @property
    def tasks_fully_solved(self) -> int:
        return sum(1 for v in self.per_task.values() if v == 1.0)

    def summary(self) -> str:
        return (f"score = {100 * self.score:.2f}%  ({self.n_pairs_solved}/{self.n_test_pairs} test pairs; "
                f"{self.tasks_fully_solved}/{self.n_tasks} tasks fully solved)")


def score_submission(submission: Submission, solutions: Solutions) -> ScoreReport:
    """Score a Kaggle-format submission against solutions. Unknown task IDs in the submission are ignored."""
    per_task: Dict[str, float] = {}
    n_pairs = 0
    n_solved = 0
    for tid, sols in solutions.items():
        s = score_task(submission.get(tid), sols)
        per_task[tid] = s
        n_pairs += len(sols)
        n_solved += round(s * len(sols))
    score = float(np.mean(list(per_task.values()))) if per_task else 0.0
    return ScoreReport(score=score, n_tasks=len(per_task), n_test_pairs=n_pairs, n_pairs_solved=n_solved, per_task=per_task)


# ---------------------------------------------------------------------------
# Non-learned floor baselines (no ground-truth output shape)
# ---------------------------------------------------------------------------

def _np(g) -> np.ndarray:
    return np.asarray(g, dtype=np.int64)


D4_PRIMITIVES: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1),
    "rot180": lambda g: np.rot90(g, 2),
    "rot270": lambda g: np.rot90(g, 3),
    "flip_lr": np.fliplr,
    "flip_ud": np.flipud,
    "transpose": lambda g: g.T,
    "anti_transpose": lambda g: np.rot90(g, 2).T,
}


def _demo_fit(fn: Callable[[np.ndarray], np.ndarray], demos: Sequence[dict]) -> tuple[int, float]:
    """(number of demos matched exactly, mean pixel agreement on shape-compatible demos)."""
    exact, pix = 0, []
    for d in demos:
        pred, out = fn(_np(d["input"])), _np(d["output"])
        if pred.shape == out.shape:
            pix.append(float((pred == out).mean()))
            exact += int(np.array_equal(pred, out))
        else:
            pix.append(0.0)
    return exact, float(np.mean(pix)) if pix else 0.0


def _floor_zeros(task: dict, test_input: np.ndarray) -> np.ndarray:
    return np.zeros_like(test_input)


def _floor_copy_input(task: dict, test_input: np.ndarray) -> np.ndarray:
    return test_input.copy()


def _floor_mode_colour(task: dict, test_input: np.ndarray) -> np.ndarray:
    colours = np.concatenate([_np(d["output"]).ravel() for d in task["train"]])
    return np.full_like(test_input, int(np.bincount(colours, minlength=10).argmax()))


def _floor_best_d4(task: dict, test_input: np.ndarray) -> np.ndarray:
    best_name = max(D4_PRIMITIVES, key=lambda k: _demo_fit(D4_PRIMITIVES[k], task["train"]))
    return D4_PRIMITIVES[best_name](test_input).copy()


FLOOR_BASELINES: Dict[str, Callable[[dict, np.ndarray], np.ndarray]] = {
    "zeros_at_input_shape": _floor_zeros,
    "copy_test_input": _floor_copy_input,
    "mode_colour_at_input_shape": _floor_mode_colour,
    "best_d4_primitive_by_demo_fit": _floor_best_d4,
}


def floor_submission(challenges: Challenges, floor: str) -> Submission:
    """Build a single-candidate submission (attempt_2 = attempt_1) for a named floor baseline."""
    fn = FLOOR_BASELINES[floor]
    sub: Submission = {}
    for tid, task in challenges.items():
        sub[tid] = []
        for pair in task["test"]:
            pred = fn(task, _np(pair["input"])).tolist()
            sub[tid].append({"attempt_1": pred, "attempt_2": pred})
    return sub


def score_floors(challenges: Challenges, solutions: Solutions) -> Dict[str, ScoreReport]:
    return {name: score_submission(floor_submission(challenges, name), solutions) for name in FLOOR_BASELINES}
