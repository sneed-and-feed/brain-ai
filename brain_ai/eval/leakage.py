"""
Split-hygiene (leakage) checks.

Two levels:
1. Raw-file level (`raw_arc_overlap`): before any dataset is built, verify that no evaluation test pair
   (input, output) appears anywhere in the training subsets, and report evaluation tasks that are exact
   duplicates of training tasks. Known positive control: ARC-AGI-2 'training2' contains ARC-AGI-1
   evaluation tasks (TRM README), so this check must flag that combination.
2. Built-dataset level (`trm_built_dataset_overlap`): after TRM's dataset builder has run, decode the
   un-augmented training examples and verify that no evaluation test pair is present in the train split.

Both functions return reports; `assert_no_leakage` raises on any test-pair overlap.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def grid_hash(grid) -> str:
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError("grid must be 2-D")
    return hashlib.sha256(bytes(arr.shape) + arr.tobytes()).hexdigest()


def pair_hash(inp, out) -> str:
    return grid_hash(inp) + ":" + grid_hash(out)


@dataclass
class LeakageReport:
    eval_test_pairs: int
    leaked_test_pairs: List[Tuple[str, int]] = field(default_factory=list)   # (eval task id, test index)
    leaked_test_inputs: List[Tuple[str, int]] = field(default_factory=list)  # input seen in training (any role)
    duplicate_tasks: List[Tuple[str, str]] = field(default_factory=list)     # (eval task id, training source)

    @property
    def clean(self) -> bool:
        return not self.leaked_test_pairs

    def summary(self) -> str:
        return (f"{self.eval_test_pairs} eval test pairs | leaked (input,output) pairs: {len(self.leaked_test_pairs)} | "
                f"test inputs seen in training: {len(self.leaked_test_inputs)} | "
                f"eval tasks duplicated in training: {len(self.duplicate_tasks)}")


def _task_signature(task: dict) -> str:
    hashes = sorted(pair_hash(p["input"], p["output"]) for p in task["train"])
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def raw_arc_overlap(eval_challenges: Dict[str, dict], eval_solutions: Dict[str, list],
                    training_sets: Dict[str, Tuple[Dict[str, dict], Optional[Dict[str, list]]]]) -> LeakageReport:
    """
    eval_*: the evaluation split (Kaggle format). training_sets: name -> (challenges, solutions or None).
    Every pair of every training task (demonstrations and, where solutions exist, test pairs) is indexed.
    """
    seen_pairs: Dict[str, str] = {}
    seen_inputs: Dict[str, str] = {}
    signatures: Dict[str, str] = {}
    for name, (chal, sols) in training_sets.items():
        for tid, task in chal.items():
            signatures.setdefault(_task_signature(task), f"{name}/{tid}")
            pairs = list(task["train"])
            if sols is not None and tid in sols:
                pairs += [{"input": t["input"], "output": o} for t, o in zip(task["test"], sols[tid])]
            for p in pairs:
                seen_pairs.setdefault(pair_hash(p["input"], p["output"]), f"{name}/{tid}")
                seen_inputs.setdefault(grid_hash(p["input"]), f"{name}/{tid}")

    report = LeakageReport(eval_test_pairs=sum(len(v) for v in eval_solutions.values()))
    for tid, task in eval_challenges.items():
        src = signatures.get(_task_signature(task))
        if src is not None:
            report.duplicate_tasks.append((tid, src))
        for i, (t, out) in enumerate(zip(task["test"], eval_solutions[tid])):
            if pair_hash(t["input"], out) in seen_pairs:
                report.leaked_test_pairs.append((tid, i))
            if grid_hash(t["input"]) in seen_inputs:
                report.leaked_test_inputs.append((tid, i))
    return report


def _decode_trm_arc_seq(seq: np.ndarray, side: int = 30) -> Optional[np.ndarray]:
    """Invert TRM's ARC encoding (PAD=0, EOS=1, colours 2..11, optional translation): crop to colour cells."""
    grid = np.asarray(seq).reshape(side, side)
    mask = grid >= 2
    if not mask.any():
        return None
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return (grid[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] - 2).astype(np.uint8)


def trm_built_dataset_overlap(data_dir: str, set_name: str = "all") -> LeakageReport:
    """
    Check a dataset produced by TRM's dataset/build_arc_dataset.py.

    Uses test_puzzles.json (the evaluation tasks with test outputs) and the train split. Within each group,
    the first puzzle is the un-augmented original, so only those examples are decoded; augmented copies are
    deterministic transforms of the same examples.
    """
    with open(os.path.join(data_dir, "test_puzzles.json")) as f:
        test_puzzles = json.load(f)

    split = os.path.join(data_dir, "train")
    inputs = np.load(os.path.join(split, f"{set_name}__inputs.npy"), mmap_mode="r")
    labels = np.load(os.path.join(split, f"{set_name}__labels.npy"), mmap_mode="r")
    puzzle_indices = np.load(os.path.join(split, f"{set_name}__puzzle_indices.npy"))
    group_indices = np.load(os.path.join(split, f"{set_name}__group_indices.npy"))

    seen_pairs, seen_inputs = set(), set()
    for g in range(len(group_indices) - 1):
        p = int(group_indices[g])
        for ex in range(int(puzzle_indices[p]), int(puzzle_indices[p + 1])):
            inp, out = _decode_trm_arc_seq(inputs[ex]), _decode_trm_arc_seq(labels[ex])
            if inp is None or out is None:
                continue
            seen_pairs.add(pair_hash(inp, out))
            seen_inputs.add(grid_hash(inp))

    report = LeakageReport(eval_test_pairs=sum(len(t["test"]) for t in test_puzzles.values()))
    for tid, task in test_puzzles.items():
        for i, pair in enumerate(task["test"]):
            if pair_hash(pair["input"], pair["output"]) in seen_pairs:
                report.leaked_test_pairs.append((tid, i))
            if grid_hash(pair["input"]) in seen_inputs:
                report.leaked_test_inputs.append((tid, i))
    return report


def trm_sudoku_overlap(data_dir: str, set_name: str = "all") -> LeakageReport:
    """Check that no Sudoku test puzzle (input board) appears among un-augmented training puzzles."""
    def originals(split: str) -> Iterable[bytes]:
        d = os.path.join(data_dir, split)
        inputs = np.load(os.path.join(d, f"{set_name}__inputs.npy"), mmap_mode="r")
        puzzle_indices = np.load(os.path.join(d, f"{set_name}__puzzle_indices.npy"))
        group_indices = np.load(os.path.join(d, f"{set_name}__group_indices.npy"))
        for g in range(len(group_indices) - 1):
            p = int(group_indices[g])
            yield np.asarray(inputs[int(puzzle_indices[p])]).tobytes()

    train = {hashlib.sha256(b).hexdigest() for b in originals("train")}
    report = LeakageReport(eval_test_pairs=0)
    for i, b in enumerate(originals("test")):
        report.eval_test_pairs += 1
        if hashlib.sha256(b).hexdigest() in train:
            report.leaked_test_pairs.append(("sudoku_test", i))
    return report


# Known benchmark-intrinsic duplicate between official ARC-AGI-1 evaluation and training sets
# (evaluation task 070dd51e is byte-identical to training task 40853293 in the original ARC release).
KNOWN_ARC_DUPLICATES = {("070dd51e", 0)}


def assert_no_leakage(report: LeakageReport, context: str, allowed: set = KNOWN_ARC_DUPLICATES) -> None:
    actual_leaks = [p for p in report.leaked_test_pairs if p not in allowed]
    if actual_leaks:
        raise RuntimeError(f"[LEAKAGE] {context}: {len(actual_leaks)} evaluation test pairs found in "
                           f"training data, e.g. {actual_leaks[:5]}. Aborting run.")
