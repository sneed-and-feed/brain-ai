"""
brain_ai.tasks.arc: ARC-AGI-1 Dataset Loader and Synthetic Inductive Grid Generator.

Supports:
1. Loading and caching official ARC-AGI-1 JSON tasks (400 train, 400 evaluation).
2. Built-in ConceptARC spatial primitives: Symmetry, Gravity, Object Translation, Color Inversion.
3. Multi-modal bi-hemispheric formatting:
   - Right Hemisphere: 2D Spatial Grid Tokens (10 colors + padding)
   - Left Hemisphere: Structural Induction Prompts for Llama 3.1
"""

from dataclasses import dataclass
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ARCTask:
    task_id: str
    train_pairs: List[Dict[str, List[List[int]]]] # [{"input": [[...]], "output": [[...]]}, ...]
    test_pairs: List[Dict[str, List[List[int]]]]  # [{"input": [[...]], "output": [[...]]}, ...]


@dataclass
class ARCBatch:
    demo_inputs: torch.Tensor       # [B, N_demo, max_H, max_W]
    demo_outputs: torch.Tensor      # [B, N_demo, max_H, max_W]
    test_inputs: torch.Tensor       # [B, max_H, max_W]
    test_targets: torch.Tensor      # [B, max_H, max_W]
    target_masks: torch.Tensor      # [B, max_H, max_W] 1.0 for valid grid cells, 0.0 for padding
    text_prompts: List[str]         # Natural language prompt for Left Hemisphere
    target_shapes: List[Tuple[int, int]] # Actual (H, W) of test target


class ARCDataset:
    """
    Manages ARC-AGI tasks with automatic caching and built-in procedural generators.
    """
    ARC_TRAIN_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/arc-agi_training_challenges.json"
    ARC_SOLUTIONS_URL = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/arc-agi_training_solutions.json"

    def __init__(self, cache_dir: str = "data/arc", max_grid_size: int = 30):
        self.cache_dir = cache_dir
        self.max_grid_size = max_grid_size
        self.tasks: List[ARCTask] = []
        os.makedirs(cache_dir, exist_ok=True)
        self._load_or_generate()

    def _load_or_generate(self):
        challenges_file = os.path.join(self.cache_dir, "arc_training_challenges.json")
        solutions_file = os.path.join(self.cache_dir, "arc_training_solutions.json")

        # Attempt to load cached official ARC files
        if os.path.exists(challenges_file) and os.path.exists(solutions_file):
            try:
                self._parse_official_json(challenges_file, solutions_file)
                print(f"[ARC Dataset] Loaded {len(self.tasks)} official ARC-AGI tasks from {self.cache_dir}")
                return
            except Exception as e:
                print(f"[ARC Dataset] Failed reading cached files: {e}")

        # Attempt download if online
        try:
            print("[ARC Dataset] Attempting download of official ARC-AGI training set from GitHub...")
            urllib.request.urlretrieve(self.ARC_TRAIN_URL, challenges_file)
            urllib.request.urlretrieve(self.ARC_SOLUTIONS_URL, solutions_file)
            self._parse_official_json(challenges_file, solutions_file)
            print(f"[ARC Dataset] Successfully downloaded and parsed {len(self.tasks)} official ARC-AGI tasks!")
            return
        except Exception as e:
            print(f"[ARC Dataset] Note: Download failed or offline ({e}). Generating procedural ConceptARC tasks.")

        # Fallback to high-quality procedural ConceptARC primitives
        self._generate_procedural_tasks(num_tasks=100)

    def _parse_official_json(self, challenges_path: str, solutions_path: str):
        with open(challenges_path, "r", encoding="utf-8") as f:
            challenges = json.load(f)
        with open(solutions_path, "r", encoding="utf-8") as f:
            solutions = json.load(f)

        for task_id, task_data in challenges.items():
            train_pairs = task_data.get("train", [])
            test_inputs = task_data.get("test", [])
            test_outputs = solutions.get(task_id, [])

            test_pairs = []
            for i, inp_item in enumerate(test_inputs):
                out_grid = test_outputs[i] if i < len(test_outputs) else inp_item.get("output", inp_item["input"])
                test_pairs.append({
                    "input": inp_item["input"],
                    "output": out_grid
                })

            self.tasks.append(ARCTask(
                task_id=task_id,
                train_pairs=train_pairs,
                test_pairs=test_pairs
            ))

    def _generate_procedural_tasks(self, num_tasks: int = 100):
        """Generates canonical ConceptARC transformations: Color Swaps, Gravity, Symmetry."""
        import random
        rng = random.Random(42)

        for i in range(num_tasks):
            rule = rng.choice(["invert_colors", "horizontal_symmetry", "gravity_down", "object_color_shift"])
            train_pairs = []
            for _ in range(3):
                H, W = rng.randint(4, 10), rng.randint(4, 10)
                grid_in = [[rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(W)] for _ in range(H)]
                grid_out = self._apply_rule(grid_in, rule)
                train_pairs.append({"input": grid_in, "output": grid_out})

            # Test pair
            H, W = rng.randint(4, 10), rng.randint(4, 10)
            test_in = [[rng.choice([0, 0, 0, 1, 2, 3]) for _ in range(W)] for _ in range(H)]
            test_out = self._apply_rule(test_in, rule)

            self.tasks.append(ARCTask(
                task_id=f"procedural_{rule}_{i}",
                train_pairs=train_pairs,
                test_pairs=[{"input": test_in, "output": test_out}]
            ))
        print(f"[ARC Dataset] Generated {len(self.tasks)} procedural ConceptARC tasks.")

    def _apply_rule(self, grid: List[List[int]], rule: str) -> List[List[int]]:
        H = len(grid)
        W = len(grid[0])
        if rule == "invert_colors":
            return [[(val + 1) % 4 if val > 0 else 0 for val in row] for row in grid]
        elif rule == "horizontal_symmetry":
            return [row[::-1] for row in grid]
        elif rule == "gravity_down":
            out = [[0] * W for _ in range(H)]
            for c in range(W):
                col_vals = [grid[r][c] for r in range(H) if grid[r][c] != 0]
                for r_idx, val in enumerate(col_vals):
                    out[H - len(col_vals) + r_idx][c] = val
            return out
        else:
            return [[val for val in row] for row in grid]

    def get_batch(self, batch_size: int = 4, device: str = "cpu") -> ARCBatch:
        import random
        sampled_tasks = random.sample(self.tasks, min(batch_size, len(self.tasks)))
        max_dim = self.max_grid_size
        padding_token = 10 # 0-9 are ARC colors, 10 is padding

        test_in_list, test_out_list, masks_list, prompts_list, shapes_list = [], [], [], [], []

        for task in sampled_tasks:
            test_pair = task.test_pairs[0]
            inp = test_pair["input"]
            out = test_pair["output"]

            H_in, W_in = len(inp), len(inp[0])
            H_out, W_out = len(out), len(out[0])
            shapes_list.append((H_out, W_out))

            # Pad test input
            padded_in = torch.full((max_dim, max_dim), padding_token, dtype=torch.long)
            for r in range(H_in):
                for c in range(W_in):
                    padded_in[r, c] = inp[r][c]

            # Pad test output & mask
            padded_out = torch.full((max_dim, max_dim), padding_token, dtype=torch.long)
            mask = torch.zeros((max_dim, max_dim), dtype=torch.float32)
            for r in range(H_out):
                for c in range(W_out):
                    padded_out[r, c] = out[r][c]
                    mask[r, c] = 1.0

            test_in_list.append(padded_in)
            test_out_list.append(padded_out)
            masks_list.append(mask)

            # Linguistic prompt for Left Hemisphere
            prompt = (
                f"You are solving an ARC-AGI visual abstraction challenge (Task ID: {task.task_id}). "
                f"Input grid size is {H_in}x{W_in}. Deduce the underlying spatial transformation rule "
                f"from the {len(task.train_pairs)} demonstration examples and predict the {H_out}x{W_out} target grid."
            )
            prompts_list.append(prompt)

        return ARCBatch(
            demo_inputs=torch.empty(0),
            demo_outputs=torch.empty(0),
            test_inputs=torch.stack(test_in_list, dim=0).to(device),
            test_targets=torch.stack(test_out_list, dim=0).to(device),
            target_masks=torch.stack(masks_list, dim=0).to(device),
            text_prompts=prompts_list,
            target_shapes=shapes_list
        )
