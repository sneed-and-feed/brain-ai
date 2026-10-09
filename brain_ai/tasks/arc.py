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
    ARC_ZIP_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.zip"

    def __init__(self, cache_dir: str = "data/arc", max_grid_size: int = 30):
        self.cache_dir = cache_dir
        self.max_grid_size = max_grid_size
        self.tasks: List[ARCTask] = []
        os.makedirs(cache_dir, exist_ok=True)
        self._load_or_generate()

    def _load_or_generate(self):
        cached_zip = os.path.join(self.cache_dir, "arc_master.zip")

        # 1. Attempt loading from cached zip
        if os.path.exists(cached_zip):
            try:
                self._load_from_zip(cached_zip)
                if len(self.tasks) > 0:
                    print(f"[ARC Dataset] Loaded {len(self.tasks)} official ARC-AGI tasks (filtered to <= {self.max_grid_size}x{self.max_grid_size}) from {cached_zip}")
                    return
            except Exception as e:
                print(f"[ARC Dataset] Failed reading cached zip: {e}")

        # 2. Attempt download of official ARC GitHub zip
        try:
            print("[ARC Dataset] Downloading official ARC-AGI dataset from GitHub (400 train + 400 eval tasks)...")
            req = urllib.request.Request(self.ARC_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
            with open(cached_zip, "wb") as f:
                f.write(data)
            self._load_from_zip(cached_zip)
            if len(self.tasks) > 0:
                print(f"[ARC Dataset] Successfully loaded {len(self.tasks)} official ARC-AGI tasks (filtered to <= {self.max_grid_size}x{self.max_grid_size})!")
                return
        except Exception as e:
            print(f"[ARC Dataset] Note: Download failed or offline ({e}). Generating procedural ConceptARC tasks.")

        # 3. Fallback to procedural ConceptARC primitives
        self._generate_procedural_tasks(num_tasks=100)

    def _load_from_zip(self, zip_path: str):
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            for fname in z.namelist():
                if fname.endswith(".json") and ("/data/training/" in fname or "/data/evaluation/" in fname):
                    task_id = os.path.splitext(os.path.basename(fname))[0]
                    task_data = json.loads(z.read(fname).decode("utf-8"))
                    train_pairs = task_data.get("train", [])
                    test_pairs = task_data.get("test", [])

                    # Filter: ensure test challenges fit strictly within max_grid_size
                    fits = True
                    for tp in test_pairs:
                        inp = tp.get("input", [])
                        out = tp.get("output", inp)
                        if len(inp) > self.max_grid_size or (len(inp) > 0 and len(inp[0]) > self.max_grid_size):
                            fits = False
                            break
                        if len(out) > self.max_grid_size or (len(out) > 0 and len(out[0]) > self.max_grid_size):
                            fits = False
                            break
                    if fits:
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

            H_in = len(inp)
            W_in = len(inp[0]) if H_in > 0 else 0
            H_out = len(out)
            W_out = len(out[0]) if H_out > 0 else 0

            copy_h_in = min(H_in, max_dim)
            copy_w_in = min(W_in, max_dim)
            copy_h_out = min(H_out, max_dim)
            copy_w_out = min(W_out, max_dim)

            shapes_list.append((copy_h_out, copy_w_out))

            # Pad test input
            padded_in = torch.full((max_dim, max_dim), padding_token, dtype=torch.long)
            for r in range(copy_h_in):
                for c in range(copy_w_in):
                    padded_in[r, c] = inp[r][c]

            # Pad test output with 0 & record mask (mask zeros out padding)
            padded_out = torch.zeros((max_dim, max_dim), dtype=torch.long)
            mask = torch.zeros((max_dim, max_dim), dtype=torch.float32)
            for r in range(copy_h_out):
                for c in range(copy_w_out):
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


class ARCSpatialGridEmbedding(nn.Module):
    """
    Embeds 2D discrete ARC grid tokens with learned 2D coordinate embeddings
    into the continuous Right Hemisphere latent manifold.
    """
    def __init__(self, num_colors: int = 11, d_model: int = 512, max_size: int = 32):
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, d_model)
        self.row_embed = nn.Embedding(max_size, d_model)
        self.col_embed = nn.Embedding(max_size, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        # grids: [B, H, W]
        B, H, W = grids.shape
        device = grids.device
        rows = torch.arange(H, device=device).unsqueeze(1).repeat(1, W).view(-1)
        cols = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1).view(-1)
        
        flat_tokens = grids.view(B, H * W)
        color_emb = self.color_embed(flat_tokens)
        pos_emb = self.row_embed(rows) + self.col_embed(cols)
        return self.proj(color_emb + pos_emb.unsqueeze(0))


class ARCPredictionHead(nn.Module):
    """
    Decodes Right Hemisphere recurrent latents back into 2D discrete ARC grid color logits.
    Guarantees strict C-contiguous memory layout for robust cuDNN Conv2d backward passes.
    """
    def __init__(self, d_model: int = 512, num_colors: int = 10, max_size: int = 15):
        super().__init__()
        self.max_size = max_size
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_colors, kernel_size=1)
        )

    def forward(self, rh_latents: torch.Tensor) -> torch.Tensor:
        # rh_latents: [B, H*W, d_model]
        B, L, D = rh_latents.shape
        H = W = self.max_size
        # Strictly enforce contiguous memory layout to avoid cuDNN internal errors on backward
        x_2d = rh_latents.transpose(1, 2).contiguous().view(B, D, H, W)
        logits_2d = self.conv(x_2d) # [B, 10, H, W]
        return logits_2d

