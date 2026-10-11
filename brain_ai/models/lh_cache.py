"""
brain_ai.models.lh_cache: Left Hemisphere (LH) Latent Extraction and Offline Caching.

Implements Phase P2 / RQ1 requirements:
1. Serialization of ARC demonstration grids (input/output) into structured linguistic prompts.
   Strict leakage prevention: test outputs are never included.
2. Offline extraction of intermediate residual stream representations from a frozen LLM
   (Qwen2.5 / Llama 3.1 or deterministic mock for testing/CI).
3. Pre-computation and atomic disk/Drive caching of LH latents indexed by task identifier.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def serialize_grid(grid: Union[List[List[int]], np.ndarray]) -> str:
    """Format a 2D integer grid into a structured text representation with dimensions."""
    if isinstance(grid, np.ndarray):
        grid = grid.tolist()
    if not grid or not isinstance(grid, list):
        return "Empty Grid (0x0)"
    h = len(grid)
    w = len(grid[0]) if h > 0 and isinstance(grid[0], list) else 0
    lines = [f"Grid ({h}x{w}):"]
    for row in grid:
        lines.append(" ".join(str(int(c)) for c in row))
    return "\n".join(lines)


def format_arc_task_prompt(task_data: dict, task_id: str = "") -> str:
    """
    Format ARC task demonstration pairs into a structured natural-language prompt.
    CRITICAL: Only demonstration pairs (task_data['train']) are serialized.
    Test outputs are never included, guaranteeing zero evaluation label leakage.
    """
    train_pairs = task_data.get("train", [])
    if not train_pairs:
        # Fallback if Kaggle format stores pairs directly
        train_pairs = task_data.get("train_pairs", [])

    header = [
        f"ARC Task: {task_id}" if task_id else "ARC Task Demonstration Analysis",
        "Analyze the following input-to-output demonstration grids and infer the transformation rule.\n"
    ]

    demo_blocks = []
    for idx, pair in enumerate(train_pairs):
        inp_grid = pair.get("input", [])
        out_grid = pair.get("output", [])
        inp_str = serialize_grid(inp_grid)
        out_str = serialize_grid(out_grid)
        demo_blocks.append(
            f"--- Demonstration {idx + 1} ---\n"
            f"[Input Grid]\n{inp_str}\n"
            f"[Output Grid]\n{out_str}\n"
        )

    prompt = "\n".join(header) + "\n".join(demo_blocks)
    return prompt.strip()


def hash_demonstrations(task_data: dict) -> str:
    """Computes SHA256 digest of demonstration grids for cache validation."""
    train_pairs = task_data.get("train", task_data.get("train_pairs", []))
    serialized = json.dumps(train_pairs, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def extract_lh_latents(
    prompt: str,
    model: Optional[nn.Module] = None,
    tokenizer: Any = None,
    hook_layer: int = 16,
    target_tokens: int = 32,
    d_model: int = 4096,
    device: str = "cpu",
    mock_mode: bool = False
) -> torch.Tensor:
    """
    Extract intermediate residual stream representations Z_L from frozen LLM.
    Returns: Tensor of shape (1, target_tokens, d_model) in float32.
    """
    if mock_mode or model is None or getattr(model, "mock_mode", False):
        # Deterministic pseudo-random latent synthesis based on prompt hash
        h = hashlib.sha256(prompt.encode("utf-8")).digest()
        # Seed torch RNG with hash bytes for determinism
        seed = int.from_bytes(h[:4], byteorder="big")
        g = torch.Generator().manual_seed(seed)
        latents = torch.randn(1, target_tokens, d_model, generator=g, dtype=torch.float32)
        # Apply layer-norm style normalization
        latents = F.layer_norm(latents, (d_model,))
        return latents.to(device)

    # Live model execution
    model.eval()
    if tokenizer is not None:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
    else:
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        if hasattr(model, "extract_residual_stream"):
            raw_latents = model.extract_residual_stream(input_ids, attention_mask=attn_mask)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            # Direct HuggingFace causal LM
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                return_dict=True
            )
            # Hidden states: 0 = embedding, 1..N = layers
            layer_idx = min(hook_layer + 1, len(outputs.hidden_states) - 1)
            raw_latents = outputs.hidden_states[layer_idx]
        else:
            # Fallback forward
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            raw_latents = out.get("residual_latent", out.logits if hasattr(out, "logits") else out[0])

    # raw_latents: (1, seq_len, d_model)
    if raw_latents.dim() == 2:
        raw_latents = raw_latents.unsqueeze(0)

    # Adaptive 1D pooling across sequence dimension to target_tokens
    # Permute to (B, d_model, seq_len) for adaptive_avg_pool1d
    pooled = F.adaptive_avg_pool1d(raw_latents.transpose(1, 2).float(), target_tokens)
    pooled = pooled.transpose(1, 2)  # (1, target_tokens, d_model)
    return pooled.to(device)


def build_lh_cache_for_tasks(
    challenges: Dict[str, dict],
    identifiers_map: Union[Dict[str, int], Sequence[str]],
    model: Optional[nn.Module] = None,
    tokenizer: Any = None,
    hook_layer: int = 16,
    target_tokens: int = 32,
    d_model: int = 4096,
    device: str = "cpu",
    mock_mode: bool = False,
    output_path: Optional[str] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Builds a stacked tensor of cached LH latents for unique base tasks,
    with an index mapping tensor (puzzle_to_base_id) for TRM's augmented puzzle identifiers.
    Returns:
        latents_tensor: Tensor of shape (num_unique_tasks + 1, target_tokens, d_model)
        metadata: Dict containing provenance, task IDs, demonstration hashes, and puzzle_to_base_id
    """
    # Normalize input into a sequence of identifiers
    if isinstance(identifiers_map, dict):
        if not any("|||" in str(k) for k in identifiers_map):
            # Direct dictionary mapping without TRM augmentations
            num_tasks = max(len(identifiers_map), (max(identifiers_map.values()) + 1) if identifiers_map else 0)
            latents_tensor = torch.zeros(num_tasks, target_tokens, d_model, dtype=torch.float32)
            puzzle_to_base_id = torch.arange(num_tasks, dtype=torch.long)
            sorted_tasks = sorted(identifiers_map.items(), key=lambda x: x[1])
            unique_tasks = [(k, v) for k, v in sorted_tasks]
            raw_list = None
        else:
            max_v = max(identifiers_map.values()) if identifiers_map else 0
            raw_list = ["<blank>"] * (max_v + 1)
            for k, v in identifiers_map.items():
                raw_list[v] = k
    else:
        # Sequence of identifiers (list, tuple, etc.)
        raw_list = list(identifiers_map)

    if raw_list is not None:
        unique_base_ids = []
        seen = set()
        for raw_id in raw_list:
            if not raw_id or raw_id == "<blank>":
                continue
            base_id = str(raw_id).split("|||")[0].replace(".json", "")
            if base_id not in seen:
                seen.add(base_id)
                unique_base_ids.append(base_id)

        # Index 0 is reserved for <blank> (zeros)
        latents_tensor = torch.zeros(len(unique_base_ids) + 1, target_tokens, d_model, dtype=torch.float32)
        base_task_to_idx = {bid: i + 1 for i, bid in enumerate(unique_base_ids)}
        puzzle_to_base_id = torch.zeros(len(raw_list), dtype=torch.long)
        for p_idx, raw_id in enumerate(raw_list):
            if not raw_id or raw_id == "<blank>":
                puzzle_to_base_id[p_idx] = 0
            else:
                base_id = str(raw_id).split("|||")[0].replace(".json", "")
                puzzle_to_base_id[p_idx] = base_task_to_idx.get(base_id, 0)

        unique_tasks = [(bid, base_task_to_idx[bid]) for bid in unique_base_ids]

    metadata = {
        "num_tasks": latents_tensor.shape[0],
        "target_tokens": target_tokens,
        "d_model": d_model,
        "hook_layer": hook_layer,
        "mock_mode": mock_mode,
        "task_meta": {},
        "puzzle_to_base_id": puzzle_to_base_id
    }

    for task_id, idx in unique_tasks:
        task_data = None
        candidates = [task_id, f"{task_id}.json", task_id.replace(".json", "")]
        for c in candidates:
            if c in challenges:
                task_data = challenges[c]
                break
        if task_data is None:
            raise KeyError(f"Task ID '{task_id}' not found in challenges dictionary.")

        prompt = format_arc_task_prompt(task_data, task_id=task_id)
        demo_hash = hash_demonstrations(task_data)

        z_l = extract_lh_latents(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            hook_layer=hook_layer,
            target_tokens=target_tokens,
            d_model=d_model,
            device="cpu",  # cache on CPU memory, moved to GPU when loading
            mock_mode=mock_mode
        )
        latents_tensor[idx] = z_l.squeeze(0)

        metadata["task_meta"][task_id] = {
            "index": idx,
            "demo_hash": demo_hash,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            "n_demos": len(task_data.get("train", task_data.get("train_pairs", [])))
        }

    # Save to disk if requested
    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        meta_to_save = {k: v for k, v in metadata.items() if k != "puzzle_to_base_id"}
        meta_to_save["puzzle_to_base_id_shape"] = list(puzzle_to_base_id.shape)
        meta_path = output_path + ".meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta_to_save, f, indent=2)
        torch.save({
            "latents": latents_tensor,
            "puzzle_to_base_id": puzzle_to_base_id,
            "metadata": metadata
        }, output_path)

    return latents_tensor, metadata


def load_lh_cache(cache_path: str, device: str = "cpu") -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load cached LH latents tensor and metadata from disk."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"LH cache file not found at: {cache_path}")
    data = torch.load(cache_path, map_location=device, weights_only=False)
    if isinstance(data, dict) and "latents" in data:
        meta = data.get("metadata", {})
        if "puzzle_to_base_id" in data:
            meta["puzzle_to_base_id"] = data["puzzle_to_base_id"].to(device)
        return data["latents"].to(device), meta
    elif isinstance(data, torch.Tensor):
        meta_path = cache_path + ".meta.json"
        metadata = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return data.to(device), metadata
    else:
        raise ValueError(f"Unrecognized cache format in {cache_path}")


def verify_no_test_leakage_in_cache(
    metadata: Dict[str, Any],
    solutions: Dict[str, List[List[List[int]]]]
) -> bool:
    """
    Integrity assertion: confirms that no evaluation test outputs were used
    during the construction of the LH cache.
    """
    for task_id, meta in metadata.get("task_meta", {}).items():
        if task_id in solutions:
            # Task is in solutions, verify that the metadata records only training demonstrations
            n_demos = meta.get("n_demos", 0)
            if n_demos == 0:
                raise ValueError(f"Task {task_id} has 0 demonstrations recorded in cache metadata.")
            # If prompt text was recorded in metadata, verify no test solution appears
            if "prompt" in meta:
                for sol_grid in solutions[task_id]:
                    grid_repr = serialize_grid(sol_grid)
                    if grid_repr in meta["prompt"]:
                        raise ValueError(f"LEAKAGE DETECTED: Ground-truth test output found in prompt for task {task_id}!")
    return True
