"""
Prototype script to test the 4-way benchmark and Amygdalar dynamic routing.
Conditions:
1. Raw Llama (pure symbolic text prompt)
2. Pure RH (ablated Left Hemisphere, z_lh = 0)
3. Bi-Hemispheric System 1 (fast feedforward, 0 TTA)
4. Bi-Hemispheric System 2 (full TTA, 30 steps)
5. Dynamic Amygdala Router (System 1 vs System 2 based on conflict/urgency)
"""

import json
import re
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from brain_ai.tasks.arc import ARCDataset, ARCSpatialGridEmbedding, ARCPredictionHead
from brain_ai.models.llama_lh import LeftHemisphereLlama
from brain_ai.models.ensemble import BiHemisphericBrain

def format_grid_prompt(grid):
    return "/".join("".join(str(c) for c in row) for row in grid)

def eval_raw_llama_mock(task, lh_model):
    # Evaluates symbolic text performance
    prompt = f"ARC Task {task.task_id}. "
    for i, p in enumerate(task.train_pairs[:3]):
        prompt += f"Ex{i+1}: {format_grid_prompt(p['input'])} -> {format_grid_prompt(p['output'])}. "
    test_in = task.test_pairs[0]['input']
    prompt += f"Test input: {format_grid_prompt(test_in)}. Output grid:"
    
    # In mock mode, returns simulated baseline ~10-20%
    if lh_model.mock_mode:
        H_out = len(task.test_pairs[0]['output'])
        W_out = len(task.test_pairs[0]['output'][0])
        # Raw LLMs often copy input or predict most common color
        pred = np.zeros((H_out, W_out), dtype=int)
        for r in range(min(len(test_in), H_out)):
            for c in range(min(len(test_in[0]), W_out)):
                pred[r, c] = test_in[r][c]
        gt = np.array(task.test_pairs[0]['output'])
        return (pred == gt).mean() * 100.0, "Input-copy baseline"
    else:
        # Full Llama instruct prompt
        full_prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are an ARC-AGI spatial reasoning expert. Given input-output examples, deduce the rule and output ONLY the test output grid as a 2D JSON array of integers, with no explanation.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        out_text = lh_model.generate_with_callosal_feedback(full_prompt, delta_lh=None, max_new_tokens=128, temperature=0.1)
        # Parse JSON array from text
        match = re.search(r"\[\[.*?\]\]", out_text, re.DOTALL)
        if match:
            try:
                pred = np.array(json.loads(match.group(0)))
                gt = np.array(task.test_pairs[0]['output'])
                if pred.shape == gt.shape:
                    return (pred == gt).mean() * 100.0, out_text[:40]
            except Exception:
                pass
        return 0.0, out_text[:40]

print("Benchmark prototype definition ready!")
