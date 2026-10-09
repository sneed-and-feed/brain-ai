"""
tests/test_llama_arc.py: Unit tests for LeftHemisphereLlama and ARCDataset.
"""

import pytest
import torch
from brain_ai.models.llama_lh import LeftHemisphereLlama
from brain_ai.models.ensemble import BiHemisphericBrain
from brain_ai.tasks.arc import ARCDataset


def test_left_hemisphere_llama_mock():
    # Test Left Hemisphere in mock mode
    lh = LeftHemisphereLlama(d_model=4096, mock_mode=True)
    
    texts = ["You are solving an ARC task.", "Navigate through the grid."]
    tokens = lh.tokenize(texts)
    assert "input_ids" in tokens
    assert tokens["input_ids"].shape[0] == 2
    
    # Extract residual stream
    z_l = lh.extract_residual_stream(tokens["input_ids"], tokens["attention_mask"])
    assert z_l.shape == (2, tokens["input_ids"].shape[1], 4096)
    
    # Generate text with callosal feedback
    feedback = torch.randn(1, tokens["input_ids"].shape[1], 4096)
    response = lh.generate_with_callosal_feedback("What is the rule?", delta_lh=feedback)
    assert isinstance(response, str)
    assert len(response) > 0


def test_arc_dataset():
    dataset = ARCDataset(cache_dir="tests/mock_arc", max_grid_size=15)
    assert len(dataset.tasks) > 0
    
    batch = dataset.get_batch(batch_size=2)
    assert batch.test_inputs.shape == (2, 15, 15)
    assert batch.test_targets.shape == (2, 15, 15)
    assert batch.target_masks.shape == (2, 15, 15)
    assert len(batch.text_prompts) == 2
    assert len(batch.target_shapes) == 2


def test_bihemispheric_llama_coupling():
    d_lh = 4096
    d_rh = 512
    d_call = 512
    
    lh = LeftHemisphereLlama(d_model=d_lh, mock_mode=True)
    brain = BiHemisphericBrain(d_lh=d_lh, d_rh=d_rh, d_callosum=d_call)
    
    prompts = ["Infer the transformation rule."]
    enc = lh.tokenize(prompts)
    z_lh = lh.extract_residual_stream(enc["input_ids"])
    
    rh_inputs = torch.randn(1, 225, d_rh)
    
    # Forward pass uniting LH Llama latents and RH spatial engine
    outputs = brain(lh_latents=z_lh, rh_inputs=rh_inputs)
    
    assert outputs["lh_latents_updated"].shape == z_lh.shape
    assert outputs["rh_latents_updated"].shape == rh_inputs.shape
    assert "affective_state" in outputs
    assert "conflict_score" in outputs


def test_arc_spatial_head_and_loss_backward():
    import torch.nn as nn
    from brain_ai.tasks.arc import ARCSpatialGridEmbedding, ARCPredictionHead

    d_rh = 512
    max_size = 15
    embedder = ARCSpatialGridEmbedding(num_colors=11, d_model=d_rh, max_size=32)
    head = ARCPredictionHead(d_model=d_rh, num_colors=10, max_size=max_size)

    # Test inputs
    grids = torch.randint(0, 10, (2, max_size, max_size))
    rh_latents = embedder(grids)
    assert rh_latents.shape == (2, max_size * max_size, d_rh)

    # Test forward head
    logits = head(rh_latents)
    assert logits.shape == (2, 10, max_size, max_size)

    # Test targets with valid labels and padding mask
    targets = torch.randint(0, 10, (2, max_size, max_size))
    mask = torch.ones((2, max_size, max_size))
    mask[:, 10:, :] = 0.0

    ce_loss_fn = nn.CrossEntropyLoss(reduction="none")
    safe_target = torch.clamp(targets, 0, 9)
    ce_matrix = ce_loss_fn(logits, safe_target)
    loss = (ce_matrix * mask).sum() / (mask.sum() + 1e-8)

    # Verify backward pass succeeds without internal errors
    loss.backward()

    for p in head.parameters():
        assert p.grad is not None

