"""
tests.test_phase2_arc2: Comprehensive Unit and Integration Tests for Phase 2.
"""

import pytest
import torch
import numpy as np

from brain_ai.models.qwen_lh import LeftHemisphereQwen
from brain_ai.models.hrm_scaled import ScaledHierarchicalReasoningModel, RoPE2D, AxialAttention2D
from brain_ai.tasks.arc_dsl import (
    apply_d4, invert_d4, expand_demos_d4, d4_symmetrized_consensus,
    get_connected_components, bounding_box, crop, paste, flood_fill, gravity,
    execute_dsl_program, verify_program_on_demos
)
from brain_ai.models.ensemble_arc2 import ScaledBiHemisphericBrainARC2, MultiCandidatePass2Selector


def test_d4_group_inversions():
    grid = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    for t in range(8):
        transformed = apply_d4(grid, t)
        inverted = invert_d4(transformed, t)
        assert np.array_equal(grid, inverted), f"D4 transform {t} failed inversion"


def test_d4_expansion():
    demos = [(np.zeros((3, 3), dtype=int), np.ones((3, 3), dtype=int))]
    expanded = expand_demos_d4(demos)
    assert len(expanded) == 8


def test_arc_dsl_primitives():
    grid = np.zeros((5, 5), dtype=int)
    grid[1:3, 1:3] = 2 # A 2x2 square of color 2
    comps = get_connected_components(grid, background=0)
    assert len(comps) == 1
    assert comps[0]["color"] == 2
    assert comps[0]["bbox"] == (1, 1, 2, 2)

    cropped = crop(grid, comps[0]["bbox"])
    assert cropped.shape == (2, 2)
    assert np.all(cropped == 2)

    filled = flood_fill(grid, 0, 0, 7)
    assert filled[0, 0] == 7
    assert filled[1, 1] == 2 # Unaffected

    grav = gravity(grid, direction="down")
    assert grav[4, 1] == 2
    assert grav[3, 1] == 2


def test_dsl_sandbox_execution():
    code = """
def transform(grid):
    return recolor(grid, 2, 5)
"""
    test_in = np.array([[2, 0], [0, 2]])
    res = execute_dsl_program(code, test_in)
    assert res is not None
    assert np.all(res == np.array([[5, 0], [0, 5]]))


def test_scaled_hrm_forward():
    d_model = 64
    model = ScaledHierarchicalReasoningModel(d_model=d_model, n_heads=4, d_ffn=128, L_cycles=2, M_max=4)
    x = torch.randn(2, 6, 6, d_model)
    out, carry, aux = model(x, max_steps=2)
    assert out.shape == (2, 6, 6, d_model)
    assert carry.step_count.max().item() in (1, 2)


def test_qwen_mock_lh():
    lh = LeftHemisphereQwen(d_model=128, mock_mode=True, device="cpu")
    out = lh(prompt_text="Solve ARC task")
    assert out["residual_latent"].shape[-1] == 128


def test_scaled_bihemispheric_brain_arc2():
    brain = ScaledBiHemisphericBrainARC2(
        d_lh=128,
        d_rh=64,
        d_callosum=64,
        callosal_heads=2,
        mock_mode=True,
        device="cpu"
    )
    # Test System 1 reflex
    x_grid = torch.randn(1, 4, 4, 64)
    z_ref, info_ref = brain.forward_system1_reflex(x_grid)
    assert z_ref.shape == (1, 4, 4, 64)
    assert info_ref["mode"] == "system1_reflex"

    # Test full cognitive pass
    z_cog, info_cog = brain.forward_cognitive(x_grid, lh_prompt="Solve ARC 123")
    assert z_cog.shape == (1, 4, 4, 64)
    assert "affective_state" in info_cog


def test_pass2_candidate_selection():
    cand1 = np.ones((3, 3), dtype=int)
    cand2 = np.zeros((3, 3), dtype=int)
    pool = [
        ("cand1", cand1, 15.0),
        ("cand2", cand2, 12.0)
    ]
    att1, att2, meta = MultiCandidatePass2Selector.select_pass2_candidates(pool)
    assert np.array_equal(att1, cand1)
    assert np.array_equal(att2, cand2)
    assert meta["attempt_1_source"] == "cand1"
    assert meta["attempt_2_source"] == "cand2"
