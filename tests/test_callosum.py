"""
Tests for brain_ai modules:
- Dale's principle sign segregation
- Homeostatic synaptic scaling
- DaleDifferentialCrossAttention
- OpenJevAmygdalarRouter & NeuromodulatoryController
- HierarchicalReasoningModel
- BiHemisphericBrain integration
"""

import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from brain_ai.models.callosum import (
    DaleLinear,
    HomeostaticSynapticScaling,
    DaleDifferentialCrossAttention,
    InterHemisphericLatentCoupling,
    CallosalConditioning,
    CallosalPuzzleEmbedding
)
from brain_ai.models.lh_cache import (
    serialize_grid,
    format_arc_task_prompt,
    hash_demonstrations,
    extract_lh_latents,
    build_lh_cache_for_tasks,
    load_lh_cache,
    verify_no_test_leakage_in_cache
)
from brain_ai.models.amygdala import (
    OpenJevAmygdalarRouter,
    NeuromodulatoryController
)
from brain_ai.models.hrm import (
    HierarchicalReasoningModel
)
from brain_ai.models.ensemble import (
    BiHemisphericBrain
)


def test_dale_linear_sign_segregation():
    in_features, out_features = 64, 32
    layer = DaleLinear(in_features, out_features, p_excitatory=0.8, mode="softplus")
    
    W = layer.get_effective_weight()
    n_e = int(round(in_features * 0.8))
    
    assert W.shape == (out_features, in_features)
    # Excitatory columns must be >= 0
    assert (W[:, :n_e] >= 0.0).all(), "Excitatory columns violate W >= 0"
    # Inhibitory columns must be <= 0
    assert (W[:, n_e:] <= 0.0).all(), "Inhibitory columns violate W <= 0"
    
    # Check forward pass
    x = torch.randn(4, 10, in_features)
    out = layer(x)
    assert out.shape == (4, 10, out_features)


def test_homeostatic_synaptic_scaling():
    d_model = 64
    scaler = HomeostaticSynapticScaling(d_model=d_model, r_target=1.0)
    
    # Large inputs (potential epilepsy)
    x = torch.randn(2, 8, d_model) * 10.0
    x_normed, loss = scaler(x)
    
    assert x_normed.shape == x.shape
    rms = torch.sqrt(torch.mean(x_normed ** 2, dim=-1))
    # RMS should be tightly bounded near r_target = 1.0
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)
    assert loss.item() > 0.0


def test_interhemispheric_coupling():
    B, T_L, T_R, D = 2, 8, 8, 32
    z_left = torch.randn(B, T_L, D)
    z_right = torch.randn(B, T_R, D)
    
    callosum = InterHemisphericLatentCoupling(
        d_latent=D,
        n_heads=2,
        cross_talk_sign=-1.0,
        p_excitatory=0.8,
        r_target=1.0
    )
    
    z_l_out, z_r_out, losses = callosum(z_left, z_right)
    
    assert z_l_out.shape == z_left.shape
    assert z_r_out.shape == z_right.shape
    assert "loss_homeostatic" in losses
    assert not torch.isnan(z_l_out).any()
    assert not torch.isnan(z_r_out).any()


def test_amygdala_and_neuromodulation():
    B, D, K = 2, 64, 4
    state = torch.randn(B, D)
    cands = torch.randn(B, K)
    
    router = OpenJevAmygdalarRouter(hidden_dim=D, mlp_dim=32)
    controller = NeuromodulatoryController()
    
    aff = router(state, candidate_logits=cands)
    assert "valence" in aff
    assert "uncertainty" in aff
    assert "urgency" in aff
    assert (aff["valence"] >= -1.0).all() and (aff["valence"] <= 1.0).all()
    assert (aff["uncertainty"] >= 0.0).all() and (aff["uncertainty"] <= 1.0).all()
    assert (aff["urgency"] >= 0.0).all() and (aff["urgency"] <= 1.0).all()
    
    ctrl = controller.compute_modulations(aff["valence"], aff["uncertainty"], aff["urgency"])
    assert "gamma_E" in ctrl
    assert "gamma_I" in ctrl
    assert "temperature" in ctrl
    assert "reasoning_steps" in ctrl


def test_hierarchical_reasoning_model():
    B, S, D = 2, 6, 64
    x = torch.randn(B, S, D)
    
    hrm = HierarchicalReasoningModel(
        d_model=D,
        n_heads=4,
        d_ffn=128,
        L_cycles=2,
        M_max=3
    )
    
    converged_z_H, carry, aux = hrm(x)
    assert converged_z_H.shape == (B, S, D)
    assert carry.step_count.max().item() <= 3


def test_full_bihemispheric_brain():
    B, S_L, S_R = 2, 8, 6
    d_lh, d_rh, d_call = 128, 64, 64
    
    brain = BiHemisphericBrain(
        d_lh=d_lh,
        d_rh=d_rh,
        d_callosum=d_call,
        callosal_heads=2,
        hrm_cycles=2,
        hrm_max_segments=3
    )
    
    lh_latents = torch.randn(B, S_L, d_lh)
    rh_inputs = torch.randn(B, S_R, d_rh)
    
    out = brain(lh_latents, rh_inputs)
    
    assert out["lh_latents_updated"].shape == lh_latents.shape
    assert out["rh_latents_updated"].shape == rh_inputs.shape
    assert "conflict_score" in out
    assert "affective_state" in out
    assert "neuromodulatory_controls" in out
    assert not torch.isnan(out["lh_latents_updated"]).any()


def test_maze_generator():
    from brain_ai.tasks.maze import MazeGenerator
    gen = MazeGenerator(size=11, seed=42)
    batch = gen.generate_batch(batch_size=4)
    
    assert batch.grid_tokens.shape == (4, 121)
    assert batch.path_targets.shape == (4, 121)
    assert len(batch.text_prompts) == 4
    assert len(batch.solution_paths) == 4
    # Start and goal must be in the solution path
    for p in batch.solution_paths:
        assert p[0] == (1, 1)
        assert p[-1] == (9, 9)


def test_spatial_modules():
    from brain_ai.tasks.maze import Spatial2DGridEmbedding, SpatialConvHead
    B, N, D = 2, 11, 64
    embedder = Spatial2DGridEmbedding(num_tokens=4, d_model=D, max_size=32)
    head = SpatialConvHead(d_model=D, size=N)
    
    tokens = torch.randint(0, 4, (B, N * N))
    latents = embedder(tokens, size=N)
    assert latents.shape == (B, N * N, D)
    
    logits = head(latents)
    assert logits.shape == (B, N * N)


def test_reflex_first_cascaded_router():
    from brain_ai.models.amygdala import ReflexFirstCascadedRouter
    router = ReflexFirstCascadedRouter(theta_bypass=0.90)

    # 1. High S1 fit -> Bypass S2
    pred_s1 = torch.tensor([1, 2, 3])
    pred_s2 = torch.tensor([1, 0, 0])
    res1 = router.decide_and_select(fit_s1=0.95, pred_test_s1=pred_s1)
    assert res1["decision"] == "System 1 (Reflex Bypass)"
    assert not res1["escalated_to_s2"]
    assert torch.equal(res1["selected_pred"], pred_s1)

    # 2. Low S1 fit, S2 improves -> S2 accepted
    res2 = router.decide_and_select(
        fit_s1=0.60,
        fit_s2=0.85,
        pred_test_s1=pred_s1,
        pred_test_s2=pred_s2
    )
    assert res2["decision"] == "System 2 (Adapted TTA)"
    assert res2["escalated_to_s2"]
    assert not res2["safety_reversion"]
    assert torch.equal(res2["selected_pred"], pred_s2)

    # 3. S2 degrades -> Reverts to S1
    res3 = router.decide_and_select(
        fit_s1=0.75,
        fit_s2=0.40,
        pred_test_s1=pred_s1,
        pred_test_s2=pred_s2
    )
    assert res3["decision"] == "System 1 Fallback (Safety Reversion)"
    assert res3["safety_reversion"]
    assert torch.equal(res3["selected_pred"], pred_s1)


def test_callosal_conditioning_dale_inhibitory():
    B, S_L, S_R = 2, 32, 64
    d_lh, d_reasoner, d_call = 128, 64, 64
    num_query = 16

    callosum = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=d_call,
        num_query_tokens=num_query,
        n_heads=2,
        p_excitatory=0.8,
        cross_talk_sign=-1.0,
        constraint_type="dale_inhibitory",
        r_target=1.0
    )

    lh_latents = torch.randn(B, S_L, d_lh, requires_grad=True)

    # 1. Test task conditioning synthesis
    task_tokens = callosum.condition_task(lh_latents)
    assert task_tokens.shape == (B, num_query, d_reasoner)
    assert not torch.isnan(task_tokens).any()
    assert not torch.isinf(task_tokens).any()

    # 2. Test reasoner state conditioning
    reasoner_state = torch.randn(B, S_R, d_reasoner)
    conditioned_state = callosum.condition_state(reasoner_state, lh_latents)
    assert conditioned_state.shape == (B, S_R, d_reasoner)
    assert not torch.isnan(conditioned_state).any()

    # 3. Test backward pass and gradient flow
    loss = task_tokens.sum() + conditioned_state.sum()
    loss.backward()

    assert callosum.coupling_gain.grad is not None
    assert callosum.task_queries.grad is not None
    assert lh_latents.grad is not None


def test_callosal_conditioning_dale_excitatory_and_unconstrained():
    B, S_L = 2, 16
    d_lh, d_reasoner, d_call = 64, 32, 32
    num_query = 8

    # Excitatory
    callosum_exc = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=d_call,
        num_query_tokens=num_query,
        n_heads=2,
        constraint_type="dale_excitatory",
        cross_talk_sign=1.0
    )
    lh = torch.randn(B, S_L, d_lh)
    out_exc = callosum_exc.condition_task(lh)
    assert out_exc.shape == (B, num_query, d_reasoner)
    assert callosum_exc.cross_talk_sign == 1.0

    # Unconstrained
    callosum_unconstrained = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=d_call,
        num_query_tokens=num_query,
        n_heads=2,
        constraint_type="unconstrained",
        cross_talk_sign=1.0
    )
    out_unconstrained = callosum_unconstrained.condition_task(lh)
    assert out_unconstrained.shape == (B, num_query, d_reasoner)


def test_callosal_puzzle_embedding_wrapper():
    N_tasks, S_L, d_lh = 10, 16, 64
    d_reasoner, num_query = 32, 8

    cached_latents = torch.randn(N_tasks, S_L, d_lh)
    callosum = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=32,
        num_query_tokens=num_query,
        n_heads=2,
        constraint_type="dale_inhibitory"
    )

    # Flattened output (for drop-in TRM compatibility)
    emb_module = CallosalPuzzleEmbedding(
        cached_lh_latents=cached_latents,
        callosal_conditioning=callosum,
        num_query_tokens=num_query,
        d_reasoner=d_reasoner,
        flatten_output=True
    )

    puzzle_ids = torch.tensor([0, 3, 7])
    out = emb_module(puzzle_ids)
    assert out.shape == (3, num_query * d_reasoner)

    # Backward pass: gradients must reach callosum parameters, but NOT cached_latents buffer
    loss = out.sum()
    loss.backward()
    assert callosum.coupling_gain.grad is not None
    assert emb_module.cached_lh_latents.grad is None

    # Unflattened output
    emb_unflattened = CallosalPuzzleEmbedding(
        cached_lh_latents=cached_latents,
        callosal_conditioning=callosum,
        num_query_tokens=num_query,
        d_reasoner=d_reasoner,
        flatten_output=False
    )
    out_unflattened = emb_unflattened(puzzle_ids)
    assert out_unflattened.shape == (3, num_query, d_reasoner)


def test_lh_grid_serialization_and_prompting():
    grid = [[1, 2], [3, 4]]
    s = serialize_grid(grid)
    assert "Grid (2x2):" in s
    assert "1 2" in s
    assert "3 4" in s

    task_data = {
        "train": [
            {"input": [[0, 1]], "output": [[1, 0]]},
            {"input": [[2, 3]], "output": [[3, 2]]}
        ],
        "test": [
            {"input": [[4, 5]], "output": [[9, 9]]}  # Must NEVER appear in prompt!
        ]
    }

    prompt = format_arc_task_prompt(task_data, task_id="demo_task_01")
    assert "ARC Task: demo_task_01" in prompt
    assert "Demonstration 1" in prompt
    assert "Demonstration 2" in prompt
    assert "[Input Grid]" in prompt
    assert "[Output Grid]" in prompt
    # Strict Leakage Assertion: test output [[9, 9]] must NOT be present!
    assert "9 9" not in prompt

    h = hash_demonstrations(task_data)
    assert isinstance(h, str) and len(h) == 16


def test_lh_latent_extraction_and_caching(tmp_path):
    prompt = "Test prompt for ARC task demonstration inference."
    d_model = 64
    target_tokens = 16

    # Test deterministic mock extraction
    z1 = extract_lh_latents(prompt, mock_mode=True, d_model=d_model, target_tokens=target_tokens)
    z2 = extract_lh_latents(prompt, mock_mode=True, d_model=d_model, target_tokens=target_tokens)
    assert z1.shape == (1, target_tokens, d_model)
    assert torch.allclose(z1, z2), "Mock extraction must be deterministic"

    # Test distinct prompts produce distinct representations
    z_diff = extract_lh_latents("Different prompt", mock_mode=True, d_model=d_model, target_tokens=target_tokens)
    assert not torch.allclose(z1, z_diff)

    # Test caching pipeline
    challenges = {
        "task_a": {"train": [{"input": [[1, 0]], "output": [[0, 1]]}]},
        "task_b": {"train": [{"input": [[2, 2]], "output": [[3, 3]]}]}
    }
    identifiers = {"task_a": 0, "task_b": 1}
    out_file = str(tmp_path / "test_lh_cache.pt")

    tensor, meta = build_lh_cache_for_tasks(
        challenges=challenges,
        identifiers_map=identifiers,
        mock_mode=True,
        d_model=d_model,
        target_tokens=target_tokens,
        output_path=out_file
    )

    assert tensor.shape == (2, target_tokens, d_model)
    assert os.path.exists(out_file)

    # Load roundtrip
    loaded_tensor, loaded_meta = load_lh_cache(out_file)
    assert torch.allclose(tensor, loaded_tensor)
    assert loaded_meta["num_tasks"] == 2

    # Leakage check
    fake_solutions = {"task_a": [[[0, 1]]], "task_b": [[[3, 3]]]}
    assert verify_no_test_leakage_in_cache(loaded_meta, fake_solutions)


def test_lh_cache_with_trm_augmented_identifiers_list(tmp_path):
    """Verify that build_lh_cache_for_tasks correctly processes TRM's identifiers.json list with <blank> and augmentations."""
    challenges = {
        "007bbfb7": {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}]
        },
        "00d62c1b": {
            "train": [{"input": [[5, 6]], "output": [[6, 5]]}]
        }
    }
    # Emulate TRM's identifiers.json list format
    identifiers_list = [
        "<blank>",
        "007bbfb7",
        "007bbfb7|||t0|||0123456789",
        "007bbfb7|||t1|||0987654321",
        "00d62c1b",
        "00d62c1b|||t3|||0246813579"
    ]
    out_file = str(tmp_path / "test_trm_lh_cache.pt")
    target_tokens, d_model = 8, 32

    tensor, meta = build_lh_cache_for_tasks(
        challenges=challenges,
        identifiers_map=identifiers_list,
        mock_mode=True,
        d_model=d_model,
        target_tokens=target_tokens,
        output_path=out_file
    )

    # Tensor shape must be (num_unique_tasks + 1, target_tokens, d_model) -> (3, 8, 32)
    assert tensor.shape == (3, target_tokens, d_model)
    # Row 0 (<blank>) must be all zeros
    assert torch.all(tensor[0] == 0.0)
    # Rows 1 and 2 must be non-zero
    assert not torch.all(tensor[1] == 0.0)
    assert not torch.all(tensor[2] == 0.0)

    # Check puzzle_to_base_id
    p2b = meta["puzzle_to_base_id"]
    assert p2b.shape == (len(identifiers_list),)
    assert p2b[0] == 0  # <blank>
    assert p2b[1] == 1  # 007bbfb7
    assert p2b[2] == 1  # 007bbfb7|||t0
    assert p2b[3] == 1  # 007bbfb7|||t1
    assert p2b[4] == 2  # 00d62c1b
    assert p2b[5] == 2  # 00d62c1b|||t3

    # Load roundtrip
    loaded_tensor, loaded_meta = load_lh_cache(out_file)
    assert torch.allclose(tensor, loaded_tensor)
    assert "puzzle_to_base_id" in loaded_meta
    assert torch.equal(p2b, loaded_meta["puzzle_to_base_id"])

    # Test CallosalPuzzleEmbedding with puzzle_to_base_id
    cond = CallosalConditioning(
        d_lh=d_model,
        d_reasoner=16,
        d_callosum=16,
        num_query_tokens=4,
        n_heads=2,
        constraint_type="dale_inhibitory"
    )
    callosal_emb = CallosalPuzzleEmbedding(
        cached_lh_latents=tensor,
        callosal_conditioning=cond,
        num_query_tokens=4,
        d_reasoner=16,
        flatten_output=False,
        puzzle_to_base_id=p2b
    )
    # Pass augmented puzzle indices (e.g. index 2 and index 3 should yield identical task tokens)
    out_2 = callosal_emb(torch.tensor([2]))
    out_3 = callosal_emb(torch.tensor([3]))
    assert torch.allclose(out_2, out_3, atol=1e-6)



def test_callosum_runner_helpers():
    import importlib.util
    import os
    runner_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts", "callosum_colab_runner.py"))
    spec = importlib.util.spec_from_file_location("callosum_colab_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    fp1 = runner._config_fingerprint({"lr": 1e-4, "run_name": "run1"})
    fp2 = runner._config_fingerprint({"lr": 1e-4, "run_name": "run2"})
    fp3 = runner._config_fingerprint({"lr": 2e-4, "run_name": "run1"})
    assert fp1 == fp2
    assert fp1 != fp3

    d = {"_orig_mod.weight": 1, "bias": 2}
    cleaned = runner._clean_keys(d)
    assert "weight" in cleaned and "_orig_mod.weight" not in cleaned


def test_callosal_conditioning_mixed_precision_bfloat16():
    """Verify that CallosalConditioning gracefully handles float32 latents when model is in bfloat16."""
    d_lh, d_reasoner, d_call = 64, 32, 32
    num_query = 8
    callosum = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=d_call,
        num_query_tokens=num_query,
        n_heads=2,
        constraint_type="dale_inhibitory"
    ).to(torch.bfloat16)

    # Input latents are float32 (e.g. freshly loaded from cache)
    lh_latents_f32 = torch.randn(2, 16, d_lh, dtype=torch.float32)

    # 1. condition_task must not raise RuntimeError: mat1 and mat2 must have the same dtype
    out = callosum.condition_task(lh_latents_f32)
    assert out.shape == (2, num_query, d_reasoner)
    assert out.dtype == torch.bfloat16

    # 2. condition_state must handle mixed precision
    reasoner_state_f32 = torch.randn(2, 16, d_reasoner, dtype=torch.float32)
    out_state = callosum.condition_state(reasoner_state_f32, lh_latents_f32)
    assert out_state.shape == (2, 16, d_reasoner)
    assert out_state.dtype == torch.bfloat16


def test_callosal_puzzle_embedding_safe_bounds_and_non_persistent_buffer():
    """Verify out-of-bounds puzzle IDs are safely clamped and cache buffer is non-persistent."""
    cached_latents = torch.randn(10, 16, 64)
    callosum = CallosalConditioning(d_lh=64, d_reasoner=32, d_callosum=32, num_query_tokens=8, n_heads=2)
    emb = CallosalPuzzleEmbedding(
        cached_lh_latents=cached_latents,
        callosal_conditioning=callosum,
        num_query_tokens=8,
        d_reasoner=32
    )

    # Buffer must not be in state_dict (persistent=False prevents 200MB checkpoint bloat)
    assert "cached_lh_latents" not in emb.state_dict()

    # Out-of-bounds IDs (e.g. 15 and -3 on size 10) must not crash
    pids = torch.tensor([0, 5, 15, -1])
    out = emb(pids)
    assert out.shape == (4, 8 * 32)
    assert not torch.isnan(out).any()


def test_trm_inner_model_callosal_wiring_and_param_partitioning():
    """Verify that replacing inner.puzzle_emb correctly wires Callosal bridge and partitions parameters without duplicates."""
    import torch.nn as nn
    from brain_ai.models.callosum import CallosalConditioning, CallosalPuzzleEmbedding

    # Mock TRM structure: outer ACT wrapper around inner module
    class MockTRMInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Linear(32, 64)
            self.puzzle_emb = nn.Embedding(20, 64)
            self.layers = nn.Linear(64, 64)

    class MockTRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = MockTRMInner()

        @property
        def puzzle_emb(self):
            return self.inner.puzzle_emb

        @puzzle_emb.setter
        def puzzle_emb(self, val):
            self.inner.puzzle_emb = val

    model = MockTRM()
    callosum_cond = CallosalConditioning(d_lh=64, d_reasoner=64, d_callosum=64, num_query_tokens=4, n_heads=2)
    callosal_emb = CallosalPuzzleEmbedding(torch.randn(20, 16, 64), callosum_cond, num_query_tokens=4, d_reasoner=64)

    # Wire into model
    if hasattr(model, "inner") and hasattr(model.inner, "puzzle_emb"):
        model.inner.puzzle_emb = callosal_emb
    model.puzzle_emb = callosal_emb

    assert model.inner.puzzle_emb is callosal_emb
    assert model.puzzle_emb is callosal_emb

    # Parameter partitioning logic
    callosum_param_set = set(callosum_cond.parameters())
    reasoner_params = [p for p in model.parameters() if p not in callosum_param_set]
    callosum_params = list(callosum_cond.parameters())

    # Assert mutual exclusion and exhaustiveness
    all_params = list(model.parameters())
    assert len(reasoner_params) + len(callosum_params) == len(all_params)
    assert set(reasoner_params).isdisjoint(callosum_param_set)
    assert len(callosum_params) == len(callosum_param_set)


def test_leakage_detector_catches_solution_in_prompt():
    """Verify that verify_no_test_leakage_in_cache raises ValueError when a test solution is in prompt."""
    meta = {
        "task_meta": {
            "task_leak": {
                "n_demos": 2,
                "prompt": "Here is demonstration 1: Grid (2x2):\n1 1\n1 1\nHere is output: Grid (2x2):\n9 9\n9 9"
            }
        }
    }
    # Test output grid [[9, 9], [9, 9]] is leaked into prompt
    solutions = {
        "task_leak": [[[9, 9], [9, 9]]]
    }

    with pytest.raises(ValueError, match="LEAKAGE DETECTED"):
        verify_no_test_leakage_in_cache(meta, solutions)


# ---------------------------------------------------------------------------
# Phase P3 / RQ3: System 2 Test-Time Latent Adaptation (TTA) Tests
# ---------------------------------------------------------------------------

def test_callosal_conditioning_with_delta_z():
    """Verify that condition_task correctly adds delta_z to task conditioning tokens."""
    d_lh, d_reasoner, d_call = 64, 32, 32
    num_query = 8
    callosum = CallosalConditioning(
        d_lh=d_lh,
        d_reasoner=d_reasoner,
        d_callosum=d_call,
        num_query_tokens=num_query,
        n_heads=2,
        constraint_type="dale_inhibitory"
    )
    lh = torch.randn(2, 16, d_lh)
    z0 = callosum.condition_task(lh)
    assert z0.shape == (2, num_query, d_reasoner)

    # 1. 2D delta_z: (num_query, d_reasoner)
    delta_z_2d = torch.randn(num_query, d_reasoner)
    z_adapted = callosum.condition_task(lh, delta_z=delta_z_2d)
    assert z_adapted.shape == (2, num_query, d_reasoner)
    assert torch.allclose(z_adapted, z0 + delta_z_2d.unsqueeze(0), atol=1e-5)

    # 2. 3D delta_z: (B, num_query, d_reasoner)
    delta_z_3d = torch.randn(2, num_query, d_reasoner)
    z_adapted_3d = callosum.condition_task(lh, delta_z=delta_z_3d)
    assert torch.allclose(z_adapted_3d, z0 + delta_z_3d, atol=1e-5)

    # 3. Flattened delta_z: (num_query * d_reasoner)
    delta_z_flat = torch.randn(num_query * d_reasoner)
    z_adapted_flat = callosum.condition_task(lh, delta_z=delta_z_flat)
    assert torch.allclose(z_adapted_flat, z0 + delta_z_flat.view(1, num_query, d_reasoner), atol=1e-5)


def test_callosal_puzzle_embedding_active_and_per_puzzle_delta_z():
    """Verify active_delta_z and per-puzzle delta_z routing in CallosalPuzzleEmbedding."""
    N_tasks, S_L, d_lh = 5, 16, 64
    d_reasoner, num_query = 32, 8
    cached_latents = torch.randn(N_tasks, S_L, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=32, num_query_tokens=num_query, n_heads=2)

    emb = CallosalPuzzleEmbedding(
        cached_lh_latents=cached_latents,
        callosal_conditioning=callosum,
        num_query_tokens=num_query,
        d_reasoner=d_reasoner,
        flatten_output=False
    )

    pids = torch.tensor([0, 1])
    z0 = emb(pids)

    # Test active_delta_z
    d_z = torch.randn(num_query, d_reasoner)
    emb.set_active_delta_z(d_z)
    z_active = emb(pids)
    assert torch.allclose(z_active, z0 + d_z.unsqueeze(0), atol=1e-5)

    emb.clear_active_delta_z()
    assert torch.allclose(emb(pids), z0, atol=1e-5)

    # Test per-puzzle delta_z routing
    dz_0 = torch.randn(num_query, d_reasoner)
    dz_1 = torch.randn(num_query, d_reasoner)
    emb.set_puzzle_delta_z(0, dz_0)
    emb.set_puzzle_delta_z(1, dz_1)

    z_per_puzzle = emb(pids)
    assert torch.allclose(z_per_puzzle[0], z0[0] + dz_0, atol=1e-5)
    assert torch.allclose(z_per_puzzle[1], z0[1] + dz_1, atol=1e-5)

    emb.clear_puzzle_delta_z()
    assert torch.allclose(emb(pids), z0, atol=1e-5)


def test_encode_decode_trm_grid_roundtrip():
    """Verify roundtrip sequence encoding and decoding of ARC grids."""
    from brain_ai.models.callosum import encode_grid_to_trm_seq, decode_trm_seq_to_grid

    grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    seq = encode_grid_to_trm_seq(grid, pad_r=0, pad_c=0, side=30)
    assert seq.shape == (900,)
    assert seq[0] == 3  # 1 + 2
    assert seq[1] == 4  # 2 + 2

    decoded = decode_trm_seq_to_grid(seq, side=30)
    assert decoded == grid


def test_system2_latent_adaptor_freezes_weights_and_reduces_loss():
    """Verify System2LatentAdaptor freezes model weights, optimizes delta_z, and bounds drift."""
    import torch.nn as nn
    from brain_ai.models.callosum import System2LatentAdaptor

    # Create mock reasoner with CallosalPuzzleEmbedding
    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(4, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    class MockARCReasoner(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.token_emb = nn.Embedding(12, 16)
            self.linear = nn.Linear(16, 12)

        def forward(self, batch):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_vec = self.puzzle_emb(pids).view(-1, 1, num_query * d_reasoner)
            h = self.token_emb(inps) + p_vec[:, :, :16]
            logits = self.linear(h)
            return {"logits": logits}

    model = MockARCReasoner(puzzle_emb)

    task_data = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            {"input": [[2, 2], [3, 3]], "output": [[3, 3], [2, 2]]}
        ],
        "test": [
            {"input": [[5, 6], [7, 8]]}
        ]
    }

    adaptor = System2LatentAdaptor(n_steps=30, lr=0.05, lambda_anchor=1.0, device="cpu")
    delta_z, diag = adaptor.adapt_task(model=model, task_data=task_data, puzzle_id=1)

    # 1. Base model weights must remain untouched and have no gradients
    for p in model.parameters():
        assert p.grad is None

    # 2. Diagnostics checks
    assert diag["accepted"] is True
    assert diag["final_demo_loss"] < diag["step0_loss"]
    assert diag["norm_delta_z"] > 0.0
    assert len(diag["step_losses"]) == 31

    # 3. Proximal anchor effect: higher lambda_anchor bounds delta_z to smaller norm
    adaptor_high_anchor = System2LatentAdaptor(n_steps=30, lr=0.05, lambda_anchor=50.0, device="cpu")
    delta_z_high, diag_high = adaptor_high_anchor.adapt_task(model=model, task_data=task_data, puzzle_id=1)
    assert diag_high["norm_delta_z"] < diag["norm_delta_z"]


def test_system2_weak_pareto_fallback_on_worse_loss():
    """Verify weak Pareto fallback: if demonstration loss does not improve, reverts to delta_z = 0."""
    from brain_ai.models.callosum import System2LatentAdaptor

    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(2, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    # 1. Test case: gradient steps cause severe loss overshoot / degradation
    class OvershootPredictor(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb

        def forward(self, batch):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_vec = self.puzzle_emb(pids).view(-1, 1, num_query * d_reasoner)
            logits = torch.zeros(inps.size(0), inps.size(1), 12)
            logits[:, :, 2] = 10.0 - torch.sum(p_vec ** 2)
            return {"logits": logits}

    model = OvershootPredictor(puzzle_emb)
    task_data = {
        "train": [{"input": [[0, 0]], "output": [[0, 0]]}],
        "test": [{"input": [[0, 0]]}]
    }

    adaptor = System2LatentAdaptor(n_steps=10, lr=50.0, lambda_anchor=0.0, device="cpu")
    delta_z, diag = adaptor.adapt_task(model=model, task_data=task_data, puzzle_id=0)

    # Must fall back to Step 0: delta_z = 0 and accepted = False
    assert diag["accepted"] is False
    assert torch.all(delta_z == 0.0)
    assert diag["norm_delta_z"] == 0.0
    assert diag["final_demo_loss"] == diag["step0_loss"]

    # 2. Test case: constant loss (no improvement over step 0)
    class ConstantPredictor(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb

        def forward(self, batch):
            inps = batch["inputs"]
            logits = torch.zeros(inps.size(0), inps.size(1), 12)
            logits[:, :, 2] = 5.0
            return {"logits": logits}

    model_const = ConstantPredictor(puzzle_emb)
    adaptor_const = System2LatentAdaptor(n_steps=5, lr=0.1, lambda_anchor=0.0, device="cpu")
    delta_z_c, diag_c = adaptor_const.adapt_task(model=model_const, task_data=task_data, puzzle_id=0)

    assert diag_c["accepted"] is False
    assert torch.all(delta_z_c == 0.0)
    assert diag_c["norm_delta_z"] == 0.0
    assert diag_c["final_demo_loss"] == diag_c["step0_loss"]


def test_system2_latent_adaptor_strict_leakage():
    """Verify test outputs are NEVER accessed or required by System2LatentAdaptor."""
    from brain_ai.models.callosum import System2LatentAdaptor

    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(2, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    class MockModel(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.linear = nn.Linear(16, 12)

        def forward(self, batch):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_vec = self.puzzle_emb(pids).view(-1, 1, num_query * d_reasoner)[:, :, :16]
            logits = self.linear(p_vec.expand(-1, inps.size(1), -1))
            return {"logits": logits}

    model = MockModel(puzzle_emb)

    task_clean = {
        "train": [{"input": [[1, 1]], "output": [[2, 2]]}],
        "test": [{"input": [[3, 3]]}]  # No output field!
    }
    task_with_fake_output = {
        "train": [{"input": [[1, 1]], "output": [[2, 2]]}],
        "test": [{"input": [[3, 3]], "output": [[9, 9]]}]  # Even if output present, must be ignored
    }

    adaptor = System2LatentAdaptor(n_steps=10, lr=0.01, device="cpu")
    torch.manual_seed(42)
    dz_clean, diag_clean = adaptor.adapt_task(model, task_clean, puzzle_id=0)

    torch.manual_seed(42)
    dz_with_out, diag_with_out = adaptor.adapt_task(model, task_with_fake_output, puzzle_id=0)

    assert torch.allclose(dz_clean, dz_with_out, atol=1e-7)
    assert diag_clean["step0_loss"] == diag_with_out["step0_loss"]
    assert diag_clean["final_demo_loss"] == diag_with_out["final_demo_loss"]


def test_system2_adapt_and_evaluate_submission_pipeline():
    """Verify adapt_and_evaluate_submission produces Kaggle-format submissions that score correctly."""
    from brain_ai.models.callosum import System2LatentAdaptor
    from brain_ai.eval.arc_scorer import score_submission

    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(3, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    class MockModel(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.token_emb = nn.Embedding(12, 16)
            self.head = nn.Linear(16, 12)

        def forward(self, batch):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_vec = self.puzzle_emb(pids).view(-1, 1, num_query * d_reasoner)[:, :, :16]
            h = self.token_emb(inps) + p_vec
            return {"logits": self.head(h)}

    model = MockModel(puzzle_emb)

    challenges = {
        "task_1": {
            "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
            "test": [{"input": [[2, 0]]}]
        },
        "task_2": {
            "train": [{"input": [[3, 3]], "output": [[4, 4]]}],
            "test": [{"input": [[5, 5]]}]
        }
    }
    solutions = {
        "task_1": [[[0, 2]]],
        "task_2": [[[4, 4]]]
    }

    adaptor = System2LatentAdaptor(n_steps=5, lr=0.01, device="cpu")
    results = adaptor.adapt_and_evaluate_submission(model, challenges)

    sub_s2 = results["submission_s2"]
    sub_s1 = results["submission_s1"]
    assert "task_1" in sub_s2 and "task_2" in sub_s2
    assert len(sub_s2["task_1"]) == 1
    assert "attempt_1" in sub_s2["task_1"][0]
    assert "attempt_2" in sub_s2["task_1"][0]

    rep = score_submission(sub_s2, solutions)
    assert rep.n_tasks == 2
    assert isinstance(rep.score, float)


def test_encode_trm_seq_edge_cases():
    """Verify encode_grid_to_trm_seq handles empty, oversized, negative, and clipped inputs gracefully."""
    from brain_ai.models.callosum import encode_grid_to_trm_seq
    import numpy as np

    # 1. Empty grid
    empty_seq = encode_grid_to_trm_seq([])
    assert empty_seq.shape == (900,)
    assert (empty_seq == 0).all()

    empty_nested = encode_grid_to_trm_seq([[]])
    assert empty_nested.shape == (900,)
    assert (empty_nested == 0).all()

    # 2. Oversized grid (>30x30)
    big_grid = [[1] * 35 for _ in range(40)]
    big_seq = encode_grid_to_trm_seq(big_grid)
    assert big_seq.shape == (900,)
    assert big_seq[0] == 3

    # 3. Non-square grid (3x5) with padding
    rect_grid = [[2, 3, 4, 5, 6], [7, 8, 9, 0, 1], [2, 3, 4, 5, 6]]
    rect_seq = encode_grid_to_trm_seq(rect_grid, pad_r=2, pad_c=3)
    assert rect_seq.shape == (900,)

    # 4. Color clamping: colors < 0 or > 9
    out_of_range_grid = [[-5, 15], [0, 9]]
    clamped_seq = encode_grid_to_trm_seq(out_of_range_grid)
    assert clamped_seq[0] == 2   # clipped to 0, + 2 = 2
    assert clamped_seq[1] == 11  # clipped to 9, + 2 = 11


def test_decode_trm_seq_no_uint8_underflow_and_edge_cases():
    """Verify decode_trm_seq_to_grid clips tokens < 2 to 0 rather than wrapping around to 254/255."""
    from brain_ai.models.callosum import decode_trm_seq_to_grid
    import numpy as np

    # 1. Token 0 (PAD) and token 1 (EOS) inside active bounding box
    arr = np.zeros((30, 30), dtype=np.uint8)
    arr[0, 0] = 3  # color 1
    arr[0, 1] = 0  # mispredicted PAD
    arr[1, 0] = 1  # mispredicted EOS
    arr[1, 1] = 4  # color 2

    grid = decode_trm_seq_to_grid(arr)
    assert grid is not None
    assert grid == [[1, 0], [0, 2]]  # Tokens 0 and 1 are safely clipped to 0 (no 254 or 255)
    for row in grid:
        for val in row:
            assert 0 <= val <= 9

    # 2. None and empty inputs
    assert decode_trm_seq_to_grid(None) is None
    assert decode_trm_seq_to_grid([]) is None
    assert decode_trm_seq_to_grid(np.zeros(900)) is None  # no tokens >= 2


def test_find_callosal_embedding_across_wrappers():
    """Verify find_callosal_embedding systematically resolves embedding across wrapper hierarchies."""
    from brain_ai.models.callosum import find_callosal_embedding

    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(2, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    # 1. Direct attribute
    class ModelDirect(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.puzzle_emb = e
    assert find_callosal_embedding(ModelDirect(emb)) is emb

    # 2. Wrapped in .inner
    class ModelInner(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.inner = ModelDirect(e)
    assert find_callosal_embedding(ModelInner(emb)) is emb

    # 3. Wrapped in DDP .module
    class ModelDDP(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.module = ModelDirect(e)
    assert find_callosal_embedding(ModelDDP(emb)) is emb

    # 4. Deeply wrapped in .module.inner
    class ModelDDPInner(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.module = ModelInner(e)
    assert find_callosal_embedding(ModelDDPInner(emb)) is emb


def test_callosal_puzzle_embedding_delta_z_validation_and_autograd():
    """Verify CallosalPuzzleEmbedding validates delta_z shapes and supports autograd without in-place errors."""
    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(2, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    p2b = torch.tensor([0, 1])
    emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner, puzzle_to_base_id=p2b)

    # 1. Invalid delta_z shape raises ValueError
    emb.set_active_delta_z(torch.randn(7, 3))
    with pytest.raises(ValueError, match="Shape mismatch for active_delta_z"):
        emb(torch.tensor([0]))
    emb.clear_active_delta_z()

    # 2. Autograd through puzzle_delta_z without in-place mutation errors
    dz0 = torch.randn(num_query, d_reasoner, requires_grad=True)
    emb.set_puzzle_delta_z(0, dz0)
    out = emb(torch.tensor([0, 1]))
    loss = out.sum()
    loss.backward()
    assert dz0.grad is not None
    emb.clear_puzzle_delta_z()


def test_system2_latent_adaptor_tuple_and_dict_output_variants():
    """Verify System2LatentAdaptor correctly handles diverse model forward signatures."""
    import torch.nn.functional as F
    from brain_ai.models.callosum import System2LatentAdaptor, BatchDict

    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(2, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    # Variant A: Model returns tuple (loss, carry, aux)
    class TupleModel(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.head = nn.Linear(num_query * d_reasoner, 12)

        def forward(self, batch):
            pids = batch["puzzle_identifiers"]
            p_vec = self.puzzle_emb(pids)
            logits = self.head(p_vec).unsqueeze(1).expand(-1, 900, -1)
            if "labels" in batch and batch["labels"] is not None:
                loss = F.cross_entropy(logits.view(-1, 12), batch["labels"].view(-1), ignore_index=0)
                return (loss, None, {"logits": logits})
            return (logits, None)

    model_tuple = TupleModel(puzzle_emb)
    task_data = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[1]]}]
    }

    adaptor = System2LatentAdaptor(n_steps=5, lr=0.01, device="cpu")
    dz, diag = adaptor.adapt_task(model_tuple, task_data, puzzle_id=0)
    assert diag["accepted"] is True
    assert len(diag["step_losses"]) >= 6

    # Test predict_test_pair with tuple output
    pred = adaptor.predict_test_pair(model_tuple, [[1]], puzzle_id=0, delta_z=dz)
    assert isinstance(pred, list)


def test_system2_adaptor_with_act_loss_head_recurrent_model():
    """Verify that System2LatentAdaptor works seamlessly with ACT/TRM models requiring carry and ACTLossHead signatures."""
    from brain_ai.models.callosum import System2LatentAdaptor
    d_lh, d_reasoner, num_query = 32, 16, 4
    cached_latents = torch.randn(2, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=16, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    class MockACTModel(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.token_emb = nn.Embedding(12, d_reasoner)
            self.head = nn.Linear(d_reasoner, 12)

        def initial_carry(self, batch):
            B = batch["inputs"].size(0)
            return {"steps": torch.zeros(B, dtype=torch.int32), "halted": torch.ones(B, dtype=torch.bool)}

        def forward(self, carry, batch, return_keys=()):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_vec = self.puzzle_emb(pids).view(-1, 1, num_query * d_reasoner)[:, :, :d_reasoner]
            h = self.token_emb(inps) + p_vec
            logits = self.head(h)
            outputs = {"logits": logits, "preds": torch.argmax(logits, dim=-1)}

            labels = batch.get("labels", None)
            if labels is not None:
                loss = F.cross_entropy(logits.view(-1, 12), labels.view(-1), ignore_index=0)
            else:
                loss = torch.tensor(0.0)

            new_carry = {"steps": carry["steps"] + 1, "halted": torch.ones_like(carry["halted"])}
            detached = {k: outputs[k].detach() for k in return_keys if k in outputs}
            # Matches ACTLossHead signature: (new_carry, loss, metrics, detached_outputs, all_finish)
            return new_carry, loss, {"loss": loss.detach()}, detached, True

    act_model = MockACTModel(puzzle_emb)
    task_data = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
            {"input": [[2, 2]], "output": [[3, 3]]}
        ],
        "test": [
            {"input": [[5, 6]]}
        ]
    }

    adaptor = System2LatentAdaptor(n_steps=10, lr=0.02, lambda_anchor=1.0, device="cpu")

    # 1. adapt_task must succeed with initial_carry
    dz, diag = adaptor.adapt_task(model=act_model, task_data=task_data, puzzle_id=0)
    assert diag["accepted"] is True
    assert diag["final_demo_loss"] < diag["step0_loss"]
    assert diag["norm_delta_z"] > 0.0

    # 2. predict_test_pair must succeed with initial_carry
    pred = adaptor.predict_test_pair(act_model, [[5, 6]], puzzle_id=0, delta_z=dz)
    assert isinstance(pred, list)
    assert len(pred) > 0


def test_system2_adaptor_real_trm_act_loss_head_hierarchy():
    """Strictly verify TRM ACTLossHead wrapping TinyRecursiveReasoningModel_ACTV1."""
    from dataclasses import dataclass
    from brain_ai.models.callosum import System2LatentAdaptor
    d_lh, d_reasoner, num_query = 64, 32, 16
    cached_latents = torch.randn(5, 8, d_lh)
    callosum = CallosalConditioning(d_lh=d_lh, d_reasoner=d_reasoner, d_callosum=32, num_query_tokens=num_query, n_heads=2)
    puzzle_emb = CallosalPuzzleEmbedding(cached_latents, callosum, num_query_tokens=num_query, d_reasoner=d_reasoner)

    @dataclass
    class MockInnerCarry:
        z_H: torch.Tensor
        z_L: torch.Tensor

    @dataclass
    class MockACTCarry:
        inner_carry: MockInnerCarry
        steps: torch.Tensor
        halted: torch.Tensor
        current_data: Dict[str, torch.Tensor]

    class RealTRMInner(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.puzzle_emb = emb
            self.embed_tokens = nn.Embedding(12, d_reasoner)
            self.lm_head = nn.Linear(d_reasoner, 12)
            self.q_head = nn.Linear(d_reasoner, 2)
            self.H_init = nn.Parameter(torch.randn(1, 1, d_reasoner))
            self.L_init = nn.Parameter(torch.randn(1, 1, d_reasoner))

        def empty_carry(self, batch_size: int):
            return MockInnerCarry(
                z_H=torch.empty(batch_size, 900 + num_query, d_reasoner),
                z_L=torch.empty(batch_size, 900 + num_query, d_reasoner),
            )

        def reset_carry(self, reset_flag: torch.Tensor, carry: MockInnerCarry):
            return MockInnerCarry(
                z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
                z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            )

        def forward(self, carry: MockInnerCarry, batch: Dict[str, torch.Tensor]):
            inps = batch["inputs"]
            pids = batch["puzzle_identifiers"]
            p_emb = self.puzzle_emb(pids).view(-1, num_query, d_reasoner)
            tok_emb = self.embed_tokens(inps.to(torch.int32))
            emb = torch.cat([p_emb, tok_emb], dim=1)
            # Mix puzzle embedding across sequence to simulate self-attention
            emb = emb + p_emb.mean(dim=1, keepdim=True)
            z_H = carry.z_H + emb
            z_L = carry.z_L + z_H
            logits = self.lm_head(z_H)[:, num_query:]
            q_logits = self.q_head(z_H[:, 0])
            return MockInnerCarry(z_H.detach(), z_L.detach()), logits, (q_logits[..., 0], q_logits[..., 1])

    class RealTRMACT(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.inner = RealTRMInner(emb)

        @property
        def puzzle_emb(self):
            return self.inner.puzzle_emb

        def initial_carry(self, batch: Dict[str, torch.Tensor]):
            batch_size = batch["inputs"].shape[0]
            return MockACTCarry(
                inner_carry=self.inner.empty_carry(batch_size),
                steps=torch.zeros((batch_size,), dtype=torch.int32),
                halted=torch.ones((batch_size,), dtype=torch.bool),
                current_data={k: torch.empty_like(v) for k, v in batch.items()}
            )

        def forward(self, carry: MockACTCarry, batch: Dict[str, torch.Tensor]):
            new_inner = self.inner.reset_carry(carry.halted, carry.inner_carry)
            new_steps = torch.where(carry.halted, 0, carry.steps)
            new_data = {
                k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
                for k, v in carry.current_data.items()
            }
            new_inner, logits, (q_halt, q_cont) = self.inner(new_inner, new_data)
            outputs = {"logits": logits, "q_halt_logits": q_halt, "q_continue_logits": q_cont}
            return MockACTCarry(new_inner, new_steps + 1, torch.ones_like(carry.halted), new_data), outputs

    class RealACTLossHead(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def initial_carry(self, *args, **kwargs):
            return self.model.initial_carry(*args, **kwargs)

        def forward(self, return_keys: Sequence[str], **model_kwargs):
            new_carry, outputs = self.model(**model_kwargs)
            labels = new_carry.current_data["labels"]
            loss = F.cross_entropy(outputs["logits"].reshape(-1, 12), labels.reshape(-1), ignore_index=0)
            detached = {k: outputs[k].detach() for k in return_keys if k in outputs}
            return new_carry, loss, {"loss": loss.detach()}, detached, True

    act_head = RealACTLossHead(RealTRMACT(puzzle_emb))

    task_data = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}
        ],
        "test": [
            {"input": [[1, 2]]}
        ]
    }

    adaptor = System2LatentAdaptor(n_steps=10, lr=0.02, lambda_anchor=1.0, device="cpu")
    dz, diag = adaptor.adapt_task(model=act_head, task_data=task_data, puzzle_id=0)
    assert diag["accepted"] is True
    assert diag["final_demo_loss"] < diag["step0_loss"]
    assert diag["norm_delta_z"] > 0.0

    pred = adaptor.predict_test_pair(act_head, [[1, 2]], puzzle_id=0, delta_z=dz)
    assert isinstance(pred, list)
    assert len(pred) > 0









