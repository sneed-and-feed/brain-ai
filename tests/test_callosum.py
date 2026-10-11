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






