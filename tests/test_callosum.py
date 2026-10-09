"""
Tests for brain_ai modules:
- Dale's principle sign segregation
- Homeostatic synaptic scaling
- DaleDifferentialCrossAttention
- OpenJevAmygdalarRouter & NeuromodulatoryController
- HierarchicalReasoningModel
- BiHemisphericBrain integration
"""

import pytest
import torch
from brain_ai.models.callosum import (
    DaleLinear,
    HomeostaticSynapticScaling,
    DaleDifferentialCrossAttention,
    InterHemisphericLatentCoupling
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

