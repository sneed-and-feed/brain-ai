"""
Unit tests for 1D Temporal CNN Diffusion Policy.
Tests forward diffusion loss, DDIM fast sampling, Franka kinematic env integration,
training helper, and checkpointing.
"""

import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from brain_ai.models.diffusion_policy import (
    SinusoidalPosEmb,
    Conv1dBlock,
    ConditionalResidualBlock1D,
    ConditionalUnet1D,
    DiffusionModel1D,
    DiffusionRoboticsPolicy,
    extract_obs_tensor,
    train_diffusion_policy,
    save_diffusion_checkpoint,
    load_diffusion_checkpoint
)
from brain_ai.tasks.robotics_sandbox import (
    FrankaKinematics,
    FrankaKinematicEnv,
    create_nominal_reach_task
)


# ==============================================================================
# 1. OBSERVATION EXTRACTION TESTS
# ==============================================================================

def test_extract_obs_tensor_from_dict():
    """Verifies that 21-dim proprioceptive state is correctly extracted from env dict."""
    env = FrankaKinematicEnv(create_nominal_reach_task())
    obs = env.reset()

    # Verify keys
    assert "joint_q" in obs
    assert "joint_dq" in obs
    assert "ee_pos" in obs
    assert "target_pos" in obs
    assert "gripper_width" in obs

    # Extract 21-dim tensor
    obs_tensor = extract_obs_tensor(obs)
    assert isinstance(obs_tensor, torch.Tensor)
    assert obs_tensor.shape == (21,)
    assert obs_tensor.dtype == torch.float32

    # Check slice components: 7 + 7 + 3 + 3 + 1 = 21
    assert torch.allclose(obs_tensor[0:7], obs["joint_q"])
    assert torch.allclose(obs_tensor[7:14], obs["joint_dq"])
    assert torch.allclose(obs_tensor[14:17], obs["ee_pos"])
    assert torch.allclose(obs_tensor[17:20], obs["target_pos"])
    assert abs(obs_tensor[20].item() - obs["gripper_width"]) < 1e-4


def test_extract_obs_tensor_batched():
    """Verifies extraction and passthrough for batched tensors and dicts."""
    B = 4
    joint_q = torch.randn(B, 7)
    joint_dq = torch.randn(B, 7)
    ee_pos = torch.randn(B, 3)
    target_pos = torch.randn(B, 3)
    gripper_width = torch.tensor([0.08] * B)

    obs_dict = {
        "joint_q": joint_q,
        "joint_dq": joint_dq,
        "ee_pos": ee_pos,
        "target_pos": target_pos,
        "gripper_width": gripper_width
    }

    obs_tensor = extract_obs_tensor(obs_dict)
    assert obs_tensor.shape == (B, 21)

    # Passthrough for already-extracted tensor
    raw_tensor = torch.randn(B, 21)
    passed = extract_obs_tensor(raw_tensor)
    assert passed.shape == (B, 21)
    assert torch.allclose(raw_tensor, passed)


# ==============================================================================
# 2. ARCHITECTURE & COMPONENT TESTS
# ==============================================================================

def test_sinusoidal_pos_emb():
    """Verifies sinusoidal time embedding outputs expected dimensions."""
    time_emb = SinusoidalPosEmb(dim=64)
    timesteps = torch.tensor([0, 10, 50, 99], dtype=torch.long)
    emb = time_emb(timesteps)
    assert emb.shape == (4, 64)
    assert not torch.isnan(emb).any()


def test_conditional_residual_block_1d():
    """Verifies 1D residual block with FiLM conditioning and dilation."""
    B, C_in, C_out, L, cond_dim = 2, 32, 64, 16, 128
    block = ConditionalResidualBlock1D(
        in_channels=C_in,
        out_channels=C_out,
        cond_dim=cond_dim,
        kernel_size=5,
        dilation=2
    )

    x = torch.randn(B, C_in, L)
    cond = torch.randn(B, cond_dim)
    out = block(x, cond)

    # Sequence length must be strictly preserved
    assert out.shape == (B, C_out, L)
    assert not torch.isnan(out).any()


def test_conditional_unet_1d_forward_and_backward():
    """Verifies U-Net forward pass, backward pass, and parameter count."""
    unet = ConditionalUnet1D(
        act_dim=7,
        obs_dim=21,
        pred_horizon=16,
        down_dims=(32, 64, 128),
        cond_dim=64,
        time_dim=32,
        dilations=(1, 2, 4)
    )

    B = 2
    noisy_actions = torch.randn(B, 16, 7)
    timesteps = torch.randint(0, 100, (B,))
    obs = torch.randn(B, 21)

    pred_noise = unet(noisy_actions, timesteps, obs)
    assert pred_noise.shape == (B, 16, 7)

    # Backward pass
    loss = pred_noise.sum()
    loss.backward()

    # Ensure all parameters have valid non-zero gradients
    for name, param in unet.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    total_params = sum(p.numel() for p in unet.parameters())
    assert total_params > 100_000


# ==============================================================================
# 3. FORWARD DIFFUSION & LOSS COMPUTATION TESTS
# ==============================================================================

def test_forward_diffusion_loss_computation():
    """Tests forward diffusion noise injection and MSE loss computation."""
    model = DiffusionModel1D(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        down_dims=(32, 64, 128),
        cond_dim=64,
        num_train_timesteps=50,
        beta_schedule="squaredcos_cap_v2"
    )

    B = 4
    clean_actions = torch.randn(B, 16, 7)
    obs = torch.randn(B, 21)

    loss = model.compute_loss(clean_actions, obs)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar loss
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    # Backpropagation verification
    loss.backward()
    grad_norm = model.unet.in_conv.weight.grad.norm().item()
    assert grad_norm > 0.0


def test_forward_diffusion_loss_with_env_obs_dict():
    """Tests that compute_loss accepts observation dict directly from FrankaKinematicEnv."""
    model = DiffusionModel1D(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        down_dims=(32, 64, 128),
        cond_dim=64,
        num_train_timesteps=50
    )

    env = FrankaKinematicEnv(create_nominal_reach_task())
    obs = env.reset()
    clean_actions = torch.randn(1, 16, 7)

    loss = model.compute_loss(clean_actions, obs)
    assert loss.dim() == 0
    assert loss.item() > 0.0


# ==============================================================================
# 4. DDIM FAST SAMPLING TESTS (K=8, 10, 16 STEPS)
# ==============================================================================

@pytest.mark.parametrize("k_steps", [8, 10, 16])
def test_ddim_fast_sampling_steps(k_steps):
    """Verifies fast DDIM sampling with K=8, K=10, and K=16 denoising steps."""
    model = DiffusionModel1D(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        down_dims=(32, 64, 128),
        cond_dim=64,
        num_train_timesteps=50,
        clip_sample=True,
        clip_val=2.6
    )

    B = 2
    obs = torch.randn(B, 21)

    sampled = model.ddim_sample(obs, num_inference_steps=k_steps, eta=0.0)
    assert sampled.shape == (B, 16, 7)
    assert not torch.isnan(sampled).any()

    # Check that clipped actions stay within reasonable bounds
    assert torch.all(sampled >= -3.0)
    assert torch.all(sampled <= 3.0)


def test_ddim_fast_sampling_determinism():
    """Tests that DDIM sampler with eta=0.0 is deterministic given fixed input and generator."""
    model = DiffusionModel1D(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        down_dims=(32, 64, 128),
        cond_dim=64,
        num_train_timesteps=50
    )

    obs = torch.randn(1, 21)
    g1 = torch.Generator().manual_seed(42)
    sample1 = model.ddim_sample(obs, num_inference_steps=10, eta=0.0, generator=g1)

    g2 = torch.Generator().manual_seed(42)
    sample2 = model.ddim_sample(obs, num_inference_steps=10, eta=0.0, generator=g2)

    assert torch.allclose(sample1, sample2, atol=1e-5)


# ==============================================================================
# 5. DIFFUSION ROBOTICS POLICY & FRANKA KINEMATIC ENV INTEGRATION
# ==============================================================================

def test_diffusion_robotics_policy_franka_interface():
    """
    Tests DiffusionRoboticsPolicy interacting with FrankaKinematicEnv across
    multiple execution horizon cycles (T_a = 8, T_p = 16).
    """
    env = FrankaKinematicEnv(create_nominal_reach_task())
    policy = DiffusionRoboticsPolicy(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        exec_horizon=8,
        num_ddim_steps=8,
        simulated_latency_ms=50.0,
        down_dims=(32, 64, 128),
        cond_dim=64,
        num_train_timesteps=50
    )

    obs = env.reset()
    policy.reset()

    # Track replanning events across 18 simulation steps
    replan_events = []
    latencies = []

    for step in range(18):
        action, telemetry = policy.compute_action(obs, env.t)

        assert action.shape == (7,)
        assert isinstance(telemetry, dict)
        assert telemetry["policy"] == "DiffusionRoboticsPolicy"

        # Action velocity limits
        max_vel = FrankaKinematics.MAX_JOINT_VELOCITY
        assert torch.all(action >= -max_vel)
        assert torch.all(action <= max_vel)

        replan_events.append(telemetry["replan_triggered"])
        latencies.append(telemetry["decision_latency_ms"])

        obs, reward, done, info = env.step(action)
        assert not torch.isnan(obs["joint_q"]).any()

    # Step 0: replan triggered (buffer empty)
    assert replan_events[0] is True
    assert latencies[0] >= 50.0  # Simulated decision latency

    # Steps 1 to 7: cached chunk execution (replan False, low latency)
    for s in range(1, 8):
        assert replan_events[s] is False
        assert latencies[s] < 5.0

    # Step 8: execution horizon T_a reached -> replan triggered
    assert replan_events[8] is True
    assert latencies[8] >= 50.0

    # Steps 9 to 15: cached chunk execution
    for s in range(9, 16):
        assert replan_events[s] is False

    # Step 16: next execution horizon -> replan triggered
    assert replan_events[16] is True


def test_policy_clear_buffer():
    """Tests that clear_buffer immediately triggers replanning on next call."""
    policy = DiffusionRoboticsPolicy(
        down_dims=(32, 64, 128),
        num_train_timesteps=50,
        num_ddim_steps=5
    )
    obs = {"joint_q": torch.zeros(7), "joint_dq": torch.zeros(7), "ee_pos": torch.zeros(3), "target_pos": torch.zeros(3)}

    _, info0 = policy.compute_action(obs, 0.0)
    assert info0["replan_triggered"] is True

    _, info1 = policy.compute_action(obs, 0.01)
    assert info1["replan_triggered"] is False

    # Clear buffer manually
    policy.clear_buffer()
    _, info2 = policy.compute_action(obs, 0.02)
    assert info2["replan_triggered"] is True


# ==============================================================================
# 6. TRAINING HELPER & CHECKPOINTING TESTS
# ==============================================================================

def test_train_diffusion_policy():
    """Tests training helper function on demonstration dataset."""
    policy = DiffusionRoboticsPolicy(
        down_dims=(32, 64, 128),
        num_train_timesteps=50,
        num_ddim_steps=5
    )

    N = 16
    dataset = {
        "obs": torch.randn(N, 21),
        "actions": torch.randn(N, 16, 7)
    }

    losses = train_diffusion_policy(
        policy=policy,
        dataset=dataset,
        epochs=3,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4
    )

    assert len(losses) == 3
    assert all(isinstance(loss, float) for loss in losses)
    assert all(loss > 0.0 for loss in losses)


def test_checkpoint_save_and_load():
    """Tests serialization and deserialization of policy checkpoints."""
    policy = DiffusionRoboticsPolicy(
        obs_dim=21,
        act_dim=7,
        pred_horizon=16,
        exec_horizon=8,
        num_ddim_steps=5,
        simulated_latency_ms=45.0,
        down_dims=(32, 64, 128),
        cond_dim=64,
        num_train_timesteps=50
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "diffusion_test.pt"

        # Save checkpoint
        saved_path = save_diffusion_checkpoint(policy, ckpt_path)
        assert Path(saved_path).exists()

        # Load into new policy instance
        loaded_policy = load_diffusion_checkpoint(saved_path)
        assert isinstance(loaded_policy, DiffusionRoboticsPolicy)
        assert loaded_policy.obs_dim == policy.obs_dim
        assert loaded_policy.act_dim == policy.act_dim
        assert loaded_policy.pred_horizon == policy.pred_horizon
        assert loaded_policy.exec_horizon == policy.exec_horizon
        assert loaded_policy.simulated_latency_ms == policy.simulated_latency_ms

        # Fixed generator sampling comparison
        obs = torch.randn(1, 21)
        g1 = torch.Generator().manual_seed(999)
        out_orig = policy.model.ddim_sample(obs, num_inference_steps=5, generator=g1)

        g2 = torch.Generator().manual_seed(999)
        out_loaded = loaded_policy.model.ddim_sample(obs, num_inference_steps=5, generator=g2)

        assert torch.allclose(out_orig, out_loaded, atol=1e-5)
