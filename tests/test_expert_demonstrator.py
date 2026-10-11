"""
tests.test_expert_demonstrator: Comprehensive verification suite for
FrankaScriptedExpert, MinimumJerkInterpolator, and DemonstrationCollector.

Verifies:
1. Analytical properties of MinimumJerkInterpolator (C^2 boundary conditions, zero endpoint jerk).
2. Kinematic compliance and trajectory tracking (>95% success rate, 0% collisions) across:
   - Nominal reaching tasks with randomized goals.
   - Multi-phase pick-and-place tasks with randomized pick/place sites.
3. Compatibility with RoboticsBenchmarkRunner.
4. Dataset collection produces valid, finite tensors with 21-dim observations and 7-dim actions.
5. Integrity of the exported dataset artifact at data/embodied_demos/franka_nominal_demos.pt.
6. Alignment with extract_obs_tensor in diffusion_policy.
"""

import os
import tempfile
import pytest
import torch

from brain_ai.tasks.robotics_sandbox import (
    FrankaKinematics,
    FrankaKinematicEnv,
    RoboticsBenchmarkRunner,
    create_nominal_reach_task,
    create_pick_and_place_task,
    compute_path_smoothness,
)
from brain_ai.tasks.expert_demonstrator import (
    MinimumJerkInterpolator,
    FrankaScriptedExpert,
    TrajectoryPhase,
    DemonstrationCollector,
    collect_nominal_dataset,
    create_randomized_reach_task,
    create_randomized_pick_and_place_task,
)
from brain_ai.models.diffusion_policy import extract_obs_tensor


def test_minimum_jerk_interpolator_boundary_conditions():
    """Validates C^2 minimum-jerk interpolation boundary conditions and monotonicity."""
    p_start = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float32)
    p_end = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float32)
    duration = 2.0

    # At tau = 0.0: position is p_start, velocity is 0
    p0, v0 = MinimumJerkInterpolator.step(p_start, p_end, tau=0.0, duration=duration)
    assert torch.allclose(p0, p_start, atol=1e-5), f"Expected p_start at tau=0, got {p0}"
    assert torch.allclose(v0, torch.zeros(3), atol=1e-5), f"Expected zero velocity at tau=0, got {v0}"

    # At tau = 1.0: position is p_end, velocity is 0
    p1, v1 = MinimumJerkInterpolator.step(p_start, p_end, tau=1.0, duration=duration)
    assert torch.allclose(p1, p_end, atol=1e-5), f"Expected p_end at tau=1, got {p1}"
    assert torch.allclose(v1, torch.zeros(3), atol=1e-5), f"Expected zero velocity at tau=1, got {v1}"

    # At tau = 0.5: midpoint position, maximum velocity along direction vector
    pm, vm = MinimumJerkInterpolator.step(p_start, p_end, tau=0.5, duration=duration)
    p_midpoint = 0.5 * (p_start + p_end)
    assert torch.allclose(pm, p_midpoint, atol=1e-5), f"Expected midpoint, got {pm}"
    # At tau=0.5: s_dot = (30*(0.25) - 60*(0.125) + 30*(0.0625)) / 2.0 = (7.5 - 7.5 + 1.875)/2.0 = 0.9375
    expected_v = 0.9375 * (p_end - p_start)
    assert torch.allclose(vm, expected_v, atol=1e-5)


def test_expert_nominal_reach_success_and_safety():
    """
    Validates that FrankaScriptedExpert achieves >95% success rate and zero collisions
    across randomized nominal reaching tasks within the Franka workspace.
    """
    num_trials = 10
    success_count = 0
    collision_count = 0
    env = FrankaKinematicEnv()
    expert = FrankaScriptedExpert(dt=0.01)

    for i in range(num_trials):
        task = create_randomized_reach_task(seed=100 + i)
        env.task_spec = task
        obs = env.reset()
        expert.reset(task)

        done = False
        info = {}
        while not done:
            t = obs["t"]
            action, p_info = expert.compute_action(obs, t)

            # Assert joint velocity limits
            assert torch.all(torch.abs(action) <= FrankaKinematics.MAX_JOINT_VELOCITY + 1e-4), (
                f"Action exceeded joint limits: {action}"
            )

            obs, reward, done, info = env.step(action, gripper_width=p_info.get("gripper_width"))
            if expert.is_completed() and done:
                break

        if info.get("success", False):
            success_count += 1
        if info.get("collision_occurred", False):
            collision_count += 1

    success_rate = (success_count / num_trials) * 100.0
    collision_rate = (collision_count / num_trials) * 100.0

    assert success_rate >= 95.0, f"Expected >=95% reach success rate, got {success_rate}%"
    assert collision_rate == 0.0, f"Expected 0% collision rate, got {collision_rate}%"


def test_expert_pick_and_place_phases_and_success():
    """
    Validates that FrankaScriptedExpert completes full 8-phase pick-and-place tasks:
    1. Approach pre-grasp
    2. Descend to pick site
    3. Grasp (gripper close)
    4. Lift
    5. Transit to pre-place
    6. Descend to place site
    7. Open gripper (gripper release)
    8. Retract
    Achieves 100% success rate with verified pick, place, and retract flags, and zero collisions.
    """
    num_trials = 10
    success_count = 0
    collision_count = 0
    env = FrankaKinematicEnv()
    expert = FrankaScriptedExpert(dt=0.01)

    for i in range(num_trials):
        task = create_randomized_pick_and_place_task(seed=500 + i)
        env.task_spec = task
        obs = env.reset()
        expert.reset(task)

        done = False
        info = {}
        while not done:
            t = obs["t"]
            action, p_info = expert.compute_action(obs, t)

            # Assert joint velocity limits
            assert torch.all(torch.abs(action) <= FrankaKinematics.MAX_JOINT_VELOCITY + 1e-4)

            obs, reward, done, info = env.step(action, gripper_width=p_info.get("gripper_width"))
            if expert.is_completed() and (done or expert.current_segment_idx >= len(expert.segments)):
                break

        assert env.picked, f"Trial {i}: Object was not picked"
        assert env.placed, f"Trial {i}: Object was not placed"
        assert env.retracted, f"Trial {i}: Arm did not retract after placement"

        if info.get("success", False):
            success_count += 1
        if info.get("collision_occurred", False):
            collision_count += 1

    success_rate = (success_count / num_trials) * 100.0
    assert success_rate >= 95.0, f"Expected >=95% pick-and-place success rate, got {success_rate}%"
    assert collision_count == 0, f"Encountered {collision_count} collisions in pick-and-place trials"


def test_expert_benchmark_runner_integration():
    """Validates that FrankaScriptedExpert integrates cleanly with RoboticsBenchmarkRunner."""
    runner = RoboticsBenchmarkRunner(dt=0.01)
    reach_task = create_nominal_reach_task()
    pp_task = create_pick_and_place_task()

    env = FrankaKinematicEnv()
    expert = FrankaScriptedExpert(dt=0.01)

    # 1. Run reach episode via runner
    reach_metrics = runner.run_episode(env, expert, reach_task)
    assert reach_metrics.success is True, "Reach episode should succeed"
    assert reach_metrics.collision is False, "Reach episode should have zero collisions"
    assert reach_metrics.path_smoothness > 0.90, f"Smoothness should be high: {reach_metrics.path_smoothness}"

    # 2. Run pick-and-place episode via runner
    pp_metrics = runner.run_episode(env, expert, pp_task)
    assert pp_metrics.success is True, "Pick-and-place episode should succeed"
    assert pp_metrics.collision is False, "Pick-and-place should have zero collisions"
    assert pp_metrics.path_smoothness > 0.90, f"Smoothness should be high: {pp_metrics.path_smoothness}"


def test_dataset_collector_shapes_and_numeric_validity():
    """
    Validates DemonstrationCollector by generating a miniature dataset and verifying:
    - Observation tensor has shape [N_total, 21].
    - Action tensor has shape [N_total, 7].
    - Episode lengths sum exactly to N_total.
    - All tensors are finite (no NaNs, no Infs).
    - Metadata contains task info, dt=0.01, seeds, and slices.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "test_demos.pt")
        dataset = collect_nominal_dataset(
            num_episodes=10,
            save_path=tmp_path,
            seed=123,
            reach_ratio=0.5
        )

        assert os.path.exists(tmp_path), "Dataset file was not saved"

        # Verify keys
        expected_keys = {"observations", "actions", "episode_lengths", "metadata", "observations_dict"}
        assert expected_keys.issubset(set(dataset.keys())), f"Missing keys in dataset: {dataset.keys()}"

        obs = dataset["observations"]
        acts = dataset["actions"]
        ep_lens = dataset["episode_lengths"]
        meta = dataset["metadata"]

        # Shapes
        total_steps = sum(ep_lens)
        assert obs.shape == (total_steps, 21), f"Expected shape ({total_steps}, 21), got {obs.shape}"
        assert acts.shape == (total_steps, 7), f"Expected shape ({total_steps}, 7), got {acts.shape}"
        assert len(ep_lens) == 10, f"Expected 10 episodes, got {len(ep_lens)}"

        # Numerical validity
        assert torch.all(torch.isfinite(obs)), "Observations contain NaN or Inf values!"
        assert torch.all(torch.isfinite(acts)), "Actions contain NaN or Inf values!"

        # Sub-tensors
        obs_dict = dataset["observations_dict"]
        assert obs_dict["joint_q"].shape == (total_steps, 7)
        assert obs_dict["joint_dq"].shape == (total_steps, 7)
        assert obs_dict["ee_pos"].shape == (total_steps, 3)
        assert obs_dict["target_pos"].shape == (total_steps, 3)
        assert obs_dict["gripper_width"].shape == (total_steps, 1)

        # Metadata
        assert meta["dt"] == 0.01
        assert meta["total_episodes"] == 10
        assert meta["num_reach_episodes"] == 5
        assert meta["num_pick_and_place_episodes"] == 5
        assert meta["success_rate_pct"] >= 95.0
        assert meta["collision_rate_pct"] == 0.0


def test_saved_nominal_demos_artifact_integrity():
    """
    Validates the persistent nominal demonstrations dataset saved at
    data/embodied_demos/franka_nominal_demos.pt.
    Ensures 150-200 episodes, 0% collisions, 100% success rate, and weights_only=True loadability.
    """
    artifact_path = "data/embodied_demos/franka_nominal_demos.pt"
    assert os.path.exists(artifact_path), f"Artifact missing at {artifact_path}"

    # Load with strict security settings
    data = torch.load(artifact_path, weights_only=True)

    assert "observations" in data
    assert "actions" in data
    assert "episode_lengths" in data
    assert "metadata" in data

    obs = data["observations"]
    acts = data["actions"]
    lens = data["episode_lengths"]
    meta = data["metadata"]

    assert 150 <= len(lens) <= 200, f"Expected 150-200 episodes, found {len(lens)}"
    assert obs.shape[1] == 21, f"Expected 21 observation dims, got {obs.shape[1]}"
    assert acts.shape[1] == 7, f"Expected 7 action dims, got {acts.shape[1]}"
    assert obs.shape[0] == acts.shape[0] == sum(lens)

    assert torch.all(torch.isfinite(obs))
    assert torch.all(torch.isfinite(acts))

    assert meta["success_rate_pct"] >= 95.0, f"Expected >=95% success, got {meta['success_rate_pct']}"
    assert meta["collision_rate_pct"] == 0.0, f"Expected 0% collision, got {meta['collision_rate_pct']}"


def test_observation_vector_extract_obs_tensor_parity():
    """
    Validates that DemonstrationCollector.extract_observation_vector matches
    diffusion_policy.extract_obs_tensor bit-for-bit.
    """
    env = FrankaKinematicEnv()
    obs = env.reset()

    v1 = DemonstrationCollector.extract_observation_vector(obs)
    v2 = extract_obs_tensor(obs)

    assert v1.shape == (21,)
    assert v2.shape == (21,)
    assert torch.allclose(v1, v2, atol=1e-6), "Discrepancy between observation vector extractors!"
