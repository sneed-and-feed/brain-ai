"""
tests.test_robotics_sandbox: Comprehensive test suite for Franka 7-DoF simulator,
Dynamic ER-2 stress-testing tasks, execution policies, and benchmark runner.
"""

import math
import pytest
import torch

from brain_ai.tasks.robotics_sandbox import (
    FrankaKinematics,
    FrankaKinematicEnv,
    GymnasiumMuJoCoRoboticsAdapter,
    make_robotics_env,
    Obstacle,
    ObstacleType,
    CapsuleCollisionChecker,
    ER2TaskType,
    RoboticsTaskSpec,
    create_nominal_reach_task,
    create_multi_hazard_reach_task,
    create_pick_and_place_task,
    create_obstacle_field_task,
    create_multi_hazard_obstacle_field_task,
    MonolithicVLAPolicy,
    BiHemisphericRoboticsPolicy,
    EpisodeMetrics,
    BenchmarkSummary,
    RoboticsBenchmarkRunner,
    compute_path_smoothness,
    distance_segment_to_point,
)


def test_franka_kinematics_fk_and_limits():
    """Validates Franka 7-DoF forward kinematics, joint clamping, and Jacobian."""
    q_home = FrankaKinematics.DEFAULT_HOME_Q.clone()
    p_ee, link_pos, J = FrankaKinematics.forward_kinematics(q_home)

    # 1. Verification of output shapes
    assert p_ee.shape == (3,), f"Expected p_ee shape (3,), got {p_ee.shape}"
    assert link_pos.shape == (9, 3), f"Expected link_pos shape (9, 3), got {link_pos.shape}"
    assert J.shape == (3, 7), f"Expected Jacobian shape (3, 7), got {J.shape}"

    # 2. Check reasonable home end-effector position (in front of robot, positive z)
    assert p_ee[0] > 0.25, "End effector should be forward along x"
    assert abs(p_ee[1]) < 0.05, "End effector should be near y=0 at home configuration"
    assert p_ee[2] > 0.35, "End effector should be elevated above base"

    # 3. Test joint limits clamping
    q_out_of_bounds = torch.tensor([10.0, -10.0, 5.0, 2.0, -5.0, 5.0, -5.0])
    q_clamped = FrankaKinematics.clamp_joints(q_out_of_bounds)
    assert torch.all(q_clamped >= FrankaKinematics.JOINT_LIMITS_LOWER)
    assert torch.all(q_clamped <= FrankaKinematics.JOINT_LIMITS_UPPER)


def test_franka_inverse_kinematics_convergence():
    """Validates that DLS inverse kinematics converges to reachable target."""
    q_curr = FrankaKinematics.DEFAULT_HOME_Q.clone()
    target_pos = torch.tensor([0.42, 0.12, 0.42], dtype=torch.float32)

    for _ in range(10):
        dq, p_ee, err = FrankaKinematics.inverse_kinematics_step(
            q_curr, target_pos, q_rest=FrankaKinematics.DEFAULT_HOME_Q
        )
        q_curr = FrankaKinematics.clamp_joints(q_curr + dq)
        if err < 1e-3:
            break

    p_final, _, _ = FrankaKinematics.forward_kinematics(q_curr)
    final_err = float(torch.norm(target_pos - p_final).item())
    assert final_err < 0.005, f"IK failed to converge, final error: {final_err:.5f} m"


def test_capsule_collision_checker():
    """Validates continuous segment-to-point and link capsule collision detection."""
    # Segment from (0,0,0) to (1,0,0)
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])

    # Point at (0.5, 0.1, 0.0) -> distance should be 0.1
    c1 = torch.tensor([0.5, 0.1, 0.0])
    dist1 = distance_segment_to_point(p1, p2, c1)
    assert abs(dist1 - 0.1) < 1e-4

    # Point off the end at (1.5, 0.0, 0.0) -> distance to p2 is 0.5
    c2 = torch.tensor([1.5, 0.0, 0.0])
    dist2 = distance_segment_to_point(p1, p2, c2)
    assert abs(dist2 - 0.5) < 1e-4

    # Test robot arm collision checking
    q_home = FrankaKinematics.DEFAULT_HOME_Q.clone()
    p_ee, link_pos, _ = FrankaKinematics.forward_kinematics(q_home)

    # Obstacle placed directly on end-effector
    obs_collide = Obstacle(
        name="direct_hit",
        position=p_ee.clone(),
        radius=0.05
    )
    is_col, clr, name = CapsuleCollisionChecker.check_collisions(link_pos, [obs_collide], t=0.0)
    assert is_col is True
    assert name == "direct_hit"
    assert clr <= 0.0

    # Obstacle far away
    obs_far = Obstacle(
        name="far_away",
        position=torch.tensor([5.0, 5.0, 5.0]),
        radius=0.05
    )
    is_col_far, clr_far, _ = CapsuleCollisionChecker.check_collisions(link_pos, [obs_far], t=0.0)
    assert is_col_far is False
    assert clr_far > 4.0


def test_franka_kinematic_env_lifecycle():
    """Validates reset, step, observation structure, and termination flags in FrankaKinematicEnv."""
    task = create_nominal_reach_task()
    env = FrankaKinematicEnv(task_spec=task)

    obs = env.reset()
    assert "joint_q" in obs
    assert "ee_pos" in obs
    assert "target_pos" in obs
    assert "obstacles" in obs
    assert "directive" in obs
    assert obs["step"] == 0

    # Execute a small step
    action = torch.zeros(7, dtype=torch.float32)
    next_obs, reward, done, info = env.step(action)

    assert next_obs["step"] == 1
    assert "is_collision" in info
    assert "min_clearance" in info
    assert "success" in info
    assert isinstance(done, (bool, torch.BoolTensor))


def test_gymnasium_mujoco_adapter_interface():
    """Validates Gymnasium / MuJoCo adapter detection and graceful headless fallback."""
    is_avail = GymnasiumMuJoCoRoboticsAdapter.is_available()
    assert isinstance(is_avail, bool)

    if not is_avail:
        # Should raise clean ImportError when instantiated directly
        with pytest.raises(ImportError):
            GymnasiumMuJoCoRoboticsAdapter()

        # Factory should gracefully fallback to pure-PyTorch kinematic env
        env = make_robotics_env(backend="mujoco")
        assert isinstance(env, FrankaKinematicEnv)
    else:
        env = make_robotics_env(backend="mujoco")
        assert isinstance(env, BaseRoboticsEnv)


def test_dynamic_er2_task_specs():
    """Validates the Dynamic ER-2 task specifications and dynamic hazard injection."""
    # 1. Nominal reach task
    reach_task = create_nominal_reach_task(with_hazard=True, t_hazard=0.35)
    assert reach_task.task_type == ER2TaskType.NOMINAL_REACH
    assert "Move the end-effector" in reach_task.directive
    assert reach_task.dynamic_hazard is not None
    assert reach_task.dynamic_hazard.is_dynamic is True
    assert reach_task.dynamic_hazard.t_activate == 0.35

    # Check hazard position before and after activation
    pos_before = reach_task.dynamic_hazard.get_position(t=0.20)
    pos_after = reach_task.dynamic_hazard.get_position(t=0.50)
    assert torch.equal(pos_before, reach_task.dynamic_hazard.position)
    assert not torch.equal(pos_after, reach_task.dynamic_hazard.position)

    # 2. Pick and place task
    pp_task = create_pick_and_place_task()
    assert pp_task.task_type == ER2TaskType.PICK_AND_PLACE
    assert pp_task.pick_pos is not None
    assert pp_task.place_pos is not None

    # 3. Obstacle field task
    of_task = create_obstacle_field_task(with_hazard=False)
    assert of_task.task_type == ER2TaskType.OBSTACLE_FIELD
    assert len(of_task.static_obstacles) >= 2


def test_monolithic_vla_policy_latency_and_hazard_vulnerability():
    """
    Validates that the Baseline Monolithic VLA policy:
    1. Exhibits ~500ms deliberation latency on re-planning ticks.
    2. Suffers collision against a dynamic hazard due to the 500ms re-planning blind spot.
    """
    task = create_nominal_reach_task(with_hazard=True, t_hazard=0.25)
    env = FrankaKinematicEnv(task_spec=task)
    policy = MonolithicVLAPolicy(replan_latency_s=0.500)

    runner = RoboticsBenchmarkRunner()
    metrics = runner.run_episode(env, policy, task)

    assert metrics.mean_decision_latency_ms > 10.0, "Should simulate heavy deliberation latency"
    assert metrics.collision is True, "Monolithic VLA should collide with moving hazard during 500ms delay"
    assert metrics.collision_steps > 0


def test_bi_hemispheric_policy_reflex_bypass_and_evasion():
    """
    Validates that the Bi-Hemispheric policy:
    1. Evaluates Amygdalar salience with sub-15ms latency.
    2. Triggers Reflex Bypass when dynamic hazard approaches.
    3. Successfully evades collision and completes the reach task.
    """
    task = create_nominal_reach_task(with_hazard=True, t_hazard=0.25)
    env = FrankaKinematicEnv(task_spec=task)
    policy = BiHemisphericRoboticsPolicy()

    runner = RoboticsBenchmarkRunner()
    metrics = runner.run_episode(env, policy, task)

    # Sub-15ms Amygdalar decision latency
    assert metrics.mean_decision_latency_ms < 15.0, (
        f"Expected sub-15ms latency, got {metrics.mean_decision_latency_ms:.2f} ms"
    )
    # Zero collision rate
    assert metrics.collision is False, "Bi-Hemispheric reflex should dodge dynamic hazard"
    assert metrics.collision_steps == 0
    assert metrics.reflex_activations > 0, "Amygdalar reflex bypass should have activated"
    assert metrics.success is True, "Should successfully reach target after dodging hazard"


def test_path_smoothness_and_metrics_logging():
    """Validates trajectory jerk and path smoothness computation."""
    dt = 0.01

    # 1. Perfectly straight smooth trajectory
    smooth_traj = [
        torch.tensor([float(i) * 0.01, 0.0, 0.0], dtype=torch.float32)
        for i in range(50)
    ]
    path_len, jerk_sq, score_smooth = compute_path_smoothness(smooth_traj, dt=dt)
    assert abs(path_len - 0.49) < 1e-4
    assert jerk_sq < 1e-2, f"Constant velocity should have negligible jerk, got {jerk_sq}"
    assert score_smooth > 0.99

    # 2. Jerky trajectory with oscillations
    jerky_traj = []
    for i in range(50):
        y_osc = 0.05 * math.sin(i * 1.5)
        jerky_traj.append(torch.tensor([float(i) * 0.01, y_osc, 0.0], dtype=torch.float32))

    _, jerk_sq_noisy, score_noisy = compute_path_smoothness(jerky_traj, dt=dt)
    assert jerk_sq_noisy > jerk_sq
    assert score_noisy < score_smooth


def test_robotics_benchmark_runner_full_suite():
    """
    Runs full benchmark runner comparing Monolithic VLA vs Bi-Hemispheric System,
    verifying metrics summary, collision rate disparity, latency, and report generation.
    """
    tasks = [
        create_nominal_reach_task(with_hazard=False),
        create_nominal_reach_task(with_hazard=True, t_hazard=0.25),
        create_obstacle_field_task(with_hazard=True, t_hazard=0.25)
    ]

    policies = {
        "Monolithic_VLA": MonolithicVLAPolicy(replan_latency_s=0.500),
        "BiHemispheric": BiHemisphericRoboticsPolicy()
    }

    runner = RoboticsBenchmarkRunner()
    summaries = runner.run_benchmark(tasks, policies, num_trials=2)

    assert "Monolithic_VLA" in summaries
    assert "BiHemispheric" in summaries

    mono_sum = summaries["Monolithic_VLA"]
    bi_sum = summaries["BiHemispheric"]

    # Bi-Hemispheric policy must have significantly lower collision rate
    assert bi_sum.collision_rate < mono_sum.collision_rate, (
        f"Bi-Hemispheric collision ({bi_sum.collision_rate}%) must be lower than Monolithic ({mono_sum.collision_rate}%)"
    )

    # Bi-Hemispheric must demonstrate sub-15ms decision latency
    assert bi_sum.mean_latency_ms < 15.0, (
        f"Bi-Hemispheric latency must be sub-15ms, got {bi_sum.mean_latency_ms:.2f} ms"
    )

    # Monolithic has higher deliberation latency
    assert mono_sum.mean_latency_ms > bi_sum.mean_latency_ms

    # Check comparison table formatting
    table = runner.format_comparison_table(summaries)
    assert "| Policy |" in table
    assert "Monolithic_VLA" in table
    assert "BiHemispheric" in table


def test_franka_yoshikawa_adaptive_damping_at_singularity():
    """
    Validates FrankaKinematics Yoshikawa manipulability computation and adaptive DLS
    pseudo-inverse near singular configurations.
    """
    q_home = FrankaKinematics.DEFAULT_HOME_Q.clone()
    _, _, J_home = FrankaKinematics.forward_kinematics(q_home)
    w_home = FrankaKinematics.yoshikawa_manipulability(J_home)
    assert w_home > 0.05, f"Nominal home configuration should have high manipulability, got {w_home}"

    # Stretched singular configuration (q3=0, elbow fully straight)
    q_singular = torch.zeros(7, dtype=torch.float32)
    _, _, J_sing = FrankaKinematics.forward_kinematics(q_singular)
    w_sing = FrankaKinematics.yoshikawa_manipulability(J_sing)
    assert w_sing < w_home, f"Singular config should have lower manipulability ({w_sing} < {w_home})"

    # Inversion test at singularity
    J_pinv = FrankaKinematics.damped_least_squares_pinv(J_sing, lambda_dls=1e-2, w_threshold=0.03, lambda_max=0.20)
    assert J_pinv.shape == (7, 3)
    assert not torch.isnan(J_pinv).any(), "Singular pseudo-inverse produced NaNs!"
    assert not torch.isinf(J_pinv).any(), "Singular pseudo-inverse produced Infs!"
    assert torch.max(torch.abs(J_pinv)).item() < 30.0, "Singular pseudo-inverse norm blew up!"


def test_multi_hazard_reach_task_protocol():
    """
    Validates the multi-hazard reach task protocol with 3 intersecting hazards
    and speeds ranging from 0.2 m/s to 1.2 m/s.
    """
    speeds = [0.35, 0.75, 1.10]
    task = create_multi_hazard_reach_task(hazard_speeds=speeds, t_hazards=[0.15, 0.30, 0.45])

    assert task.task_type == ER2TaskType.NOMINAL_REACH
    assert len(task.dynamic_hazards) == 3
    assert len(task.all_obstacles) == 3

    # Check speeds
    for i, speed in enumerate(speeds):
        h = task.dynamic_hazards[i]
        assert h.is_dynamic is True
        v_mag = float(torch.norm(h.velocity).item())
        assert math.isclose(v_mag, speed, rel_tol=0.05)


def test_multi_hazard_obstacle_field_task_protocol():
    """Validates multi-hazard obstacle field with static barriers and dynamic hazards."""
    task = create_multi_hazard_obstacle_field_task()
    assert task.task_type == ER2TaskType.OBSTACLE_FIELD
    assert len(task.static_obstacles) == 2
    assert len(task.dynamic_hazards) == 2
    assert len(task.all_obstacles) == 4


def test_bi_hemispheric_policy_multi_hazard_evasion():
    """
    Validates that BiHemisphericRoboticsPolicy successfully dodges multiple intersecting
    hazards with zero collisions and sub-15ms decision latency.
    """
    task = create_multi_hazard_reach_task(hazard_speeds=[0.40, 0.80], t_hazards=[0.20, 0.35])
    env = FrankaKinematicEnv(task_spec=task)
    policy = BiHemisphericRoboticsPolicy()

    runner = RoboticsBenchmarkRunner()
    metrics = runner.run_episode(env, policy, task)

    assert metrics.collision is False, "Bi-Hemispheric policy should evade all multi-hazard obstacles"
    assert metrics.collision_steps == 0
    assert metrics.reflex_activations > 0, "Reflex bypass should have triggered"
    assert metrics.mean_decision_latency_ms < 15.0
    assert metrics.success is True


def test_monolithic_vla_multi_hazard_failure():
    """
    Validates that MonolithicVLAPolicy collides when subjected to multi-hazard
    dynamic obstacles due to deliberation latency blind spots.
    """
    task = create_multi_hazard_reach_task(hazard_speeds=[0.40, 0.80], t_hazards=[0.20, 0.35])
    env = FrankaKinematicEnv(task_spec=task)
    policy = MonolithicVLAPolicy(replan_latency_s=0.500)

    runner = RoboticsBenchmarkRunner()
    metrics = runner.run_episode(env, policy, task)

    assert metrics.collision is True, "Monolithic VLA should collide under multi-hazard conditions"
    assert metrics.collision_steps > 0

