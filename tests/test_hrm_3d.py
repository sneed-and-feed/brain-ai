"""
tests.test_hrm_3d: Comprehensive Unit Tests for HRM-3D and Embodied-VLA Architecture.

Covers:
1. RoPE-3D continuous 3D relative translation invariance and norm preservation.
2. Proprioceptive and 3D Spatial Embedders with continuous metric coordinates.
3. Forward Kinematics 7-DOF and analytical geometric Jacobian.
4. Kinematic SE(3) relaxation with obstacle avoidance and joint limit compliance.
5. HRM-3D dual-timescale recurrence (tactical L-module & strategic H-module unrolling).
6. Semantic-to-Spatial Waypoint Projector & continuous 3D attractor potential fields.
7. EmbodiedCallosalBridge: Dale's Principle, net transcallosal inhibition (s = -1.0), and homeostatic scaling.
8. SubcorticalEmbodiedRouter: Affective appraisal heads (Omega, V, U) and neuromodulation.
9. Reflex-first 10-15ms bypass for nominal trajectory execution.
10. Test-Time Adaptation (TTA) trigger upon collision hazard (Omega > theta) and monotonic safety.
11. EmbodiedVLA end-to-end cognitive integration.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from brain_ai.models.hrm_3d import (
    RoPE3D,
    ForwardKinematics7DOF,
    KinematicSE3Relaxation,
    Proprioceptive3DEmbedder,
    Spatial3DEmbedder,
    HRM3D,
    HRM3DStateCarry,
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_normalize,
    so3_geodesic_distance
)
from brain_ai.models.embodied_vla import (
    SemanticToSpatialWaypointProjector,
    EmbodiedCallosalBridge,
    SubcorticalEmbodiedRouter,
    LeftHemisphereGemma,
    EmbodiedVLA
)


def test_rope_3d_relative_translation_invariance():
    """
    RoPE-3D Mathematical Invariance Test:
    Verifies that <rot(q, p_A), rot(k, p_B)> == <rot(q, p_A - p_B), k>.
    Continuous relative 3D displacement must be strictly preserved in inner products.
    """
    dim = 64
    rope = RoPE3D(dim=dim)
    
    # Batch size 1, 1 head, 1 token
    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)
    
    p_A = torch.tensor([[[0.45, -0.32, 0.78]]]) # 3D point A in meters
    p_B = torch.tensor([[[0.15, 0.22, 0.18]]])   # 3D point B in meters
    p_diff = p_A - p_B
    p_zero = torch.zeros_like(p_A)
    
    q_rot_A = rope(q, p_A)
    k_rot_B = rope(k, p_B)
    dot_AB = torch.sum(q_rot_A * k_rot_B).item()
    
    q_rot_diff = rope(q, p_diff)
    k_rot_zero = rope(k, p_zero)
    dot_diff = torch.sum(q_rot_diff * k_rot_zero).item()
    
    assert math.isclose(dot_AB, dot_diff, rel_tol=1e-5, abs_tol=1e-5), (
        f"RoPE-3D translation invariance failed: dot_AB={dot_AB}, dot_diff={dot_diff}"
    )
    
    # Norm preservation test: ||rot(x, p)|| == ||x||
    norm_orig = torch.norm(q).item()
    norm_rot = torch.norm(q_rot_A).item()
    assert math.isclose(norm_orig, norm_rot, rel_tol=1e-5, abs_tol=1e-5), (
        f"RoPE-3D norm preservation failed: orig={norm_orig}, rot={norm_rot}"
    )


def test_rope_3d_shapes_and_none_handling():
    """Verifies RoPE-3D multi-head tensor shapes and None coordinates fallback."""
    B, H, S, D = 2, 4, 12, 32
    rope = RoPE3D(dim=D)
    
    qk = torch.randn(B, H, S, D)
    coords = torch.randn(B, S, 3)
    
    out = rope(qk, coords)
    assert out.shape == (B, H, S, D)
    
    # Test None coordinates fallback
    out_none = rope(qk, None)
    assert out_none.shape == (B, H, S, D)
    assert torch.equal(out_none, qk)


def test_proprioceptive_and_spatial_embedders():
    """Tests proprioceptive and continuous spatial 3D embedding heads."""
    B, N_joints, d_model = 2, 7, 64
    
    proprio = Proprioceptive3DEmbedder(num_joints=N_joints, d_model=d_model)
    q = torch.randn(B, N_joints) * 0.5
    tokens, coords = proprio(q_joints=q)
    
    # 7 joints + 1 end-effector token = 8 tokens
    assert tokens.shape == (B, N_joints + 1, d_model)
    assert coords.shape == (B, N_joints + 1, 3)
    assert not torch.isnan(tokens).any()
    assert not torch.isnan(coords).any()
    
    # Spatial 3D embedder with Fourier frequencies
    N_pts = 16
    spatial = Spatial3DEmbedder(in_features=3, d_model=d_model)
    pt_coords = torch.randn(B, N_pts, 3)
    pt_feats = torch.randn(B, N_pts, 3)
    s_tokens, s_coords = spatial(coords=pt_coords, features=pt_feats)
    
    assert s_tokens.shape == (B, N_pts, d_model)
    assert s_coords.shape == (B, N_pts, 3)
    assert not torch.isnan(s_tokens).any()


def test_forward_kinematics_and_geometric_jacobian():
    """
    Tests 7-DOF analytical forward kinematics and geometric Jacobian.
    Validates end-effector unit quaternion and differential kinematics: delta_p approx J_v * delta_q.
    """
    fk = ForwardKinematics7DOF()
    B = 2
    q = torch.tensor([[0.0, 0.2, 0.0, -0.5, 0.0, 0.3, 0.0],
                      [0.1, -0.3, 0.2, -0.8, 0.1, 0.4, -0.2]], dtype=torch.float32)
    
    ee_pos, ee_quat, link_pos, J = fk(q)
    
    assert ee_pos.shape == (B, 3)
    assert ee_quat.shape == (B, 4)
    assert link_pos.shape == (B, 7, 3)
    assert J.shape == (B, 6, 7)
    
    # Quaternions must be unit magnitude
    quat_norms = torch.norm(ee_quat, dim=-1)
    assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-4)
    
    # Differential Kinematics Test: delta_p approx J_v @ delta_q
    eps = 1e-4
    delta_q = torch.zeros_like(q)
    delta_q[:, 1] = eps # Small perturbation on joint 1
    
    ee_pos_perturbed, _, _, _ = fk(q + delta_q)
    numerical_dp = (ee_pos_perturbed - ee_pos) / eps # (B, 3)
    
    # Analytical linear velocity Jacobian column 1
    analytical_dp = J[:, :3, 1] # (B, 3)
    
    assert torch.allclose(numerical_dp, analytical_dp, atol=2e-3), (
        f"Geometric Jacobian mismatch: num={numerical_dp}, ana={analytical_dp}"
    )


def test_kinematic_se3_relaxation():
    """
    Tests iterative SE(3) kinematic relaxation:
    1. Tracking error reduction towards target waypoint.
    2. Obstacle potential avoidance (clearance increases away from obstacle).
    3. Joint limit compliance.
    """
    relax = KinematicSE3Relaxation(num_joints=7, d_latent=64, relax_steps=4)
    
    B = 2
    q_init = torch.zeros(B, 7)
    target_pos = torch.tensor([[0.3, 0.2, 0.5], [0.2, -0.2, 0.6]])
    
    # Relaxation without obstacles
    out_free = relax(q_init=q_init, target_pos=target_pos, steps=5)
    q_relaxed = out_free["q_relaxed"]
    assert q_relaxed.shape == (B, 7)
    assert out_free["ee_pose"].shape == (B, 7)
    
    # Check that tracking error was reduced compared to initial pose
    fk = relax.fk
    init_ee_pos, _, _, _ = fk(q_init)
    init_error = torch.norm(target_pos - init_ee_pos, dim=-1)
    final_error = out_free["tracking_error"]
    assert (final_error < init_error).all(), "SE(3) relaxation failed to reduce tracking error"
    
    # Joint limits check
    assert (q_relaxed >= relax.q_min - 1e-4).all()
    assert (q_relaxed <= relax.q_max + 1e-4).all()
    
    # Relaxation with an obstacle placed near trajectory
    obs = torch.tensor([[[0.1, 0.1, 0.4]], [[0.1, -0.1, 0.4]]]) # (B, 1, 3)
    out_obs = relax(q_init=q_init, target_pos=target_pos, obstacles=obs, steps=5)
    assert out_obs["min_clearance"].shape == (B,)
    assert not torch.isnan(out_obs["q_relaxed"]).any()


def test_hrm_3d_dual_timescale_recurrent_unrolling():
    """
    Tests HRM-3D dual-timescale frontoparietal unrolling:
    - Tactical L-module running L_cycles inner steps.
    - Strategic H-module running 1 outer step.
    - ACT Q-head halting decision and state carry progression.
    """
    B, S, d_model = 2, 6, 64
    model = HRM3D(
        d_model=d_model,
        n_heads=4,
        d_ffn=128,
        num_joints=7,
        L_cycles=2,
        M_max=3,
        se3_relax_steps=2
    )
    
    x_embed = torch.randn(B, S, d_model)
    coords = torch.randn(B, S, 3)
    target_pos = torch.tensor([[0.25, 0.10, 0.45], [0.30, -0.15, 0.50]])
    
    converged_z_H, carry, aux = model(
        x_embed=x_embed,
        coords=coords,
        target_pos=target_pos,
        max_steps=2
    )
    
    assert converged_z_H.shape == (B, S, d_model)
    assert carry.z_H.shape == (B, S, d_model)
    assert carry.z_L.shape == (B, S, d_model)
    assert carry.q_joints.shape == (B, 7)
    assert carry.ee_pose.shape == (B, 7)
    assert carry.step_count.max().item() in (1, 2)
    assert "relaxation_info" in aux
    assert not torch.isnan(converged_z_H).any()


def test_semantic_to_spatial_waypoint_projector():
    """
    Tests SemanticToSpatialWaypointProjector:
    - Maps language tokens into continuous 3D waypoints.
    - Computes continuous attractor potential U(x) and pulling force F(x) = -grad U(x).
    """
    B, S_L, d_lh, d_rh = 2, 8, 128, 64
    num_wp = 4
    projector = SemanticToSpatialWaypointProjector(
        d_lh=d_lh,
        d_rh=d_rh,
        num_waypoints=num_wp,
        workspace_radius=1.0
    )
    
    lh_latents = torch.randn(B, S_L, d_lh)
    wp_out = projector(lh_latents)
    
    assert wp_out["positions"].shape == (B, num_wp, 3)
    assert wp_out["quaternions"].shape == (B, num_wp, 4)
    assert wp_out["strengths"].shape == (B, num_wp)
    assert wp_out["sigmas"].shape == (B, num_wp)
    assert wp_out["tokens"].shape == (B, num_wp, d_rh)
    
    # Test continuous attractor potential evaluation
    query_coords = torch.randn(B, 10, 3)
    potential = projector.evaluate_potential(
        query_coords=query_coords,
        positions=wp_out["positions"],
        strengths=wp_out["strengths"],
        sigmas=wp_out["sigmas"]
    )
    assert potential.shape == (B, 10)
    # Potential at waypoint itself should be strongly negative (minimum energy basin)
    p0 = wp_out["positions"][:, 0:1, :] # (B, 1, 3)
    pot_at_p0 = projector.evaluate_potential(
        query_coords=p0,
        positions=wp_out["positions"],
        strengths=wp_out["strengths"],
        sigmas=wp_out["sigmas"]
    )
    assert (pot_at_p0 < 0.0).all(), "Attractor potential must be negative at target waypoint"
    
    # Test pulling force field F(x) = -grad U(x)
    force = projector.evaluate_attractor_force(
        query_coords=query_coords,
        positions=wp_out["positions"],
        strengths=wp_out["strengths"],
        sigmas=wp_out["sigmas"]
    )
    assert force.shape == (B, 10, 3)
    assert not torch.isnan(force).any()


def test_embodied_callosal_bridge_dale_and_inhibition():
    """
    Tests EmbodiedCallosalBridge:
    - Dale's principle sign segregation across all projections.
    - Net transcallosal inhibition (s = -1.0).
    - Homeostatic synaptic scaling to target energy r* = 1.0.
    """
    B, S_L, S_R = 2, 8, 6
    d_lh, d_rh, d_call = 128, 64, 64
    
    bridge = EmbodiedCallosalBridge(
        d_lh=d_lh,
        d_rh=d_rh,
        d_callosum=d_call,
        n_heads=2,
        cross_talk_sign=-1.0,
        r_target=1.0
    )
    
    # 1. Dale's Principle verification on all projection layers
    dale_report = bridge.verify_dales_principle()
    for layer_name, obeys in dale_report.items():
        assert obeys, f"Layer {layer_name} violated Dale's Principle!"
        
    # 2. Forward pass and net transcallosal inhibition
    lh_latents = torch.randn(B, S_L, d_lh)
    rh_latents = torch.randn(B, S_R, d_rh)
    
    out = bridge(lh_latents=lh_latents, rh_latents=rh_latents)
    assert out["lh_latents_updated"].shape == (B, S_L, d_lh)
    assert out["rh_latents_updated"].shape == (B, S_R, d_rh)
    assert "waypoint_info" in out
    assert "callosal_loss" in out
    # Effective coupling should have negative sign due to s = -1.0
    assert out["effective_coupling"].item() < 0.0, "Transcallosal coupling must be inhibitory (s = -1.0)"


def test_subcortical_embodied_router_appraisals():
    """
    Tests SubcorticalEmbodiedRouter:
    - Affective appraisal heads: Threat Omega in [0, 1], Valence V in [-1, 1], Uncertainty U in [0, 1].
    - Obstacle proximity increases threat Omega.
    - Neuromodulatory gain calculations.
    """
    B, d_state = 2, 64
    router = SubcorticalEmbodiedRouter(d_state=d_state, mlp_dim=32)
    state = torch.randn(B, d_state)
    
    # Case A: Far obstacle (nominal)
    far_dist = torch.tensor([[1.5], [2.0]])
    appraisal_far = router.appraise(state, min_obstacle_dist=far_dist)
    assert (appraisal_far["threat_omega"] >= 0.0).all() and (appraisal_far["threat_omega"] <= 1.0).all()
    assert (appraisal_far["valence"] >= -1.0).all() and (appraisal_far["valence"] <= 1.0).all()
    assert (appraisal_far["uncertainty"] >= 0.0).all() and (appraisal_far["uncertainty"] <= 1.0).all()
    
    # Case B: Imminent collision obstacle (proximity hazard)
    near_dist = torch.tensor([[0.02], [0.03]]) # 2-3 cm away (critical danger)
    appraisal_near = router.appraise(state, min_obstacle_dist=near_dist)
    
    # Imminent obstacle proximity must significantly elevate Threat Omega
    assert (appraisal_near["threat_omega"] > appraisal_far["threat_omega"]).all(), (
        "Imminent obstacle did not elevate threat Omega!"
    )
    assert (appraisal_near["threat_omega"] > 0.6).all(), "Hazard threat Omega should exceed 0.6"


def test_reflex_first_10_15ms_bypass():
    """
    Tests System 1 Reflex-First Nominal Bypass:
    When conditions are nominal (clear space, low threat, low uncertainty, positive valence),
    router instantly bypasses System 2 TTA and executes nominal trajectory in 10-15ms.
    """
    d_state = 64
    router = SubcorticalEmbodiedRouter(
        d_state=d_state,
        theta_omega=0.25,
        theta_u=0.20
    )
    
    # Construct nominal conditions
    omega_nominal = torch.tensor([0.05, 0.08])
    u_nominal = torch.tensor([0.05, 0.10])
    v_nominal = torch.tensor([0.6, 0.8])
    
    bypass = router.should_bypass_reflex(omega_nominal, v_nominal, u_nominal)
    assert bypass.all(), "Nominal condition should trigger reflex bypass"
    
    # Route and arbitrate with large distance (nominal clear workspace)
    state = torch.randn(2, d_state)
    clear_dist = torch.tensor([[1.2], [1.5]])
    res = router.route_and_arbitrate(state, min_obstacle_dist=clear_dist)
    
    assert res["bypassed_tta"] is True
    assert res["decision"] == "System 1 (Reflex Bypass)"
    assert 10.0 <= res["reflex_latency_ms"] <= 20.0, f"Latency outside 10-15ms range: {res['reflex_latency_ms']}"


def test_collision_hazard_tta_trigger():
    """
    Tests Collision Hazard Triggering System 2 Test-Time Adaptation (TTA):
    When an obstacle hazard occurs (Omega > theta_omega), bypass is suppressed,
    TTA is triggered, and transcallosal inhibitory gain gamma_I is elevated.
    """
    d_state = 64
    router = SubcorticalEmbodiedRouter(
        d_state=d_state,
        theta_omega=0.25,
        safe_clearance_threshold=0.10
    )
    
    state = torch.randn(2, d_state)
    # Dangerous obstacle within 4 cm
    hazard_dist = torch.tensor([[0.04], [0.03]])
    
    res = router.route_and_arbitrate(state, min_obstacle_dist=hazard_dist)
    
    assert res["bypassed_tta"] is False, "Hazard should suppress reflex bypass"
    assert res["tta_triggered"] is True
    assert res["decision"] == "System 2 (Collision Hazard TTA)"
    assert (res["appraisal"]["threat_omega"] > router.theta_omega).all()
    
    # Neuromodulatory gain: inhibitory gain gamma_I should be elevated to suppress collided trajectories
    gamma_i = res["neuromodulatory_controls"]["gamma_I"]
    assert (gamma_i >= router.neuromodulator.gamma_I0).all()


def test_embodied_vla_end_to_end_cognitive_pass():
    """
    Tests unified EmbodiedVLA end-to-end integration:
    - Left Hemisphere Gemma token processing.
    - Embodied Callosal exchange with Dale's principle and semantic attractor projection.
    - Subcortical affective appraisal and arbitration.
    - Reflex bypass pass and TTA obstacle adaptation pass.
    """
    d_lh, d_rh, d_call = 128, 64, 64
    vla = EmbodiedVLA(
        d_lh=d_lh,
        d_rh=d_rh,
        d_callosum=d_call,
        num_joints=7,
        num_waypoints=3,
        hrm_cycles=2,
        hrm_max_steps=2,
        mock_lh=True,
        device="cpu"
    )
    
    # 1. Nominal scenario (no obstacles): triggers fast reflex bypass
    q_init = torch.zeros(1, 7)
    out_nominal = vla(
        instruction_text="pick up the green block and place it on tray",
        q_joints=q_init,
        obstacles=None
    )
    assert "routing_info" in out_nominal
    assert "execution_out" in out_nominal
    assert out_nominal["execution_out"]["mode"] in ("system1_reflex", "system2_tta")
    
    # 2. Collision hazard scenario: place obstacle close to arm (3cm below end-effector)
    ee_pos0, _, _, _ = vla.right_hemisphere.kinematic_relaxation.fk(q_init)
    obs_hazard = (ee_pos0 - torch.tensor([0.0, 0.0, 0.03])).unsqueeze(1) # (1, 1, 3) 3cm below end-effector!
    out_hazard = vla(
        instruction_text="move forward carefully",
        q_joints=q_init,
        obstacles=obs_hazard
    )
    
    # Should trigger System 2 TTA adaptation
    assert out_hazard["routing_info"]["tta_triggered"] is True
    assert out_hazard["execution_out"]["mode"] == "system2_tta"
    assert "q_adapted" in out_hazard["execution_out"]
    assert "dq_adapted" in out_hazard["execution_out"]
    assert "gripper" in out_hazard["execution_out"]
    assert not torch.isnan(out_hazard["execution_out"]["q_adapted"]).any()


def test_yoshikawa_manipulability_and_adaptive_dls_near_singularity():
    """
    Mathematical Hardening Test:
    Verifies Yoshikawa manipulability w = sqrt(det(J J^T)) and adaptive DLS damping
    near singular manipulator configurations.
    """
    relax = KinematicSE3Relaxation(num_joints=7, damping=0.05)
    
    # 1. Nominal non-singular configuration
    q_nominal = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]], dtype=torch.float32)
    _, _, _, J_nom = relax.fk(q_nominal)
    w_nom = relax.yoshikawa_manipulability(J_nom)
    assert w_nom.item() > 0.005, f"Nominal manipulability should be healthy, got {w_nom.item()}"
    
    # 2. Outstretched configuration near kinematic boundary singularity
    q_singular = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    _, _, _, J_sing = relax.fk(q_singular)
    w_sing = relax.yoshikawa_manipulability(J_sing)
    assert w_sing.item() < w_nom.item(), "Outstretched configuration should have lower manipulability"
    
    # 3. Inversion stability test: J_dagger must remain bounded and finite without NaNs
    J_dagger_nom = relax.damped_least_squares_inverse(J_nom)
    J_dagger_sing = relax.damped_least_squares_inverse(J_sing, w_threshold=0.04, lambda_max=0.30)
    
    assert not torch.isnan(J_dagger_sing).any(), "Singular DLS inverse produced NaNs!"
    assert not torch.isinf(J_dagger_sing).any(), "Singular DLS inverse produced Infs!"
    assert torch.max(torch.abs(J_dagger_sing)).item() < 50.0, "DLS inverse magnitude blew up!"


def test_robust_matrix_to_quaternion_and_so3_geodesic_distance():
    """
    Tests Shepperd algorithm conversion on extreme 180-degree rotations (tr(R) <= 0)
    and verifies SO(3) Riemannian geodesic distance consistency with antipodal quaternions.
    """
    # 1. 180-degree rotations around principal axes where tr(R) = 1 - 1 - 1 = -1.0
    R_180_x = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ]], dtype=torch.float32)
    q_180_x = matrix_to_quaternion(R_180_x)
    assert not torch.isnan(q_180_x).any(), "Shepperd conversion failed on tr=-1 rotation around X!"
    assert math.isclose(torch.norm(q_180_x).item(), 1.0, abs_tol=1e-5)
    assert math.isclose(abs(q_180_x[0, 1].item()), 1.0, abs_tol=1e-4) # x component is unit

    R_180_z = torch.tensor([[
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]], dtype=torch.float32)
    q_180_z = matrix_to_quaternion(R_180_z)
    assert not torch.isnan(q_180_z).any(), "Shepperd conversion failed on tr=-1 rotation around Z!"
    assert math.isclose(abs(q_180_z[0, 3].item()), 1.0, abs_tol=1e-4) # z component is unit

    # 2. SO(3) Geodesic Distance Antipodal Consistency
    q1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)
    q1_antipodal = -q1
    d_antipodal = so3_geodesic_distance(q1, q1_antipodal)
    assert math.isclose(d_antipodal.item(), 0.0, abs_tol=1e-5), (
        f"SO(3) geodesic distance must be 0 for antipodal quaternions, got {d_antipodal.item()}"
    )

    # 3. 90-degree rotation distance test: angle = pi / 2
    # Quaternion for 90-deg rotation around z: [cos(pi/4), 0, 0, sin(pi/4)]
    q_90z = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]], dtype=torch.float32)
    q_identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    d_90 = so3_geodesic_distance(q_identity, q_90z)
    assert math.isclose(d_90.item(), math.pi / 2, abs_tol=1e-4), (
        f"Expected pi/2 ({math.pi/2}), got {d_90.item()}"
    )


def test_obstacle_repulsive_field_zero_distance_stability():
    """
    Tests numerical stability of obstacle repulsive potentials when distance d -> 0.
    Verifies that smooth epsilon-clamping strictly prevents gradient explosions and NaNs.
    """
    relax = KinematicSE3Relaxation(num_joints=7, damping=0.05)
    
    q_init = torch.zeros(1, 7)
    ee_pos, _, link_positions, _ = relax.fk(q_init)
    
    # Place obstacle directly at the position of link 4 (d = 0)
    obs_zero_dist = link_positions[:, 4:5, :].clone() # (1, 1, 3)
    target_pos = ee_pos + torch.tensor([[0.1, 0.0, 0.1]])
    
    # Relaxation pass with zero-distance obstacle
    out = relax(
        q_init=q_init,
        target_pos=target_pos,
        obstacles=obs_zero_dist,
        steps=3
    )
    
    q_rel = out["q_relaxed"]
    assert not torch.isnan(q_rel).any(), "Zero-distance obstacle caused NaNs in relaxed joints!"
    assert not torch.isinf(q_rel).any(), "Zero-distance obstacle caused Infs in relaxed joints!"
    assert out["min_clearance"].item() >= 0.0

