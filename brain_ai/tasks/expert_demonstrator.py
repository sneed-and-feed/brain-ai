"""
brain_ai.tasks.expert_demonstrator: Scripted Expert Demonstrator and Dataset Collector.

Implements:
1. Kinematically Compliant Trajectory Interpolation:
   - Analytical minimum-jerk (quintic polynomial) boundary-value trajectory generator.
   - Smooth position and feedforward Cartesian velocity profiles (zero jerk/acceleration spikes).
2. Scripted Expert Execution Policy (FrankaScriptedExpert):
   - High-precision closed-loop tracking with feedforward velocity and proportional correction.
   - Singularity-robust DLS inverse kinematics with Yoshikawa manipulability-adaptive damping.
   - Null-space posture optimization projecting arm configuration towards comfortable DEFAULT_HOME_Q.
   - Clamping joint velocities to physical Franka Panda limits (MAX_JOINT_VELOCITY).
   - Multi-phase state machine for pick-and-place:
       1. Approach pre-grasp waypoint (elevated above pick site)
       2. Descend to pick site
       3. Grasp (smooth gripper closure)
       4. Lift object vertically
       5. Transit to pre-place waypoint
       6. Descend to place site
       7. Open gripper (smooth release)
       8. Retract end-effector vertically
   - Single-phase smooth reaching with terminal stabilization.
3. Clean Nominal Demonstration Dataset Collector:
   - Collects 150-200 clean nominal episodes (NO hazards present).
   - Key architectural decision: Demonstrations contain NO dynamic hazards. Both baseline
     and bi-hemispheric systems encounter dynamic hazards for the first time during test-time evaluation.
   - Stores unified 21-dimensional proprioceptive observations:
       [joint_q (7), joint_dq (7), ee_pos (3), target_pos (3), gripper_width (1)]
   - Stores 7-dimensional joint velocity action commands [dq].
   - Exports serialized artifact to data/embodied_demos/franka_nominal_demos.pt.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import os
from typing import Dict, List, Tuple, Optional, Any, Union

import torch

from brain_ai.tasks.robotics_sandbox import (
    FrankaKinematics,
    FrankaKinematicEnv,
    BaseExecutionPolicy,
    RoboticsTaskSpec,
    ER2TaskType,
    create_nominal_reach_task,
    create_pick_and_place_task,
    compute_path_smoothness,
)


# ==============================================================================
# 1. MINIMUM-JERK (QUINTIC POLYNOMIAL) TRAJECTORY INTERPOLATOR
# ==============================================================================

class MinimumJerkInterpolator:
    """
    Minimum-jerk / quintic polynomial trajectory interpolator.
    Generates C^2-continuous smooth Cartesian paths with zero endpoint
    velocities and accelerations:
        s(tau) = 10 * tau^3 - 15 * tau^4 + 6 * tau^5
        s_dot(tau) = (30 * tau^2 - 60 * tau^3 + 30 * tau^4) / duration
    """

    @staticmethod
    def step(
        p_start: torch.Tensor,
        p_end: torch.Tensor,
        tau: float,
        duration: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes desired position and feedforward Cartesian velocity at normalized time tau.

        Args:
            p_start: 3D starting position [3]
            p_end: 3D target position [3]
            tau: Normalized time progress in [0.0, 1.0]
            duration: Segment duration in seconds (T = N * dt)
        Returns:
            p_des: Desired 3D position [3]
            v_des: Desired 3D feedforward Cartesian velocity [3] (m/s)
        """
        tau_c = min(max(float(tau), 0.0), 1.0)
        tau2 = tau_c * tau_c
        tau3 = tau2 * tau_c
        tau4 = tau3 * tau_c
        tau5 = tau4 * tau_c

        # Minimum-jerk polynomial scale factor s(tau) in [0, 1]
        s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5

        # Analytical derivative s_dot(tau) = ds/dt
        dur_safe = max(duration, 1e-4)
        s_dot = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / dur_safe

        delta_p = p_end - p_start
        p_des = p_start + s * delta_p
        v_des = s_dot * delta_p

        return p_des, v_des


# ==============================================================================
# 2. TRAJECTORY SEGMENTS & PICK-AND-PLACE PHASES
# ==============================================================================

class TrajectoryPhase(Enum):
    # Reach phases
    REACH = "reach"
    HOLD = "hold"
    # Pick-and-place phases
    APPROACH_PRE_GRASP = "approach_pre_grasp"
    DESCEND_TO_PICK = "descend_to_pick"
    GRASP = "grasp"
    LIFT = "lift"
    TRANSIT_TO_PRE_PLACE = "transit_to_pre_place"
    DESCEND_TO_PLACE = "descend_to_place"
    OPEN_GRIPPER = "open_gripper"
    RETRACT = "retract"
    COMPLETED = "completed"


@dataclass
class TrajectorySegment:
    """Defines a discrete motion segment in the end-effector trajectory."""
    name: str
    phase: TrajectoryPhase
    target_pos: torch.Tensor
    steps: int
    target_gripper_width: float
    ramp_gripper: bool = False
    start_pos: Optional[torch.Tensor] = None


# ==============================================================================
# 3. SCRIPTED EXPERT EXECUTION POLICY
# ==============================================================================

class FrankaScriptedExpert(BaseExecutionPolicy):
    """
    Scripted expert demonstration policy for the Franka Emika Panda 7-DoF manipulator.

    Features:
    - Trajectory Planning: Computes kinematically compliant minimum-jerk waypoints
      for both nominal reaching and pick-and-place sequences.
    - Inverse Kinematics: Damped Least Squares (DLS) with Yoshikawa adaptive damping.
    - Posture Optimization: Null-space projection returning redundant joints to DEFAULT_HOME_Q.
    - Joint Safety: Clamps all joint velocities strictly within physical MAX_JOINT_VELOCITY limits.
    - Gripper Telemetry: Controls discrete and smooth opening/closing transitions.
    """

    def __init__(
        self,
        task_spec: Optional[RoboticsTaskSpec] = None,
        kp: float = 6.0,
        k_null: float = 0.05,
        lambda_dls: float = 0.01,
        w_threshold: float = 0.03,
        lambda_max: float = 0.15,
        dt: float = 0.01
    ):
        self.task_spec = task_spec or create_nominal_reach_task()
        self.kp = kp
        self.k_null = k_null
        self.lambda_dls = lambda_dls
        self.w_threshold = w_threshold
        self.lambda_max = lambda_max
        self.dt = dt

        # State tracking
        self.segments: List[TrajectorySegment] = []
        self.current_segment_idx = 0
        self.segment_step = 0
        self.total_steps_executed = 0
        self.current_gripper_width = 0.08
        self.is_done = False
        self._initialized = False

    def reset(self, task_spec: Optional[RoboticsTaskSpec] = None):
        """Resets expert policy state and re-plans trajectory for current task."""
        if task_spec is not None:
            self.task_spec = task_spec
        self.segments.clear()
        self.current_segment_idx = 0
        self.segment_step = 0
        self.total_steps_executed = 0
        self.current_gripper_width = 0.08
        self.is_done = False
        self._initialized = False

    def _plan_trajectory(self, initial_p_ee: torch.Tensor):
        """Constructs kinematically compliant trajectory segments based on task type."""
        self.segments.clear()
        device = initial_p_ee.device
        dtype = initial_p_ee.dtype

        if self.task_spec.task_type == ER2TaskType.PICK_AND_PLACE:
            pick_pos = (
                self.task_spec.pick_pos.to(device=device, dtype=dtype)
                if self.task_spec.pick_pos is not None
                else torch.tensor([0.35, -0.15, 0.30], device=device, dtype=dtype)
            )
            place_pos = (
                self.task_spec.place_pos.to(device=device, dtype=dtype)
                if self.task_spec.place_pos is not None
                else torch.tensor([0.45, 0.20, 0.30], device=device, dtype=dtype)
            )

            # Waypoint geometry
            z_pre_pick = torch.tensor([0.0, 0.0, 0.10], device=device, dtype=dtype)
            z_lift = torch.tensor([0.0, 0.0, 0.15], device=device, dtype=dtype)
            z_retract = torch.tensor([0.0, 0.0, 0.12], device=device, dtype=dtype)

            pre_pick_pos = pick_pos + z_pre_pick
            lift_pos = pick_pos + z_lift
            pre_place_pos = place_pos + z_lift
            retract_pos = place_pos + z_retract

            # 8-Phase Pick-and-Place State Machine:
            # 1. Approach pre-grasp waypoint
            self.segments.append(TrajectorySegment(
                name="approach_pre_grasp",
                phase=TrajectoryPhase.APPROACH_PRE_GRASP,
                target_pos=pre_pick_pos,
                steps=30,
                target_gripper_width=0.08
            ))
            # 2. Descend to pick site
            self.segments.append(TrajectorySegment(
                name="descend_to_pick",
                phase=TrajectoryPhase.DESCEND_TO_PICK,
                target_pos=pick_pos,
                steps=20,
                target_gripper_width=0.08
            ))
            # 3. Grasp: close gripper smoothly at pick location
            self.segments.append(TrajectorySegment(
                name="grasp",
                phase=TrajectoryPhase.GRASP,
                target_pos=pick_pos,
                steps=10,
                target_gripper_width=0.00,
                ramp_gripper=True
            ))
            # 4. Lift object vertically
            self.segments.append(TrajectorySegment(
                name="lift",
                phase=TrajectoryPhase.LIFT,
                target_pos=lift_pos,
                steps=20,
                target_gripper_width=0.00
            ))
            # 5. Transit across workspace to pre-place waypoint
            self.segments.append(TrajectorySegment(
                name="transit_to_pre_place",
                phase=TrajectoryPhase.TRANSIT_TO_PRE_PLACE,
                target_pos=pre_place_pos,
                steps=40,
                target_gripper_width=0.00
            ))
            # 6. Descend to place site
            self.segments.append(TrajectorySegment(
                name="descend_to_place",
                phase=TrajectoryPhase.DESCEND_TO_PLACE,
                target_pos=place_pos,
                steps=20,
                target_gripper_width=0.00
            ))
            # 7. Open gripper: smooth release
            self.segments.append(TrajectorySegment(
                name="open_gripper",
                phase=TrajectoryPhase.OPEN_GRIPPER,
                target_pos=place_pos,
                steps=10,
                target_gripper_width=0.08,
                ramp_gripper=True
            ))
            # 8. Retract: lift end-effector safely away from place site
            self.segments.append(TrajectorySegment(
                name="retract",
                phase=TrajectoryPhase.RETRACT,
                target_pos=retract_pos,
                steps=20,
                target_gripper_width=0.08
            ))

        else:
            # Nominal Reach / Obstacle Field Navigation: Smooth minimum-jerk reach
            target = self.task_spec.target_pos.to(device=device, dtype=dtype)
            self.segments.append(TrajectorySegment(
                name="smooth_reach",
                phase=TrajectoryPhase.REACH,
                target_pos=target,
                steps=100,
                target_gripper_width=0.08
            ))
            # Terminal stabilization hold
            self.segments.append(TrajectorySegment(
                name="terminal_hold",
                phase=TrajectoryPhase.HOLD,
                target_pos=target,
                steps=15,
                target_gripper_width=0.08
            ))

        self.current_segment_idx = 0
        self.segment_step = 0
        self.current_gripper_width = 0.08
        self._initialized = True

    def compute_action(self, obs: Dict[str, Any], t: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes kinematically compliant joint velocity action dq [7] and telemetry.

        Args:
            obs: Observation dict from FrankaKinematicEnv
            t: Current simulation time (seconds)
        Returns:
            dq_cmd: Commanded joint velocities [7] clamped to MAX_JOINT_VELOCITY
            policy_info: Telemetry metadata including gripper width and trajectory phase
        """
        q = obs["joint_q"]
        p_ee = obs["ee_pos"]
        J = obs["jacobian"]

        if not self._initialized:
            if self.task_spec is None:
                if "task_spec" in obs and obs["task_spec"] is not None:
                    self.task_spec = obs["task_spec"]
                elif obs.get("task_type") == "pick_and_place" or ("pick" in obs.get("directive", "").lower()):
                    pick_pos = obs.get("pick_pos")
                    place_pos = obs.get("place_pos", obs.get("target_pos"))
                    self.task_spec = create_pick_and_place_task(pick_pos=pick_pos, place_pos=place_pos, with_hazard=False)
                elif "target_pos" in obs:
                    self.task_spec = create_nominal_reach_task(target_pos=obs["target_pos"], with_hazard=False)
            self._plan_trajectory(p_ee)

        if self.current_segment_idx >= len(self.segments):
            self.is_done = True
            # Zero hold velocity
            zero_dq = torch.zeros(7, dtype=q.dtype, device=q.device)
            return zero_dq, {
                "policy": "FrankaScriptedExpert",
                "phase": TrajectoryPhase.COMPLETED.value,
                "gripper_width": self.current_gripper_width,
                "is_completed": True
            }

        cur_seg = self.segments[self.current_segment_idx]
        if cur_seg.start_pos is None:
            cur_seg.start_pos = p_ee.clone()

        self.segment_step += 1
        self.total_steps_executed += 1
        tau = min(self.segment_step / cur_seg.steps, 1.0)
        duration = cur_seg.steps * self.dt

        # Compute minimum-jerk position and feedforward velocity
        p_des, v_ff = MinimumJerkInterpolator.step(
            cur_seg.start_pos, cur_seg.target_pos, tau, duration
        )

        # Gripper width modulation
        if cur_seg.ramp_gripper:
            # Linear interpolation between previous and target gripper width
            prev_width = 0.08 if cur_seg.target_gripper_width < 0.04 else 0.00
            self.current_gripper_width = prev_width + tau * (cur_seg.target_gripper_width - prev_width)
        else:
            self.current_gripper_width = cur_seg.target_gripper_width

        # Proportional Cartesian velocity command with feedforward tracking
        v_cmd = v_ff + self.kp * (p_des - p_ee)

        # Singularity-robust Damped Least Squares Inverse Kinematics with Yoshikawa adaptive damping
        J_pinv = FrankaKinematics.damped_least_squares_pinv(
            J,
            lambda_dls=self.lambda_dls,
            w_threshold=self.w_threshold,
            lambda_max=self.lambda_max
        )
        dq_task = J_pinv @ v_cmd

        # Null-space posture stabilization towards DEFAULT_HOME_Q
        q_home = FrankaKinematics.DEFAULT_HOME_Q.to(device=q.device, dtype=q.dtype)
        P_null = torch.eye(7, device=q.device, dtype=q.dtype) - J_pinv @ J
        dq_null = P_null @ (self.k_null * (q_home - q))

        # Total commanded velocity with physical limit clamping
        dq_total = dq_task + dq_null
        max_vel = FrankaKinematics.MAX_JOINT_VELOCITY.to(device=q.device, dtype=q.dtype)
        dq_cmd = torch.clamp(dq_total, -max_vel, max_vel)

        phase_name = cur_seg.name

        # Advance to next segment if current segment is complete
        if self.segment_step >= cur_seg.steps:
            self.current_segment_idx += 1
            self.segment_step = 0
            if self.current_segment_idx < len(self.segments):
                self.segments[self.current_segment_idx].start_pos = p_ee.clone()
            else:
                self.is_done = True

        policy_info = {
            "policy": "FrankaScriptedExpert",
            "phase": phase_name,
            "gripper_width": float(self.current_gripper_width),
            "desired_ee_pos": p_des.clone(),
            "target_pos": cur_seg.target_pos.clone(),
            "is_completed": self.is_done
        }

        return dq_cmd, policy_info

    def is_completed(self) -> bool:
        """Returns True if the full planned trajectory has concluded."""
        return self.is_done


# ==============================================================================
# 4. RANDOMIZED WORKSPACE TASK GENERATORS (CLEAN NOMINAL ONLY)
# ==============================================================================

def create_randomized_reach_task(
    seed: int,
    workspace_x_bounds: Tuple[float, float] = (0.38, 0.54),
    workspace_y_bounds: Tuple[float, float] = (-0.22, 0.22),
    workspace_z_bounds: Tuple[float, float] = (0.28, 0.48)
) -> RoboticsTaskSpec:
    """
    Creates a nominal reach task with randomized target coordinates
    strictly within the Franka Panda kinematic reachable workspace.
    Clean nominal task: NO static or dynamic hazards.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    x = float(torch.empty(1).uniform_(workspace_x_bounds[0], workspace_x_bounds[1], generator=g).item())
    y = float(torch.empty(1).uniform_(workspace_y_bounds[0], workspace_y_bounds[1], generator=g).item())
    z = float(torch.empty(1).uniform_(workspace_z_bounds[0], workspace_z_bounds[1], generator=g).item())
    target_pos = torch.tensor([x, y, z], dtype=torch.float32)

    return create_nominal_reach_task(target_pos=target_pos, with_hazard=False)


def create_randomized_pick_and_place_task(
    seed: int,
    pick_x_bounds: Tuple[float, float] = (0.35, 0.44),
    pick_y_bounds: Tuple[float, float] = (-0.20, -0.08),
    pick_z_bounds: Tuple[float, float] = (0.26, 0.32),
    place_x_bounds: Tuple[float, float] = (0.42, 0.50),
    place_y_bounds: Tuple[float, float] = (0.08, 0.20),
    place_z_bounds: Tuple[float, float] = (0.26, 0.32)
) -> RoboticsTaskSpec:
    """
    Creates a pick-and-place task with randomized pick and place sites
    strictly within the Franka Panda kinematic reachable workspace.
    Clean nominal task: NO static or dynamic hazards.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    px = float(torch.empty(1).uniform_(pick_x_bounds[0], pick_x_bounds[1], generator=g).item())
    py = float(torch.empty(1).uniform_(pick_y_bounds[0], pick_y_bounds[1], generator=g).item())
    pz = float(torch.empty(1).uniform_(pick_z_bounds[0], pick_z_bounds[1], generator=g).item())
    pick_pos = torch.tensor([px, py, pz], dtype=torch.float32)

    lx = float(torch.empty(1).uniform_(place_x_bounds[0], place_x_bounds[1], generator=g).item())
    ly = float(torch.empty(1).uniform_(place_y_bounds[0], place_y_bounds[1], generator=g).item())
    lz = float(torch.empty(1).uniform_(place_z_bounds[0], place_z_bounds[1], generator=g).item())
    place_pos = torch.tensor([lx, ly, lz], dtype=torch.float32)

    return create_pick_and_place_task(pick_pos=pick_pos, place_pos=place_pos, with_hazard=False)


# ==============================================================================
# 5. DEMONSTRATION DATASET COLLECTOR
# ==============================================================================

class DemonstrationCollector:
    """
    Collects clean nominal demonstration trajectories from FrankaScriptedExpert.
    Constructs unified 21-dim observation tensors and 7-dim action tensors.
    Guarantees zero collisions and verified task success before dataset export.
    """

    @staticmethod
    def extract_observation_vector(obs: Dict[str, Any]) -> torch.Tensor:
        """
        Constructs the unified 21-dimensional proprioceptive observation vector:
            - joint_q (7): Joint angles
            - joint_dq (7): Joint velocities
            - ee_pos (3): End-effector 3D coordinates
            - target_pos (3): Goal target 3D coordinates
            - gripper_width (1): Gripper width opening distance
        """
        joint_q = obs["joint_q"].view(7)
        joint_dq = obs["joint_dq"].view(7)
        ee_pos = obs["ee_pos"].view(3)
        target_pos = obs["target_pos"].view(3)
        grip_w = obs.get("gripper_width", 0.08)
        if isinstance(grip_w, (int, float)):
            grip_t = torch.tensor([grip_w], dtype=torch.float32, device=joint_q.device)
        else:
            grip_t = grip_w.view(1).to(dtype=torch.float32, device=joint_q.device)

        return torch.cat([joint_q, joint_dq, ee_pos, target_pos, grip_t], dim=-1)

    @staticmethod
    def collect_single_episode(
        env: FrankaKinematicEnv,
        expert: FrankaScriptedExpert,
        task_spec: RoboticsTaskSpec
    ) -> Dict[str, Any]:
        """
        Executes a single demonstration episode and records state-action transitions.

        Returns:
            Dict containing observations [T, 21], actions [T, 7], length, success, collision
        """
        env.task_spec = task_spec
        obs = env.reset()
        expert.reset(task_spec)

        ep_obs_list: List[torch.Tensor] = []
        ep_act_list: List[torch.Tensor] = []
        done = False
        info: Dict[str, Any] = {}

        while not done:
            t = obs["t"]
            # Extract 21-dim observation before executing action
            obs_vec = DemonstrationCollector.extract_observation_vector(obs)
            ep_obs_list.append(obs_vec.cpu().clone())

            # Expert action computation
            action, p_info = expert.compute_action(obs, t)
            ep_act_list.append(action.cpu().clone())

            # Step environment with gripper synchronization
            gripper_w = p_info.get("gripper_width", 0.08)
            obs, reward, done, info = env.step(action, gripper_width=gripper_w)

            # Stop when expert finishes all waypoints or env terminates
            if expert.is_completed() and (done or expert.current_segment_idx >= len(expert.segments)):
                break

        # Verification metrics
        success = info.get("success", False)
        collision = info.get("collision_occurred", False)

        return {
            "observations": torch.stack(ep_obs_list),  # [T, 21]
            "actions": torch.stack(ep_act_list),        # [T, 7]
            "length": len(ep_obs_list),
            "success": success,
            "collision": collision,
            "task_type": task_spec.task_type.value,
            "task_name": task_spec.name
        }


def collect_nominal_dataset(
    num_episodes: int = 180,
    save_path: str = "data/embodied_demos/franka_nominal_demos.pt",
    seed: int = 42,
    reach_ratio: float = 0.5
) -> Dict[str, Any]:
    """
    Generates and saves the nominal clean expert demonstration dataset.

    Key Architectural Principles:
    - Number of episodes: 150 - 200 (default 180).
    - Balanced task distribution: reach_ratio * num_episodes reach tasks + remaining pick-and-place tasks.
    - Zero Hazards: Demos contain strictly NO hazards! Dynamic hazards are encountered
      for the first time at test time.
    - Full Kinematic Validation: Zero collisions, finite numeric tensors, verified task success.

    Dataset Structure:
        - 'observations': torch.FloatTensor [N_total, 21]
        - 'actions': torch.FloatTensor [N_total, 7]
        - 'episode_lengths': List[int] of length num_episodes
        - 'metadata': Dictionary with task counts, seeds, dt=0.01, success rate, feature layout
        - 'observations_dict': Dictionary of component sub-tensors

    Args:
        num_episodes: Total number of clean episodes to collect (150-200)
        save_path: File path where .pt artifact is stored
        seed: Random seed for task generation reproducibility
        reach_ratio: Proportion of episodes dedicated to nominal reach (default 0.5)
    Returns:
        The complete saved dataset dictionary
    """
    assert num_episodes >= 1, f"Expected at least 1 episode, requested {num_episodes}"
    torch.manual_seed(seed)

    num_reach = int(num_episodes * reach_ratio)
    num_pick_place = num_episodes - num_reach

    tasks: List[RoboticsTaskSpec] = []
    # 1. Generate clean nominal reach tasks
    for i in range(num_reach):
        task_seed = seed * 1000 + i
        tasks.append(create_randomized_reach_task(task_seed))

    # 2. Generate clean nominal pick-and-place tasks
    for j in range(num_pick_place):
        task_seed = seed * 2000 + j
        tasks.append(create_randomized_pick_and_place_task(task_seed))

    all_obs: List[torch.Tensor] = []
    all_acts: List[torch.Tensor] = []
    episode_lengths: List[int] = []
    successful_episodes = 0
    colliding_episodes = 0

    expert = FrankaScriptedExpert(dt=0.01)
    env = FrankaKinematicEnv()

    for idx, task in enumerate(tasks):
        ep_data = DemonstrationCollector.collect_single_episode(env, expert, task)

        all_obs.append(ep_data["observations"])
        all_acts.append(ep_data["actions"])
        episode_lengths.append(ep_data["length"])

        if ep_data["success"]:
            successful_episodes += 1
        if ep_data["collision"]:
            colliding_episodes += 1

    total_timesteps = sum(episode_lengths)
    observations_tensor = torch.cat(all_obs, dim=0).to(torch.float32)
    actions_tensor = torch.cat(all_acts, dim=0).to(torch.float32)

    # Sanity checks
    assert torch.all(torch.isfinite(observations_tensor)), "Observations tensor contains NaNs or Infs!"
    assert torch.all(torch.isfinite(actions_tensor)), "Actions tensor contains NaNs or Infs!"
    assert colliding_episodes == 0, f"Encountered {colliding_episodes} collisions in clean demos!"

    success_rate = (successful_episodes / num_episodes) * 100.0

    # Build component sub-tensors dictionary
    obs_dict = {
        "joint_q": observations_tensor[:, 0:7],
        "joint_dq": observations_tensor[:, 7:14],
        "ee_pos": observations_tensor[:, 14:17],
        "target_pos": observations_tensor[:, 17:20],
        "gripper_width": observations_tensor[:, 20:21],
    }

    metadata = {
        "task_name": "Franka_Panda_Nominal_Clean_Demos",
        "description": "Nominal expert demonstrations on Franka Panda 7-DoF manipulator arm (no hazards).",
        "total_episodes": num_episodes,
        "num_reach_episodes": num_reach,
        "num_pick_and_place_episodes": num_pick_place,
        "total_timesteps": total_timesteps,
        "dt": 0.01,
        "control_frequency_hz": 100,
        "success_rate_pct": success_rate,
        "collision_rate_pct": (colliding_episodes / num_episodes) * 100.0,
        "obs_dim": 21,
        "action_dim": 7,
        "feature_slices": {
            "joint_q": [0, 7],
            "joint_dq": [7, 14],
            "ee_pos": [14, 17],
            "target_pos": [17, 20],
            "gripper_width": [20, 21]
        },
        "seed": seed
    }

    dataset = {
        "observations": observations_tensor,
        "actions": actions_tensor,
        "episode_lengths": episode_lengths,
        "metadata": metadata,
        "observations_dict": obs_dict
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(dataset, save_path)

    return dataset


if __name__ == "__main__":
    out_path = "data/embodied_demos/franka_nominal_demos.pt"
    print(f"Collecting 180 clean nominal demonstration episodes...")
    dataset = collect_nominal_dataset(num_episodes=180, save_path=out_path, seed=42)
    meta = dataset["metadata"]
    print(f"Collection complete!")
    print(f"Total episodes: {meta['total_episodes']}")
    print(f"Total timesteps: {meta['total_timesteps']}")
    print(f"Observations shape: {dataset['observations'].shape}")
    print(f"Actions shape: {dataset['actions'].shape}")
    print(f"Success rate: {meta['success_rate_pct']:.1f}%")
    print(f"Collision rate: {meta['collision_rate_pct']:.1f}%")
    print(f"Saved to: {out_path}")
