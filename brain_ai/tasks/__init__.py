"""
brain_ai.tasks: Benchmark and diagnostic task generators.
"""

from brain_ai.tasks.maze import (
    MazeGenerator, 
    MazeBatch, 
    Spatial2DGridEmbedding, 
    SpatialConvHead
)
from brain_ai.tasks.arc import (
    ARCDataset,
    ARCBatch,
    ARCTask,
    ARCSpatialGridEmbedding,
    ARCPredictionHead
)

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
    create_pick_and_place_task,
    create_obstacle_field_task,
    BaseExecutionPolicy,
    MonolithicVLAPolicy,
    BiHemisphericRoboticsPolicy,
    EpisodeMetrics,
    BenchmarkSummary,
    RoboticsBenchmarkRunner,
    compute_path_smoothness,
)

__all__ = [
    "MazeGenerator", 
    "MazeBatch", 
    "Spatial2DGridEmbedding", 
    "SpatialConvHead",
    "ARCDataset",
    "ARCBatch",
    "ARCTask",
    "ARCSpatialGridEmbedding",
    "ARCPredictionHead",
    "FrankaKinematics",
    "FrankaKinematicEnv",
    "GymnasiumMuJoCoRoboticsAdapter",
    "make_robotics_env",
    "Obstacle",
    "ObstacleType",
    "CapsuleCollisionChecker",
    "ER2TaskType",
    "RoboticsTaskSpec",
    "create_nominal_reach_task",
    "create_pick_and_place_task",
    "create_obstacle_field_task",
    "BaseExecutionPolicy",
    "MonolithicVLAPolicy",
    "BiHemisphericRoboticsPolicy",
    "EpisodeMetrics",
    "BenchmarkSummary",
    "RoboticsBenchmarkRunner",
    "compute_path_smoothness",
]


