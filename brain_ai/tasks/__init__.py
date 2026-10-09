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
    ARCTask
)

__all__ = [
    "MazeGenerator", 
    "MazeBatch", 
    "Spatial2DGridEmbedding", 
    "SpatialConvHead",
    "ARCDataset",
    "ARCBatch",
    "ARCTask"
]


