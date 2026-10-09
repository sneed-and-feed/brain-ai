"""
brain_ai.tasks: Benchmark and diagnostic task generators.
"""

from brain_ai.tasks.maze import (
    MazeGenerator, 
    MazeBatch, 
    Spatial2DGridEmbedding, 
    SpatialConvHead
)

__all__ = [
    "MazeGenerator", 
    "MazeBatch", 
    "Spatial2DGridEmbedding", 
    "SpatialConvHead"
]

