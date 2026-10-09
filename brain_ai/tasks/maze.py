"""
brain_ai.tasks.maze: Programmatic 2D Maze Constraint Generator for Tier 1 Diagnostics.

Generates random N x N mazes with unique or multi-branch shortest paths.
Provides synchronized inputs for:
- Right Hemisphere (HRM): Grid cell spatial embeddings (Wall, Path, Start, Goal)
- Left Hemisphere (LLM): Natural language problem prompt & coordinates
- Ground Truth: Shortest path cell mask and direction sequence
"""

from collections import deque
from dataclasses import dataclass
import random
from typing import List, Tuple, Dict
import torch


@dataclass
class MazeBatch:
    grid_tokens: torch.Tensor       # [B, N*N] integer categories (0: Empty, 1: Wall, 2: Start, 3: Goal)
    path_targets: torch.Tensor      # [B, N*N] binary mask of shortest path cells
    text_prompts: List[str]         # Natural language prompt for LLM Left Hemisphere
    solution_paths: List[List[Tuple[int, int]]] # Coordinates [(r, c), ...]
    path_directions: List[str]      # E.g. "R, R, D, D, R, U..."


class MazeGenerator:
    """
    Randomized DFS maze generator with BFS shortest-path solver.
    Produces guaranteed solvable mazes with dead-ends.
    """
    def __init__(self, size: int = 15, seed: int = 42):
        assert size % 2 == 1, "Maze size must be odd for standard wall/corridor grids"
        self.size = size
        self.rng = random.Random(seed)

    def generate_single_maze(self) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]], str]:
        N = self.size
        # 1: Wall, 0: Passage
        grid = torch.ones((N, N), dtype=torch.long)
        
        # Carve passages using randomized DFS
        start_cell = (1, 1)
        grid[start_cell[0], start_cell[1]] = 0
        stack = [start_cell]
        visited = {start_cell}
        
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        
        while stack:
            curr_r, curr_c = stack[-1]
            neighbors = []
            
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                if 1 <= nr < N - 1 and 1 <= nc < N - 1 and (nr, nc) not in visited:
                    neighbors.append((nr, nc, dr, dc))
                    
            if neighbors:
                nr, nc, dr, dc = self.rng.choice(neighbors)
                # Carve wall between
                grid[curr_r + dr // 2, curr_c + dc // 2] = 0
                grid[nr, nc] = 0
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()
                
        # Set start and goal
        start_pos = (1, 1)
        goal_pos = (N - 2, N - 2)
        grid[start_pos[0], start_pos[1]] = 2 # Start
        grid[goal_pos[0], goal_pos[1]] = 3   # Goal
        
        # BFS Shortest Path
        queue = deque([(start_pos[0], start_pos[1], [start_pos])])
        bfs_visited = {start_pos}
        shortest_path = []
        
        step_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            r, c, path = queue.popleft()
            if (r, c) == goal_pos:
                shortest_path = path
                break
                
            for dr, dc in step_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in bfs_visited:
                    # Traversible if passage (0), start (2), or goal (3)
                    if grid[nr, nc].item() in (0, 2, 3):
                        bfs_visited.add((nr, nc))
                        queue.append((nr, nc, path + [(nr, nc)]))
                        
        # Target path binary mask
        path_mask = torch.zeros((N, N), dtype=torch.float32)
        for r, c in shortest_path:
            path_mask[r, c] = 1.0
            
        # Direction string
        dir_strs = []
        for i in range(len(shortest_path) - 1):
            r1, c1 = shortest_path[i]
            r2, c2 = shortest_path[i+1]
            if r2 == r1 + 1: dir_strs.append("D")
            elif r2 == r1 - 1: dir_strs.append("U")
            elif c2 == c1 + 1: dir_strs.append("R")
            elif c2 == c1 - 1: dir_strs.append("L")
            
        dir_sequence = ", ".join(dir_strs)
        return grid.view(-1), path_mask.view(-1), shortest_path, dir_sequence

    def generate_batch(self, batch_size: int = 8) -> MazeBatch:
        grids, masks, paths, directions, prompts = [], [], [], [], []
        
        for b in range(batch_size):
            g, m, p, d = self.generate_single_maze()
            grids.append(g)
            masks.append(m)
            paths.append(p)
            directions.append(d)
            
            prompt = (
                f"You are navigating a {self.size}x{self.size} grid maze from Start (1,1) to Goal ({self.size-2},{self.size-2}). "
                f"Identify the optimal shortest path without hitting walls. Steps required: {len(p) - 1} moves."
            )
            prompts.append(prompt)
            
        return MazeBatch(
            grid_tokens=torch.stack(grids, dim=0),
            path_targets=torch.stack(masks, dim=0),
            text_prompts=prompts,
            solution_paths=paths,
            path_directions=directions
        )
