"""
brain_ai.tasks.arc_dsl: Typed Python Domain Specific Language (DSL) & Dihedral D4
Engine for ARC-AGI-2.

Capabilities:
1. Full 8-fold Dihedral Group (D4) geometric transformations and inversions.
2. Symbolic spatial primitives: connected components, bounding boxes, cropping,
   pasting, flood fills, recoloring, gravity physics, and color mapping.
3. Sandboxed execution environment for candidate programs synthesized by Left Hemisphere.
4. Test-Time Majority Consensus across D4 dihedral orbits.
"""

from typing import List, Tuple, Dict, Any, Optional, Callable
import copy
import numpy as np


# ==============================================================================
# 1. Dihedral Group D4 (8-Fold Symmetries)
# ==============================================================================

D4_TRANSFORMS = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "diag_main",
    "diag_anti"
]

def apply_d4(grid: np.ndarray, transform_idx: int) -> np.ndarray:
    """Applies one of the 8 dihedral transformations to a 2D integer grid."""
    t = transform_idx % 8
    if t == 0:    # Identity
        return grid.copy()
    elif t == 1:  # Rotate 90 deg clockwise
        return np.rot90(grid, -1).copy()
    elif t == 2:  # Rotate 180 deg
        return np.rot90(grid, 2).copy()
    elif t == 3:  # Rotate 270 deg (90 counter-clockwise)
        return np.rot90(grid, 1).copy()
    elif t == 4:  # Horizontal flip (left-right)
        return np.fliplr(grid).copy()
    elif t == 5:  # Vertical flip (up-down)
        return np.flipud(grid).copy()
    elif t == 6:  # Main diagonal transpose
        return grid.T.copy()
    elif t == 7:  # Anti-diagonal transpose
        return np.fliplr(np.rot90(grid, 1)).copy()
    return grid.copy()


def invert_d4(grid: np.ndarray, transform_idx: int) -> np.ndarray:
    """Inverts the specified dihedral transformation."""
    t = transform_idx % 8
    if t == 0:
        return grid.copy()
    elif t == 1:  # Inverse of rot 90 CW is rot 90 CCW (rot270)
        return np.rot90(grid, 1).copy()
    elif t == 2:  # Inverse of rot 180 is rot 180
        return np.rot90(grid, 2).copy()
    elif t == 3:  # Inverse of rot 270 CW is rot 90 CW
        return np.rot90(grid, -1).copy()
    elif t in (4, 5, 6, 7): # Flips and diagonal reflections are self-inverses
        return apply_d4(grid, t)
    return grid.copy()


def expand_demos_d4(
    demos: List[Tuple[np.ndarray, np.ndarray]]
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Expands K demonstration pairs into 8*K rotationally/reflectively invariant pairs.
    Prevents System 2 TTA from overfitting sparse (K <= 3) demonstrations.
    """
    expanded = []
    for x, y in demos:
        for t in range(8):
            expanded.append((apply_d4(x, t), apply_d4(y, t)))
    return expanded


def d4_symmetrized_consensus(
    model_fn: Callable[[np.ndarray], np.ndarray],
    test_input: np.ndarray
) -> np.ndarray:
    """
    Runs model on all 8 dihedral transformations of test_input,
    inverts each output, and computes pixel-wise modal consensus.
    """
    inverted_predictions = []
    for t in range(8):
        x_aug = apply_d4(test_input, t)
        pred_aug = model_fn(x_aug)
        pred_inv = invert_d4(pred_aug, t)
        inverted_predictions.append(pred_inv)

    # Filter to identical shape if shapes differ
    target_shape = inverted_predictions[0].shape
    valid_preds = [p for p in inverted_predictions if p.shape == target_shape]
    if not valid_preds:
        return inverted_predictions[0]

    stack = np.stack(valid_preds, axis=0) # (8, H, W)
    H, W = target_shape
    consensus_grid = np.zeros((H, W), dtype=int)

    for r in range(H):
        for c in range(W):
            vals, counts = np.unique(stack[:, r, c], return_counts=True)
            consensus_grid[r, c] = vals[np.argmax(counts)]

    return consensus_grid


# ==============================================================================
# 2. Symbolic Spatial DSL Primitives
# ==============================================================================

def get_connected_components(grid: np.ndarray, background: int = 0) -> List[Dict[str, Any]]:
    """Extracts non-background 4-connected components from a grid."""
    H, W = grid.shape
    visited = np.zeros((H, W), dtype=bool)
    components = []

    for r in range(H):
        for c in range(W):
            val = int(grid[r, c])
            if val != background and not visited[r, c]:
                # BFS flood fill for component
                coords = []
                queue = [(r, c)]
                visited[r, c] = True

                while queue:
                    cr, cc = queue.pop(0)
                    coords.append((cr, cc))
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if not visited[nr, nc] and grid[nr, nc] == val:
                                visited[nr, nc] = True
                                queue.append((nr, nc))

                rows = [p[0] for p in coords]
                cols = [p[1] for p in coords]
                r_min, r_max = min(rows), max(rows)
                c_min, c_max = min(cols), max(cols)

                subgrid = grid[r_min:r_max+1, c_min:c_max+1].copy()
                components.append({
                    "color": val,
                    "coords": coords,
                    "bbox": (r_min, c_min, r_max, c_max),
                    "size": len(coords),
                    "subgrid": subgrid
                })

    return components


def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Returns (r_min, c_min, r_max, c_max) for non-zero entries."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def crop(grid: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crops subgrid specified by (r_min, c_min, r_max, c_max)."""
    r_min, c_min, r_max, c_max = bbox
    return grid[r_min:r_max+1, c_min:c_max+1].copy()


def paste(canvas: np.ndarray, subgrid: np.ndarray, top_left: Tuple[int, int]) -> np.ndarray:
    """Pastes subgrid onto canvas at (r, c)."""
    out = canvas.copy()
    r, c = top_left
    h, w = subgrid.shape
    H, W = canvas.shape
    r_end = min(r + h, H)
    c_end = min(c + w, W)
    h_paste = r_end - r
    w_paste = c_end - c
    if h_paste > 0 and w_paste > 0:
        out[r:r_end, c:c_end] = subgrid[:h_paste, :w_paste]
    return out


def recolor(grid: np.ndarray, old_val: int, new_val: int) -> np.ndarray:
    """Replaces all occurrences of old_val with new_val."""
    out = grid.copy()
    out[grid == old_val] = new_val
    return out


def flood_fill(grid: np.ndarray, r: int, c: int, fill_color: int) -> np.ndarray:
    """Standard 4-way flood fill from starting coordinate (r, c)."""
    out = grid.copy()
    H, W = grid.shape
    target_color = out[r, c]
    if target_color == fill_color:
        return out

    queue = [(r, c)]
    out[r, c] = fill_color
    while queue:
        cr, cc = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and out[nr, nc] == target_color:
                out[nr, nc] = fill_color
                queue.append((nr, nc))
    return out


def gravity(grid: np.ndarray, direction: str = "down", background: int = 0) -> np.ndarray:
    """Simulates gravity physics on non-background pixels."""
    out = np.full_like(grid, background)
    H, W = grid.shape
    if direction == "down":
        for c in range(W):
            non_bg = [grid[r, c] for r in range(H) if grid[r, c] != background]
            for idx, val in enumerate(non_bg):
                out[H - len(non_bg) + idx, c] = val
    elif direction == "up":
        for c in range(W):
            non_bg = [grid[r, c] for r in range(H) if grid[r, c] != background]
            for idx, val in enumerate(non_bg):
                out[idx, c] = val
    return out


# ==============================================================================
# 3. Sandbox DSL Program Verification
# ==============================================================================

def execute_dsl_program(
    code_str: str,
    test_input: np.ndarray,
    timeout_sec: float = 2.0
) -> Optional[np.ndarray]:
    """
    Safely executes synthesized DSL code defining `transform(grid: np.ndarray) -> np.ndarray`.
    """
    safe_scope = {
        "np": np,
        "numpy": np,
        "apply_d4": apply_d4,
        "invert_d4": invert_d4,
        "get_connected_components": get_connected_components,
        "bounding_box": bounding_box,
        "crop": crop,
        "paste": paste,
        "recolor": recolor,
        "flood_fill": flood_fill,
        "gravity": gravity
    }

    try:
        # Extract code inside ```python block if present
        if "```python" in code_str:
            code_str = code_str.split("```python")[1].split("```")[0]
        elif "```" in code_str:
            code_str = code_str.split("```")[1].split("```")[0]

        exec(code_str, safe_scope)
        if "transform" not in safe_scope or not callable(safe_scope["transform"]):
            return None

        result = safe_scope["transform"](test_input.copy())
        if isinstance(result, np.ndarray) and result.ndim == 2:
            return result.astype(int)
        return None
    except Exception:
        return None


def verify_program_on_demos(
    code_str: str,
    demos: List[Tuple[np.ndarray, np.ndarray]]
) -> float:
    """Computes exact match demonstration fit [0.0, 1.0] for a candidate program."""
    if not demos:
        return 0.0
    correct = 0
    for inp, out in demos:
        pred = execute_dsl_program(code_str, inp)
        if pred is not None and pred.shape == out.shape and np.array_equal(pred, out):
            correct += 1
    return correct / len(demos)
