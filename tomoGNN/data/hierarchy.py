# hierarchy.py
# Hierarchy-aware edge weight computation
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


def extract_hierarchy_path(node_name: str) -> List[str]:
    """
    Extract hierarchy path from node name.

    Args:
        node_name: Node name like 'module_a/module_b/inst'

    Returns:
        List of hierarchy levels ['module_a', 'module_b', 'inst']

    Example:
        'ex_stage_i/alu_i/adder_0' ->
            ['ex_stage_i', 'alu_i', 'adder_0']
    """
    if '/' not in str(node_name):
        return [str(node_name)]

    return str(node_name).split('/')


def compute_lca_depth(path_a: List[str],
                      path_b: List[str]) -> int:
    """
    Compute depth of Lowest Common Ancestor (LCA).

    Args:
        path_a: Hierarchy path for node A
        path_b: Hierarchy path for node B

    Returns:
        Depth of LCA (number of matching prefix levels)

    Example:
        path_a = ['ex', 'alu', 'add']
        path_b = ['ex', 'alu', 'mul']
        -> LCA depth = 2 (match 'ex', 'alu')
    """
    lca_depth = 0
    for a, b in zip(path_a, path_b):
        if a == b:
            lca_depth += 1
        else:
            break
    return lca_depth


def compute_hierarchy_weight(path_a: List[str],
                             path_b: List[str]) -> float:
    """
    Compute hierarchy weight for edge (a, b).

    Formula:
        hij = 2 * (dLCA / max(depth_a, depth_b)) - 1

    Properties:
        - Range: [-1, 1]
        - hij ≈ 1: Same hierarchy (high similarity)
        - hij ≈ -1: Different hierarchy (low similarity)

    Args:
        path_a: Hierarchy path for node A
        path_b: Hierarchy path for node B

    Returns:
        Hierarchy weight in [-1, 1]
    """
    depth_a = len(path_a)
    depth_b = len(path_b)
    max_depth = max(depth_a, depth_b)

    if max_depth == 0:
        return 0.0

    lca_depth = compute_lca_depth(path_a, path_b)
    hij = 2.0 * (lca_depth / max_depth) - 1.0

    return hij


def compute_hierarchy_weights(
    edges_df: pd.DataFrame,
    id_to_name: Dict[int, str],
    show_progress: bool = True
) -> np.ndarray:
    """
    Compute hierarchy weights for all edges.

    Args:
        edges_df: DataFrame with 'src', 'dst' (integer IDs)
        id_to_name: Mapping from integer ID to node name
        show_progress: Whether to show progress bar

    Returns:
        Array of hierarchy weights [num_edges]

    Example:
        hij = compute_hierarchy_weights(edges_df, id_to_name)
        # Shape: [num_edges]
        # Range: [-1, 1]
    """
    print("Computing hierarchy-aware weights...")

    num_edges = len(edges_df)
    hij_weights = np.zeros(num_edges, dtype=np.float32)

    # Pre-compute hierarchy paths for all nodes
    node_paths = {}
    unique_ids = set(edges_df['src']) | set(edges_df['dst'])

    print(f"  Extracting paths for {len(unique_ids)} nodes...")
    for nid in unique_ids:
        if nid in id_to_name:
            node_paths[nid] = extract_hierarchy_path(
                id_to_name[nid]
            )
        else:
            node_paths[nid] = ['unknown']

    # Compute weights for each edge
    print(f"  Computing weights for {num_edges} edges...")

    iterator = enumerate(zip(edges_df['src'], edges_df['dst']))
    if show_progress and num_edges > 10000:
        iterator = tqdm(
            iterator,
            total=num_edges,
            desc="  Hierarchy weights"
        )

    for idx, (src_id, dst_id) in iterator:
        path_src = node_paths.get(src_id, ['unknown'])
        path_dst = node_paths.get(dst_id, ['unknown'])

        hij_weights[idx] = compute_hierarchy_weight(
            path_src,
            path_dst
        )

    # Print statistics
    print(f"  Hierarchy weight statistics:")
    print(f"    Range: [{hij_weights.min():.3f}, "
          f"{hij_weights.max():.3f}]")
    print(f"    Mean: {hij_weights.mean():.3f}")
    print(f"    Std: {hij_weights.std():.3f}")

    # Distribution analysis
    same_hier = (hij_weights > 0.5).sum()
    diff_hier = (hij_weights < -0.5).sum()
    mixed = num_edges - same_hier - diff_hier

    print(f"  Distribution:")
    print(f"    Same hierarchy (>0.5): {same_hier} "
          f"({100*same_hier/num_edges:.1f}%)")
    print(f"    Mixed (-0.5 to 0.5): {mixed} "
          f"({100*mixed/num_edges:.1f}%)")
    print(f"    Different (<-0.5): {diff_hier} "
          f"({100*diff_hier/num_edges:.1f}%)")

    return hij_weights


def add_hierarchy_to_weights(
    omega_plus_base: np.ndarray,
    omega_minus_base: np.ndarray,
    hij: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate hierarchy weights into base weights while preserving sign.

    Formula:
        omega_plus_final = omega_plus_base + hij
        omega_minus_final = omega_minus_base - hij

    Effect:
        - Same hierarchy (hij > 0):
          Increase attraction, decrease repulsion
        - Different hierarchy (hij < 0):
          Decrease attraction, increase repulsion (can flip sign)

    Args:
        omega_plus_base: Base attraction weights
        omega_minus_base: Base repulsion weights
        hij: Hierarchy weights [-1, 1]

    Returns:
        Tuple of (omega_plus_final, omega_minus_final) which may be negative
        if hierarchy adjustments outweigh the base terms.
    """
    # Additive hierarchy integration (preserve signed information)
    omega_plus_final = omega_plus_base + hij
    omega_minus_final = omega_minus_base - hij

    return omega_plus_final, omega_minus_final
