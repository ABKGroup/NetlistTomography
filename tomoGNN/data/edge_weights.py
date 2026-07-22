# edge_weights.py
# Multi-run edge weight computation
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from config import TrainConfig
from data.hierarchy import compute_hierarchy_weights, \
    add_hierarchy_to_weights
from data.multi_run_module import MultiRunWeightingModule


def compute_criticality(
    slack: np.ndarray,
    cp: float
) -> np.ndarray:
    """
    Compute criticality from slack values.

    Formula:
        c = clip(max(0, -slack) / cp, 0, 1)

    Args:
        slack: Slack values (negative = critical)
        cp: Clock period

    Returns:
        Criticality in [0, 1]
    """
    c = np.maximum(0.0, -slack) / max(cp, 1e-9)
    return np.clip(c, 0.0, 1.0)


def normalize_length(
    length: np.ndarray,
    max_clip: float = 3.0
) -> np.ndarray:
    """
    Normalize length using 95th percentile.

    Formula:
        L = clip(length / p95, 0, max_clip)

    Args:
        length: Raw length values
        max_clip: Maximum normalized length

    Returns:
        Normalized length in [0, max_clip]
    """
    if len(length) == 0:
        return length

    p95 = np.percentile(length, 95)
    L = length / max(p95, 1e-9)
    return np.clip(L, 0.0, max_clip)


def prepare_run_data(
    runs_data: List[Dict],
    cp: float,
    cfg: TrainConfig
) -> List[Dict]:
    """
    Prepare per-run criticality and normalized length.

    Args:
        runs_data: List of {length: array, slack: array}
        cp: Clock period
        cfg: Training configuration

    Returns:
        List of {criticality: array, length: array}
    """
    processed_runs = []

    for idx, run_data in enumerate(runs_data):
        slack = run_data['slack']
        length = run_data['length']

        # Compute criticality
        criticality = compute_criticality(slack, cp)

        # Normalize length
        norm_length = normalize_length(
            length,
            cfg.max_len_clip
        )

        processed_runs.append({
            'criticality': criticality,
            'length': norm_length
        })

    return processed_runs


def compute_multi_run_weights(
    processed_runs: List[Dict],
    cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute aggregated multi-run weights.

    Uses MultiRunWeightingModule to aggregate across runs:
        - Learnable mode: softmax(alpha), softmax(beta)
        - Naive mode: equal weights

    Args:
        processed_runs: Per-run criticality/length data
        cfg: Training configuration

    Returns:
        Tuple of (omega_plus_base, omega_minus_base)
    """
    print("Computing multi-run weights...")

    # Create weighting module
    module = MultiRunWeightingModule(cfg)

    # Stack data into tensors
    criticality_runs = torch.stack([
        torch.from_numpy(run['criticality']).float()
        for run in processed_runs
    ])  # [num_runs, num_edges]

    length_runs = torch.stack([
        torch.from_numpy(run['length']).float()
        for run in processed_runs
    ])  # [num_runs, num_edges]

    # Compute aggregated weights
    with torch.no_grad():
        omega_plus_base, omega_minus_base = module(
            criticality_runs,
            length_runs
        )

    # Convert to numpy
    omega_plus_base = omega_plus_base.numpy()
    omega_minus_base = omega_minus_base.numpy()

    print(f"  omega_plus_base: "
          f"[{omega_plus_base.min():.3f}, "
          f"{omega_plus_base.max():.3f}], "
          f"mean={omega_plus_base.mean():.3f}")
    print(f"  omega_minus_base: "
          f"[{omega_minus_base.min():.3f}, "
          f"{omega_minus_base.max():.3f}], "
          f"mean={omega_minus_base.mean():.3f}")

    return omega_plus_base, omega_minus_base


def compute_edge_weights_with_hierarchy(
    edges_df: pd.DataFrame,
    runs_data: List[Dict],
    id_to_name: Dict,
    cp: float,
    cfg: TrainConfig
) -> pd.DataFrame:
    """
    Complete edge weight computation with hierarchy.

    Pipeline:
    1. Prepare per-run criticality and normalized length
    2. Aggregate multi-run weights (learnable or naive)
    3. Compute hierarchy weights
    4. Integrate hierarchy into final weights

    Args:
        edges_df: DataFrame with src, dst
        runs_data: Per-run slack/length data
        id_to_name: Node ID to name mapping
        cp: Clock period
        cfg: Training configuration

    Returns:
        DataFrame with computed weights and attributes
    """
    print("="*60)
    print("Computing Edge Weights")
    print("="*60)

    # Step 1: Prepare run data
    processed_runs = prepare_run_data(runs_data, cp, cfg)

    # Step 2: Compute multi-run base weights
    omega_plus_base, omega_minus_base = compute_multi_run_weights(
        processed_runs,
        cfg
    )

    # Step 3: Compute hierarchy weights
    hij = compute_hierarchy_weights(
        edges_df,
        id_to_name,
        show_progress=True
    )

    # Step 4: Integrate hierarchy
    omega_plus_final, omega_minus_final = \
        add_hierarchy_to_weights(
            omega_plus_base,
            omega_minus_base,
            hij
        )

    # Build result DataFrame
    result_df = edges_df.copy()

    # Store intermediate values (from first run for reference)
    result_df['c'] = processed_runs[0]['criticality']
    result_df['L'] = processed_runs[0]['length']

    # Store base weights
    result_df['omega_plus_base'] = omega_plus_base
    result_df['omega_minus_base'] = omega_minus_base

    # Store hierarchy weight
    result_df['hij'] = hij

    # Store FINAL weights (used in training)
    result_df['omega_plus'] = omega_plus_final
    result_df['omega_minus'] = omega_minus_final

    print("="*60)
    print("Edge Weight Computation Complete")
    print(f"  Final omega_plus: "
          f"[{omega_plus_final.min():.3f}, "
          f"{omega_plus_final.max():.3f}]")
    print(f"  Final omega_minus: "
          f"[{omega_minus_final.min():.3f}, "
          f"{omega_minus_final.max():.3f}]")
    print("="*60)

    return result_df
