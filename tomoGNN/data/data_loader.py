# data_loader.py
# Load multi-run CSV data for netlist tomography
from __future__ import annotations
import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict, Optional


def detect_num_runs(df: pd.DataFrame,
                    prefix: str = 'slack') -> int:
    """
    Detect number of runs from column names.

    Args:
        df: DataFrame with columns like slack_1, slack_2, ...
        prefix: Column prefix to search for

    Returns:
        Number of detected runs
    """
    pattern = re.compile(f'{prefix}_(\\d+)')
    run_numbers = []

    for col in df.columns:
        match = pattern.match(col)
        if match:
            run_numbers.append(int(match.group(1)))

    if not run_numbers:
        return 0

    return max(run_numbers)


def load_nodes_csv(nodes_csv: str) -> Tuple[pd.DataFrame, int]:
    """
    Load node data from merged CSV.

    Expected columns:
        Instance, Cell, ClockPeriod, Width, Height,
        slack_1...slack_N, pt_x_1...pt_N, pt_y_1...pt_N

    Returns:
        - nodes_df: DataFrame with Instance, x, y
        - num_runs: Detected number of runs
    """
    print(f"Loading nodes from {nodes_csv}")
    df = pd.read_csv(nodes_csv)

    # Detect number of runs
    num_runs = detect_num_runs(df, 'slack')
    if num_runs == 0:
        raise ValueError(
            "No slack_N columns found in nodes CSV"
        )

    print(f"  Detected {num_runs} runs")
    print(f"  Loaded {len(df)} nodes")

    # Keep all per-run coordinates
    x_cols = [f'pt_x_{i}' for i in range(1, num_runs + 1)]
    y_cols = [f'pt_y_{i}' for i in range(1, num_runs + 1)]

    # Check if coordinate columns exist
    missing_x = [c for c in x_cols if c not in df.columns]
    missing_y = [c for c in y_cols if c not in df.columns]

    if missing_x or missing_y:
        raise ValueError(
            f"Missing coordinate columns: {missing_x + missing_y}"
        )

    # Extract all per-run coordinates (no averaging)
    coord_cols = x_cols + y_cols
    nodes_df = df[['Instance'] + coord_cols].copy()

    # Rename Instance to node_id
    nodes_df.rename(columns={'Instance': 'node_id'}, inplace=True)

    # Compute average for display purposes only
    x_avg = df[x_cols].mean(axis=1)
    y_avg = df[y_cols].mean(axis=1)

    print(f"  Coordinate ranges (averaged for display):")
    print(f"    x: [{x_avg.min():.1f}, {x_avg.max():.1f}]")
    print(f"    y: [{y_avg.min():.1f}, {y_avg.max():.1f}]")

    return nodes_df, num_runs


def load_edges_csv(edges_csv: str,
                   num_runs: int) -> Tuple[pd.DataFrame,
                                            List[Dict]]:
    """
    Load edge data from merged CSV.

    Expected columns:
        Net, Source, Sink, ClockPeriod,
        length_1...length_N, slack_1...slack_N

    Returns:
        - edges_df: DataFrame with src, dst
        - runs_data: List of {length: array, slack: array}
    """
    print(f"Loading edges from {edges_csv}")
    df = pd.read_csv(edges_csv)

    # Verify expected columns
    required = ['Source', 'Sink']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"  Loaded {len(df)} edges")

    # Rename columns
    edges_df = df[['Source', 'Sink']].copy()
    edges_df.columns = ['src', 'dst']

    # Extract per-run data
    runs_data = []
    for run_idx in range(1, num_runs + 1):
        length_col = f'length_{run_idx}'
        slack_col = f'slack_{run_idx}'

        if length_col not in df.columns:
            raise ValueError(f"Missing column: {length_col}")
        if slack_col not in df.columns:
            raise ValueError(f"Missing column: {slack_col}")

        runs_data.append({
            'length': df[length_col].to_numpy(dtype=float),
            'slack': df[slack_col].to_numpy(dtype=float)
        })

    print(f"  Extracted data for {num_runs} runs")

    # Print statistics for first run
    lengths = runs_data[0]['length']
    slacks = runs_data[0]['slack']
    print(f"  Run 1 statistics:")
    print(f"    Length: [{lengths.min():.2f}, "
          f"{lengths.max():.2f}], "
          f"mean={lengths.mean():.2f}")
    print(f"    Slack: [{slacks.min():.3f}, "
          f"{slacks.max():.3f}], "
          f"mean={slacks.mean():.3f}")

    return edges_df, runs_data


def reindex_node_ids(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convert node IDs to contiguous integers [0..N-1].

    Args:
        nodes_df: DataFrame with 'node_id', x, y
        edges_df: DataFrame with 'src', 'dst'

    Returns:
        - nodes_df: with added 'nid' column [0..N-1]
        - edges_df: with src/dst mapped to integer IDs
        - id_to_name: Dict mapping int ID -> original name
    """
    print("Reindexing node IDs...")

    # Create categorical mapping
    node_names = nodes_df['node_id'].astype('category')
    nodes_df['nid'] = node_names.cat.codes

    # Create mapping dictionaries
    name_to_id = dict(
        zip(node_names, nodes_df['nid'])
    )
    id_to_name = dict(
        zip(nodes_df['nid'], nodes_df['node_id'])
    )

    # Remap edges
    edges_df['src'] = edges_df['src'].map(name_to_id)
    edges_df['dst'] = edges_df['dst'].map(name_to_id)

    # Check for unmapped edges
    unmapped = edges_df[
        edges_df['src'].isna() | edges_df['dst'].isna()
    ]
    if len(unmapped) > 0:
        print(f"  Warning: {len(unmapped)} edges with "
              f"unknown nodes")
        edges_df = edges_df.dropna(subset=['src', 'dst'])

    # Convert to integers
    edges_df['src'] = edges_df['src'].astype(int)
    edges_df['dst'] = edges_df['dst'].astype(int)

    print(f"  Mapped {len(nodes_df)} nodes to IDs [0..{len(nodes_df)-1}]")
    print(f"  Mapped {len(edges_df)} edges")

    return nodes_df, edges_df, id_to_name


# def load_multi_run_data(
#     nodes_csv: str,
#     edges_csv: str
# ) -> Tuple[pd.DataFrame,
#            pd.DataFrame,
#            List[Dict],
#            Dict,
#            int]:
#     """
#     Load complete multi-run dataset.

#     Args:
#         nodes_csv: Path to merged nodes CSV
#         edges_csv: Path to merged edges CSV

#     Returns:
#         - nodes_df: With nid, x, y, node_id
#         - edges_df: With src, dst (integer IDs)
#         - runs_data: Per-run length/slack arrays
#         - id_to_name: Mapping from integer ID to node name
#         - num_runs: Number of tomography runs

#     Example:
#         nodes_df, edges_df, runs_data, id_map, N = \\
#             load_multi_run_data(
#                 'ariane_merged_nodes.csv',
#                 'ariane_merged_edges.csv'
#             )
#     """
#     print("="*60)
#     print("Loading Multi-Run Data")
#     print("="*60)

#     # Load nodes
#     nodes_df, num_runs = load_nodes_csv(nodes_csv)

#     # Load edges
#     edges_df, runs_data = load_edges_csv(edges_csv, num_runs)

#     # Reindex IDs
#     nodes_df, edges_df, id_to_name = reindex_node_ids(
#         nodes_df,
#         edges_df
#     )

#     print("="*60)
#     print(f"Data Loading Complete: {len(nodes_df)} nodes, "
#           f"{len(edges_df)} edges, {num_runs} runs")
#     print("="*60)

#     return nodes_df, edges_df, runs_data, id_to_name, num_runs


def load_multi_run_data(
    nodes_csv: str,
    edges_csv: str,
    max_runs: Optional[int] = None
) -> Tuple[pd.DataFrame,
           pd.DataFrame,
           List[Dict],
           Dict,
           int]:
    """
    Load complete multi-run dataset.

    Args:
        nodes_csv: Path to merged nodes CSV
        edges_csv: Path to merged edges CSV
        max_runs: Optional cap on number of placement runs to load (use
            the first `max_runs` runs; None loads all available runs)

    Returns:
        - nodes_df: With nid, x, y, node_id
        - edges_df: With src, dst (integer IDs)
        - runs_data: Per-run length/slack arrays
        - id_to_name: Mapping from integer ID to node name
        - num_runs: Number of tomography runs

    Example:
        nodes_df, edges_df, runs_data, id_map, N = \\
            load_multi_run_data(
                'ariane_merged_nodes.csv',
                'ariane_merged_edges.csv'
            )
    """
    print("="*60)
    print("Loading Multi-Run Data")
    print("="*60)

    # Load nodes
    nodes_df, num_runs_detected = load_nodes_csv(nodes_csv)
    
    # Determine number of runs to keep
    if max_runs is not None and max_runs > 0:
        num_runs = min(max_runs, num_runs_detected)
        if num_runs < num_runs_detected:
            print(f"Limiting to first {num_runs} runs out of {num_runs_detected} available")
    else:
        num_runs = num_runs_detected
    
    if num_runs <= 0:
        raise ValueError("At least one placement run is required to build the dataset")
    
    # Load edges (only up to the selected number of runs)
    edges_df, runs_data = load_edges_csv(edges_csv, num_runs)

    # If we limited runs, drop unused per-run coordinate columns
    if num_runs < num_runs_detected:
        x_cols = [f'pt_x_{i}' for i in range(1, num_runs + 1)]
        y_cols = [f'pt_y_{i}' for i in range(1, num_runs + 1)]
        nodes_df = nodes_df[['node_id'] + x_cols + y_cols].copy()
    
    # Reindex IDs
    nodes_df, edges_df, id_to_name = reindex_node_ids(
        nodes_df,
        edges_df
    )

    print("="*60)
    print(f"Data Loading Complete: {len(nodes_df)} nodes, "
          f"{len(edges_df)} edges, {num_runs} runs (detected {num_runs_detected})")
    print("="*60)

    return nodes_df, edges_df, runs_data, id_to_name, num_runs



def get_clock_period(nodes_csv: str) -> float:
    """
    Extract clock period from nodes CSV.

    Args:
        nodes_csv: Path to nodes CSV

    Returns:
        Clock period value
    """
    df = pd.read_csv(nodes_csv, nrows=1)

    if 'ClockPeriod' not in df.columns:
        print("Warning: ClockPeriod column not found, "
              "using default 1.3")
        return 1.3

    cp = float(df['ClockPeriod'].iloc[0])
    print(f"Clock period: {cp}")
    return cp
