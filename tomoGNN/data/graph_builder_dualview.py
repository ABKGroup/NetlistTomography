# graph_builder_dualview.py
# Build dual-view graph data for multi-run GNN
from __future__ import annotations
import torch
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple
from config import TrainConfig
from data.data_loader import load_multi_run_data
from data.edge_weights import compute_edge_weights_with_hierarchy
from data.graph_builder import (
    build_node_features,
    normalize_features,
    make_undirected,
    compute_node_degrees
)
from data.colocation_graph import (
    build_colocation_graphs_all_runs,
    make_undirected_colocation_graph,
    compute_run_quality_scores
)


def prepare_netlist_graph_features(
    nodes_df: pd.DataFrame,
    edges_with_weights: pd.DataFrame,
    num_runs: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare netlist graph G features.
    
    Returns:
        x: [N, feature_dim] node features (coordinates + degree)
        edge_index_G: [2, E_G] netlist edges
        edge_attr_G: [E_G, 2] edge features [criticality, length]
        edge_weight_G: [E_G] scalar edge weights (omega_plus)
    """
    num_nodes = len(nodes_df)
    
    # Build node features: [pt_x_1, ..., pt_x_N, pt_y_1, ..., pt_y_N, degree]
    x = build_node_features(nodes_df, num_runs)
    
    # Build edge index (convert to stacked array first to avoid warning)
    edge_index_G = torch.tensor(
        np.stack([
            edges_with_weights['src'].to_numpy(),
            edges_with_weights['dst'].to_numpy()
        ], axis=0),
        dtype=torch.long
    )  # [2, E_G]
    
    # Edge attributes: [c, L] (criticality and length from first run)
    edge_attr_G = torch.tensor(
        edges_with_weights[['c', 'L']].to_numpy(),
        dtype=torch.float
    )  # [E_G, 2]
    
    # Edge weights: use omega_plus (attraction weight) as scalar weight for model
    # omega_plus may be negative after hierarchy integration; message passing
    # expects non-negative magnitudes, so we clamp for the GraphSAGE layer but
    # keep the signed values in the full weight tensor for sampling and losses.
    omega_plus_np = edges_with_weights['omega_plus'].to_numpy()
    edge_weight_G_signed = torch.tensor(omega_plus_np, dtype=torch.float)
    edge_weight_G = torch.clamp(edge_weight_G_signed, min=0.0)  # [E_G]
    
    # Store both omega_plus and omega_minus (signed) for pair sampling / loss
    edge_weights_G_full = torch.tensor(
        edges_with_weights[['omega_plus', 'omega_minus']].to_numpy(),
        dtype=torch.float
    )  # [E_G, 2] - signed weights preserved
    
    # Add degree feature
    degrees = compute_node_degrees(edge_index_G, num_nodes)
    x[:, -1] = torch.log(1.0 + degrees)
    
    # Normalize features
    x = normalize_features(x)
    
    return x, edge_index_G, edge_attr_G, edge_weight_G, edge_weights_G_full


def prepare_per_run_data(
    x: torch.Tensor,
    edge_index_G: torch.Tensor,
    edge_attr_G: torch.Tensor,
    edge_weight_G: torch.Tensor,
    edge_weights_G_full: torch.Tensor,
    colocation_graphs: List[Dict],
    undirected: bool = True
) -> List[Dict]:
    """
    Prepare data for each run.
    
    Args:
        x: [N, feature_dim] node features
        edge_index_G: [2, E_G] netlist edges
        edge_attr_G: [E_G, 2] netlist edge features
        edge_weight_G: [E_G] non-negative netlist edge weights for message passing
        edge_weights_G_full: [E_G, 2] signed weights [omega_plus, omega_minus]
        colocation_graphs: List of co-location graphs per run
        undirected: Whether to make graphs undirected
    """
    data_per_run = []
    
    # Process netlist graph G (shared across runs)
    if undirected:
        edge_index_G_proc, edge_attr_G_proc, edge_weight_G_proc = \
            make_undirected(edge_index_G, edge_attr_G, edge_weight_G)
        # Also process full weights for sampling
        _, _, edge_weights_G_full_proc = \
            make_undirected(edge_index_G, edge_attr_G, edge_weights_G_full)
    else:
        edge_index_G_proc = edge_index_G
        edge_attr_G_proc = edge_attr_G
        edge_weight_G_proc = edge_weight_G
        edge_weights_G_full_proc = edge_weights_G_full
    
    # Process each run's co-location graph
    for run_id, coloc_data in enumerate(colocation_graphs):
        edge_index_H = coloc_data['edge_index_H']
        edge_attr_H = coloc_data['edge_attr_H']
        edge_weight_H = coloc_data['edge_weight_H']
        
        # Make undirected if needed
        if undirected:
            edge_index_H, edge_attr_H, edge_weight_H = \
                make_undirected_colocation_graph(
                    edge_index_H, 
                    edge_attr_H, 
                    edge_weight_H
                )
        
        # Create data dict for this run
        run_data = {
            'x': x,  # Shared node features
            'edge_index_G': edge_index_G_proc,
            'edge_attr_G': edge_attr_G_proc,
            'edge_weight_G': edge_weight_G_proc,  # [E_G] scalar weights for model
            'edge_weights_G_full': edge_weights_G_full_proc,  # [E_G, 2] full weights for sampling
            'edge_index_H': edge_index_H,
            'edge_attr_H': edge_attr_H,
            'edge_weight_H': edge_weight_H
        }
        
        data_per_run.append(run_data)
    
    return data_per_run


def load_and_build_dualview_graph(
    nodes_csv: str,
    edges_csv: str,
    cp: float,
    cfg: TrainConfig,
    colocation_q: float = None,
    colocation_radius: float = None,
    colocation_sigma: float = None,
    colocation_M: int = None,
    colocation_S: int = None,
    colocation_n_jobs: int = None
) -> Tuple[List[Dict], torch.Tensor, Dict, List[np.ndarray], List[float]]:
    """
    Complete pipeline for dual-view graph construction.
    
    All colocation parameters default to values from cfg if not explicitly provided.
    
    Args:
        nodes_csv: Path to nodes CSV
        edges_csv: Path to edges CSV
        cp: Clock period
        cfg: Training configuration
        colocation_q: Adaptive radius multiplier (None = use cfg.colocation_q)
        colocation_radius: Fixed radius (None = use cfg.colocation_radius)
        colocation_sigma: Gaussian kernel bandwidth (None = use cfg.colocation_sigma)
        colocation_M: Query points for radius (None = use cfg.colocation_M)
        colocation_S: Subsample size for NN index (None = use cfg.colocation_S)
        colocation_n_jobs: Number of workers (None = use cfg.colocation_n_jobs)
        
    Returns:
        data_per_run: List of dicts with dual-graph data per run
        run_quality: [num_runs] quality scores
        id_to_name: Node ID to name mapping
        coordinates_all_runs: List of [N, 2] coordinate arrays for each run
        sigmas_all_runs: List of sigma values for each run
    """
    # Use config values if not provided
    if colocation_q is None:
        colocation_q = getattr(cfg, 'colocation_q', 1.0)
    if colocation_radius is None:
        colocation_radius = getattr(cfg, 'colocation_radius', None)
    if colocation_sigma is None:
        colocation_sigma = getattr(cfg, 'colocation_sigma', 10.0)
    if colocation_M is None:
        colocation_M = getattr(cfg, 'colocation_M', 10000)
    if colocation_S is None:
        colocation_S = getattr(cfg, 'colocation_S', 100000)
    if colocation_n_jobs is None:
        colocation_n_jobs = getattr(cfg, 'colocation_n_jobs', 1)
    print("="*60)
    print("Building Dual-View Multi-Run Graph")
    print("="*60)
    
    # Load multi-run data
    t_start = time.time()
    # nodes_df, edges_df, runs_data, id_to_name, num_runs = \
    #     load_multi_run_data(nodes_csv, edges_csv)
    nodes_df, edges_df, runs_data, id_to_name, num_runs = \
        load_multi_run_data(
            nodes_csv,
            edges_csv,
            getattr(cfg, 'max_runs', None)
        )
    
    print(f"⏱️  Data loading time: {time.time() - t_start:.2f}s")
    
    # Set num_runs in config (required for MultiRunWeightingModule)
    cfg.num_runs = num_runs
    
    # Compute edge weights for netlist graph
    t_start = time.time()
    edges_with_weights = compute_edge_weights_with_hierarchy(
        edges_df,
        runs_data,
        id_to_name,
        cp,
        cfg
    )
    print(f"⏱️  Edge weight computation time: {time.time() - t_start:.2f}s")
    
    # Prepare netlist graph G
    x, edge_index_G, edge_attr_G, edge_weight_G, edge_weights_G_full = \
        prepare_netlist_graph_features(
            nodes_df,
            edges_with_weights,
            num_runs
        )
    
    print("="*60)
    print("Netlist Graph G:")
    print(f"  Nodes: {x.size(0)}")
    print(f"  Edges: {edge_index_G.size(1)}")
    print(f"  Node features: {x.size(1)}")
    print(f"  Edge features: {edge_attr_G.size(1)}")
    print("="*60)
    
    # Build co-location graphs H for each run (using config parameters)
    t_start = time.time()
    colocation_graphs = build_colocation_graphs_all_runs(
        nodes_df,
        num_runs,
        radius=colocation_radius,
        sigma=colocation_sigma,
        q=colocation_q,
        M=colocation_M,
        n_jobs=colocation_n_jobs,
        sample_index_size=colocation_S
    )
    print(f"⏱️  Co-location graph building time: {time.time() - t_start:.2f}s ({(time.time() - t_start) / num_runs:.2f}s per run)")
    
    # Compute run quality scores
    t_start = time.time()
    run_quality = compute_run_quality_scores(
        nodes_df,
        edges_df,
        runs_data,
        num_runs,
        metric='slack'
    )
    print(f"⏱️  Quality score computation time: {time.time() - t_start:.2f}s")
    
    # Prepare per-run data
    data_per_run = prepare_per_run_data(
        x,
        edge_index_G,
        edge_attr_G,
        edge_weight_G,
        edge_weights_G_full,
        colocation_graphs,
        undirected=not cfg.directed
    )
    
    print("="*60)
    print("Dual-View Graph Construction Complete")
    print(f"  Total runs: {len(data_per_run)}")
    print(f"  Run quality range: [{run_quality.min():.3f}, {run_quality.max():.3f}]")
    print("="*60)
    
    # Extract coordinates for all runs (for co-location consistency computation)
    coordinates_all_runs = []
    for run_idx in range(num_runs):
        pt_x_col = f'pt_x_{run_idx + 1}'
        pt_y_col = f'pt_y_{run_idx + 1}'
        coords = nodes_df[[pt_x_col, pt_y_col]].to_numpy(dtype=np.float32)
        coordinates_all_runs.append(coords)
    
    # Sigmas for all runs (use same sigma for all runs for simplicity)
    sigmas_all_runs = [colocation_sigma] * num_runs
    
    return data_per_run, run_quality, id_to_name, coordinates_all_runs, sigmas_all_runs

