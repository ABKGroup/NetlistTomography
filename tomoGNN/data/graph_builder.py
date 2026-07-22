# graph_builder.py
# Build PyTorch Geometric graph from processed data
from __future__ import annotations
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian
from scipy.sparse.linalg import eigsh
from typing import Dict, List
from config import TrainConfig
from data.data_loader import load_multi_run_data
from data.edge_weights import compute_edge_weights_with_hierarchy


def build_node_features(
    nodes_df: pd.DataFrame,
    num_runs: int
) -> torch.Tensor:
    """
    Build node feature matrix.

    Features: [pt_x_1, ..., pt_x_N, pt_y_1, ..., pt_y_N, log(1+degree)]

    Args:
        nodes_df: DataFrame with node_id, pt_x_1...pt_x_N, pt_y_1...pt_y_N
        num_runs: Number of tomography runs

    Returns:
        Feature tensor [N, 2*num_runs + 1]
    """
    num_nodes = len(nodes_df)

    # Extract all per-run coordinates
    x_features = []
    for i in range(1, num_runs + 1):
        x_features.append(nodes_df[f'pt_x_{i}'].to_numpy(dtype=float))

    y_features = []
    for i in range(1, num_runs + 1):
        y_features.append(nodes_df[f'pt_y_{i}'].to_numpy(dtype=float))

    # Stack: [pt_x_1, ..., pt_x_N, pt_y_1, ..., pt_y_N, degree]
    features = np.column_stack(
        x_features + y_features + [np.zeros(num_nodes)]
    )

    return torch.from_numpy(features).float()


def compute_node_degrees(
    edge_index: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """
    Compute node degrees from edge_index.

    Args:
        edge_index: [2, E] edge connectivity
        num_nodes: Total number of nodes

    Returns:
        Degree tensor [num_nodes]
    """
    degrees = torch.zeros(num_nodes, dtype=torch.float)

    # Count outgoing edges
    degrees.index_add_(
        0,
        edge_index[0],
        torch.ones(edge_index.size(1))
    )

    # Count incoming edges
    degrees.index_add_(
        0,
        edge_index[1],
        torch.ones(edge_index.size(1))
    )

    return degrees


def make_undirected(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_weights: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Convert directed graph to undirected.

    Args:
        edge_index: [2, E] edge connectivity
        edge_attr: [E, 2] edge attributes [c, L]
        edge_weights: [E, 2] edge weights [omega_plus, omega_minus]

    Returns:
        Tuple of (edge_index_undirected, edge_attr_undirected, edge_weights_undirected)
    """
    # Mirror edges: (i,j) -> (j,i)
    edge_index_rev = torch.stack([
        edge_index[1],
        edge_index[0]
    ], dim=0)

    # Concatenate original and reversed
    edge_index_undir = torch.cat([
        edge_index,
        edge_index_rev
    ], dim=1)

    edge_attr_undir = torch.cat([
        edge_attr,
        edge_attr
    ], dim=0)
    
    # Mirror edge weights if provided
    edge_weights_undir = None
    if edge_weights is not None:
        edge_weights_undir = torch.cat([
            edge_weights,
            edge_weights
        ], dim=0)

    return edge_index_undir, edge_attr_undir, edge_weights_undir


def topk_neighbor_pruning(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_weights: torch.Tensor,
    num_nodes: int,
    topk: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Keep only top-K most important neighbors per node.

    Importance = max(omega_plus, omega_minus)

    Args:
        edge_index: [2, E] edge connectivity
        edge_attr: [E, 2] edge attributes [c, L]
        edge_weights: [E, 2] edge weights [omega_plus, omega_minus]
        num_nodes: Total number of nodes
        topk: Number of neighbors to keep

    Returns:
        Pruned (edge_index, edge_attr, edge_weights)
    """
    print(f"Pruning to top-{topk} neighbors per node...")

    # Compute edge importance from edge_weights
    importance = torch.max(
        edge_weights[:, 0],  # omega_plus
        edge_weights[:, 1]   # omega_minus
    )

    # Build adjacency lists per source node
    keep_mask = torch.zeros(
        edge_index.size(1),
        dtype=torch.bool
    )

    for node_id in range(num_nodes):
        # Find edges from this node
        mask = (edge_index[0] == node_id)
        edge_indices = torch.where(mask)[0]

        if len(edge_indices) == 0:
            continue

        # Get importance scores
        scores = importance[edge_indices]

        # Keep top-K
        k = min(topk, len(edge_indices))
        _, topk_idx = torch.topk(scores, k)

        # Mark these edges to keep
        keep_mask[edge_indices[topk_idx]] = True

    # Filter edges
    edge_index_pruned = edge_index[:, keep_mask]
    edge_attr_pruned = edge_attr[keep_mask]
    edge_weights_pruned = edge_weights[keep_mask]

    print(f"  Kept {keep_mask.sum()}/{len(keep_mask)} edges")

    return edge_index_pruned, edge_attr_pruned, edge_weights_pruned


def compute_net_based_edge_weights(
    edge_index: torch.Tensor,
    num_nodes: int
) -> torch.Tensor:
    """
    Compute edge weights for Laplacian based on net size: w = 2/n
    
    For netlist hypergraphs:
    - Each net connects a driver to multiple sinks
    - If a driver has 'fanout' sinks, the net has n = fanout + 1 nodes
    - Each edge in the net gets weight w = 2/n
    
    This normalization ensures larger nets (high fanout) contribute less
    per edge than smaller nets.
    
    Args:
        edge_index: [2, E] edge connectivity
        num_nodes: Total number of nodes
        
    Returns:
        edge_weights: [E] weights for Laplacian computation
    """
    # Count fanout per source node (driver)
    src_nodes = edge_index[0]
    
    # Compute fanout: number of outgoing edges per source
    fanout = torch.zeros(num_nodes, dtype=torch.float)
    fanout.scatter_add_(0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float))
    
    # For each edge, get fanout of its source
    edge_fanout = fanout[src_nodes]  # [E]
    
    # n = fanout + 1 (driver + sinks)
    n = edge_fanout + 1
    
    # Edge weight = 2/n
    edge_weights = 2.0 / n
    
    print(f"  Net-based edge weights: min={edge_weights.min():.4f}, "
          f"max={edge_weights.max():.4f}, mean={edge_weights.mean():.4f}")
    
    return edge_weights


def compute_laplacian_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    pe_dim: int = 16,
    edge_weights: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute Laplacian positional encoding using smallest eigenvectors.
    
    Args:
        edge_index: [2, E] edge connectivity
        num_nodes: Total number of nodes
        pe_dim: Number of eigenvectors to use (default 16)
        edge_weights: [E] optional edge weights for Laplacian
                      (e.g., net-based weights 2/n)
        
    Returns:
        Positional encoding tensor [num_nodes, pe_dim]
    """
    print(f"Computing Laplacian PE with {pe_dim} dimensions...")
    
    if edge_weights is not None:
        print(f"  Using weighted Laplacian (net-based edge weights)")
    
    # Get normalized Laplacian with optional edge weights
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index,
        edge_weight=edge_weights,  # Pass net-based weights if provided
        num_nodes=num_nodes,
        normalization='sym'  # Symmetric normalization: L = I - D^(-1/2) A D^(-1/2)
    )
    
    # Convert to scipy sparse matrix
    laplacian = to_scipy_sparse_matrix(
        edge_index_lap, 
        edge_attr=edge_weight_lap,
        num_nodes=num_nodes
    )
    
    # Compute smallest eigenvalues and eigenvectors
    # We want the smallest eigenvalues (excluding 0 if it exists)
    k = min(pe_dim + 1, num_nodes - 1)  # +1 to account for possible 0 eigenvalue
    
    try:
        eigenvals, eigenvecs = eigsh(
            laplacian, 
            k=k, 
            which='SA',  # Smallest Algebraic (faster than SM)
            sigma=None,   # No shift
            tol=1e-3
        )
        
        # Remove the first eigenvector if it corresponds to eigenvalue 0
        # (which represents the constant vector in connected components)
        if eigenvals[0] < 1e-6:
            eigenvecs = eigenvecs[:, 1:]
            eigenvals = eigenvals[1:]
        
        # Take the first pe_dim eigenvectors
        if eigenvecs.shape[1] > pe_dim:
            eigenvecs = eigenvecs[:, :pe_dim]
        elif eigenvecs.shape[1] < pe_dim:
            # Pad with zeros if we don't have enough eigenvectors
            padding = np.zeros((num_nodes, pe_dim - eigenvecs.shape[1]))
            eigenvecs = np.hstack([eigenvecs, padding])
        
        pos_enc = torch.from_numpy(eigenvecs).float()
        
        print(f"  Laplacian PE computed: {pos_enc.shape}")
        print(f"  Eigenvalue range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
        
    except Exception as e:
        print(f"  Warning: Failed to compute Laplacian PE: {e}")
        print(f"  Using zero initialization instead")
        pos_enc = torch.zeros(num_nodes, pe_dim, dtype=torch.float)
    
    return pos_enc


def normalize_features(
    features: torch.Tensor
) -> torch.Tensor:
    """
    Standardize features (zero mean, unit variance).

    Args:
        features: [N, D] feature tensor

    Returns:
        Normalized features
    """
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True) + 1e-9
    return (features - mean) / std


def build_pyg_graph(
    nodes_df: pd.DataFrame,
    edges_df_with_weights: pd.DataFrame,
    cfg: TrainConfig,
    num_runs: int
) -> Data:
    """
    Build complete PyTorch Geometric Data object.

    Args:
        nodes_df: With node_id, pt_x_1...pt_x_N, pt_y_1...pt_y_N
        edges_df_with_weights: With src, dst, c, L,
                                omega_plus, omega_minus
        cfg: Training configuration
        num_runs: Number of tomography runs

    Returns:
        PyG Data object
    """
    print("="*60)
    print("Building PyTorch Geometric Graph")
    print("="*60)

    num_nodes = len(nodes_df)
    num_edges = len(edges_df_with_weights)

    # Build node features
    x = build_node_features(nodes_df, num_runs)  # [N, 2*num_runs+1]

    # Build edge index
    edge_index = torch.tensor([
        edges_df_with_weights['src'].to_numpy(),
        edges_df_with_weights['dst'].to_numpy()
    ], dtype=torch.long)  # [2, E]

    # Build edge attributes: [c, L] ONLY (observable features)
    edge_attr = torch.tensor(
        edges_df_with_weights[[
            'c', 'L'
        ]].to_numpy(),
        dtype=torch.float
    )  # [E, 2]
    
    # Build edge weights: [omega_plus, omega_minus] (for loss computation only)
    edge_weights = torch.tensor(
        edges_df_with_weights[[
            'omega_plus', 'omega_minus'
        ]].to_numpy(),
        dtype=torch.float
    )  # [E, 2]

    print(f"  Initial graph: {num_nodes} nodes, {num_edges} edges")

    # Make undirected
    if not cfg.directed:
        edge_index, edge_attr, edge_weights = make_undirected(
            edge_index,
            edge_attr,
            edge_weights
        )
        print(f"  Undirected: {edge_index.size(1)} edges")

    # Top-K pruning
    edge_index, edge_attr, edge_weights = topk_neighbor_pruning(
        edge_index,
        edge_attr,
        edge_weights,
        num_nodes,
        cfg.topk_per_node
    )

    # Add degree feature
    degrees = compute_node_degrees(edge_index, num_nodes)
    x[:, 2] = torch.log(1.0 + degrees)

    # Normalize features
    x = normalize_features(x)

    # Compute net-based edge weights for Laplacian (w = 2/n)
    laplacian_edge_weights = compute_net_based_edge_weights(
        edge_index,
        num_nodes
    )

    # Compute Laplacian positional encoding with net-based weights
    pos_enc = compute_laplacian_pe(
        edge_index,
        num_nodes,
        pe_dim=cfg.lappe_dim,
        edge_weights=laplacian_edge_weights
    )

    # Create PyG Data
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,      # [E, 2]: [c, L] only - observable features
        edge_weights=edge_weights, # [E, 2]: [omega_plus, omega_minus] for loss
        pos_enc=pos_enc,
        num_nodes=num_nodes
    )

    print("="*60)
    print(f"PyG Graph Built: {data.num_nodes} nodes, "
          f"{data.num_edges} edges")
    print(f"  Node features: {data.x.shape}")
    print(f"  Edge attributes: {data.edge_attr.shape} [c, L]")
    print(f"  Edge weights: {data.edge_weights.shape} [omega_plus, omega_minus]")
    print(f"  Positional encoding: {data.pos_enc.shape}")
    print("="*60)

    return data


def load_and_build_graph(
    nodes_csv: str,
    edges_csv: str,
    cp: float,
    cfg: TrainConfig
) -> tuple[Data, Dict]:
    """
    Complete pipeline: load data → compute weights → build graph.

    Args:
        nodes_csv: Path to nodes CSV
        edges_csv: Path to edges CSV
        cp: Clock period
        cfg: Training configuration

    Returns:
        Tuple of (PyG Data, id_to_name mapping)
    """
    # Load multi-run data
    nodes_df, edges_df, runs_data, id_to_name, num_runs = \
        load_multi_run_data(nodes_csv, edges_csv)

    # Compute edge weights
    edges_with_weights = compute_edge_weights_with_hierarchy(
        edges_df,
        runs_data,
        id_to_name,
        cp,
        cfg
    )

    # Build PyG graph
    graph = build_pyg_graph(
        nodes_df,
        edges_with_weights,
        cfg,
        num_runs
    )

    return graph, id_to_name
