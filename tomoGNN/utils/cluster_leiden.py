# cluster_leiden.py
# Leiden clustering with GPU cosine similarity
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, Optional
try:
    import igraph as ig
    import leidenalg
except ImportError:
    print("Warning: igraph or leidenalg not installed")
    ig = None
    leidenalg = None
from utils.cluster_utils import (
    print_cluster_statistics,
    analyze_cluster_quality
)


def compute_cosine_similarity_gpu(
    embeddings: np.ndarray,
    edges_df: pd.DataFrame,
    node_to_idx: Dict,
    batch_size: int = 20000,
    device: str = 'auto'
) -> tuple[np.ndarray, list]:
    """
    Compute edge cosine similarity on GPU (16-dim embeddings).

    Args:
        embeddings: [N, 16] embedding vectors
        edges_df: DataFrame with 'src', 'dst' columns
        node_to_idx: Mapping node name -> embedding index
        batch_size: Process edges in batches (larger for 16-dim)
        device: 'auto', 'cuda', or 'cpu'

    Returns:
        Tuple of (similarities, valid_edge_indices)
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Computing cosine similarity on {device}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Batch size: {batch_size}")

    # Move embeddings to GPU
    emb_tensor = torch.from_numpy(embeddings).to(device)

    # Get edge indices
    edge_indices = []
    valid_edges = []

    for idx, row in edges_df.iterrows():
        src_name, dst_name = row['src'], row['dst']

        if src_name in node_to_idx and dst_name in node_to_idx:
            src_idx = node_to_idx[src_name]
            dst_idx = node_to_idx[dst_name]
            edge_indices.append([src_idx, dst_idx])
            valid_edges.append(idx)

    if not edge_indices:
        raise ValueError("No valid edges found")

    edge_indices = np.array(edge_indices)
    num_edges = len(edge_indices)
    similarities = np.zeros(num_edges, dtype=np.float32)

    print(f"  Valid edges: {num_edges:,}")

    # Process in batches
    for i in range(0, num_edges, batch_size):
        end_idx = min(i + batch_size, num_edges)
        batch_edges = edge_indices[i:end_idx]

        # Get embeddings
        src_idx = torch.from_numpy(batch_edges[:, 0]).to(device)
        dst_idx = torch.from_numpy(batch_edges[:, 1]).to(device)

        src_emb = emb_tensor[src_idx]  # [batch, 16]
        dst_emb = emb_tensor[dst_idx]  # [batch, 16]

        # Cosine similarity (embeddings are L2-normalized)
        batch_sim = F.cosine_similarity(src_emb, dst_emb, dim=1)

        # Store results
        similarities[i:end_idx] = batch_sim.cpu().numpy()

        if (i // batch_size) % 50 == 0:
            print(f"    Processed {end_idx:,}/{num_edges:,} edges")

    print(f"  Similarity range: "
          f"[{similarities.min():.3f}, {similarities.max():.3f}]")

    return similarities, valid_edges


def build_leiden_graph(
    edges_df: pd.DataFrame,
    similarities: np.ndarray,
    valid_edges: list,
    min_weight: float = 0.001
) -> ig.Graph:
    """
    Build weighted igraph for Leiden clustering.

    Weight formula:
        w = (cosine_sim + 1) / 2
    This maps [-1, 1] to [0, 1]

    Args:
        edges_df: Original edges DataFrame
        similarities: Cosine similarity values
        valid_edges: Valid edge indices
        min_weight: Minimum weight threshold

    Returns:
        igraph Graph object
    """
    print("Building weighted graph for Leiden...")

    # Get valid edges
    df_valid = edges_df.iloc[valid_edges].copy()

    # Compute weights: normalize to [0, 1]
    weights = (similarities + 1.0) / 2.0

    # Filter low weights
    mask = weights > min_weight
    df_filtered = df_valid[mask]
    weights_filtered = weights[mask]

    print(f"  Edges after weight filter: {len(df_filtered):,}")

    # Get unique nodes
    nodes = sorted(set(df_filtered['src']) | set(df_filtered['dst']))
    node_to_id = {name: idx for idx, name in enumerate(nodes)}

    # Map edges to integer IDs
    edge_list = [
        (node_to_id[row['src']], node_to_id[row['dst']])
        for _, row in df_filtered.iterrows()
    ]

    # Build igraph
    g = ig.Graph(n=len(nodes), edges=edge_list, directed=False)
    g.es['weight'] = weights_filtered.tolist()

    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")

    return g


def run_leiden_clustering(
    g: ig.Graph,
    resolution: float = 1.0,
    seed: int = 42
) -> tuple[np.ndarray, float]:
    """
    Run Leiden community detection.

    Args:
        g: igraph Graph object
        resolution: Resolution parameter (higher = more clusters)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (labels, modularity)
    """
    print(f"Running Leiden algorithm (resolution={resolution})...")

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=seed
    )

    labels = np.array(partition.membership)
    modularity = partition.modularity

    print(f"  Modularity: {modularity:.4f}")
    print(f"  Communities: {len(set(labels))}")

    return labels, modularity


def cluster_leiden_gpu(
    embeddings: np.ndarray,
    edges_df: pd.DataFrame,
    node_to_idx: Dict,
    min_cluster_size: int = 200,
    resolution: float = 1.0,
    device: str = 'auto'
) -> np.ndarray:
    """
    Complete Leiden clustering pipeline with GPU cosine.

    Pipeline:
    1. Compute cosine similarity on GPU (16-dim embeddings)
    2. Build weighted igraph
    3. Run Leiden community detection
    4. Return cluster labels

    Args:
        embeddings: [N, 16] embedding vectors
        edges_df: Edge DataFrame with src, dst
        node_to_idx: Node name to index mapping
        min_cluster_size: Minimum cluster size (for stats only)
        resolution: Leiden resolution parameter
        device: GPU device

    Returns:
        labels: [N] cluster assignments
    """
    if ig is None or leidenalg is None:
        raise ImportError(
            "igraph and leidenalg required for Leiden clustering"
        )

    print("="*60)
    print("Leiden Clustering with GPU Cosine Similarity")
    print("="*60)

    # Step 1: Compute cosine similarity
    similarities, valid_edges = compute_cosine_similarity_gpu(
        embeddings,
        edges_df,
        node_to_idx,
        batch_size=20000,
        device=device
    )

    # Step 2: Build graph
    g = build_leiden_graph(
        edges_df,
        similarities,
        valid_edges,
        min_weight=0.001
    )

    # Step 3: Run Leiden
    labels_subset, modularity = run_leiden_clustering(
        g,
        resolution=resolution
    )

    # Map back to full node set
    num_nodes = len(embeddings)
    labels_full = np.full(num_nodes, -1, dtype=int)

    # Get node names from graph
    nodes_in_graph = sorted(
        set(edges_df.iloc[valid_edges]['src']) |
        set(edges_df.iloc[valid_edges]['dst'])
    )

    for node_name, label in zip(nodes_in_graph, labels_subset):
        if node_name in node_to_idx:
            idx = node_to_idx[node_name]
            labels_full[idx] = label

    # Print statistics
    print_cluster_statistics(labels_full, "Leiden")

    # Analyze quality
    quality = analyze_cluster_quality(embeddings, labels_full)
    print(f"Quality Metrics:")
    print(f"  Silhouette: {quality['silhouette']:.3f}")
    print(f"  Davies-Bouldin: {quality['davies_bouldin']:.3f}")

    return labels_full
