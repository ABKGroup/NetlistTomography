# cluster_utils.py
# Shared clustering utilities
from __future__ import annotations
import numpy as np
from typing import Dict


def print_cluster_statistics(
    labels: np.ndarray,
    method_name: str = "Clustering"
) -> None:
    """
    Print detailed clustering statistics.

    Args:
        labels: Cluster labels (can include -1 for noise)
        method_name: Name of clustering method
    """
    unique_labels = set(labels)
    num_clusters = len(unique_labels - {-1})
    noise_count = (labels == -1).sum()
    clustered_count = len(labels) - noise_count

    print("="*60)
    print(f"{method_name} Statistics")
    print("="*60)
    print(f"  Total instances: {len(labels):,}")
    print(f"  Clustered instances: {clustered_count:,}")
    print(f"  Noise points (-1): {noise_count:,}")
    print(f"  Number of clusters: {num_clusters}")

    if num_clusters > 0:
        # Compute cluster sizes (excluding noise)
        valid_labels = labels[labels != -1]
        unique, counts = np.unique(valid_labels, return_counts=True)

        print(f"  Cluster size range: "
              f"{counts.min()} - {counts.max()}")
        print(f"  Mean cluster size: {counts.mean():.1f}")
        print(f"  Median cluster size: {np.median(counts):.1f}")

        # Distribution
        small = (counts < 200).sum()
        medium = ((counts >= 200) & (counts < 500)).sum()
        large = (counts >= 500).sum()

        print(f"  Size distribution:")
        print(f"    Small (<200): {small}")
        print(f"    Medium (200-500): {medium}")
        print(f"    Large (>=500): {large}")

    print("="*60)


def analyze_cluster_quality(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Analyze cluster quality metrics.

    Computes:
    - Silhouette score
    - Davies-Bouldin index

    Args:
        embeddings: [N, D] embedding vectors
        labels: [N] cluster assignments

    Returns:
        Dictionary of quality metrics
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score
    )

    # Filter out noise points
    mask = labels != -1
    if mask.sum() < 2:
        return {
            'silhouette': 0.0,
            'davies_bouldin': float('inf')
        }

    emb_filtered = embeddings[mask]
    labels_filtered = labels[mask]

    # Silhouette score (higher is better, range [-1, 1])
    try:
        silhouette = silhouette_score(
            emb_filtered,
            labels_filtered,
            metric='euclidean'
        )
    except:
        silhouette = 0.0

    # Davies-Bouldin index (lower is better)
    try:
        db_index = davies_bouldin_score(
            emb_filtered,
            labels_filtered
        )
    except:
        db_index = float('inf')

    return {
        'silhouette': silhouette,
        'davies_bouldin': db_index
    }
