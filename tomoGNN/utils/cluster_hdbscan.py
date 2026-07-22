# cluster_hdbscan.py
# HDBSCAN-based clustering on embeddings
from __future__ import annotations
import numpy as np
import hdbscan
from utils.cluster_utils import (
    print_cluster_statistics,
    analyze_cluster_quality
)


def cluster_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = 200,
    min_samples: int = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    HDBSCAN density-based clustering on 16-dim embeddings.

    HDBSCAN Properties:
    - Automatically determines number of clusters
    - Robust to noise (assigns -1 label)
    - Works well with variable density clusters
    - No assumption of cluster shape

    Args:
        embeddings: [N, 16] embedding vectors
        min_cluster_size: Minimum points in a cluster
        min_samples: Core point threshold (default: min_cluster_size//20)
        metric: Distance metric (default: euclidean)

    Returns:
        labels: [N] cluster assignments (-1 = noise)
    """
    print("="*60)
    print("HDBSCAN Clustering")
    print("="*60)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Min cluster size: {min_cluster_size}")

    # Set min_samples if not specified
    if min_samples is None:
        min_samples = max(10, min_cluster_size // 20)

    print(f"  Min samples: {min_samples}")
    print(f"  Metric: {metric}")

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1  # Use all CPUs
    )

    labels = clusterer.fit_predict(embeddings)

    # Print statistics
    print_cluster_statistics(labels, "HDBSCAN")

    # Analyze quality
    quality = analyze_cluster_quality(embeddings, labels)
    print(f"Quality Metrics:")
    print(f"  Silhouette score: {quality['silhouette']:.3f}")
    print(f"  Davies-Bouldin index: {quality['davies_bouldin']:.3f}")

    return labels
