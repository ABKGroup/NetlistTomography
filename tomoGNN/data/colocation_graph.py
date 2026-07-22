# colocation_graph.py
# Build co-location graphs based on physical coordinates per run
# colocation_graph.py (FAST)
from __future__ import annotations
import torch
import numpy as np
import time
import os
import pickle
import hashlib
import json
from typing import Tuple, List, Optional, Dict
from multiprocessing import Pool, cpu_count
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x


# ------------------------------- Utilities -------------------------------

def _infer_grid_pitch_1d(vals: np.ndarray, sample: int = 20000) -> float:
    """Infer a robust grid pitch along one axis from a sample."""
    n = vals.shape[0]
    if n <= 1:
        return 1.0
    s = min(sample, n)
    idx = np.random.choice(n, size=s, replace=False)
    u = np.unique(vals[idx])
    if u.size < 2:
        return 1.0
    diffs = np.diff(np.sort(u))
    # Keep small positive gaps; ignore outliers by percentile clipping
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    hi = np.percentile(diffs, 60)  # robust central chunk
    small = diffs[diffs <= hi]
    if small.size == 0:
        small = diffs
    pitch = float(np.median(small))
    if not np.isfinite(pitch) or pitch <= 0:
        pitch = 1.0
    return pitch


def compute_adaptive_radius_fast(
    coordinates: np.ndarray,
    q: float = 0.9,
    M: int = 10000,
    S: int = 100000,
) -> float:
    """
    FAST adaptive radius:
    - Build NN index on a subsample of size S << N
    - Query 1-NN for Q=min(M,S) points inside that subsample
    - tau = q * median(1-NN)
    """
    from sklearn.neighbors import NearestNeighbors  # local import

    N = coordinates.shape[0]
    if N == 0:
        return 1.0
    S = min(S, N)
    Q = min(M, S)

    idx_index = np.random.choice(N, size=S, replace=False)
    pts_index = coordinates[idx_index]

    q_idx = np.random.choice(S, size=Q, replace=False)
    pts_query = pts_index[q_idx]

    nbrs = NearestNeighbors(n_neighbors=2, metric='manhattan', algorithm='auto')
    nbrs.fit(pts_index)                              # O(S log S)
    dists, _ = nbrs.kneighbors(pts_query)           # [Q, 2]
    nn1 = dists[:, 1]
    tau = q * float(np.median(nn1))
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0
    return tau


# ------------------------------- Caching Functions -------------------------------

def _compute_colocation_cache_key(
    coordinates: np.ndarray,
    radius: Optional[float],
    sigma: float,
    q: float,
    M: int,
    sample_index_size: int
) -> str:
    """
    Compute a unique cache key for co-location graph parameters.
    
    Args:
        coordinates: Node coordinates
        radius: Fixed radius (or None for adaptive)
        sigma: Gaussian kernel bandwidth
        q: Adaptive radius multiplier
        M: Number of query points for adaptive radius
        sample_index_size: Subsample size for NN index
    
    Returns:
        Hexadecimal hash string
    """
    # Create a dictionary of all parameters that affect the graph
    params = {
        'num_nodes': coordinates.shape[0],
        'coord_hash': hashlib.md5(coordinates.tobytes()).hexdigest()[:16],  # First 16 chars
        'radius': radius if radius is not None else 'adaptive',
        'sigma': float(sigma),
        'q': float(q),
        'M': int(M),
        'sample_index_size': int(sample_index_size),
    }
    
    # Create a canonical JSON string
    params_str = json.dumps(params, sort_keys=True)
    
    # Compute MD5 hash
    hash_obj = hashlib.md5(params_str.encode('utf-8'))
    return hash_obj.hexdigest()


def _get_colocation_cache_dir(base_cache_dir: Optional[str] = None) -> str:
    """Get or create the co-location cache directory."""
    if base_cache_dir is None:
        # Default: use .cache/ in the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_cache_dir = os.path.join(script_dir, '.cache', 'colocation_graphs')
    
    os.makedirs(base_cache_dir, exist_ok=True)
    return base_cache_dir


def save_colocation_graph_cache(
    run_idx: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_weight: torch.Tensor,
    cache_key: str,
    cache_dir: Optional[str] = None
) -> str:
    """
    Save a co-location graph to cache.
    
    Returns:
        Path to the saved cache file
    """
    cache_dir = _get_colocation_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f'run_{run_idx}_{cache_key}.pkl')
    
    cache_data = {
        'run_idx': run_idx,
        'edge_index': edge_index.cpu(),
        'edge_attr': edge_attr.cpu(),
        'edge_weight': edge_weight.cpu(),
        'cache_key': cache_key,
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return cache_file


def load_colocation_graph_cache(
    run_idx: int,
    cache_key: str,
    cache_dir: Optional[str] = None
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load a co-location graph from cache if it exists.
    
    Returns:
        (edge_index, edge_attr, edge_weight) if cache hit, None if cache miss
    """
    cache_dir = _get_colocation_cache_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f'run_{run_idx}_{cache_key}.pkl')
    
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify cache key matches
        if cache_data.get('cache_key') != cache_key:
            print(f"  Warning: Cache key mismatch for run {run_idx}, rebuilding...")
            return None
        
        return (
            cache_data['edge_index'],
            cache_data['edge_attr'],
            cache_data['edge_weight']
        )
    except Exception as e:
        print(f"  Warning: Failed to load cache for run {run_idx}: {e}")
        return None


# ---------------- Row-sweep (grid hashing, two-pointer window) ------------

def _group_by_row_sorted(
    x: np.ndarray, y: np.ndarray, W_site: float, H_row: float
):
    """
    Quantize (x, y) -> (site_idx, row_idx), group nodes by row, and sort each row by site.
    Returns:
      rows: dict[row -> {'idx': np.int32[], 'site': np.int32[]}]
      sorted_rows: np.int32[]
    """
    site_idx = np.rint(x / W_site).astype(np.int32)
    row_idx  = np.rint(y / H_row).astype(np.int32)

    # Sort once by (row, site)
    sort_key = row_idx.astype(np.int64) * (1 << 32) + site_idx.astype(np.int64)
    order = np.argsort(sort_key, kind='mergesort')
    sorted_nodes = np.arange(x.shape[0], dtype=np.int32)[order]
    sorted_rows  = row_idx[order]
    sorted_sites = site_idx[order]

    # Build row buckets via boundaries
    unique_rows, row_starts = np.unique(sorted_rows, return_index=True)
    row_ends = np.append(row_starts[1:], sorted_nodes.size)

    rows = {}
    for r, s0, s1 in zip(unique_rows, row_starts, row_ends):
        rows[int(r)] = {
            'idx':  sorted_nodes[s0:s1].copy(),
            'site': sorted_sites[s0:s1].copy(),
        }
    return rows, unique_rows.astype(np.int32)


def _emit_pairs_between_rows_sorted(
    A_idx: np.ndarray, A_site: np.ndarray,
    B_idx: np.ndarray, B_site: np.ndarray,
    dy_rows: int,
    radius: float,            # physical L1 radius (same unit as x,y)
    W_site: float, H_row: float,
    out_src: list, out_dst: list, out_dist: list,
    same_row: bool,
):
    """
    Two-pointer sweep on two sorted rows; emit directed edges A->B within physical radius.
    Physical L1 distance: d = |Δsite|*W_site + |Δrow|*H_row
    Horizontal budget in sites at this dy: s_max = floor((radius - dy*H_row)/W_site)
    """
    vert = dy_rows * H_row
    rem = radius - vert
    if rem < 0:
        return
    s_max = int(np.floor(rem / W_site))
    if s_max < 0:
        return

    nb = B_site.shape[0]
    j_left = 0
    j_right = 0

    for i in range(A_site.shape[0]):
        a_site = A_site[i]
        a_idx  = A_idx[i]

        # Slide window in B: keep sites within [a_site - s_max, a_site + s_max]
        while j_left < nb and B_site[j_left] < a_site - s_max:
            j_left += 1
        while j_right < nb and B_site[j_right] <= a_site + s_max:
            j_right += 1

        if j_left >= j_right:
            continue

        cand_sites = B_site[j_left:j_right]
        cand_idx   = B_idx[j_left:j_right]

        # same-row duplicate filter: only keep j>i when A==B
        if same_row:
            mask = cand_sites > a_site  # strictly to the right
            if not np.any(mask):
                continue
            cand_sites = cand_sites[mask]
            cand_idx   = cand_idx[mask]

        dx_sites = np.abs(cand_sites - a_site).astype(np.float32)
        d = dx_sites * W_site + vert
        keep = (d <= radius) & (cand_idx != a_idx)
        if not np.any(keep):
            continue

        kept_idx = cand_idx[keep]
        kept_d   = d[keep].astype(np.float32)

        out_src.append(np.full(kept_idx.size, a_idx, dtype=np.int32))
        out_dst.append(kept_idx.astype(np.int32))
        out_dist.append(kept_d)


def build_colocation_graph_row_sweep(
    coordinates: np.ndarray,
    radius: Optional[float] = None,   # physical L1 (same unit as coordinates)
    sigma: float = 1.0,
    q: float = 0.9,
    M: int = 10000,
    W_site: Optional[float] = None,
    H_row: Optional[float]  = None,
    sample_index_size: int = 100000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FAST co-location graph:
      - Estimate W_site/H_row if not provided
      - Compute tau (radius) via subsampled 1-NN median
      - Row sweep with two-pointer windows; directed edges (A->B) without duplicates
      - Use Gaussian weights: w = exp(-(d/sigma)^2)
    """
    N = coordinates.shape[0]
    if N == 0:
        return (torch.empty((2,0), dtype=torch.long),
                torch.empty((0,1), dtype=torch.float32),
                torch.empty((0,),  dtype=torch.float32))

    x = coordinates[:, 0].astype(np.float64, copy=False)
    y = coordinates[:, 1].astype(np.float64, copy=False)

    # Infer grid if not provided
    if W_site is None:
        W_site = _infer_grid_pitch_1d(x)
    if H_row is None:
        H_row = _infer_grid_pitch_1d(y)

    # Adaptive radius (physical units)
    if radius is None:
        radius = compute_adaptive_radius_fast(
            coordinates, q=q, M=M, S=sample_index_size
        )

    # Group nodes by row, sorted by site
    rows, unique_rows = _group_by_row_sorted(x, y, W_site, H_row)

    # Maximum vertical rows to check
    R_rows = int(np.floor(radius / H_row))

    out_src, out_dst, out_dist = [], [], []

    # Sweep rows; emit A->A and A->B with B> A (avoid duplicates)
    for r in unique_rows:
        A = rows[int(r)]
        # same row
        _emit_pairs_between_rows_sorted(
            A['idx'], A['site'],
            A['idx'], A['site'],
            dy_rows=0, radius=radius, W_site=W_site, H_row=H_row,
            out_src=out_src, out_dst=out_dst, out_dist=out_dist,
            same_row=True,
        )
        # upper rows only
        for dy in range(1, R_rows+1):
            r2 = int(r) + dy
            if r2 not in rows:
                continue
            B = rows[r2]
            _emit_pairs_between_rows_sorted(
                A['idx'], A['site'],
                B['idx'], B['site'],
                dy_rows=dy, radius=radius, W_site=W_site, H_row=H_row,
                out_src=out_src, out_dst=out_dst, out_dist=out_dist,
                same_row=False,
            )

    if len(out_src) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0,1), dtype=torch.float32)
        edge_weight= torch.empty((0,),  dtype=torch.float32)
        return edge_index, edge_attr, edge_weight

    src = np.concatenate(out_src); dst = np.concatenate(out_dst); dist = np.concatenate(out_dist).astype(np.float32)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0).astype(np.int64))
    edge_attr  = torch.from_numpy(dist).unsqueeze(1)  # [E,1]
    edge_weight= torch.exp(-(edge_attr.squeeze(1) / float(sigma))**2)
    return edge_index, edge_attr, edge_weight


# ---------------------- Build all runs (no recompute tau) -----------------

def _build_single_run_graph(args):
    """Multiprocessing helper (keeps arguments small)."""
    (run_id, coordinates, radius, sigma, q, M, W_site, H_row, sample_index_size) = args
    ei, ea, ew = build_colocation_graph_row_sweep(
        coordinates=coordinates,
        radius=radius,
        sigma=sigma,
        q=q,
        M=M,
        W_site=W_site,
        H_row=H_row,
        sample_index_size=sample_index_size,
    )
    return (run_id, ei, ea, ew)


def build_colocation_graphs_all_runs(
    nodes_df,
    num_runs: int,
    radius: Optional[float] = None,   # physical units; if None, computed per run (fast)
    sigma: float = 1.0,
    q: float = 1,
    M: int = 10000,
    n_jobs: int = 1,
    W_site: Optional[float] = None,   # if known, pass once to lock grid
    H_row: Optional[float]  = None,
    sample_index_size: int = 100000,
) -> list:
    """
    Build co-location graphs for all runs via row-sweep (fast).
    Returns a list of dicts per run: {edge_index_H, edge_attr_H, edge_weight_H}
    """
    print("="*60)
    print("Building Co-Location Graphs (Row-sweep, Manhattan L1)")
    print("="*60)
    if radius is None:
        print(f"  Adaptive radius per run (q={q}, M={M}, S={sample_index_size})")
    else:
        print(f"  Fixed radius: {radius}")
    print(f"  Gaussian sigma: {sigma}")
    if W_site is not None and H_row is not None:
        print(f"  Using provided grid: W_site={W_site}, H_row={H_row}")

    # Prepare all coordinate arrays
    all_coords = []
    for r in range(1, num_runs + 1):
        x_col, y_col = f'pt_x_{r}', f'pt_y_{r}'
        if x_col not in nodes_df.columns or y_col not in nodes_df.columns:
            raise ValueError(f"Missing coordinate columns for run {r}")
        coords = np.column_stack([nodes_df[x_col].to_numpy(), nodes_df[y_col].to_numpy()]).astype(np.float64)
        all_coords.append(coords)

    # Choose workers
    if n_jobs == -1:
        n_workers = cpu_count()
    elif n_jobs > 0:
        n_workers = min(n_jobs, cpu_count())
    else:
        n_workers = 1
    use_parallel = (n_workers > 1 and num_runs >= 4)
    print("  " + (f"Parallel processing with {n_workers} workers" if use_parallel else "Sequential processing"))

    # Compute cache key (same for all runs with same parameters, but different run_idx)
    # Use first run's coordinates to compute cache key template
    cache_enabled = True  # Can be controlled via parameter if needed
    cache_hits = 0
    cache_misses = 0
    
    t0 = time.time()
    results = [None] * num_runs

    if use_parallel:
        # Note: Parallel mode doesn't support caching yet (could be added later)
        print("  Note: Caching not yet supported in parallel mode")
        args_list = [
            (r, all_coords[r-1], radius, sigma, q, M, W_site, H_row, sample_index_size)
            for r in range(1, num_runs+1)
        ]
        with Pool(processes=n_workers) as pool:
            for (run_id, ei, ea, ew) in pool.imap_unordered(_build_single_run_graph, args_list, chunksize=1):
                results[run_id-1] = {'edge_index_H': ei, 'edge_attr_H': ea, 'edge_weight_H': ew}
    else:
        results = []
        print()  # Empty line before progress bar
        for r in tqdm(range(1, num_runs+1), desc="Building co-location graphs", unit="run"):
            coords = all_coords[r-1]
            
            # Try to load from cache first
            cache_key = None
            cached_graph = None
            if cache_enabled:
                cache_key = _compute_colocation_cache_key(
                    coords, radius, sigma, q, M, sample_index_size
                )
                cached_graph = load_colocation_graph_cache(r, cache_key)
            
            if cached_graph is not None:
                # Cache hit!
                ei, ea, ew = cached_graph
                cache_hits += 1
                results.append({'edge_index_H': ei, 'edge_attr_H': ea, 'edge_weight_H': ew})
                # Print cache hit (suppressed in tqdm loop, will print summary later)
            else:
                # Cache miss - build graph
                t_run0 = time.time()
                ei, ea, ew = build_colocation_graph_row_sweep(
                    coords, radius=radius, sigma=sigma, q=q, M=M,
                    W_site=W_site, H_row=H_row, sample_index_size=sample_index_size,
                )
                build_time = time.time() - t_run0
                cache_misses += 1
                
                # Save to cache
                if cache_enabled and cache_key is not None:
                    try:
                        save_colocation_graph_cache(r, ei, ea, ew, cache_key)
                    except Exception as e:
                        print(f"  Warning: Failed to save cache for run {r}: {e}")
                
                results.append({'edge_index_H': ei, 'edge_attr_H': ea, 'edge_weight_H': ew})
                # Print build time (suppressed in tqdm loop)
        
        print(f"\n  Cache statistics: {cache_hits} hits, {cache_misses} misses")
        if cache_hits > 0:
            print(f"  ⚡ Saved ~{cache_hits * 13:.1f}s by reusing cached graphs!")

    # Print stats for run 1 (no extra adaptive-radius recompute!)
    ei0 = results[0]['edge_index_H']
    deg0 = torch.bincount(ei0[0], minlength=all_coords[0].shape[0])
    num_iso = int((deg0 == 0).sum().item())
    avg_deg = float(deg0.float().mean().item())
    print("\n  Run 1 statistics:")
    print(f"    Nodes: {all_coords[0].shape[0]}")
    print(f"    Edges: {ei0.size(1)}")
    print(f"    Avg out-degree (directed): {avg_deg:.3f}")
    print(f"    Isolated nodes: {num_iso} ({100.0*num_iso/all_coords[0].shape[0]:.2f}%)")

    print("="*60)
    print(f"✅ Co-Location Graphs Built: {num_runs} runs in {time.time()-t0:.2f}s")
    print("="*60)
    return results



def make_undirected_colocation_graph(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_weight: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make co-location graph undirected by adding reverse edges.
    
    Args:
        edge_index: [2, E] directed edges
        edge_attr: [E, 1] edge attributes
        edge_weight: [E] edge weights
        
    Returns:
        Undirected versions of inputs
    """
    # Reverse edges
    edge_index_rev = torch.stack([
        edge_index[1],
        edge_index[0]
    ], dim=0)
    
    # Concatenate
    edge_index_undir = torch.cat([
        edge_index,
        edge_index_rev
    ], dim=1)
    
    edge_attr_undir = torch.cat([
        edge_attr,
        edge_attr
    ], dim=0)
    
    edge_weight_undir = torch.cat([
        edge_weight,
        edge_weight
    ], dim=0)
    
    return edge_index_undir, edge_attr_undir, edge_weight_undir


def compute_run_quality_scores(
    nodes_df,
    edges_df,
    runs_data: list,
    num_runs: int,
    metric: str = 'slack'
) -> torch.Tensor:
    """
    Compute PPA-derived quality scores for each run.
    
    Simple heuristic: runs with better average slack get higher quality.
    
    Args:
        nodes_df: Node dataframe (not used currently)
        edges_df: Edge dataframe (not used currently)
        runs_data: Per-run data with slack/length
        num_runs: Number of runs
        metric: Quality metric ('slack' or 'length')
        
    Returns:
        quality: [num_runs] normalized quality scores (sum to num_runs)
    """
    print("Computing run quality scores using Total Negative Slack (TNS)...")
    
    qualities = []
    tns_values = []  # Store TNS for debugging
    
    for run_id, run_data in enumerate(runs_data):
        if metric == 'slack':
            # Calculate Total Negative Slack (TNS)
            # TNS = sum of all negative slacks (timing violations)
            # Less negative TNS = better quality
            slack_values = run_data['slack']
            finite_slack = slack_values[np.isfinite(slack_values)]
            
            if len(finite_slack) > 0:
                # TNS: sum of only negative slacks (violations)
                negative_slacks = finite_slack[finite_slack < 0]
                
                if len(negative_slacks) > 0:
                    tns = negative_slacks.sum()  # Total Negative Slack (typically negative)
                else:
                    # No violations - perfect timing
                    tns = 0.0
                
                # Map TNS to quality: less negative TNS = higher quality
                # Use softer mapping to avoid extreme skewness
                # Formula: quality = 1 / (1 + |TNS| / 3000)
                # TNS=0: quality=1.0 (perfect timing)
                # TNS=-3000: quality=0.5 (moderate violations)
                # TNS=-6000: quality=0.33 (significant violations)
                # This gives more balanced weights across runs (less extreme than exp)
                quality = 1.0 / (1.0 + np.abs(tns) / 3000.0)
                tns_values.append(tns)
            else:
                quality = 0.5  # Neutral quality if all inf
                tns_values.append(float('nan'))
        elif metric == 'length':
            # Lower average length = better quality
            avg_length = run_data['length'].mean()
            quality = 1.0 / (1.0 + avg_length)  # Inverse relationship
        else:
            raise ValueError(f"Unknown quality metric: {metric}")
        
        qualities.append(quality)
    
    # Convert to tensor and normalize
    qualities = torch.tensor(qualities, dtype=torch.float)
    
    # Print TNS statistics
    if metric == 'slack' and len(tns_values) > 0:
        valid_tns = [t for t in tns_values if not np.isnan(t)]
        if len(valid_tns) > 0:
            print(f"  TNS range: [{min(valid_tns):.3f}, {max(valid_tns):.3f}]")
            print(f"  TNS mean: {np.mean(valid_tns):.3f}")
            print(f"  Runs with violations: {sum(1 for t in valid_tns if t < 0)}/{len(valid_tns)}")
            print(f"  Perfect timing runs (TNS=0): {sum(1 for t in valid_tns if t == 0)}")
    
    # Normalize so they sum to num_runs (maintain scale)
    qualities = qualities / qualities.sum() * num_runs
    
    print(f"  Quality scores range: [{qualities.min():.3f}, {qualities.max():.3f}]")
    print(f"  Quality scores mean: {qualities.mean():.3f}")
    
    return qualities


def compute_colocation_consistency(
    pairs: torch.Tensor,
    coordinates_all_runs: List[np.ndarray],
    quality_scores: torch.Tensor,
    sigmas: List[float]
) -> torch.Tensor:
    """
    Compute co-location consistency for given pairs.
    
    C_ij = Σ_r (π_r * exp(-(d_ij^r / τ_r)^2)) / Σ_r π_r
    
    Quality-weighted average of how often pairs are close across runs.
    High C_ij means nodes are consistently close in high-quality runs.
    
    Args:
        pairs: [P, 2] node pair indices
        coordinates_all_runs: List of [N, 2] coordinate arrays for each run
        quality_scores: [R] quality scores π_r for each run
        sigmas: List of sigma (τ_r) values for each run
    
    Returns:
        C_ij: [P] co-location consistency scores in [0, 1]
    """
    device = pairs.device
    num_pairs = pairs.size(0)
    num_runs = len(coordinates_all_runs)
    
    # Extract pair indices
    i_idx = pairs[:, 0].cpu().numpy()  # [P]
    j_idx = pairs[:, 1].cpu().numpy()  # [P]
    
    # Initialize accumulator
    weighted_proximity_sum = np.zeros(num_pairs, dtype=np.float32)
    quality_sum = 0.0
    
    # Convert quality scores to numpy
    quality_np = quality_scores.cpu().numpy()  # [R]
    
    # Accumulate quality-weighted proximity across runs
    for r in range(num_runs):
        coords_r = coordinates_all_runs[r]  # [N, 2]
        pi_r = quality_np[r]  # Quality score for run r
        tau_r = sigmas[r]  # Sigma for run r
        
        # Get coordinates for pairs
        coords_i = coords_r[i_idx]  # [P, 2]
        coords_j = coords_r[j_idx]  # [P, 2]
        
        # Manhattan distance
        dist_ij = np.abs(coords_i - coords_j).sum(axis=1)  # [P]
        
        # Gaussian proximity: exp(-(d/τ)^2)
        proximity = np.exp(-(dist_ij / tau_r) ** 2)  # [P]
        
        # Weight by quality score
        weighted_proximity_sum += pi_r * proximity
        quality_sum += pi_r
    
    # Normalize by total quality
    C_ij = weighted_proximity_sum / (quality_sum + 1e-9)  # [P]
    
    # Clip to [0, 1] for safety
    C_ij = np.clip(C_ij, 0.0, 1.0)
    
    # Convert to tensor
    C_ij_tensor = torch.tensor(C_ij, dtype=torch.float32, device=device)
    
    return C_ij_tensor

