#!/usr/bin/env python3
"""
Analyze timing paths and clustering data to compute:
1. Number of cuts per path (transitions between different clusters)
2. Number of unique clusters per path
3. Statistics: average, max, median, std for both metrics
"""

import argparse
import csv
import re
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics


def load_cluster_mapping(csv_file: str) -> Dict[str, int]:
    """Load instance to cluster mapping from CSV file."""
    cluster_map = {}
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance = row['Instance']
            cluster_id = int(row['Cluster_ID'])
            cluster_map[instance] = cluster_id
    print(f"Loaded {len(cluster_map)} instance-to-cluster mappings")
    return cluster_map


def parse_timing_paths(rpt_file: str) -> List[Tuple[int, List[str]]]:
    """Parse timing paths from report file."""
    paths = []
    current_path_id = None
    current_instances = []

    with open(rpt_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Check for path header
            path_match = re.match(r'^Path (\d+):$', line)
            if path_match:
                # Save previous path if exists
                if current_path_id is not None and current_instances:
                    paths.append((current_path_id, current_instances))

                # Start new path
                current_path_id = int(path_match.group(1))
                current_instances = []
                continue

            # Check for instances line
            if line.startswith('Instances:'):
                # Extract instances
                instances_str = line[len('Instances:'):].strip()
                instances = [inst.strip() for inst in instances_str.split('->')]
                current_instances = instances
                continue

    # Add last path
    if current_path_id is not None and current_instances:
        paths.append((current_path_id, current_instances))

    print(f"Parsed {len(paths)} timing paths")
    return paths


def analyze_path(instances: List[str], cluster_map: Dict[str, int]) -> Tuple[int, int, List[int], List[str]]:
    """
    Analyze a single path to compute:
    - Number of cuts (cluster transitions)
    - Number of unique clusters
    - List of unique clusters
    - List of missing instances (not in cluster map, assigned to -1)

    Returns: (num_cuts, num_clusters, cluster_list, missing_instances)
    """
    if not instances:
        return 0, 0, [], []

    # Get cluster assignments for all instances
    # Assign -1 to instances not found in cluster map
    clusters = []
    missing_instances = []

    for inst in instances:
        if inst in cluster_map:
            clusters.append(cluster_map[inst])
        else:
            clusters.append(-1)
            missing_instances.append(inst)

    # Count cuts (transitions between different clusters)
    num_cuts = 0
    for i in range(len(clusters) - 1):
        if clusters[i] != clusters[i + 1]:
            num_cuts += 1

    # Get unique clusters (sorted)
    unique_clusters = sorted(set(clusters))
    num_clusters = len(unique_clusters)

    return num_cuts, num_clusters, unique_clusters, missing_instances


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze timing paths and clustering data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Use default paths
  python3.9 analyze_path_clustering.py

  # Specify custom input/output files
  python3.9 analyze_path_clustering.py \\
    --cluster-csv path/to/cluster.csv \\
    --paths-rpt path/to/paths.rpt \\
    --output-csv results.csv
        '''
    )

    parser.add_argument(
        '--cluster-csv',
        type=str,
        default='test_input/ca53_cpu_leiden_timing_cluster_map.csv',
        help='Path to cluster mapping CSV file (default: test_input/ca53_cpu_leiden_timing_cluster_map.csv)'
    )

    parser.add_argument(
        '--paths-rpt',
        type=str,
        default='test_input/top_10000_paths.rpt',
        help='Path to timing paths report file (default: test_input/top_10000_paths.rpt)'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        default='path_clustering_analysis.csv',
        help='Path to output CSV file (default: path_clustering_analysis.csv)'
    )

    args = parser.parse_args()

    # Get file paths from arguments
    cluster_csv = args.cluster_csv
    paths_rpt = args.paths_rpt
    output_csv = args.output_csv

    print("=" * 80)
    print("Path Clustering Analysis")
    print("=" * 80)
    print()
    print(f"Cluster CSV:  {cluster_csv}")
    print(f"Paths Report: {paths_rpt}")
    print(f"Output CSV:   {output_csv}")
    print()

    # Load cluster mapping
    print("Step 1: Loading cluster mapping...")
    cluster_map = load_cluster_mapping(cluster_csv)
    print()

    # Parse timing paths
    print("Step 2: Parsing timing paths...")
    paths = parse_timing_paths(paths_rpt)
    print()

    # Analyze each path
    print("Step 3: Analyzing paths...")
    results = []
    all_cuts = []
    all_clusters = []
    all_missing_instances = set()  # Track unique missing instances

    for path_id, instances in paths:
        num_cuts, num_clusters, cluster_list, missing_instances = analyze_path(instances, cluster_map)
        results.append((path_id, num_cuts, num_clusters, cluster_list))
        all_cuts.append(num_cuts)
        all_clusters.append(num_clusters)
        all_missing_instances.update(missing_instances)

    print(f"Analyzed {len(results)} paths")

    # Print summary of missing instances (assigned to cluster -1)
    if all_missing_instances:
        print(f"\nNote: {len(all_missing_instances)} unique instances not found in cluster map (assigned to cluster -1)")
        if len(all_missing_instances) <= 5:
            print("  Examples:")
            for inst in sorted(list(all_missing_instances))[:5]:
                print(f"    - {inst}")
    print()

    # Write CSV output
    print(f"Step 4: Writing results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PathID', '#Cut', '#Cluster', 'ClusterList'])
        for path_id, num_cuts, num_clusters, cluster_list in results:
            cluster_list_str = ','.join(map(str, cluster_list))
            writer.writerow([path_id, num_cuts, num_clusters, cluster_list_str])
    print(f"Wrote {len(results)} rows to {output_csv}")
    print()

    # Calculate statistics
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print()

    # Cuts statistics
    print("CUT Statistics:")
    print(f"  Average:        {statistics.mean(all_cuts):.2f}")
    print(f"  Maximum:        {max(all_cuts)}")
    print(f"  Median:         {statistics.median(all_cuts):.2f}")
    print(f"  Std Deviation:  {statistics.stdev(all_cuts):.2f}")
    print()

    # Clusters statistics
    print("CLUSTER Statistics:")
    print(f"  Average:        {statistics.mean(all_clusters):.2f}")
    print(f"  Maximum:        {max(all_clusters)}")
    print(f"  Median:         {statistics.median(all_clusters):.2f}")
    print(f"  Std Deviation:  {statistics.stdev(all_clusters):.2f}")
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
