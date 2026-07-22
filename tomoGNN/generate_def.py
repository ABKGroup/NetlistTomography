# generate_def.py
# Generate DEF files from cluster assignments
from __future__ import annotations
import pandas as pd
import os
import re
import csv
from collections import defaultdict
from typing import Dict, List


def write_cluster_def_file(
    cluster_map_csv: str,
    design_name: str,
    output_def: str,
    min_cluster_size: int = 200
) -> None:
    """
    Generate DEF file from cluster map CSV.

    Expected CSV format:
        Instance,Cluster_ID
        inst1,0
        inst2,0
        inst3,1

    DEF Format:
        VERSION 5.8 ;
        DESIGN design_name ;
        GROUPS num_clusters ;
        - cluster_0
            instance1
            instance2
        ;
        - cluster_1
            instance3
        ;
        END GROUPS
        END DESIGN

    Filtering:
    - Only clusters with size >= min_cluster_size are written
    - Instances from small clusters are excluded from DEF

    Args:
        cluster_map_csv: Path to cluster CSV (Instance, Cluster_ID)
        design_name: Design name for DEF file
        output_def: Output DEF file path
        min_cluster_size: Minimum cluster size (default 200)
    """
    print("="*60)
    print("Generating DEF File")
    print("="*60)
    print(f"  Input: {cluster_map_csv}")
    print(f"  Design: {design_name}")
    print(f"  Output: {output_def}")
    print(f"  Min cluster size: {min_cluster_size}")

    # Read cluster map
    cluster_map = defaultdict(list)  # cluster_id -> [instances]
    total_count = 0

    with open(cluster_map_csv, 'r') as f:
        reader = csv.DictReader(f)

        # Validate columns
        if 'Instance' not in reader.fieldnames:
            raise ValueError(
                "CSV must have 'Instance' column"
            )
        if 'Cluster_ID' not in reader.fieldnames:
            raise ValueError(
                "CSV must have 'Cluster_ID' column"
            )

        # Read all mappings
        for row in reader:
            instance = row['Instance']
            cluster_id = int(row['Cluster_ID'])

            # Skip noise points (cluster_id = -1 from HDBSCAN)
            if cluster_id >= 0:
                cluster_map[cluster_id].append(instance)
                total_count += 1

    print(f"  Loaded {total_count:,} valid instance mappings")
    print(f"  Total unique clusters: {len(cluster_map)}")

    # Filter by cluster size
    valid_clusters = {
        cid: instances
        for cid, instances in cluster_map.items()
        if len(instances) >= min_cluster_size
    }

    filtered_out = len(cluster_map) - len(valid_clusters)
    valid_instances = sum(
        len(insts) for insts in valid_clusters.values()
    )
    excluded_instances = total_count - valid_instances

    cluster_ids = sorted(valid_clusters.keys())
    num_clusters = len(cluster_ids)

    print(f"  Clusters after filtering: {num_clusters}")
    print(f"  Filtered out {filtered_out} small clusters")
    print(f"  Valid instances: {valid_instances:,}")
    print(f"  Excluded instances: {excluded_instances:,}")

    if num_clusters == 0:
        print("  WARNING: No valid clusters to write!")
        print("  Consider lowering min_cluster_size")
        return

    # Create output directory
    output_dir = os.path.dirname(output_def)
    if output_dir and not os.path.exists(output_dir):
        print(f"  Creating directory: {output_dir}")
        os.makedirs(output_dir)

    # Write DEF file
    with open(output_def, 'w') as fp:
        # Header
        fp.write(f"VERSION 5.8 ;\n")
        fp.write(f"DESIGN {design_name} ;\n")
        fp.write(f"GROUPS {num_clusters} ;\n")

        # Write each cluster
        for cluster_id in cluster_ids:
            fp.write(f"- cluster_{cluster_id}\n")

            # Sort instances for consistent output
            instances = sorted(valid_clusters[cluster_id])

            for instance in instances:
                # Clean instance name (remove braces if present)
                instance_clean = re.sub(
                    r'^\\{|\\}$',
                    '',
                    str(instance)
                )
                fp.write(f"    {instance_clean}\n")

            fp.write(f";\n")

        # Footer
        fp.write(f"END GROUPS\n")
        fp.write(f"END DESIGN\n")

    print("="*60)
    print("DEF File Generated Successfully!")
    print(f"  Output: {output_def}")
    print(f"  Clusters written: {num_clusters}")
    print(f"  Instances written: {valid_instances:,}")
    print("="*60)


def save_cluster_map_csv(
    node_names: List[str],
    labels: List[int],
    output_csv: str
) -> None:
    """
    Save cluster assignments to CSV.

    Format:
        Instance,Cluster_ID
        inst1,0
        inst2,0
        inst3,1

    Args:
        node_names: List of node/instance names
        labels: Cluster labels (can include -1 for noise)
        output_csv: Output CSV path
    """
    df = pd.DataFrame({
        'Instance': node_names,
        'Cluster_ID': labels
    })

    df.to_csv(output_csv, index=False)

    # Print statistics
    cluster_counts = df['Cluster_ID'].value_counts().sort_index()
    noise_count = (df['Cluster_ID'] == -1).sum()

    print(f"Cluster map saved: {output_csv}")
    print(f"  Total instances: {len(df):,}")
    print(f"  Unique clusters: {len(cluster_counts)}")
    print(f"  Noise points (-1): {noise_count}")

    if len(cluster_counts) > 0:
        print(f"  Cluster size range: "
              f"{cluster_counts.min()} - {cluster_counts.max()}")
        print(f"  Mean cluster size: {cluster_counts.mean():.1f}")
