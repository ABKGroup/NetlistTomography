#!/usr/bin/env python3
"""
Generate DEF file from cluster map CSV

This module provides functionality to convert cluster mapping CSV files
into DEF (Design Exchange Format) files for use with EDA tools.
"""

import argparse
import csv
import os
import re
from collections import defaultdict


def write_cluster_def_file(cluster_map_csv: str,
                           design_name: str,
                           output_def: str) -> None:
    """
    Generate DEF file from cluster map CSV.

    Args:
        cluster_map_csv: Path to cluster map CSV file (Instance, Cluster_ID columns)
        design_name: Design name to use in DEF file
        output_def: Output DEF file path

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
    """
    print(f"Generating DEF file from cluster map...")
    print(f"  Input: {cluster_map_csv}")
    print(f"  Design: {design_name}")
    print(f"  Output: {output_def}")

    # Read cluster map CSV
    cluster_map = defaultdict(list)  # cluster_id -> [instances]
    total_count = 0

    with open(cluster_map_csv, 'r') as f:
        reader = csv.DictReader(f)

        # Validate columns
        if 'Instance' not in reader.fieldnames or 'Cluster_ID' not in reader.fieldnames:
            raise ValueError("CSV must contain 'Instance' and 'Cluster_ID' columns")

        for row in reader:
            instance = row['Instance']
            cluster_id = int(row['Cluster_ID'])
            total_count += 1
            cluster_map[cluster_id].append(instance)

    print(f"  Loaded {total_count:,} instance-cluster mappings")

    # Get unique cluster IDs (sorted)
    cluster_ids = sorted(cluster_map.keys())
    num_clusters = len(cluster_ids)
    valid_instances = sum(len(instances) for instances in cluster_map.values())

    print(f"  Writing {num_clusters} clusters with {valid_instances:,} instances")

    # Create output directory if needed
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

            # Get instances in this cluster (sorted for consistent output)
            instances = sorted(cluster_map[cluster_id])

            for instance in instances:
                # Clean instance name (remove leading/trailing braces if present)
                instance_clean = re.sub(r'^\\{|\\}$', '', str(instance))
                fp.write(f"    {instance_clean}\n")

            fp.write(f";\n")

        # Footer
        fp.write(f"END GROUPS\n")
        fp.write(f"END DESIGN\n")

    print(f"  DEF file written successfully!")
    print(f"  Output: {output_def}")


def main():
    """Command-line interface for DEF file generation."""
    parser = argparse.ArgumentParser(
        description='Generate DEF file from cluster map CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate DEF from optimized cluster map
  python3.9 generate_cluster_def.py \\
      --cluster-map optimized_cluster_map.csv \\
      --design ariane \\
      --output ariane_clusters.def

  # Generate DEF from FM optimizer output
  python3.9 generate_cluster_def.py \\
      --cluster-map optimized_cluster_map_fm.csv \\
      --design ariane \\
      --output ariane_clusters_fm.def
"""
    )

    parser.add_argument(
        '--cluster-map',
        type=str,
        required=True,
        help='Input cluster map CSV file (must have Instance and Cluster_ID columns)'
    )

    parser.add_argument(
        '--design',
        type=str,
        required=True,
        help='Design name for DEF file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output DEF file path'
    )

    args = parser.parse_args()

    try:
        write_cluster_def_file(
            cluster_map_csv=args.cluster_map,
            design_name=args.design,
            output_def=args.output
        )
        print("\nDEF generation complete!")

    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
