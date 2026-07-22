#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Complete synthesis mapping pipeline: run both node location and edge mapping')
    parser.add_argument('--dir_a', required=True, help='Directory A (with known locations, e.g., design_place_opt)')
    parser.add_argument('--dir_b', required=True, help='Directory B (missing locations, e.g., design_post_synth)')
    parser.add_argument('--design', required=True, help='Design name (e.g., ca53_cpu)')
    parser.add_argument('--output_dir', required=True, help='Output directory for results')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for node mapping (default: no timeout)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output file paths
    nodes_output = output_dir / f"{args.design}_complete_nodes.csv"
    edges_output = output_dir / f"{args.design}_complete_edges.csv"
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Step 1: Run node location mapping
    print("="*60)
    print("STEP 1: Running node location mapping")
    print("="*60)
    
    if args.timeout is not None:
        node_cmd = [
            'timeout', str(args.timeout),
            'python3', str(script_dir / 'syn_map_loc_generation.py'),
            '--dir_a', args.dir_a,
            '--dir_b', args.dir_b,
            '--design', args.design,
            '--output', str(nodes_output)
        ]
    else:
        node_cmd = [
            'python3', str(script_dir / 'syn_map_loc_generation.py'),
            '--dir_a', args.dir_a,
            '--dir_b', args.dir_b,
            '--design', args.design,
            '--output', str(nodes_output)
        ]
    
    if not run_command(node_cmd, "Generating node location estimates..."):
        print("Error: Node location mapping failed!")
        sys.exit(1)
    
    # Verify nodes output exists
    if not nodes_output.exists():
        print(f"Error: Expected output file {nodes_output} was not created!")
        sys.exit(1)
    
    print(f"✓ Node location mapping completed successfully!")
    print(f"  Output: {nodes_output}")
    
    # Step 2: Run edge mapping
    print("\n" + "="*60)
    print("STEP 2: Running edge mapping")
    print("="*60)
    
    edge_cmd = [
        'python3', str(script_dir / 'syn_map_edge_generation.py'),
        '--dir_a', args.dir_a,
        '--dir_b', args.dir_b,
        '--design', args.design,
        '--mapped_nodes', str(nodes_output),
        '--output', str(edges_output)
    ]
    
    if not run_command(edge_cmd, "Generating edge mapping with timing estimates..."):
        print("Error: Edge mapping failed!")
        sys.exit(1)
    
    # Verify edges output exists
    if not edges_output.exists():
        print(f"Error: Expected output file {edges_output} was not created!")
        sys.exit(1)
    
    print(f"✓ Edge mapping completed successfully!")
    print(f"  Output: {edges_output}")
    
    # Final summary
    print("\n" + "="*60)
    print("SYNTHESIS MAPPING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Input directories:")
    print(f"  Directory A (known locations): {args.dir_a}")
    print(f"  Directory B (to be mapped):    {args.dir_b}")
    print(f"  Design name:                   {args.design}")
    print(f"\nOutput files:")
    print(f"  Complete nodes: {nodes_output}")
    print(f"  Complete edges: {edges_output}")
    
    # Show file sizes for verification
    try:
        nodes_size = nodes_output.stat().st_size
        edges_size = edges_output.stat().st_size
        print(f"\nOutput file sizes:")
        print(f"  Nodes: {nodes_size:,} bytes")
        print(f"  Edges: {edges_size:,} bytes")
    except:
        pass

if __name__ == "__main__":
    main()