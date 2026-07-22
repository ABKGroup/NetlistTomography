#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
import argparse
import time
import pandas as pd
from pathlib import Path
import concurrent.futures
from typing import List, Tuple

def find_dir_a_directories(base_dir: str, pattern: str) -> List[str]:
    """Find all DIR_A directories matching the pattern."""
    search_pattern = os.path.join(base_dir, f"{pattern}", "post_place_opt")
    matched_dirs = glob.glob(search_pattern)
    return sorted(matched_dirs)

def run_syn_map_complete(dir_a: str, dir_b: str, design: str, timeout: int = 300) -> Tuple[str, bool, str]:
    """Run syn_map_complete.py for a single DIR_A to DIR_B mapping."""
    # Extract the run directory name for output
    run_dir = os.path.dirname(dir_a)
    synth_map_dir = os.path.join(run_dir, "synth_map")
    
    # Create synth_map directory if it doesn't exist
    os.makedirs(synth_map_dir, exist_ok=True)
    
    # Run syn_map_complete.py
    cmd = [
        "python3", "syn_map_complete.py",
        "--dir_a", dir_a,
        "--dir_b", dir_b, 
        "--design", design,
        "--output_dir", synth_map_dir,
        "--timeout", str(timeout)
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout + 60  # Add buffer time
        )
        
        success = result.returncode == 0
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        
        return run_dir, success, output
    
    except subprocess.TimeoutExpired:
        return run_dir, False, f"Process timed out after {timeout + 60} seconds"
    except Exception as e:
        return run_dir, False, f"Error running command: {str(e)}"

def merge_node_files(synth_map_dirs: List[str], design: str, output_dir: str) -> str:
    """Merge all node files with numbered column headers."""
    node_files = []
    
    # Collect all node files
    for synth_map_dir in synth_map_dirs:
        node_file = os.path.join(synth_map_dir, f"{design}_complete_nodes.csv")
        if os.path.exists(node_file):
            node_files.append(node_file)
    
    if not node_files:
        raise ValueError("No node files found to merge")
    
    print(f"Found {len(node_files)} node files to merge")
    
    # Read first file to get the base structure
    base_df = pd.read_csv(node_files[0])
    
    # Columns to merge with numbering
    merge_columns = ['Slack', 'PT_X', 'PT_Y']
    
    # Create the merged dataframe starting with Instance and Cell columns
    merged_df = base_df[['Instance', 'Cell']].copy()
    
    # Check ClockPeriod consistency and find minimum > 0.0
    clock_periods = []
    for i, file_path in enumerate(node_files):
        df = pd.read_csv(file_path)
        clock_periods.extend(df['ClockPeriod'].unique())
    
    unique_clock_periods = list(set(clock_periods))
    # Filter out 0.0 values and find minimum
    valid_clock_periods = [cp for cp in unique_clock_periods if cp > 0.0]
    
    if len(unique_clock_periods) > 1:
        if valid_clock_periods:
            min_clock_period = min(valid_clock_periods)
            print(f"Multiple clock periods found: {unique_clock_periods}")
            print(f"Using minimum clock period > 0.0: {min_clock_period}")
        else:
            min_clock_period = unique_clock_periods[0]  # Fallback if all are 0.0
            print(f"Warning: All clock periods are 0.0, using: {min_clock_period}")
    else:
        min_clock_period = unique_clock_periods[0]
        print(f"Clock period consistent across all files: {min_clock_period}")
    
    # Add ClockPeriod using the determined value (rounded to 6 decimal places)
    merged_df['ClockPeriod'] = round(min_clock_period, 6)
    
    # Add Width and Height from first file (rounded to 6 decimal places)
    if 'Width' in base_df.columns:
        merged_df['Width'] = base_df['Width'].round(6)
    if 'Height' in base_df.columns:
        merged_df['Height'] = base_df['Height'].round(6)
    
    # Merge numbered columns
    for col in merge_columns:
        for i, file_path in enumerate(node_files, 1):
            df = pd.read_csv(file_path)
            # Merge on Instance to handle potentially different row orders
            col_name = f"{col.lower()}_{i}"
            temp_df = df[['Instance', col]].rename(columns={col: col_name})
            # Round numerical values to 6 decimal places
            if col in ['Slack', 'PT_X', 'PT_Y']:
                temp_df[col_name] = temp_df[col_name].round(6)
            merged_df = merged_df.merge(temp_df, on='Instance', how='left')
    
    # Save merged file
    output_file = os.path.join(output_dir, f"{design}_merged_nodes.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged file saved to: {output_file}")
    print(f"Merged dataframe shape: {merged_df.shape}")
    
    return output_file

def merge_edge_files(synth_map_dirs: List[str], design: str, output_dir: str) -> str:
    """Merge all edge files with numbered column headers."""
    import time
    
    merge_start = time.time()  # Initialize merge timing
    edge_files = []
    
    # Collect all edge files
    collect_start = time.time()
    for synth_map_dir in synth_map_dirs:
        edge_file = os.path.join(synth_map_dir, f"{design}_complete_edges.csv")
        if os.path.exists(edge_file):
            edge_files.append(edge_file)
    
    if not edge_files:
        raise ValueError("No edge files found to merge")
    
    collect_time = time.time() - collect_start
    print(f"Found {len(edge_files)} edge files to merge (collected in {collect_time:.1f}s)")
    
    # Read first file to get the base structure and determine available columns
    read_start = time.time()
    base_df = pd.read_csv(edge_files[0])
    read_time = time.time() - read_start
    print(f"Edge file columns: {list(base_df.columns)} (first file read in {read_time:.1f}s)")
    print(f"First file has {len(base_df)} rows")
    
    # Check which columns are available for merging
    available_merge_columns = []
    if 'Slack' in base_df.columns:
        available_merge_columns.append('Slack')
    if 'ManhattanDistance' in base_df.columns:
        available_merge_columns.append('ManhattanDistance')
    
    print(f"Columns available for merging: {available_merge_columns}")
    
    if not available_merge_columns:
        raise ValueError("No mergeable columns (Slack or ManhattanDistance) found in edge files")
    
    # Create the base structure from first file
    setup_start = time.time()
    base_edges = base_df[['Net', 'Source', 'Sink']].copy()
    
    # Handle ClockPeriod consistency (same logic as nodes)
    clock_start = time.time()
    clock_periods = []
    for i, file_path in enumerate(edge_files):
        df = pd.read_csv(file_path)
        if 'ClockPeriod' in df.columns:
            clock_periods.extend(df['ClockPeriod'].unique())
    
    unique_clock_periods = list(set(clock_periods))
    valid_clock_periods = [cp for cp in unique_clock_periods if cp > 0.0]
    
    if len(unique_clock_periods) > 1:
        if valid_clock_periods:
            min_clock_period = min(valid_clock_periods)
            print(f"Multiple clock periods found: {unique_clock_periods}")
            print(f"Using minimum clock period > 0.0: {min_clock_period}")
        else:
            min_clock_period = unique_clock_periods[0]
            print(f"Warning: All clock periods are 0.0, using: {min_clock_period}")
    else:
        min_clock_period = unique_clock_periods[0] if unique_clock_periods else 1.0
        print(f"Clock period consistent across all files: {min_clock_period}")
    
    clock_time = time.time() - clock_start
    setup_time = time.time() - setup_start
    print(f"Clock period analysis completed in {clock_time:.1f}s, setup total: {setup_time:.1f}s")
    
    # Read all files and concatenate for pivot approach
    read_all_start = time.time()
    print(f"Reading all {len(edge_files)} edge files for pivot merge...")
    
    all_dfs = []
    for i, file_path in enumerate(edge_files, 1):
        file_start = time.time()
        df = pd.read_csv(file_path)
        
        # Add file identifier
        df['file_id'] = i
        
        # Handle special values in Slack column before pivot
        if 'Slack' in df.columns and 'Slack' in available_merge_columns:
            df['Slack'] = df['Slack'].replace('INFINITY', float('inf'))
            df['Slack'] = pd.to_numeric(df['Slack'], errors='coerce')
        
        # Round numerical values to 6 decimal places for available columns only
        for col in available_merge_columns:
            if col in df.columns:
                if col in ['Slack', 'ManhattanDistance']:
                    df[col] = df[col].round(6)
        
        all_dfs.append(df)
        
        read_time = time.time() - file_start
        if i <= 3 or i % 5 == 0:  # Show progress for first few files and every 5th
            print(f"    File {i}/{len(edge_files)}: read and processed in {read_time:.1f}s")
    
    read_all_time = time.time() - read_all_start
    print(f"All files read in {read_all_time:.1f}s")
    
    # Concatenate all dataframes
    concat_start = time.time()
    print(f"Concatenating {len(all_dfs)} dataframes...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    concat_time = time.time() - concat_start
    print(f"Concatenation completed in {concat_time:.1f}s. Combined shape: {combined_df.shape}")
    
    # Pivot to create numbered columns
    pivot_start = time.time()
    print(f"Creating pivot table for columns: {available_merge_columns}")
    
    try:
        # Create pivot table
        pivot_df = combined_df.pivot_table(
            index=['Net', 'Source', 'Sink'],
            columns='file_id',
            values=available_merge_columns,
            aggfunc='first'  # Take first value if duplicates exist
        )
        
        pivot_time = time.time() - pivot_start
        print(f"Pivot table created in {pivot_time:.1f}s. Pivot shape: {pivot_df.shape}")
        
        # Flatten column names and rename to match expected format
        flatten_start = time.time()
        pivot_df.columns = [f"{col[0].lower()}_{col[1]}" if col[0] == 'Slack' 
                           else f"length_{col[1]}" if col[0] == 'ManhattanDistance'
                           else f"{col[0].lower()}_{col[1]}" 
                           for col in pivot_df.columns]
        
        # Reset index to make Net, Source, Sink regular columns
        final_merged_df = pivot_df.reset_index()
        
        flatten_time = time.time() - flatten_start
        print(f"Column flattening completed in {flatten_time:.1f}s")
        
    except Exception as e:
        print(f"Pivot operation failed: {e}")
        print("Falling back to iterative merge approach...")
        
        # Fallback: iterative merge approach
        final_merged_df = base_edges.copy()
        
        for i, file_path in enumerate(edge_files, 1):
            df = pd.read_csv(file_path)
            
            # Handle special values
            if 'Slack' in df.columns and 'Slack' in available_merge_columns:
                df['Slack'] = df['Slack'].replace('INFINITY', float('inf'))
                df['Slack'] = pd.to_numeric(df['Slack'], errors='coerce')
            
            # Merge each available column
            for col in available_merge_columns:
                if col in df.columns:
                    col_name = f"{col.lower()}_{i}" if col == 'Slack' else f"length_{i}"
                    merge_df = df[['Net', 'Source', 'Sink', col]].rename(columns={col: col_name})
                    final_merged_df = final_merged_df.merge(
                        merge_df, 
                        on=['Net', 'Source', 'Sink'], 
                        how='outer'
                    )
    
    cleanup_start = time.time()
    # Add ClockPeriod column
    final_merged_df.insert(3, 'ClockPeriod', round(min_clock_period, 6))
    cleanup_time = time.time() - cleanup_start
    
    merge_time = time.time() - merge_start
    print(f"All column merging completed in {merge_time:.1f}s, cleanup in {cleanup_time:.1f}s")
    
    # Save merged file
    save_start = time.time()
    output_file = os.path.join(output_dir, f"{design}_merged_edges.csv")
    final_merged_df.to_csv(output_file, index=False)
    save_time = time.time() - save_start
    
    print(f"Merged edge file saved to: {output_file} (saved in {save_time:.1f}s)")
    print(f"Merged dataframe shape: {final_merged_df.shape}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Batch synthesis mapping with parallel execution')
    parser.add_argument('--base_dir', required=True, help='Base directory containing DIR_A directories')
    parser.add_argument('--pattern', required=True, help='Pattern to match DIR_A directories (e.g., run_tomo_cp_1.3_util_0.8)')
    parser.add_argument('--dir_b', required=True, help='DIR_B path')
    parser.add_argument('--design', required=True, help='Design name (e.g., ariane)')
    parser.add_argument('--output_dir', required=True, help='Output directory for merged files')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per mapping job in seconds (default: 300)')
    parser.add_argument('--max_workers', type=int, default=30, help='Maximum number of parallel workers (default: 30)')
    parser.add_argument('--test_run', action='store_true', help='Run with only first 3 directories for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all DIR_A directories
    print(f"Searching for directories matching pattern: {args.pattern}")
    dir_a_list = find_dir_a_directories(args.base_dir, args.pattern)
    
    if not dir_a_list:
        print(f"No directories found matching pattern: {args.pattern}")
        return 1
    
    print(f"Found {len(dir_a_list)} directories:")
    for dir_a in dir_a_list:
        print(f"  {dir_a}")
    
    # Test run with subset
    if args.test_run:
        dir_a_list = dir_a_list[:3]
        print(f"\nTest run: Using only first {len(dir_a_list)} directories")
    
    # Run parallel mapping
    print(f"\nRunning {len(dir_a_list)} mapping jobs in parallel (max_workers={args.max_workers})...")
    start_time = time.time()
    
    successful_jobs = []
    failed_jobs = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all jobs
        future_to_dir = {
            executor.submit(run_syn_map_complete, dir_a, args.dir_b, args.design, args.timeout): dir_a
            for dir_a in dir_a_list
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_dir):
            dir_a = future_to_dir[future]
            try:
                run_dir, success, output = future.result()
                
                if success:
                    successful_jobs.append(run_dir)
                    print(f"✓ SUCCESS: {os.path.basename(run_dir)}")
                else:
                    failed_jobs.append((run_dir, output))
                    print(f"✗ FAILED: {os.path.basename(run_dir)}")
                    
            except Exception as exc:
                failed_jobs.append((dir_a, str(exc)))
                print(f"✗ EXCEPTION: {os.path.basename(dir_a)} - {exc}")
    
    elapsed_time = time.time() - start_time
    print(f"\nParallel execution completed in {elapsed_time:.1f} seconds")
    print(f"Successful jobs: {len(successful_jobs)}/{len(dir_a_list)}")
    
    if failed_jobs:
        print("\nFailed jobs:")
        for run_dir, error in failed_jobs:
            print(f"  {os.path.basename(run_dir)}: {error}")
    
    # Merge node and edge files from successful jobs
    if successful_jobs:
        synth_map_dirs = [os.path.join(run_dir, "synth_map") for run_dir in successful_jobs]
        
        # Merge node files
        print(f"\nMerging node files from {len(successful_jobs)} successful jobs...")
        node_merge_start = time.time()
        try:
            merged_node_file = merge_node_files(synth_map_dirs, args.design, args.output_dir)
            node_merge_time = time.time() - node_merge_start
            print(f"\n✓ Node merge completed successfully in {node_merge_time:.1f} seconds: {merged_node_file}")
        except Exception as e:
            node_merge_time = time.time() - node_merge_start
            print(f"\n✗ Node merge failed after {node_merge_time:.1f} seconds: {e}")
            return 1
        
        # Merge edge files
        print(f"\nMerging edge files from {len(successful_jobs)} successful jobs...")
        edge_merge_start = time.time()
        try:
            merged_edge_file = merge_edge_files(synth_map_dirs, args.design, args.output_dir)
            edge_merge_time = time.time() - edge_merge_start
            print(f"\n✓ Edge merge completed successfully in {edge_merge_time:.1f} seconds: {merged_edge_file}")
        except Exception as e:
            edge_merge_time = time.time() - edge_merge_start
            print(f"\n✗ Edge merge failed after {edge_merge_time:.1f} seconds: {e}")
            # Continue even if edge merge fails, since node merge succeeded
            print(f"Note: Node merge was successful, only edge merge failed")
    else:
        print("\n✗ No successful jobs to merge")
        return 1
    
    # Print timing summary
    total_time = time.time() - start_time
    print(f"\n" + "="*50)
    print(f"TIMING SUMMARY:")
    print(f"  Parallel mapping execution: {elapsed_time:.1f} seconds")
    if successful_jobs:
        print(f"  Node file merging: {node_merge_time:.1f} seconds")
        print(f"  Edge file merging: {edge_merge_time:.1f} seconds")
        print(f"  Total merge time: {node_merge_time + edge_merge_time:.1f} seconds")
    print(f"  Total batch processing time: {total_time:.1f} seconds")
    print(f"="*50)
    
    print(f"\nBatch processing completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())