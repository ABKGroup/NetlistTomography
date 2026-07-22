import csv
import sys
import argparse
from collections import defaultdict

def read_nodes_csv(filepath):
    """Read nodes CSV and return dict of instance -> {timing and location info}"""
    nodes = {}
    skipped_rows = 0
    total_rows = 0
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            total_rows += 1
            try:
                instance = row['Instance']
                
                # Handle slack - could be numeric or "INFINITY"
                slack_raw = row['Slack'].strip()
                if slack_raw.upper() == 'INFINITY' or slack_raw == '':
                    slack = float('inf')
                else:
                    slack = float(slack_raw)
                
                # Handle clock period - could be empty
                clock_period_raw = row['ClockPeriod'].strip()
                if clock_period_raw == '':
                    clock_period = 0.0
                else:
                    clock_period = float(clock_period_raw)
                
                nodes[instance] = {
                    'cell': row['Cell'],
                    'pt_x': float(row['PT_X']),
                    'pt_y': float(row['PT_Y']),
                    'slack': slack,
                    'clock_period': clock_period,
                    'width': row.get('Width', ''),
                    'height': row.get('Height', '')
                }
            except Exception as e:
                skipped_rows += 1
                print(f"Warning: Skipped row {row_num} due to error: {e}")
                if skipped_rows > 10:  # Limit debug output
                    print(f"  ... (stopping debug output after 10 errors)")
                    break
    
    if skipped_rows > 0:
        print(f"Total rows processed: {total_rows}, Successfully parsed: {len(nodes)}, Skipped: {skipped_rows}")
    
    return nodes

def read_edges_csv(filepath):
    """Read edges CSV and return list of edge dictionaries"""
    edges = []
    skipped_rows = 0
    total_rows = 0
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            total_rows += 1
            try:
                # Handle slack - could be numeric or "INFINITY"
                slack_raw = row['Slack'].strip()
                if slack_raw.upper() == 'INFINITY' or slack_raw == '':
                    slack = float('inf')
                else:
                    slack = float(slack_raw)
                
                # Handle clock period - could be empty
                clock_period_raw = row['ClockPeriod'].strip()
                if clock_period_raw == '':
                    clock_period = 0.0
                else:
                    clock_period = float(clock_period_raw)
                
                edges.append({
                    'Net': row['Net'],
                    'Source': row['Source'],
                    'Sink': row['Sink'],
                    'Slack': slack,
                    'ClockPeriod': clock_period
                })
            except Exception as e:
                skipped_rows += 1
                print(f"Warning: Skipped edge row {row_num} due to error: {e}")
                if skipped_rows > 10:
                    print(f"  ... (stopping debug output after 10 errors)")
                    break
    
    if skipped_rows > 0:
        print(f"Total edge rows processed: {total_rows}, Successfully parsed: {len(edges)}, Skipped: {skipped_rows}")
    
    return edges

def calculate_manhattan_distance(source_node, sink_node, complete_nodes, nodes_a):
    """
    Calculate Manhattan distance between source and sink nodes.
    Uses mapped coordinates from complete_nodes for consistency with timing data.
    For nodes not in mapping but available in DIR_A, falls back to original coordinates.
    """
    # Get source coordinates - prefer mapped data for consistency with timing
    if source_node in complete_nodes:
        source_x = complete_nodes[source_node]['pt_x']
        source_y = complete_nodes[source_node]['pt_y']
    elif source_node in nodes_a:
        # Fallback to original DIR_A coordinates if not in mapping
        source_x = nodes_a[source_node]['pt_x']
        source_y = nodes_a[source_node]['pt_y']
    else:
        raise ValueError(f"Source node {source_node} not found in either mapped or original data")
    
    # Get sink coordinates - prefer mapped data for consistency with timing
    if sink_node in complete_nodes:
        sink_x = complete_nodes[sink_node]['pt_x']
        sink_y = complete_nodes[sink_node]['pt_y']
    elif sink_node in nodes_a:
        # Fallback to original DIR_A coordinates if not in mapping
        sink_x = nodes_a[sink_node]['pt_x']
        sink_y = nodes_a[sink_node]['pt_y']
    else:
        raise ValueError(f"Sink node {sink_node} not found in either mapped or original data")
    
    return abs(source_x - sink_x) + abs(source_y - sink_y)

def map_missing_edges(edges_b, nodes_a, complete_nodes, edges_a_lookup):
    """
    Map edges from design B using timing data based on node availability
    """
    mapped_edges = []
    case_counts = {'both_available': 0, 'source_missing': 0, 'sink_missing': 0, 'both_missing': 0}
    
    for edge in edges_b:
        source = edge['Source']
        sink = edge['Sink']
        
        # Determine node availability
        source_available = source in nodes_a
        sink_available = sink in nodes_a
        
        # Calculate Manhattan distance using consistent coordinate source
        manhattan_dist = calculate_manhattan_distance(source, sink, complete_nodes, nodes_a)
        
        if source_available and sink_available:
            # Case 1: Both available - use DIR_A edge timing if available, else derive from nodes
            edge_key = (source, sink)
            if edge_key in edges_a_lookup:
                # Use DIR_A edge timing
                dir_a_edge = edges_a_lookup[edge_key]
                slack = dir_a_edge['Slack']
                clock_period = dir_a_edge['ClockPeriod']
                estimation_source = 'both_available_dir_a'
            else:
                # Edge doesn't exist in DIR_A, derive from mapped nodes
                source_slack = complete_nodes[source]['slack']
                sink_slack = complete_nodes[sink]['slack']
                slack = min(source_slack, sink_slack)
                
                # Use minimum clock period > 0.0 from both nodes
                source_cp = complete_nodes[source]['clock_period']
                sink_cp = complete_nodes[sink]['clock_period']
                valid_cps = [cp for cp in [source_cp, sink_cp] if cp > 0.0]
                clock_period = min(valid_cps) if valid_cps else max(source_cp, sink_cp)
                estimation_source = 'both_available_derived'
            
            mapped_edge = {
                'Net': edge['Net'],
                'Source': source,
                'Sink': sink,
                'Slack': slack,
                'ClockPeriod': clock_period,
                'ManhattanDistance': manhattan_dist,
                'EstimationSource': estimation_source
            }
            case_counts['both_available'] += 1
            
        elif not source_available and sink_available:
            # Case 2: Source missing, sink available
            source_slack = complete_nodes[source]['slack']
            sink_slack = complete_nodes[sink]['slack']
            
            # Use minimum clock period > 0.0 from both nodes
            source_cp = complete_nodes[source]['clock_period']
            sink_cp = complete_nodes[sink]['clock_period']
            valid_cps = [cp for cp in [source_cp, sink_cp] if cp > 0.0]
            clock_period = min(valid_cps) if valid_cps else max(source_cp, sink_cp)
            
            mapped_edge = {
                'Net': edge['Net'],
                'Source': source,
                'Sink': sink,
                'Slack': min(source_slack, sink_slack),
                'ClockPeriod': clock_period,
                'ManhattanDistance': manhattan_dist,
                'EstimationSource': 'source_missing'
            }
            case_counts['source_missing'] += 1
            
        elif source_available and not sink_available:
            # Case 3: Source available, sink missing
            source_slack = complete_nodes[source]['slack']
            sink_slack = complete_nodes[sink]['slack']
            
            # Use minimum clock period > 0.0 from both nodes
            source_cp = complete_nodes[source]['clock_period']
            sink_cp = complete_nodes[sink]['clock_period']
            valid_cps = [cp for cp in [source_cp, sink_cp] if cp > 0.0]
            clock_period = min(valid_cps) if valid_cps else max(source_cp, sink_cp)
            
            mapped_edge = {
                'Net': edge['Net'],
                'Source': source,
                'Sink': sink,
                'Slack': min(source_slack, sink_slack),
                'ClockPeriod': clock_period,
                'ManhattanDistance': manhattan_dist,
                'EstimationSource': 'sink_missing'
            }
            case_counts['sink_missing'] += 1
            
        else:
            # Case 4: Both missing
            source_slack = complete_nodes[source]['slack']
            sink_slack = complete_nodes[sink]['slack']
            
            # Use minimum clock period > 0.0 from both nodes
            source_cp = complete_nodes[source]['clock_period']
            sink_cp = complete_nodes[sink]['clock_period']
            valid_cps = [cp for cp in [source_cp, sink_cp] if cp > 0.0]
            clock_period = min(valid_cps) if valid_cps else max(source_cp, sink_cp)
            
            mapped_edge = {
                'Net': edge['Net'],
                'Source': source,
                'Sink': sink,
                'Slack': min(source_slack, sink_slack),
                'ClockPeriod': clock_period,
                'ManhattanDistance': manhattan_dist,
                'EstimationSource': 'both_missing'
            }
            case_counts['both_missing'] += 1
        
        mapped_edges.append(mapped_edge)
    
    # Print statistics
    print(f"Edge mapping statistics:")
    print(f"  Both nodes available: {case_counts['both_available']}")
    print(f"  Source missing: {case_counts['source_missing']}")
    print(f"  Sink missing: {case_counts['sink_missing']}")
    print(f"  Both missing: {case_counts['both_missing']}")
    print(f"  Total edges: {len(mapped_edges)}")
    
    return mapped_edges

def validate_edge_mapping(edges, complete_nodes):
    """Validate that all edge endpoints exist in complete nodes"""
    missing_nodes = set()
    
    for edge in edges:
        source = edge['Source']
        sink = edge['Sink']
        
        if source not in complete_nodes:
            missing_nodes.add(source)
        if sink not in complete_nodes:
            missing_nodes.add(sink)
    
    if missing_nodes:
        print(f"Error: {len(missing_nodes)} edge endpoints not found in complete nodes mapping:")
        for node in list(missing_nodes)[:10]:  # Show first 10
            print(f"  {node}")
        if len(missing_nodes) > 10:
            print(f"  ... and {len(missing_nodes) - 10} more")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate edge mapping with timing estimates for missing nodes')
    parser.add_argument('--dir_a', required=True, help='Directory A (with original timing, e.g., design_place_opt)')
    parser.add_argument('--dir_b', required=True, help='Directory B (edges to be mapped, e.g., design_post_synth)')  
    parser.add_argument('--design', required=True, help='Design name (e.g., ca53_cpu)')
    parser.add_argument('--mapped_nodes', required=True, help='Complete mapped nodes CSV file (e.g., ca53_cpu_complete_nodes.csv)')
    parser.add_argument('--output', default='mapped_edges.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # File paths
    nodes_a_path = f"{args.dir_a}/{args.design}_nodes.csv"
    edges_a_path = f"{args.dir_a}/{args.design}_edges.csv"
    edges_b_path = f"{args.dir_b}/{args.design}_edges.csv"
    
    print("Reading node and edge data...")
    
    # Read data
    nodes_a = read_nodes_csv(nodes_a_path)
    complete_nodes = read_nodes_csv(args.mapped_nodes)
    edges_a = read_edges_csv(edges_a_path)
    edges_b = read_edges_csv(edges_b_path)
    
    # Create lookup for DIR_A edges: (source, sink) -> edge_data
    edges_a_lookup = {}
    for edge in edges_a:
        key = (edge['Source'], edge['Sink'])
        edges_a_lookup[key] = edge
    
    print(f"Nodes in A: {len(nodes_a)}")
    print(f"Complete nodes (mapped): {len(complete_nodes)}")
    print(f"Edges in A: {len(edges_a)}")
    print(f"Edges in B: {len(edges_b)}")
    
    # Validate edge mapping
    print("Validating edge endpoints...")
    if not validate_edge_mapping(edges_b, complete_nodes):
        print("Validation failed. Exiting.")
        sys.exit(1)
    
    # Map edges
    print("Mapping edges...")
    mapped_edges = map_missing_edges(edges_b, nodes_a, complete_nodes, edges_a_lookup)
    
    # Write results
    print(f"Writing results to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Net', 'Source', 'Sink', 'Slack', 'ClockPeriod', 'ManhattanDistance', 'EstimationSource'])
        
        for edge in mapped_edges:
            # Format slack for output (handle infinity)
            slack_str = 'INFINITY' if edge['Slack'] == float('inf') else f"{edge['Slack']:.3f}"
            clock_period_str = f"{edge['ClockPeriod']:.3f}"
            
            manhattan_dist_str = f"{edge['ManhattanDistance']:.3f}"
            
            writer.writerow([
                edge['Net'],
                edge['Source'],
                edge['Sink'],
                slack_str,
                clock_period_str,
                manhattan_dist_str,
                edge['EstimationSource']
            ])
    
    print(f"Complete! Mapped {len(mapped_edges)} edges.")

if __name__ == "__main__":
    main()