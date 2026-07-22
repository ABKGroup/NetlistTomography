import csv
import sys
from collections import defaultdict, deque
import argparse

def read_nodes_csv(filepath):
    """Read nodes CSV and return dict of instance -> {coordinates, cell info}"""
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
                print(f"  Row data: {row}")
                if skipped_rows > 10:  # Limit debug output
                    print(f"  ... (stopping debug output after 10 errors)")
                    break
    
    if skipped_rows > 0:
        print(f"Total rows processed: {total_rows}, Successfully parsed: {len(nodes)}, Skipped: {skipped_rows}")
    
    return nodes

def read_edges_csv(filepath):
    """Read edges CSV and return forward and reverse graphs"""
    forward_graph = defaultdict(set)  # node -> set of fanout nodes
    reverse_graph = defaultdict(set)  # node -> set of fanin nodes
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row['Source']
            sink = row['Sink']
            forward_graph[source].add(sink)
            reverse_graph[sink].add(source)
    
    return forward_graph, reverse_graph

def smart_multi_source_bfs_with_timing(missing_nodes, forward_graph, reverse_graph, nodes_with_locations):
    """
    Enhanced BFS that collects both location and timing data with intelligent caching
    """
    fanin_matches = defaultdict(set)   # missing_node -> set of fanin nodes with locations
    fanout_matches = defaultdict(set)  # missing_node -> set of fanout nodes with locations
    
    # Cache for BFS results to avoid redundant traversals
    fanin_cache = {}   # node -> set of reachable fanin nodes with locations
    fanout_cache = {}  # node -> set of reachable fanout nodes with locations
    
    def cached_fanin_bfs(start_node):
        """BFS to find all reachable fanin nodes with locations, with caching"""
        if start_node in fanin_cache:
            return fanin_cache[start_node]
        
        reachable_fanins = set()
        queue = deque([start_node])
        visited = set([start_node])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in reverse_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    if neighbor in nodes_with_locations:
                        reachable_fanins.add(neighbor)
                    else:
                        queue.append(neighbor)
        
        fanin_cache[start_node] = reachable_fanins
        return reachable_fanins
    
    def cached_fanout_bfs(start_node):
        """BFS to find all reachable fanout nodes with locations, with caching"""
        if start_node in fanout_cache:
            return fanout_cache[start_node]
        
        reachable_fanouts = set()
        queue = deque([start_node])
        visited = set([start_node])
        
        while queue:
            current = queue.popleft()
            
            for neighbor in forward_graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    if neighbor in nodes_with_locations:
                        reachable_fanouts.add(neighbor)
                    else:
                        queue.append(neighbor)
        
        fanout_cache[start_node] = reachable_fanouts
        return reachable_fanouts
    
    # Process all missing nodes using cached BFS
    for missing_node in missing_nodes:
        fanin_matches[missing_node] = cached_fanin_bfs(missing_node)
        fanout_matches[missing_node] = cached_fanout_bfs(missing_node)
    
    return fanin_matches, fanout_matches

def calculate_locations_and_timing(missing_nodes, fanin_matches, fanout_matches, nodes_with_locations):
    """Calculate average locations and timing for missing nodes based on fanin/fanout matches"""
    estimated_data = {}
    
    for missing_node in missing_nodes:
        all_matches = fanin_matches[missing_node] | fanout_matches[missing_node]
        
        if not all_matches:
            print(f"Warning: No matches found for {missing_node}")
            estimated_data[missing_node] = {
                'pt_x': 0.0, 'pt_y': 0.0, 'match_count': 0,
                'slack': float('inf'), 'clock_period': 0.0,
                'timing_source': 'none'
            }
            continue
        
        # Calculate average location
        total_x = 0.0
        total_y = 0.0
        count = 0
        
        for match_node in all_matches:
            if match_node in nodes_with_locations:
                total_x += nodes_with_locations[match_node]['pt_x']
                total_y += nodes_with_locations[match_node]['pt_y']
                count += 1
        
        if count > 0:
            avg_x = total_x / count
            avg_y = total_y / count
        else:
            avg_x = avg_y = 0.0
        
        # Determine timing - prioritize fanins (worst slack), fallback to fanouts
        timing_slack = float('inf')
        timing_clock_period = 0.0
        timing_source = 'none'
        
        # First, check fanins for worst slack
        if fanin_matches[missing_node]:
            worst_slack = float('inf')
            selected_fanin = None
            
            for fanin_node in fanin_matches[missing_node]:
                if fanin_node in nodes_with_locations:
                    fanin_slack = nodes_with_locations[fanin_node]['slack']
                    if fanin_slack < worst_slack:  # Lower slack is worse
                        worst_slack = fanin_slack
                        selected_fanin = fanin_node
            
            if selected_fanin:
                timing_slack = nodes_with_locations[selected_fanin]['slack']
                timing_clock_period = nodes_with_locations[selected_fanin]['clock_period']
                timing_source = f'fanin:{selected_fanin}'
        
        # If no fanins or fanins have infinite slack, check fanouts
        if timing_slack == float('inf') and fanout_matches[missing_node]:
            worst_slack = float('inf')
            selected_fanout = None
            
            for fanout_node in fanout_matches[missing_node]:
                if fanout_node in nodes_with_locations:
                    fanout_slack = nodes_with_locations[fanout_node]['slack']
                    if fanout_slack < worst_slack:  # Lower slack is worse
                        worst_slack = fanout_slack
                        selected_fanout = fanout_node
            
            if selected_fanout:
                timing_slack = nodes_with_locations[selected_fanout]['slack']
                timing_clock_period = nodes_with_locations[selected_fanout]['clock_period']
                timing_source = f'fanout:{selected_fanout}'
        
        estimated_data[missing_node] = {
            'pt_x': avg_x, 
            'pt_y': avg_y, 
            'match_count': count,
            'fanin_matches': len(fanin_matches[missing_node]),
            'fanout_matches': len(fanout_matches[missing_node]),
            'slack': timing_slack,
            'clock_period': timing_clock_period,
            'timing_source': timing_source
        }
    
    return estimated_data

def main():
    parser = argparse.ArgumentParser(description='Generate location estimates for missing nodes using smart BFS')
    parser.add_argument('--dir_a', required=True, help='Directory A (with locations, e.g., design_place_opt)')
    parser.add_argument('--dir_b', required=True, help='Directory B (missing locations, e.g., design_post_synth)')  
    parser.add_argument('--design', required=True, help='Design name (e.g., ca53_cpu)')
    parser.add_argument('--output', default='estimated_locations.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # File paths
    nodes_a_path = f"{args.dir_a}/{args.design}_nodes.csv"
    edges_a_path = f"{args.dir_a}/{args.design}_edges.csv"
    nodes_b_path = f"{args.dir_b}/{args.design}_nodes.csv"
    edges_b_path = f"{args.dir_b}/{args.design}_edges.csv"
    
    print("Reading node and edge data...")
    
    # Read data
    nodes_a = read_nodes_csv(nodes_a_path)
    nodes_b = read_nodes_csv(nodes_b_path)
    forward_graph_b, reverse_graph_b = read_edges_csv(edges_b_path)
    
    print(f"Nodes in A: {len(nodes_a)}")
    print(f"Nodes in B: {len(nodes_b)}")
    
    # Find missing nodes (in B but not in A)
    missing_nodes = set()
    for node in nodes_b:
        if node not in nodes_a:
            missing_nodes.add(node)
    
    print(f"Missing nodes (in B but not in A): {len(missing_nodes)}")
    
    if not missing_nodes:
        print("No missing nodes found! Will output all nodes with original data.")
        estimated_data = {}
    else:
        # Use graph B for connectivity, nodes A for locations
        print("Running smart multi-source BFS with timing...")
        fanin_matches, fanout_matches = smart_multi_source_bfs_with_timing(
            missing_nodes, forward_graph_b, reverse_graph_b, nodes_a
        )
        
        print("Calculating locations and timing...")
        estimated_data = calculate_locations_and_timing(
            missing_nodes, fanin_matches, fanout_matches, nodes_a
        )
    
    # Write results - include ALL nodes from B with estimated/original data
    print(f"Writing results to {args.output}...")
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        # Use exact same header as node CSV files
        writer.writerow(['Instance', 'Cell', 'Slack', 'ClockPeriod', 'Width', 'Height', 'PT_X', 'PT_Y'])
        
        for node in nodes_b:
            if node in missing_nodes:
                # Use estimated data for missing nodes
                data = estimated_data[node]
                # Format slack for output (handle infinity)
                slack_str = 'INFINITY' if data['slack'] == float('inf') else f"{data['slack']:.3f}"
                
                writer.writerow([
                    node,
                    nodes_b[node]['cell'],
                    slack_str,
                    f"{data['clock_period']:.3f}",
                    nodes_b[node]['width'],
                    nodes_b[node]['height'],
                    f"{data['pt_x']:.3f}",
                    f"{data['pt_y']:.3f}"
                ])
            else:
                # Node exists in both A and B - prioritize data from A if available
                node_data_b = nodes_b[node]
                
                # Use data from DIR_A if available, otherwise use DIR_B
                if node in nodes_a:
                    node_data_a = nodes_a[node]
                    slack_str = 'INFINITY' if node_data_a['slack'] == float('inf') else f"{node_data_a['slack']:.3f}"
                    pt_x = node_data_a['pt_x']
                    pt_y = node_data_a['pt_y']
                    clock_period = node_data_a['clock_period']
                else:
                    slack_str = 'INFINITY' if node_data_b['slack'] == float('inf') else f"{node_data_b['slack']:.3f}"
                    pt_x = node_data_b['pt_x']
                    pt_y = node_data_b['pt_y']
                    clock_period = node_data_b['clock_period']
                
                writer.writerow([
                    node,
                    node_data_b['cell'],  # Always use cell from DIR_B (should be same anyway)
                    slack_str,
                    f"{clock_period:.3f}",
                    node_data_b['width'],   # Always use physical dimensions from DIR_B
                    node_data_b['height'],
                    f"{pt_x:.3f}",
                    f"{pt_y:.3f}"
                ])
    
    print(f"Complete! Estimated locations and timing for {len(missing_nodes)} missing nodes.")
    
    # Print some statistics
    successful_estimates = sum(1 for data in estimated_data.values() if data['match_count'] > 0)
    timing_from_fanins = sum(1 for data in estimated_data.values() if 'fanin:' in data['timing_source'])
    timing_from_fanouts = sum(1 for data in estimated_data.values() if 'fanout:' in data['timing_source'])
    
    print(f"Successful estimates: {successful_estimates}/{len(missing_nodes)}")
    print(f"Timing from fanins: {timing_from_fanins}, from fanouts: {timing_from_fanouts}")

if __name__ == "__main__":
    main()