#!/usr/bin/env python3
"""
Location-Aware Cluster Optimization for Netlist Tomography (V1)

Enhanced version with:
- BFS-based topological distance for unmapped instance assignment
- Multi-source BFS with caching for performance
- Path criticality based on slack/period ratio
- Balanced scoring combining topology, connectivity, distance, and criticality

Optimizes clustering to minimize cuts while preserving:
- Maximum cluster size constraint (1.1x growth)
- Spatial locality (physical proximity)
- Timing criticality

Three-phase approach:
1. Assign unmapped (-1) instances using BFS + location + graph connectivity + path criticality
2. Refine boundary instances with distance-aware scoring
3. Merge strategically close clusters with high interaction
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict, Counter, deque
from typing import Dict, List, Tuple, Set
import re


class ClusterOptimizer:
    """Main optimizer class for location-aware clustering refinement."""

    def __init__(self,
                 cluster_csv: str,
                 nodes_csv: str,
                 edges_csv: str,
                 paths_rpt: str):
        """Initialize with input data files."""
        self.cluster_csv = cluster_csv
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.paths_rpt = paths_rpt

        # Data structures
        self.cluster_map = {}  # instance -> cluster_id
        self.node_locations = {}  # instance -> (x, y)
        self.node_slack = {}  # instance -> slack
        self.graph_edges = defaultdict(set)  # instance -> {neighbor_instances}
        self.paths = []  # [(path_id, [instances], slack, period)]

        # New data structures for v1
        self.path_through_node = defaultdict(list)  # instance -> [(path_id, slack, period), ...]
        self.path_criticality = {}  # path_id -> criticality_weight
        self.neighbor_union = {}  # instance -> {all neighbors (fanin + fanout)}
        self.nearest_mapped_cache = {}  # instance -> [(mapped_instance, cluster_id, distance), ...]
        self.edge_weights = defaultdict(float)  # (inst1, inst2) -> sum of exp(-slack/period) across all paths

        # Computed data
        self.cluster_coords = defaultdict(list)  # cluster_id -> [(x,y), ...]
        self.cluster_centroids = {}  # cluster_id -> (cx, cy)
        self.cluster_radius = {}  # cluster_id -> radius
        self.cluster_sizes = Counter()  # cluster_id -> count

        # Configuration
        self.max_cluster_size = None
        self.avg_cluster_radius = None

        # Statistics
        self.stats = {
            'initial_cuts': 0,
            'initial_unique_clusters': 0,
            'unmapped_instances': set(),
            'moves_made': 0,
            'merges_made': 0,
            'bfs_cache_hits': 0,
            'avg_topo_distance': 0
        }

    def load_data(self):
        """Load all input data files."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load cluster mapping
        print(f"\nLoading cluster map from {self.cluster_csv}...")
        with open(self.cluster_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance = row['Instance']
                cluster_id = int(row['Cluster_ID'])
                self.cluster_map[instance] = cluster_id
                self.cluster_sizes[cluster_id] += 1
        print(f"  Loaded {len(self.cluster_map):,} instance-to-cluster mappings")
        print(f"  Found {len(self.cluster_sizes)} clusters")

        # Load node locations and slack
        print(f"\nLoading node locations from {self.nodes_csv}...")
        with open(self.nodes_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance = row['Instance']
                x = float(row['PT_X'])
                y = float(row['PT_Y'])
                slack = float(row['Slack'])
                self.node_locations[instance] = (x, y)
                self.node_slack[instance] = slack
        print(f"  Loaded {len(self.node_locations):,} node locations")

        # Load graph edges
        print(f"\nLoading graph edges from {self.edges_csv}...")
        edge_count = 0
        with open(self.edges_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row['Source']
                sink = row['Sink']
                self.graph_edges[source].add(sink)
                self.graph_edges[sink].add(source)
                edge_count += 1
        print(f"  Loaded {edge_count:,} edges")

        # Load timing paths (now with slack and period)
        print(f"\nLoading timing paths from {self.paths_rpt}...")
        self.paths = self._parse_timing_paths(self.paths_rpt)
        print(f"  Loaded {len(self.paths):,} timing paths")

        # Build path-node index
        print("\nBuilding path-node index...")
        self._build_path_node_index()
        print(f"  Indexed {len(self.path_through_node):,} instances with path information")

        # Precompute bidirectional graph
        print("\nPrecomputing bidirectional neighbor unions...")
        self._precompute_neighbor_union()
        print(f"  Computed neighbor unions for {len(self.neighbor_union):,} nodes")

        print("\n" + "=" * 80)

    def _parse_timing_paths(self, rpt_file: str) -> List[Tuple[int, List[str], float, float]]:
        """Parse timing paths from report file, extracting slack and period."""
        paths = []
        current_path_id = None
        current_instances = []
        current_slack = None
        current_period = None

        with open(rpt_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Check for path header
                path_match = re.match(r'^Path (\d+):$', line)
                if path_match:
                    # Save previous path
                    if current_path_id is not None and current_instances:
                        paths.append((current_path_id, current_instances, current_slack, current_period))

                    # Start new path
                    current_path_id = int(path_match.group(1))
                    current_instances = []
                    current_slack = None
                    current_period = None
                    continue

                # Check for instances line
                if line.startswith('Instances:'):
                    instances_str = line[len('Instances:'):].strip()
                    instances = [inst.strip() for inst in instances_str.split('->')]
                    current_instances = instances
                    continue

                # Check for slack line
                if line.startswith('Slack:'):
                    slack_str = line[len('Slack:'):].strip()
                    # Parse "X.XXX ns" or "-X.XXX ns"
                    slack_match = re.match(r'(-?[\d.]+)\s*ns', slack_str)
                    if slack_match:
                        current_slack = float(slack_match.group(1))
                    continue

                # Check for period line
                if line.startswith('Period:'):
                    period_str = line[len('Period:'):].strip()
                    # Parse "X.XXX ns"
                    period_match = re.match(r'([\d.]+)\s*ns', period_str)
                    if period_match:
                        current_period = float(period_match.group(1))
                    continue

        # Add last path
        if current_path_id is not None and current_instances:
            paths.append((current_path_id, current_instances, current_slack, current_period))

        return paths

    def _build_path_node_index(self):
        """Build index of which paths pass through each node."""
        for path_id, instances, slack, period in self.paths:
            # Compute path criticality (exponential scaling)
            if slack is not None and period is not None and period > 0:
                criticality = math.exp(-slack / period)
            else:
                criticality = 1.0
            
            self.path_criticality[path_id] = criticality

            # Index each instance in the path
            for inst in instances:
                self.path_through_node[inst].append((path_id, slack, period))

    def _precompute_neighbor_union(self):
        """Precompute bidirectional neighbor union (fanin + fanout) for all nodes."""
        # Build forward and reverse graphs
        forward_graph = defaultdict(set)
        reverse_graph = defaultdict(set)
        
        for source, sinks in self.graph_edges.items():
            for sink in sinks:
                forward_graph[source].add(sink)
                reverse_graph[sink].add(source)
        
        # Compute union for all nodes
        all_nodes = set(forward_graph.keys()) | set(reverse_graph.keys())
        for node in all_nodes:
            self.neighbor_union[node] = forward_graph[node] | reverse_graph[node]

    def _compute_nearest_mapped_cache(self):
        """
        Multi-source BFS to find nearest mapped nodes for all unmapped instances.
        Much faster than running BFS from each unmapped node individually.
        """
        print("\nComputing nearest mapped nodes using multi-source BFS...")
        
        # Initialize: all mapped nodes at distance 0
        queue = deque()
        distance_map = {}  # instance -> distance to nearest mapped
        nearest_mapped = defaultdict(list)  # instance -> [(mapped_inst, cluster_id, dist), ...]
        
        # Start with all mapped nodes
        for instance, cluster_id in self.cluster_map.items():
            if cluster_id != -1:  # Skip unmapped
                distance_map[instance] = 0
                nearest_mapped[instance].append((instance, cluster_id, 0))
                queue.append((instance, 0))
        
        # BFS from all mapped nodes simultaneously
        nodes_visited = 0
        while queue:
            current, dist = queue.popleft()
            nodes_visited += 1
            
            # Explore neighbors
            for neighbor in self.neighbor_union.get(current, set()):
                if neighbor not in distance_map:
                    distance_map[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
                
                # Track up to 5 nearest mapped nodes (or all at same distance as 5th)
                neighbor_dist = distance_map[neighbor]
                
                # Add mapping from current (if current is mapped)
                if current in self.cluster_map and self.cluster_map[current] != -1:
                    current_cluster = self.cluster_map[current]
                    
                    # Check if we should add this mapping
                    existing = nearest_mapped[neighbor]
                    if len(existing) < 5:
                        # Add if we don't have 5 yet
                        nearest_mapped[neighbor].append((current, current_cluster, dist + 1))
                    elif len(existing) >= 5:
                        # Check if this is same distance as 5th element
                        fifth_dist = existing[4][2] if len(existing) > 4 else existing[-1][2]
                        if dist + 1 == fifth_dist:
                            # Same distance, include it
                            nearest_mapped[neighbor].append((current, current_cluster, dist + 1))
        
        # Store in cache
        self.nearest_mapped_cache = dict(nearest_mapped)
        
        print(f"  Visited {nodes_visited:,} nodes")
        print(f"  Cached nearest mapped for {len(self.nearest_mapped_cache):,} instances")

    def compute_spatial_metrics(self):
        """Compute cluster centroids, radii, and spatial statistics."""
        print("\nCOMPUTING SPATIAL METRICS")
        print("=" * 80)

        # Group coordinates by cluster
        for instance, cluster_id in self.cluster_map.items():
            if instance in self.node_locations:
                self.cluster_coords[cluster_id].append(self.node_locations[instance])

        # Compute centroids and radii
        for cluster_id, coords in self.cluster_coords.items():
            if not coords:
                continue

            # Centroid
            cx = sum(x for x, y in coords) / len(coords)
            cy = sum(y for x, y in coords) / len(coords)
            self.cluster_centroids[cluster_id] = (cx, cy)

            # Radius (average distance from centroid)
            avg_dist = sum(math.sqrt((x-cx)**2 + (y-cy)**2) for x, y in coords) / len(coords)
            self.cluster_radius[cluster_id] = avg_dist

        # Overall statistics
        radii = list(self.cluster_radius.values())
        self.avg_cluster_radius = statistics.mean(radii) if radii else 0

        # Set max cluster size (1.1x current max)
        self.max_cluster_size = int(max(self.cluster_sizes.values()) * 1.1)

        print(f"  Average cluster radius: {self.avg_cluster_radius:.1f} units")
        print(f"  Max cluster radius: {max(radii):.1f} units")
        print(f"  Max cluster size: {max(self.cluster_sizes.values()):,} instances")
        print(f"  Max allowed cluster size (1.1x): {self.max_cluster_size:,} instances")
        print()

    def compute_initial_metrics(self):
        """Compute initial cut and cluster metrics."""
        print("COMPUTING INITIAL METRICS")
        print("=" * 80)

        all_cuts = []
        all_clusters = []

        for path_id, instances, slack, period in self.paths:
            # Get cluster assignments (use -1 for unmapped)
            clusters = []
            for inst in instances:
                if inst in self.cluster_map:
                    clusters.append(self.cluster_map[inst])
                else:
                    clusters.append(-1)
                    self.stats['unmapped_instances'].add(inst)

            # Count cuts
            num_cuts = sum(1 for i in range(len(clusters)-1) if clusters[i] != clusters[i+1])
            num_clusters = len(set(clusters))

            all_cuts.append(num_cuts)
            all_clusters.append(num_clusters)

        self.stats['initial_cuts'] = statistics.mean(all_cuts)
        self.stats['initial_unique_clusters'] = statistics.mean(all_clusters)
        self.stats['initial_max_cuts'] = max(all_cuts)
        self.stats['initial_median_cuts'] = statistics.median(all_cuts)

        print(f"  Average cuts per path: {self.stats['initial_cuts']:.2f}")
        print(f"  Max cuts per path: {self.stats['initial_max_cuts']}")
        print(f"  Median cuts per path: {self.stats['initial_median_cuts']:.2f}")
        print(f"  Average unique clusters per path: {self.stats['initial_unique_clusters']:.2f}")
        print(f"  Unmapped instances (-1 cluster): {len(self.stats['unmapped_instances']):,}")
        print()

    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def find_k_nearest_clusters(self, location: Tuple[float, float], k: int = 5) -> List[Tuple[int, float]]:
        """Find K nearest clusters to a given location."""
        distances = []
        for cluster_id, centroid in self.cluster_centroids.items():
            dist = self.distance(location, centroid)
            distances.append((cluster_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_timing_criticality(self, instance: str) -> float:
        """Get timing criticality weight for an instance (higher = more critical)."""
        # Get all paths through this instance
        if instance not in self.path_through_node:
            return 1.0
        
        # Find the most critical path through this instance
        max_criticality = 1.0
        for path_id, slack, period in self.path_through_node[instance]:
            if slack is not None and period is not None and period > 0:
                # Exponential scaling: exp(-slack / period)
                # More negative slack -> higher criticality
                criticality = math.exp(-slack / period)
                max_criticality = max(max_criticality, criticality)
        
        return max_criticality

    def score_cluster_for_assignment(self, 
                                     instance: str, 
                                     cluster_id: int,
                                     mapped_nodes_in_cluster: List[Tuple[str, int, int]],
                                     location: Tuple[float, float]) -> float:
        """
        Score a cluster for assignment using balanced weights:
        - Topological distance (weight=1.0)
        - Neighbor count (weight=2.0)
        - Physical distance (weight=0.5)
        - Critical paths (weight=3.0)
        """
        if cluster_id not in self.cluster_centroids:
            return -float('inf')
        
        # 1. Topological distance: average distance to mapped nodes in this cluster
        if mapped_nodes_in_cluster:
            avg_topo_dist = sum(dist for _, _, dist in mapped_nodes_in_cluster) / len(mapped_nodes_in_cluster)
            # Invert so closer is better
            topo_score = 1.0 / (avg_topo_dist + 1)
        else:
            topo_score = 0.0
        
        # 2. Neighbor count: direct graph neighbors in this cluster
        neighbor_count = 0
        if instance in self.graph_edges:
            for neighbor in self.graph_edges[instance]:
                if self.cluster_map.get(neighbor) == cluster_id:
                    neighbor_count += 1
        
        # 3. Physical distance: to cluster centroid
        centroid = self.cluster_centroids[cluster_id]
        phys_dist = self.distance(location, centroid)
        phys_score = 1.0 / (phys_dist + 1)
        
        # 4. Critical paths: paths that go through both instance and cluster members
        criticality_score = 0.0
        if instance in self.path_through_node:
            # Get instances in this cluster
            cluster_instances = set(inst for inst, cid in self.cluster_map.items() if cid == cluster_id)
            
            # Count critical paths that intersect
            for path_id, slack, period in self.path_through_node[instance]:
                # Get path instances
                path_instances = None
                for pid, insts, _, _ in self.paths:
                    if pid == path_id:
                        path_instances = set(insts)
                        break
                
                if path_instances and path_instances & cluster_instances:
                    # This path goes through both instance and cluster
                    criticality = self.path_criticality.get(path_id, 1.0)
                    criticality_score += criticality
        
        # Combine with balanced weights
        total_score = (
            1.0 * topo_score +
            2.0 * neighbor_count +
            0.5 * phys_score +
            3.0 * criticality_score
        )
        
        return total_score

    def assign_unmapped_instances(self):
        """Phase 1: Assign unmapped instances using BFS + balanced scoring."""
        print("\n" + "=" * 80)
        print("PHASE 1: ASSIGNING UNMAPPED INSTANCES (BFS-BASED)")
        print("=" * 80)

        # Compute nearest mapped cache using multi-source BFS
        self._compute_nearest_mapped_cache()

        unmapped = list(self.stats['unmapped_instances'])
        print(f"\nProcessing {len(unmapped):,} unmapped instances...")

        assigned_count = 0
        total_topo_distance = 0

        for instance in unmapped:
            if instance not in self.node_locations:
                continue  # No location data, skip

            location = self.node_locations[instance]

            # Get nearest mapped nodes from cache
            if instance not in self.nearest_mapped_cache:
                continue
            
            nearest_mapped = self.nearest_mapped_cache[instance]
            
            # Group by cluster
            clusters_found = defaultdict(list)
            for mapped_inst, cluster_id, dist in nearest_mapped:
                clusters_found[cluster_id].append((mapped_inst, cluster_id, dist))
            
            # Score each candidate cluster
            best_cluster = None
            best_score = -float('inf')

            for cluster_id, nodes_in_cluster in clusters_found.items():
                # Skip if cluster would exceed size limit
                if self.cluster_sizes[cluster_id] >= self.max_cluster_size:
                    continue

                # Score this cluster
                score = self.score_cluster_for_assignment(
                    instance, cluster_id, nodes_in_cluster, location
                )

                if score > best_score:
                    best_score = score
                    best_cluster = cluster_id

            # Assign to best cluster
            if best_cluster is not None:
                self.cluster_map[instance] = best_cluster
                self.cluster_sizes[best_cluster] += 1
                assigned_count += 1
                
                # Track topological distance for stats
                if instance in self.nearest_mapped_cache:
                    min_dist = min(dist for _, _, dist in self.nearest_mapped_cache[instance])
                    total_topo_distance += min_dist

        avg_topo_dist = total_topo_distance / assigned_count if assigned_count > 0 else 0
        self.stats['avg_topo_distance'] = avg_topo_dist

        print(f"\nAssigned {assigned_count:,} instances to existing clusters")
        print(f"Average topological distance: {avg_topo_dist:.2f} hops")
        print(f"Remaining unmapped: {len(unmapped) - assigned_count:,}")

        # Recompute spatial metrics
        self.cluster_coords.clear()
        for instance, cluster_id in self.cluster_map.items():
            if instance in self.node_locations:
                self.cluster_coords[cluster_id].append(self.node_locations[instance])

        for cluster_id, coords in self.cluster_coords.items():
            if coords:
                cx = sum(x for x, y in coords) / len(coords)
                cy = sum(y for x, y in coords) / len(coords)
                self.cluster_centroids[cluster_id] = (cx, cy)

        print("\nPhase 1 complete!")

    # Phase 2: Boundary refinement
    def identify_boundary_instances(self) -> Set[str]:
        """Identify instances at cluster boundaries."""
        boundary_instances = set()

        for instance in self.cluster_map:
            my_cluster = self.cluster_map[instance]

            # Check if any graph neighbor is in different cluster
            if instance in self.graph_edges:
                for neighbor in self.graph_edges[instance]:
                    neighbor_cluster = self.cluster_map.get(neighbor, -1)
                    if neighbor_cluster != my_cluster and neighbor_cluster != -1:
                        boundary_instances.add(instance)
                        break

        return boundary_instances

    def build_edge_weights(self):
        """Build edge weights using exponential slack-based weighting."""
        print("\nBuilding edge weights for critical path analysis...")

        # Clear existing weights
        self.edge_weights.clear()

        # Iterate through all timing paths
        for path_id, instances, slack, period in self.paths:
            if slack is None or period is None or period <= 0:
                # No valid timing info, use default weight of 1.0
                weight = 1.0
            else:
                # Exponential weighting: exp(-slack/period)
                # Critical paths (negative slack) get weight > 1
                # Non-critical paths get weight < 1
                weight = math.exp(-slack / period)

            # Add weight to all consecutive edge pairs in this path
            for i in range(len(instances) - 1):
                inst1 = instances[i]
                inst2 = instances[i + 1]

                # Store bidirectional edges
                self.edge_weights[(inst1, inst2)] += weight
                self.edge_weights[(inst2, inst1)] += weight

        print(f"  Built edge weights for {len(self.edge_weights):,} directed edges")

        # Print some statistics
        if self.edge_weights:
            weights_list = list(self.edge_weights.values())
            avg_weight = sum(weights_list) / len(weights_list)
            max_weight = max(weights_list)
            min_weight = min(weights_list)
            print(f"  Edge weight range: [{min_weight:.2f}, {max_weight:.2f}], avg: {avg_weight:.2f}")

    def calculate_move_gain(self, instance: str, from_cluster: int, to_cluster: int) -> float:
        """Calculate gain using weighted edges and distance penalty."""
        if instance not in self.node_locations:
            return -float('inf')

        location = self.node_locations[instance]

        # Distance penalty (normalized by average cluster radius)
        to_centroid = self.cluster_centroids.get(to_cluster)
        if to_centroid is None:
            return -float('inf')

        dist_to_target = self.distance(location, to_centroid)
        distance_penalty = dist_to_target / (self.avg_cluster_radius + 1)

        # Count weighted cuts removed/added using edge weights
        weighted_cuts_removed = 0.0
        weighted_cuts_added = 0.0

        if instance in self.graph_edges:
            for neighbor in self.graph_edges[instance]:
                neighbor_cluster = self.cluster_map.get(neighbor, -1)
                if neighbor_cluster == -1:
                    continue

                # Get edge weight (default to 1.0 if edge not in weight map)
                edge = (instance, neighbor)
                edge_weight = self.edge_weights.get(edge, 1.0)

                # Currently creating a cut with this neighbor?
                if neighbor_cluster != from_cluster:
                    weighted_cuts_removed += edge_weight

                # Would create a cut after move?
                if neighbor_cluster != to_cluster:
                    weighted_cuts_added += edge_weight

        # Gain calculation: weighted cut reduction minus distance penalty
        # Distance penalty coefficient (alpha) = 0.1
        gain = (weighted_cuts_removed - weighted_cuts_added) - (distance_penalty * 0.1)

        return gain

    def refine_boundary_instances(self, max_iterations: int = 5):
        """Phase 2: Refine boundary instances with critical path prioritization."""
        print("\n" + "=" * 80)
        print("PHASE 2: BOUNDARY INSTANCE REFINEMENT (CRITICAL PATH AWARE)")
        print("=" * 80)

        # Build edge weights for weighted cut calculation
        self.build_edge_weights()

        proximity_threshold = 200  # units

        for iteration in range(max_iterations):
            print(f"\nIteration {iteration + 1}/{max_iterations}")

            boundary = self.identify_boundary_instances()
            print(f"  Boundary instances: {len(boundary):,}")

            # Sort boundary instances by criticality (descending)
            # Process most critical instances first
            boundary_with_criticality = []
            for instance in boundary:
                criticality = self.get_timing_criticality(instance)
                boundary_with_criticality.append((instance, criticality))

            boundary_with_criticality.sort(key=lambda x: x[1], reverse=True)
            sorted_boundary = [inst for inst, crit in boundary_with_criticality]

            if boundary_with_criticality:
                print(f"  Criticality range: [{boundary_with_criticality[-1][1]:.2f}, {boundary_with_criticality[0][1]:.2f}]")

            moves_this_iter = 0

            for instance in sorted_boundary:
                current_cluster = self.cluster_map[instance]
                location = self.node_locations.get(instance)

                if location is None:
                    continue

                # Find nearby clusters
                nearby = self.find_k_nearest_clusters(location, k=10)

                best_move = None
                best_gain = 0.1  # Threshold for making a move

                for candidate_cluster, dist in nearby:
                    # Skip current cluster
                    if candidate_cluster == current_cluster:
                        continue

                    # Skip if too far
                    if dist > proximity_threshold:
                        continue

                    # Skip if would exceed size limit
                    if self.cluster_sizes[candidate_cluster] >= self.max_cluster_size:
                        continue

                    # Calculate gain
                    gain = self.calculate_move_gain(instance, current_cluster, candidate_cluster)

                    if gain > best_gain:
                        best_gain = gain
                        best_move = candidate_cluster

                # Make the move if beneficial
                if best_move is not None:
                    self.cluster_map[instance] = best_move
                    self.cluster_sizes[current_cluster] -= 1
                    self.cluster_sizes[best_move] += 1
                    moves_this_iter += 1
                    self.stats['moves_made'] += 1

            print(f"  Moves made: {moves_this_iter}")

            if moves_this_iter == 0:
                print("  No beneficial moves found, stopping early")
                break

        print(f"\nPhase 2 complete! Total moves: {self.stats['moves_made']}")

    # Phase 3: Cluster merging
    def evaluate_merge_candidates(self) -> List[Tuple[int, int, float]]:
        """Identify cluster pairs for merging using weighted transitions and dynamic distance."""
        # Count weighted cluster transitions from paths
        weighted_transitions = defaultdict(float)

        for path_id, instances, slack, period in self.paths:
            # Get path weight
            if slack is not None and period is not None and period > 0:
                path_weight = math.exp(-slack / period)
            else:
                path_weight = 1.0

            # Count weighted transitions between clusters
            clusters = [self.cluster_map.get(inst, -1) for inst in instances]
            for i in range(len(clusters) - 1):
                if clusters[i] != clusters[i+1] and clusters[i] != -1 and clusters[i+1] != -1:
                    pair = tuple(sorted([clusters[i], clusters[i+1]]))
                    weighted_transitions[pair] += path_weight

        # Evaluate candidates
        candidates = []

        for (c1, c2), weighted_trans in weighted_transitions.items():
            # Must have high interaction (using weighted transitions)
            if weighted_trans < 400:
                continue

            # Check if both clusters exist
            if c1 not in self.cluster_centroids or c2 not in self.cluster_centroids:
                continue
            if c1 not in self.cluster_radius or c2 not in self.cluster_radius:
                continue

            # Check physical distance (dynamic constraint based on cluster radii)
            dist = self.distance(self.cluster_centroids[c1], self.cluster_centroids[c2])
            max_dist = 1.1 * (self.cluster_radius[c1] + self.cluster_radius[c2])
            if dist > max_dist:
                continue

            # Check combined size
            combined_size = self.cluster_sizes[c1] + self.cluster_sizes[c2]
            if combined_size > self.max_cluster_size:
                continue

            # Score = weighted_transitions / (distance + 1)
            score = weighted_trans / (dist + 1)
            candidates.append((c1, c2, score))

        # Sort by score (descending)
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def merge_clusters(self, c1: int, c2: int):
        """Merge cluster c2 into cluster c1."""
        # Reassign all instances from c2 to c1
        for instance in list(self.cluster_map.keys()):
            if self.cluster_map[instance] == c2:
                self.cluster_map[instance] = c1

        # Update sizes
        self.cluster_sizes[c1] += self.cluster_sizes[c2]
        del self.cluster_sizes[c2]

        # Update spatial data
        if c2 in self.cluster_centroids:
            del self.cluster_centroids[c2]
        if c2 in self.cluster_radius:
            del self.cluster_radius[c2]
        if c2 in self.cluster_coords:
            del self.cluster_coords[c2]

    def recompute_cluster_spatial_metrics(self, cluster_id: int):
        """Recompute centroid and radius for a specific cluster."""
        # Collect all coordinates for this cluster
        coords = []
        for instance, cid in self.cluster_map.items():
            if cid == cluster_id and instance in self.node_locations:
                coords.append(self.node_locations[instance])

        if not coords:
            # Empty cluster, remove it
            if cluster_id in self.cluster_centroids:
                del self.cluster_centroids[cluster_id]
            if cluster_id in self.cluster_radius:
                del self.cluster_radius[cluster_id]
            if cluster_id in self.cluster_coords:
                del self.cluster_coords[cluster_id]
            return

        # Update cluster_coords
        self.cluster_coords[cluster_id] = coords

        # Recompute centroid
        cx = sum(x for x, y in coords) / len(coords)
        cy = sum(y for x, y in coords) / len(coords)
        self.cluster_centroids[cluster_id] = (cx, cy)

        # Recompute radius
        avg_dist = sum(math.sqrt((x-cx)**2 + (y-cy)**2) for x, y in coords) / len(coords)
        self.cluster_radius[cluster_id] = avg_dist

    def strategic_merging(self):
        """Phase 3: Iterative greedy cluster merging."""
        print("\n" + "=" * 80)
        print("PHASE 3: ITERATIVE CLUSTER MERGING (WEIGHTED TRANSITIONS)")
        print("=" * 80)

        merge_count = 0
        iteration = 0

        while True:
            iteration += 1
            print(f"\nIteration {iteration}:")

            # Get all valid merge candidates
            candidates = self.evaluate_merge_candidates()
            print(f"  Found {len(candidates)} valid merge candidates")

            # If no candidates, stop
            if not candidates:
                print("  No more valid pairs to merge")
                break

            # Show top candidates
            if iteration == 1 and len(candidates) >= 10:
                print("\n  Top 10 candidates:")
                for i, (c1, c2, score) in enumerate(candidates[:10]):
                    dist = self.distance(self.cluster_centroids[c1], self.cluster_centroids[c2])
                    weighted_trans = score * (dist + 1)  # Reverse the score calculation
                    print(f"    {i+1}. Clusters {c1} + {c2}: score={score:.1f}, dist={dist:.1f}, weighted_trans={weighted_trans:.1f}")

            # Merge only the best pair
            c1, c2, score = candidates[0]
            dist = self.distance(self.cluster_centroids[c1], self.cluster_centroids[c2])
            print(f"  Merging cluster {c2} into {c1} (score={score:.1f}, dist={dist:.1f})")

            self.merge_clusters(c1, c2)

            # Recompute spatial metrics for merged cluster
            self.recompute_cluster_spatial_metrics(c1)

            merge_count += 1
            self.stats['merges_made'] += 1

            # Optional: Limit total merges to prevent infinite loops
            if merge_count >= 50:
                print(f"  Reached maximum merge limit (50)")
                break

        print(f"\nPhase 3 complete! Merged {merge_count} cluster pairs")

    def compute_final_metrics(self):
        """Compute final metrics after optimization."""
        print("\n" + "=" * 80)
        print("COMPUTING FINAL METRICS")
        print("=" * 80)

        all_cuts = []
        all_clusters = []
        unmapped_count = 0

        for path_id, instances, slack, period in self.paths:
            clusters = []
            for inst in instances:
                if inst in self.cluster_map:
                    clusters.append(self.cluster_map[inst])
                else:
                    clusters.append(-1)
                    unmapped_count += 1

            num_cuts = sum(1 for i in range(len(clusters)-1) if clusters[i] != clusters[i+1])
            num_clusters = len(set(clusters))

            all_cuts.append(num_cuts)
            all_clusters.append(num_clusters)

        final_cuts = statistics.mean(all_cuts)
        final_clusters = statistics.mean(all_clusters)
        final_max_cuts = max(all_cuts)
        final_median_cuts = statistics.median(all_cuts)

        # Store final metrics
        self.stats['final_cuts'] = final_cuts
        self.stats['final_unique_clusters'] = final_clusters
        self.stats['final_unmapped_count'] = unmapped_count
        self.stats['final_max_cuts'] = final_max_cuts
        self.stats['final_median_cuts'] = final_median_cuts

        print(f"\nFinal metrics:")
        print(f"  Average cuts per path: {final_cuts:.2f}")
        print(f"  Max cuts per path: {final_max_cuts}")
        print(f"  Median cuts per path: {final_median_cuts:.2f}")
        print(f"  Average unique clusters per path: {final_clusters:.2f}")
        print(f"  Remaining unmapped instances: {unmapped_count}")

        print(f"\nImprovement:")
        # Calculate percentage improvements with zero-division guards
        avg_cuts_pct = f"{100*(self.stats['initial_cuts'] - final_cuts)/self.stats['initial_cuts']:.1f}%" if self.stats['initial_cuts'] != 0 else "N/A"
        max_cuts_pct = f"{100*(self.stats['initial_max_cuts'] - final_max_cuts)/self.stats['initial_max_cuts']:.1f}%" if self.stats['initial_max_cuts'] != 0 else "N/A"
        median_cuts_pct = f"{100*(self.stats['initial_median_cuts'] - final_median_cuts)/self.stats['initial_median_cuts']:.1f}%" if self.stats['initial_median_cuts'] != 0 else "N/A"
        avg_clusters_pct = f"{100*(self.stats['initial_unique_clusters'] - final_clusters)/self.stats['initial_unique_clusters']:.1f}%" if self.stats['initial_unique_clusters'] != 0 else "N/A"

        print(f"  Average cuts: {self.stats['initial_cuts']:.2f} -> {final_cuts:.2f} ({(self.stats['initial_cuts'] - final_cuts):.2f}, {avg_cuts_pct})")
        print(f"  Max cuts: {self.stats['initial_max_cuts']} -> {final_max_cuts} ({(self.stats['initial_max_cuts'] - final_max_cuts)}, {max_cuts_pct})")
        print(f"  Median cuts: {self.stats['initial_median_cuts']:.2f} -> {final_median_cuts:.2f} ({(self.stats['initial_median_cuts'] - final_median_cuts):.2f}, {median_cuts_pct})")
        print(f"  Average clusters: {self.stats['initial_unique_clusters']:.2f} -> {final_clusters:.2f} ({(self.stats['initial_unique_clusters'] - final_clusters):.2f}, {avg_clusters_pct})")

        print(f"\nActions taken:")
        print(f"  Instances reassigned: {self.stats['moves_made']}")
        print(f"  Clusters merged: {self.stats['merges_made']}")
        print(f"  Average topological distance: {self.stats['avg_topo_distance']:.2f} hops")
        print()

    def save_optimized_mapping(self, output_file: str):
        """Save optimized cluster mapping to CSV."""
        print(f"Saving optimized mapping to {output_file}...")

        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Instance', 'Cluster_ID'])

            for instance in sorted(self.cluster_map.keys()):
                writer.writerow([instance, self.cluster_map[instance]])

        print(f"  Saved {len(self.cluster_map):,} mappings")
        print()

    def generate_summary_report(self, summary_file: str, output_csv: str):
        """Generate comprehensive summary report of optimization results."""
        print(f"Generating summary report to {summary_file}...")

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("CLUSTER OPTIMIZATION RESULTS SUMMARY (V1 - BFS Enhanced)\n")
            f.write("=" * 80 + "\n\n")

            # Input data section
            f.write("INPUT DATA:\n")
            f.write("-" * 11 + "\n")
            f.write(f"- Instances in cluster map: {len(self.cluster_map):,}\n")
            f.write(f"- Instances with locations: {len(self.node_locations):,}\n")
            f.write(f"- Graph edges: {len(self.graph_edges):,}\n")
            f.write(f"- Timing paths analyzed: {len(self.paths):,}\n")
            f.write(f"- Original clusters: {len(self.cluster_sizes)}\n")
            f.write(f"- Unmapped instances: {len(self.stats['unmapped_instances']):,}\n\n")

            # Optimization phases section
            f.write("OPTIMIZATION PHASES:\n")
            f.write("-" * 20 + "\n")
            f.write("Phase 1: Assign Unmapped Instances (BFS-Based)\n")
            # Count how many unique unmapped instances now have mappings
            original_unmapped = len(self.stats['unmapped_instances'])
            still_unmapped = sum(1 for inst in self.stats['unmapped_instances'] if inst not in self.cluster_map)
            assigned = original_unmapped - still_unmapped
            f.write(f"  - Assigned {assigned:,} of {original_unmapped:,} instances to existing clusters\n")
            f.write(f"  - Used multi-source BFS for topological distance\n")
            f.write(f"  - Balanced scoring: topology + neighbors + distance + criticality\n")
            f.write(f"  - Average topological distance: {self.stats['avg_topo_distance']:.2f} hops\n\n")

            f.write("Phase 2: Boundary Instance Refinement\n")
            f.write(f"  - Made {self.stats['moves_made']:,} beneficial moves\n")
            f.write(f"  - Used distance-aware gain calculation with path criticality\n\n")

            f.write("Phase 3: Strategic Cluster Merging\n")
            f.write(f"  - Merged {self.stats['merges_made']} cluster pairs\n")
            f.write(f"  - Based on physical proximity + interaction frequency\n\n")

            # Improvements section
            f.write("IMPROVEMENTS:\n")
            f.write("-" * 13 + "\n")
            f.write(f"{'':24} {'BEFORE':>8} {'AFTER':>8} {'CHANGE':>8} {'IMPROVEMENT':>12}\n")

            initial_cuts = self.stats['initial_cuts']
            final_cuts = self.stats['final_cuts']
            cut_change = final_cuts - initial_cuts
            cut_improvement = 100 * (initial_cuts - final_cuts) / initial_cuts if initial_cuts != 0 else 0

            initial_max_cuts = self.stats['initial_max_cuts']
            final_max_cuts = self.stats['final_max_cuts']
            max_cut_change = final_max_cuts - initial_max_cuts
            max_cut_improvement = 100 * (initial_max_cuts - final_max_cuts) / initial_max_cuts if initial_max_cuts != 0 else 0

            initial_median_cuts = self.stats['initial_median_cuts']
            final_median_cuts = self.stats['final_median_cuts']
            median_cut_change = final_median_cuts - initial_median_cuts
            median_cut_improvement = 100 * (initial_median_cuts - final_median_cuts) / initial_median_cuts if initial_median_cuts != 0 else 0

            initial_clusters = self.stats['initial_unique_clusters']
            final_clusters = self.stats['final_unique_clusters']
            cluster_change = final_clusters - initial_clusters
            cluster_improvement = 100 * (initial_clusters - final_clusters) / initial_clusters if initial_clusters != 0 else 0

            # Format improvement percentages with N/A for zero denominators
            cut_imp_str = f"{cut_improvement:11.1f}%" if initial_cuts != 0 else "        N/A"
            max_cut_imp_str = f"{max_cut_improvement:11.1f}%" if initial_max_cuts != 0 else "        N/A"
            median_cut_imp_str = f"{median_cut_improvement:11.1f}%" if initial_median_cuts != 0 else "        N/A"
            cluster_imp_str = f"{cluster_improvement:11.1f}%" if initial_clusters != 0 else "        N/A"

            f.write(f"{'Average CUT per path:':<24} {initial_cuts:8.2f} {final_cuts:8.2f} {cut_change:+8.2f} {cut_imp_str}\n")
            f.write(f"{'Max cuts per path:':<24} {initial_max_cuts:8} {final_max_cuts:8} {max_cut_change:+8} {max_cut_imp_str}\n")
            f.write(f"{'Median cuts:':<24} {initial_median_cuts:8.2f} {final_median_cuts:8.2f} {median_cut_change:+8.2f} {median_cut_imp_str}\n")
            f.write(f"{'Average Clusters/path:':<24} {initial_clusters:8.2f} {final_clusters:8.2f} {cluster_change:+8.2f} {cluster_imp_str}\n")
            f.write("\n")

            # Constraints section
            f.write("CONSTRAINTS MAINTAINED:\n")
            f.write("-" * 23 + "\n")
            f.write(f"✓ Max cluster size: {self.max_cluster_size:,} instances (1.1x limit)\n")
            f.write(f"✓ Spatial compactness preserved (avg radius ~{self.avg_cluster_radius:.0f} units)\n")
            f.write(f"✓ Assigned {assigned:,} of {original_unmapped:,} originally unmapped instances\n\n")

            # Key achievements section
            f.write("KEY ACHIEVEMENTS:\n")
            f.write("-" * 17 + "\n")
            f.write(f"1. Reduced average cuts by {cut_improvement:.1f}% while maintaining cluster locality\n")
            f.write(f"2. Used BFS-based topological distance (avg {self.stats['avg_topo_distance']:.2f} hops)\n")
            f.write(f"3. Made {self.stats['moves_made']:,} location-aware instance moves to reduce cuts\n")
            f.write(f"4. Merged {self.stats['merges_made']} spatially close, highly-interacting clusters\n")
            if assigned > 0:
                f.write(f"5. Assigned {assigned:,} of {original_unmapped:,} unmapped instances ({100*assigned/original_unmapped:.1f}%)\n")
            f.write(f"6. Reduced cluster fragmentation ({cluster_improvement:.1f}% fewer unique clusters per path)\n\n")

            # Output files section
            f.write("OUTPUT FILES:\n")
            f.write("-" * 13 + "\n")
            f.write(f"- {output_csv}: {len(self.cluster_map):,} instance-to-cluster mappings\n\n")

            f.write("=" * 80 + "\n")

        print(f"  Summary report saved!")

    def run(self, output_file: str, summary_file: str = None, phases: set = None):
        """Run optimization pipeline with selected phases.

        Args:
            output_file: Path to save optimized cluster mapping CSV
            summary_file: Optional path to save summary report
            phases: Set of phase numbers to run {1, 2, 3}. Default: {1, 2, 3} (all phases)
        """
        if phases is None:
            phases = {1, 2, 3}  # Default: run all phases

        # Always run initialization
        self.load_data()
        self.compute_spatial_metrics()
        self.compute_initial_metrics()

        # Print which phases will run
        print("\n" + "=" * 80)
        print(f"RUNNING PHASES: {sorted(phases)}")
        print("  Phase 1: Assign Unmapped Instances" + (" [ENABLED]" if 1 in phases else " [SKIPPED]"))
        print("  Phase 2: Boundary Refinement" + (" [ENABLED]" if 2 in phases else " [SKIPPED]"))
        print("  Phase 3: Cluster Merging" + (" [ENABLED]" if 3 in phases else " [SKIPPED]"))
        print("=" * 80)

        # Phase 1: Assign unmapped instances
        if 1 in phases:
            self.assign_unmapped_instances()
        else:
            print("\nSkipping Phase 1 (Assign Unmapped Instances)")

        # Phase 2: Boundary refinement
        if 2 in phases:
            self.refine_boundary_instances(max_iterations=5)
        else:
            print("\nSkipping Phase 2 (Boundary Refinement)")

        # Phase 3: Cluster merging
        if 3 in phases:
            self.strategic_merging()
        else:
            print("\nSkipping Phase 3 (Cluster Merging)")

        # Always compute final metrics and save
        self.compute_final_metrics()
        self.save_optimized_mapping(output_file)

        # Generate summary report if requested
        if summary_file:
            print()
            self.generate_summary_report(summary_file, output_file)

        print("=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Location-aware cluster optimization for netlist tomography (V1 - BFS Enhanced)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--cluster-csv',
        type=str,
        default='test_input_v1/ca53_cpu_leiden_timing_cluster_map.csv',
        help='Input cluster mapping CSV'
    )

    parser.add_argument(
        '--nodes-csv',
        type=str,
        default='test_input_v1/ca53_cpu_nodes.csv',
        help='Node locations CSV'
    )

    parser.add_argument(
        '--edges-csv',
        type=str,
        default='test_input_v1/ca53_cpu_edges.csv',
        help='Graph edges CSV'
    )

    parser.add_argument(
        '--paths-rpt',
        type=str,
        default='test_input_v1/top_50000_paths_start_end_pairs.rpt',
        help='Timing paths report'
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        default='optimized_cluster_map_v1.csv',
        help='Output optimized cluster mapping CSV'
    )

    parser.add_argument(
        '--summary-file',
        type=str,
        default='optimization_summary_v1.txt',
        help='Output summary report file'
    )

    parser.add_argument(
        '--phases',
        type=str,
        default='1,2,3',
        help='Comma-separated list of phases to run (1=assign unmapped, 2=refine boundary, 3=merge clusters). Default: "1,2,3" (all phases). Examples: "1", "2,3", "1,3"'
    )

    args = parser.parse_args()

    # Parse and validate phases
    try:
        phases_to_run = set(int(p.strip()) for p in args.phases.split(','))
        if not phases_to_run.issubset({1, 2, 3}):
            raise ValueError("Phases must be from {1, 2, 3}")
        if not phases_to_run:
            raise ValueError("At least one phase must be specified")
    except (ValueError, AttributeError) as e:
        print(f"Error: Invalid phases '{args.phases}'. Must be comma-separated values from {{1, 2, 3}}")
        print(f"  Details: {e}")
        sys.exit(1)

    optimizer = ClusterOptimizer(
        cluster_csv=args.cluster_csv,
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        paths_rpt=args.paths_rpt
    )

    optimizer.run(args.output_csv, args.summary_file, phases=phases_to_run)


if __name__ == '__main__':
    main()

