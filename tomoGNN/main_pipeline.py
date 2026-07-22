# main_pipeline_dualview.py
# End-to-end pipeline for Dual-View Multi-Run GNN
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
from datetime import datetime
from config import DualViewConfig, TrainConfig
from data.data_loader import get_clock_period
from data.graph_builder_dualview import load_and_build_dualview_graph
from train import train_tomognn
from utils.cluster_hdbscan import cluster_hdbscan
from utils.cluster_leiden import cluster_leiden_gpu
from generate_def import (
    save_cluster_map_csv,
    write_cluster_def_file
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Dual-View Netlist Tomography ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: HDBSCAN clustering
  python main_pipeline_dualview.py \\
      --design ariane \\
      --nodes input_data/ariane_merged_nodes.csv \\
      --edges input_data/ariane_merged_edges.csv \\
      --output results/ariane_dualview

  # With custom memory settings
  python main_pipeline_dualview.py \\
      --design ariane \\
      --nodes input_data/ariane_merged_nodes.csv \\
      --edges input_data/ariane_merged_edges.csv \\
      --output results/ariane_dualview \\
      --colocation-q 1.5 \\
      --batch-size-nodes 4096 \\
      --runs-per-epoch 5

  # With Leiden clustering
  python main_pipeline_dualview.py \\
      --design ariane \\
      --nodes input_data/ariane_merged_nodes.csv \\
      --edges input_data/ariane_merged_edges.csv \\
      --output results/ariane_dualview \\
      --use-leiden

Note: Command-line arguments override defaults in config_dualview.py
"""
    )

    # ==================== Required Arguments ====================
    parser.add_argument(
        '--design',
        type=str,
        required=True,
        help='Design name'
    )

    parser.add_argument(
        '--nodes',
        type=str,
        required=True,
        help='Path to nodes CSV file'
    )

    parser.add_argument(
        '--edges',
        type=str,
        required=True,
        help='Path to edges CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory'
    )

    parser.add_argument(
        '--cuda-id',
        type=int,
        default=None,
        help='CUDA device index to use (default: 0)'
    )

    # ==================== Graph Building Parameters ====================
    parser.add_argument(
        '--colocation-q',
        type=float,
        default=None,
        help='Co-location radius multiplier (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--colocation-sigma',
        type=float,
        default=None,
        help='Co-location Gaussian kernel bandwidth (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--colocation-n-jobs',
        type=int,
        default=None,
        help='Number of parallel workers for co-location (-1=all CPUs, default: from config_dualview.py)'
    )

    # ==================== Training Parameters ====================
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--batch-size-nodes',
        type=int,
        default=None,
        help='Nodes per mini-batch for NeighborLoader (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--fanouts',
        type=str,
        default=None,
        help='Neighbor sampling fanouts, comma-separated (e.g., "8,8,4") (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--runs-per-epoch',
        type=int,
        default=None,
        help='Number of runs to process per epoch (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--processing-batch-size',
        type=int,
        default=None,
        help='Number of runs to process at once in forward pass (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--max-runs',
        type=int,
        default=None,
        help='Maximum number of placement runs to load (default: all runs)'
    )

    # ==================== Model Architecture ====================
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=None,
        help='Hidden dimension (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--output-dim',
        type=int,
        default=None,
        help='Output embedding dimension (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=None,
        help='Number of GNN layers (default: from config_dualview.py)'
    )

    # ==================== Edge Weight Parameters ====================
    parser.add_argument(
        '--p-crit',
        type=float,
        default=None,
        help='Criticality exponent (1.0=linear, 1.5=emphasize critical, default: from config_dualview.py)'
    )

    parser.add_argument(
        '--k-len',
        type=float,
        default=None,
        help='Length decay factor (0.3=gentle, 1.0=strong, default: from config_dualview.py)'
    )

    parser.add_argument(
        '--max-len-clip',
        type=float,
        default=None,
        help='Maximum normalized length clipping (default: from config_dualview.py)'
    )

    # ==================== Loss Function ====================
    parser.add_argument(
        '--temperature-init',
        type=float,
        default=None,
        help='Initial InfoNCE temperature (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--temperature-min',
        type=float,
        default=None,
        help='Minimum temperature bound (0.15-0.30, higher for weak designs, default: from config_dualview.py)'
    )

    parser.add_argument(
        '--temperature-max',
        type=float,
        default=None,
        help='Maximum temperature bound (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--temperature-lr',
        type=float,
        default=None,
        help='Temperature learning rate (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--lambda-len-push',
        type=float,
        default=None,
        help='Length push regularization weight (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--push-margin',
        type=float,
        default=None,
        help='Margin for push loss (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--push-tau',
        type=float,
        default=None,
        help='Softness of margin hinge (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--lambda-coloc',
        type=float,
        default=None,
        help='Calibration loss weight (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--tau-calibration',
        type=float,
        default=None,
        help='Calibration sigmoid temperature (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--focal-gamma',
        type=float,
        default=None,
        help='Focal weighting exponent (1.0=conservative, 2.0=aggressive, default: from config_dualview.py)'
    )

    parser.add_argument(
        '--tau-plus',
        type=float,
        default=None,
        help='Debiased InfoNCE prior probability of false negatives (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=None,
        help='L2 regularization weight decay (default: from config_dualview.py)'
    )

    # ==================== Clustering ====================
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=None,
        help='Minimum cluster size for HDBSCAN (default: from config_dualview.py)'
    )

    parser.add_argument(
        '--leiden-resolution',
        type=float,
        default=None,
        help='Leiden resolution parameter (default: from config_dualview.py)'
    )

    # ==================== Flags ====================
    parser.add_argument(
        '--naive-weights',
        action='store_true',
        help='Use naive (equal) weights instead of learnable'
    )

    parser.add_argument(
        '--use-leiden',
        action='store_true',
        help='Use Leiden instead of HDBSCAN clustering'
    )

    return parser.parse_args()


def create_config_from_args(args) -> DualViewConfig:
    """
    Create DualViewConfig from command-line arguments.
    Command-line arguments override defaults in config_dualview.py.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        DualViewConfig with overrides applied
    """
    # Start with defaults from config_dualview.py
    cfg = DualViewConfig()
    
    # Override with command-line arguments (only if specified)
    # Graph building
    if args.colocation_q is not None:
        cfg.colocation_q = args.colocation_q
    if args.colocation_sigma is not None:
        cfg.colocation_sigma = args.colocation_sigma
    if args.colocation_n_jobs is not None:
        cfg.colocation_n_jobs = args.colocation_n_jobs
    
    # Edge weight parameters
    if args.p_crit is not None:
        cfg.p_crit = args.p_crit
    if args.k_len is not None:
        cfg.k_len = args.k_len
    if args.max_len_clip is not None:
        cfg.max_len_clip = args.max_len_clip
    
    # Training
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.batch_size_nodes is not None:
        cfg.batch_size_nodes = args.batch_size_nodes
    if args.fanouts is not None:
        # Parse comma-separated fanouts (e.g., "8,8,4" -> (8, 8, 4))
        cfg.fanouts = tuple(int(x.strip()) for x in args.fanouts.split(','))
    if args.runs_per_epoch is not None:
        cfg.runs_per_epoch = args.runs_per_epoch
    if args.processing_batch_size is not None:
        cfg.processing_batch_size = args.processing_batch_size
    if args.max_runs is not None:
        cfg.max_runs = args.max_runs
    
    # Device selection
    if args.cuda_id is not None:
        cfg.cuda_id = args.cuda_id
        if isinstance(cfg.device, str) and cfg.device.startswith('cuda'):
            cfg.device = f'cuda:{args.cuda_id}'

    # Model architecture
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.output_dim is not None:
        cfg.output_dim = args.output_dim
    if args.num_layers is not None:
        cfg.num_layers = args.num_layers
    
    # Loss function - Temperature
    if args.temperature_init is not None:
        cfg.temperature_init = args.temperature_init
    if args.temperature_min is not None:
        cfg.temperature_min = args.temperature_min
    if args.temperature_max is not None:
        cfg.temperature_max = args.temperature_max
    if args.temperature_lr is not None:
        cfg.temperature_lr = args.temperature_lr
    
    # Loss function - Push
    if args.lambda_len_push is not None:
        cfg.lambda_len_push = args.lambda_len_push
    if args.push_margin is not None:
        cfg.push_margin = args.push_margin
    if args.push_tau is not None:
        cfg.push_tau = args.push_tau
    
    # Loss function - Calibration
    if args.lambda_coloc is not None:
        cfg.lambda_coloc = args.lambda_coloc
    if args.tau_calibration is not None:
        cfg.tau_calibration = args.tau_calibration
    if args.focal_gamma is not None:
        cfg.focal_gamma = args.focal_gamma
    
    # Loss function - Debiased InfoNCE
    if args.tau_plus is not None:
        cfg.tau_plus = args.tau_plus
    
    # Clustering
    if args.min_cluster_size is not None:
        cfg.min_cluster_size = args.min_cluster_size
    if args.leiden_resolution is not None:
        cfg.leiden_resolution = args.leiden_resolution
    
    # Flags (always set from args)
    cfg.naive_weights = args.naive_weights
    cfg.use_leiden = args.use_leiden
    
    return cfg


class TeeOutput:
    """Write to both file and stdout."""
    def __init__(self, file, terminal):
        self.file = file
        self.terminal = terminal
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()


def setup_logging(output_dir: str, log_file_name: str = "run.log") -> TeeOutput:
    """Setup logging to both file and stdout.
    
    Args:
        output_dir: Output directory path
        log_file_name: Name of log file (default: "run.log")
    """
    log_file_path = os.path.join(output_dir, log_file_name)
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # Write header
    log_file.write("="*80 + "\n")
    log_file.write("Dual-View Netlist Tomography ML Pipeline - Run Log\n")
    log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    # Create tee output
    tee = TeeOutput(log_file, sys.stdout)
    
    return tee, log_file


def run_pipeline(args):
    """
    Execute complete dual-view ML pipeline.

    Pipeline steps:
    1. Load multi-run CSV data
    2. Build dual-view graphs (netlist G + co-location H per run)
    3. Compute run quality scores
    4. Train dual-view GNN
    5. Generate fused embeddings
    6. Cluster embeddings (HDBSCAN or Leiden)
    7. Save cluster map CSV
    8. Generate DEF file
    """
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Note: Logging is now set up after config is loaded (to use config.log_file_name)
    # Initialize variables
    tee = None
    log_file = None
    log_file_name = "run.log"  # Default, will be updated from config
    original_stdout = sys.stdout
    
    try:
        print("="*60)
        print("Dual-View Netlist Tomography ML Pipeline")
        print("="*60)
        print(f"  Design: {args.design}")
        print(f"  Nodes: {args.nodes}")
        print(f"  Edges: {args.edges}")
        print(f"  Output: {args.output}")
        
        # Setup configuration: use defaults from config_dualview.py, 
        # override with command-line arguments if specified
        cfg = create_config_from_args(args)
        
        # Get log file name from config
        log_file_name = cfg.log_file_name if hasattr(cfg, 'log_file_name') else "run.log"
        
        # Setup logging with config-defined log file name
        tee, log_file = setup_logging(args.output, log_file_name)
        original_stdout = sys.stdout
        sys.stdout = tee
        
        print(f"  Mode: "
              f"{'NAIVE' if cfg.naive_weights else 'LEARNABLE'}")
        print(f"  Clustering: "
              f"{'Leiden' if cfg.use_leiden else 'HDBSCAN'}")
        print(f"  Min cluster size: {cfg.min_cluster_size}")
        print(f"  Co-location q: {cfg.colocation_q}")
        print(f"  Co-location sigma: {cfg.colocation_sigma}")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Device: {cfg.device}")
        
        # Show which parameters were overridden from command line
        overridden = []
        
        # Graph building
        if args.colocation_q is not None:
            overridden.append(f"colocation_q={cfg.colocation_q}")
        if args.colocation_sigma is not None:
            overridden.append(f"colocation_sigma={cfg.colocation_sigma}")
        if args.colocation_n_jobs is not None:
            overridden.append(f"colocation_n_jobs={cfg.colocation_n_jobs}")
        
        # Edge weights
        if args.p_crit is not None:
            overridden.append(f"p_crit={cfg.p_crit}")
        if args.k_len is not None:
            overridden.append(f"k_len={cfg.k_len}")
        if args.max_len_clip is not None:
            overridden.append(f"max_len_clip={cfg.max_len_clip}")
        
        # Training
        if args.epochs is not None:
            overridden.append(f"epochs={cfg.epochs}")
        if args.lr is not None:
            overridden.append(f"lr={cfg.lr}")
        if args.weight_decay is not None:
            overridden.append(f"weight_decay={cfg.weight_decay}")
        if args.batch_size_nodes is not None:
            overridden.append(f"batch_size_nodes={cfg.batch_size_nodes}")
        if args.fanouts is not None:
            overridden.append(f"fanouts={cfg.fanouts}")
        if args.runs_per_epoch is not None:
            overridden.append(f"runs_per_epoch={cfg.runs_per_epoch}")
        if args.processing_batch_size is not None:
            overridden.append(f"processing_batch_size={cfg.processing_batch_size}")
        if args.max_runs is not None:
            overridden.append(f"max_runs={cfg.max_runs}")

        if args.cuda_id is not None:
            overridden.append(f"cuda_id={cfg.cuda_id}")
        
        # Model
        if args.hidden_dim is not None:
            overridden.append(f"hidden_dim={cfg.hidden_dim}")
        if args.output_dim is not None:
            overridden.append(f"output_dim={cfg.output_dim}")
        if args.num_layers is not None:
            overridden.append(f"num_layers={cfg.num_layers}")
        
        # Loss - Temperature
        if args.temperature_init is not None:
            overridden.append(f"temperature_init={cfg.temperature_init}")
        if args.temperature_min is not None:
            overridden.append(f"temperature_min={cfg.temperature_min}")
        if args.temperature_max is not None:
            overridden.append(f"temperature_max={cfg.temperature_max}")
        if args.temperature_lr is not None:
            overridden.append(f"temperature_lr={cfg.temperature_lr}")
        
        # Loss - Push
        if args.lambda_len_push is not None:
            overridden.append(f"lambda_len_push={cfg.lambda_len_push}")
        if args.push_margin is not None:
            overridden.append(f"push_margin={cfg.push_margin}")
        if args.push_tau is not None:
            overridden.append(f"push_tau={cfg.push_tau}")
        
        # Loss - Calibration
        if args.lambda_coloc is not None:
            overridden.append(f"lambda_coloc={cfg.lambda_coloc}")
        if args.tau_calibration is not None:
            overridden.append(f"tau_calibration={cfg.tau_calibration}")
        if args.focal_gamma is not None:
            overridden.append(f"focal_gamma={cfg.focal_gamma}")
        
        # Loss - Debiased
        if args.tau_plus is not None:
            overridden.append(f"tau_plus={cfg.tau_plus}")
        
        # Clustering
        if args.min_cluster_size is not None:
            overridden.append(f"min_cluster_size={cfg.min_cluster_size}")
        if args.leiden_resolution is not None:
            overridden.append(f"leiden_resolution={cfg.leiden_resolution}")
        
        if overridden:
            print(f"  Overridden from command line:")
            # Group overrides for readability
            for i in range(0, len(overridden), 3):
                print(f"    {', '.join(overridden[i:i+3])}")
        else:
            print(f"  Using all defaults from config_dualview.py")
        
        print("="*60)

        # Get clock period
        cp = get_clock_period(args.nodes)

        # Step 1-2: Load data and build dual-view graphs
        print("\n" + "="*60)
        print("STEP 1: Load Data and Build Dual-View Graphs")
        print("="*60)

        data_per_run, run_quality, id_to_name, coordinates_all_runs, sigmas_all_runs = \
            load_and_build_dualview_graph(
                args.nodes,
                args.edges,
                cp,
                cfg
                # All colocation parameters now taken from cfg automatically
            )

        # Step 3: Train dual-view GNN
        print("\n" + "="*60)
        print("STEP 2: Train TomoGNN Model")
        print("="*60)

        model, embeddings = train_tomognn(
            data_per_run,
            run_quality,
            cfg,
            output_dir=args.output,
            coordinates_all_runs=coordinates_all_runs,
            sigmas_all_runs=sigmas_all_runs
        )

        # Save embeddings
        emb_path = os.path.join(args.output, 'embeddings_tomognn.npy')
        np.save(emb_path, embeddings)
        print(f"\nEmbeddings saved: {emb_path}")

        # Step 4: Clustering
        print("\n" + "="*60)
        print("STEP 3: Clustering")
        print("="*60)

        # Get node names in order
        num_nodes = embeddings.shape[0]
        node_names = [id_to_name[i] for i in range(num_nodes)]

        if args.use_leiden:
            # Leiden clustering
            from data.data_loader import load_multi_run_data
            import pandas as pd

            # Create node_to_idx mapping
            node_to_idx = {name: idx for idx, name in enumerate(node_names)}

            # Load edges (need src/dst as names for Leiden)
            nodes_df, edges_df, _, _, _ = load_multi_run_data(
                args.nodes,
                args.edges
            )

            # Map integer IDs back to names
            edges_for_leiden = pd.DataFrame({
                'src': [id_to_name[i] for i in edges_df['src']],
                'dst': [id_to_name[i] for i in edges_df['dst']]
            })

            labels = cluster_leiden_gpu(
                embeddings,
                edges_for_leiden,
                node_to_idx,
                min_cluster_size=cfg.min_cluster_size,
                resolution=cfg.leiden_resolution
            )
        else:
            # HDBSCAN clustering
            labels = cluster_hdbscan(
                embeddings,
                min_cluster_size=cfg.min_cluster_size
            )

        # Step 5: Save cluster map
        print("\n" + "="*60)
        print("STEP 4: Save Cluster Map")
        print("="*60)

        cluster_csv = os.path.join(args.output, 'cluster_map_dualview.csv')
        save_cluster_map_csv(node_names, labels, cluster_csv)

        # Step 6: Generate DEF file
        print("\n" + "="*60)
        print("STEP 5: Generate DEF File")
        print("="*60)

        def_file = os.path.join(
            args.output,
            f'{args.design}_clusters_tomognn.def'
        )

        write_cluster_def_file(
            cluster_map_csv=cluster_csv,
            design_name=args.design,
            output_def=def_file,
            min_cluster_size=cfg.min_cluster_size
        )

        # Final summary
        print("\n" + "="*60)
        print("DUAL-VIEW PIPELINE COMPLETE!")
        print("="*60)
        print(f"  Embeddings: {emb_path}")
        print(f"  Cluster map: {cluster_csv}")
        print(f"  DEF file: {def_file}")
        print(f"  Log file: {os.path.join(args.output, log_file_name)}")
        print("="*60)
    finally:
        # Restore stdout and close log file
        if sys.stdout != original_stdout:
            sys.stdout = original_stdout
        if log_file is not None:
            log_file.write("\n" + "="*80 + "\n")
            log_file.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("="*80 + "\n")
            log_file.close()
            print(f"\nLog saved to: {os.path.join(args.output, log_file_name)}")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        run_pipeline(args)
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

