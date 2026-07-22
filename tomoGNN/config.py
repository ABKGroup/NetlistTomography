# config_dualview.py
# Complete hyperparameter configuration for Dual-View Multi-Run GNN
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class DualViewConfig:
    """
    Complete hyperparameter configuration for Dual-View Multi-Run GNN.
    
    This includes all hyperparameters used in:
    - Graph building (netlist G + co-location H)
    - Model architecture (GraphSAGE backbone)
    - Training (optimization, loss, sampling)
    - Clustering (HDBSCAN/Leiden)
    """
    
    # ==================== Graph Building ====================
    # Co-location graph construction
    #!memory issue
    colocation_q: float = 3.0              # Adaptive radius multiplier: q * median(1-NN distances)
    colocation_sigma: float = 10.0         # Gaussian kernel bandwidth: w_ij = exp(-(d_ij/σ)^2)
    colocation_radius: float = None        # Optional fixed radius (if None, computed adaptively)
    colocation_M: int = 10000              # Number of query points for adaptive radius median
    colocation_S: int = 100000             # Subsample size for NN index building (S in compute_adaptive_radius_fast)
    colocation_n_jobs: int = 1             # Number of parallel workers (-1=all CPUs, 1=sequential)
    
    # Netlist graph edge weights (from edge_weights.py)
    p_crit: float = 1.5                   # Criticality exponent: c^p_crit
    k_len: float = 1.0                    # Length decay factor: exp(-k_len * L)
    max_len_clip: float = 3.0             # Max normalized length clipping
    
    # Graph directionality
    undirected: bool = True                # Make graphs undirected
    # Note: 'directed' is computed in __post_init__ for compatibility with TrainConfig
    
    # ==================== Model Architecture ====================
    # Input/Output dimensions
    input_dim: int = None                  # Auto-determined from data (coordinates + degree)
    hidden_dim: int = 128                 # GraphSAGE hidden dimension
    output_dim: int = 16                   # Final embedding dimension
    num_layers: int = 3                    # Number of dual-graph layers (K=3)
    
    # GraphSAGE backbone
    sage_aggr: str = "mean"                # Aggregation function: mean/sum/max
    sage_normalize: bool = True            # L2 normalize GraphSAGE outputs
    
    # Edge feature dimensions
    edge_feat_dim_G: int = 2               # Netlist edge features: [criticality, length]
    edge_feat_dim_H: int = 1               # Co-location edge features: [distance]
    
    # Regularization
    dropout: float = 0.1                   # Dropout rate
    
    # Run-conditioned FiLM
    num_runs: int = None                   # Auto-determined from data
    
    # Quality-aware fusion
    fusion_hidden_dim: int = None          # Auto: same as output_dim
    
    # ==================== Training Configuration ====================
    # Optimization
    lr: float = 8e-3                      # Learning rate (AdamW) - increased from 1e-3 for faster convergence
    weight_decay: float = 1e-4            # L2 regularization weight decay
    epochs: int = 500                      # Maximum training epochs
    
    # Run selection / ablations
    max_runs: Optional[int] = None         # Limit number of placement runs to load (None = use all)
    
 
    
    # Memory management
    run_batch_size: int = None             # Auto: adaptive based on design size (see train_dualview.py)
    processing_batch_size: int = 3         # Number of runs to process at once (in model.forward) - reduced for memory
    # !memory issue
    runs_per_epoch: int = 10             # Number of runs to process per epoch (None = auto: min(3, num_runs) if num_runs >= 5 else num_runs)
    
    # Subgraph sampling (NeighborLoader)
    #!memory issue
    batch_size_nodes: int = 8192           # Nodes per mini-batch for NeighborLoader (increased for efficiency)
    fanouts: tuple = (10, 10, 5)           # Neighbor sampling fanouts per layer (reduced for speed)
    #!memory issue
    # fanouts: tuple = (8, 8, 4)
    
    # Early stopping
    patience: int = 50                     # Early stopping patience (epochs) - increased for temperature adaptation
    min_delta: float = 0.0                # Minimum loss improvement threshold
    
    # ==================== Loss Function ====================
    # Enhanced Contrastive Loss (InfoNCE + Length Push + Calibration)
    # Learnable temperature
    temperature_init: float = 0.8         # Initial InfoNCE temperature (increased for better exploration)
    temperature_min: float = 0.15         # Minimum temperature (increased from 0.07 to prevent collapse)
    temperature_max: float = 1.2          # Maximum temperature (reduced to prevent over-smoothing)
    learn_temperature: bool = True        # Learn temperature during training (HIGHLY RECOMMENDED)
    temperature_lr: float = 1.25e-4       # Temperature learning rate (much slower than model: lr/80)
    
    # Push term
    lambda_len_push: float = 3.0          # Length push regularization weight (increased from 1.5 to prevent collapse)
    push_margin: float = 0.20             # Margin for push loss (increased from 0.10 for stronger separation)
    push_tau: float = 0.5                 # Softness of margin hinge
    
    # Calibration
    lambda_coloc: float = 1.0             # Calibration loss weight (balanced to work with temperature-scaled push)
    tau_calibration: float = 0.6          # Calibration sigmoid temperature (0.6 is optimal)
    focal_gamma: float = 1.0              # Focal weighting exponent (1.0 = conservative, 2.0 = aggressive)
    
    # Debiased contrastive
    use_debiased: bool = True             # Use debiased InfoNCE (corrects false negatives)
    tau_plus: float = 0.08                # Prior probability of false negatives (5-10% for netlist graphs)
    
    # Self-consistency (optional)
    lambda_self: float = 0.0              # Self-consistency regularization (keep 0.0 unless needed)
    
    # General
    normalize_loss: bool = True           # Normalize loss by number of pairs (for better convergence)
    
    # ==================== Pair Sampling ====================
    # Sampling for contrastive loss
    n_pos_samples: int = None              # Auto: max(256, num_nodes // 10)
    n_neg_samples: int = None              # Auto: max(256, num_nodes // 10)
    sampling_replace: bool = False         # Sampling with or without replacement
    
    # ==================== Run Quality Calculation ====================
    # Quality score based on Total Negative Slack (TNS)
    quality_metric: str = "slack"          # Metric: "slack" (TNS-based)
    tns_scale_factor: float = 1000.0       # TNS scaling: quality = exp(TNS / scale), typical TNS: [-10K, 0]
    
    # ==================== Clustering ====================
    # HDBSCAN
    min_cluster_size: int = 200            # Minimum cluster size for DEF output
    cluster_selection_epsilon: float = 0.0 # HDBSCAN cluster selection epsilon
    
    # Leiden (alternative)
    leiden_resolution: float = 1.0         # Leiden resolution (higher = more clusters)
    use_leiden: bool = False               # Use Leiden instead of HDBSCAN
    
    # ==================== Multi-Run Weights ====================
    naive_weights: bool = False            # True=equal weights, False=learnable quality-aware
    
    # ==================== Feature Normalization ====================
    normalize_features: bool = True        # Normalize node features
    
    # ==================== Miscellaneous ====================
    seed: int = 42                         # Random seed for reproducibility
    device: str = "cuda"                   # Device: "cuda" or "cpu"
    cuda_id: Optional[int] = 0             # CUDA device index (None=use default)
    
    # ==================== Output Configuration ====================
    save_checkpoints: bool = True          # Save model checkpoints
    save_embeddings: bool = True           # Save final embeddings
    save_cluster_map: bool = True          # Save cluster assignments
    save_def_file: bool = True             # Generate DEF file
    save_training_curves: bool = True       # Save training curve plots
    
    # ==================== Logging ====================
    log_file: str = None                   # Auto: output_dir/{log_file_name}
    log_file_name: str = "run2.log"         # Log file name (relative to output_dir)
    verbose: bool = True                   # Print detailed logs
    
    def __post_init__(self):
        """Auto-set dependent parameters."""
        # Set fusion hidden dim to output_dim if not specified
        if self.fusion_hidden_dim is None:
            self.fusion_hidden_dim = self.output_dim
        
        # Sync directed/undirected for compatibility
        # directed=False means undirected=True (compatible with TrainConfig)
        self.directed = not self.undirected

        # Normalize device / cuda_id pairing
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            if ":" in self.device:
                # Derive cuda_id from explicit device string
                try:
                    self.cuda_id = int(self.device.split(":", 1)[1])
                except (IndexError, ValueError):
                    pass  # Keep existing cuda_id if parsing fails
            elif self.cuda_id is not None:
                self.device = f"cuda:{self.cuda_id}"
            else:
                self.cuda_id = 0
                self.device = "cuda:0"
        else:
            # Non-CUDA device; clear cuda_id to avoid confusion
            self.cuda_id = None
        
        # Set run batch size based on model defaults (handled in train_dualview.py)
        if self.run_batch_size is None:
            # Will be set dynamically based on num_nodes
            pass
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            'graph_building': {
                'colocation_q': self.colocation_q,
                'colocation_sigma': self.colocation_sigma,
                'colocation_radius': self.colocation_radius,
                'colocation_M': self.colocation_M,
                'colocation_S': self.colocation_S,
                'colocation_n_jobs': self.colocation_n_jobs,
                'p_crit': self.p_crit,
                'k_len': self.k_len,
                'max_len_clip': self.max_len_clip,
                'undirected': self.undirected,
            },
            'model': {
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'sage_aggr': self.sage_aggr,
                'sage_normalize': self.sage_normalize,
                'edge_feat_dim_G': self.edge_feat_dim_G,
                'edge_feat_dim_H': self.edge_feat_dim_H,
                'dropout': self.dropout,
            },
            'training': {
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'epochs': self.epochs,
                'patience': self.patience,
                'min_delta': self.min_delta,
                'processing_batch_size': self.processing_batch_size,
                'max_runs': self.max_runs,
            },
            'loss': {
                'temperature_init': self.temperature_init,
                'temperature_min': self.temperature_min,
                'temperature_max': self.temperature_max,
                'learn_temperature': self.learn_temperature,
                'temperature_lr': self.temperature_lr,
                'lambda_len_push': self.lambda_len_push,
                'push_margin': self.push_margin,
                'push_tau': self.push_tau,
                'lambda_coloc': self.lambda_coloc,
                'tau_calibration': self.tau_calibration,
                'focal_gamma': self.focal_gamma,
                'use_debiased': self.use_debiased,
                'tau_plus': self.tau_plus,
                'lambda_self': self.lambda_self,
                'normalize_loss': self.normalize_loss,
            },
            'quality': {
                'quality_metric': self.quality_metric,
                'tns_scale_factor': self.tns_scale_factor,
            },
            'clustering': {
                'min_cluster_size': self.min_cluster_size,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'leiden_resolution': self.leiden_resolution,
                'use_leiden': self.use_leiden,
            },
            'misc': {
                'seed': self.seed,
                'device': self.device,
                'cuda_id': self.cuda_id,
                'naive_weights': self.naive_weights,
            }
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'DualViewConfig':
        """Create config from dictionary."""
        # Flatten nested dict
        graph = d.get('graph_building', {})
        model = d.get('model', {})
        training = d.get('training', {})
        loss = d.get('loss', {})
        quality = d.get('quality', {})
        clustering = d.get('clustering', {})
        misc = d.get('misc', {})
        
        return cls(
            # Graph building
            colocation_q=graph.get('colocation_q', 1.0),
            colocation_sigma=graph.get('colocation_sigma', 10.0),
            colocation_radius=graph.get('colocation_radius', None),
            colocation_M=graph.get('colocation_M', 10000),
            colocation_S=graph.get('colocation_S', 100000),
            colocation_n_jobs=graph.get('colocation_n_jobs', 1),
            p_crit=graph.get('p_crit', 1.5),
            k_len=graph.get('k_len', 1.0),
            max_len_clip=graph.get('max_len_clip', 3.0),
            undirected=graph.get('undirected', True),
            # Model
            hidden_dim=model.get('hidden_dim', 128),
            output_dim=model.get('output_dim', 16),
            num_layers=model.get('num_layers', 3),
            sage_aggr=model.get('sage_aggr', 'mean'),
            sage_normalize=model.get('sage_normalize', True),
            edge_feat_dim_G=model.get('edge_feat_dim_G', 2),
            edge_feat_dim_H=model.get('edge_feat_dim_H', 1),
            dropout=model.get('dropout', 0.1),
            # Training
            lr=training.get('lr', 1e-3),
            weight_decay=training.get('weight_decay', 1e-4),
            epochs=training.get('epochs', 50),
            patience=training.get('patience', 20),
            min_delta=training.get('min_delta', 0.0),
            processing_batch_size=training.get('processing_batch_size', 5),
            max_runs=training.get('max_runs', None),
            # Loss
            temperature_init=loss.get('temperature_init', 0.8),
            temperature_min=loss.get('temperature_min', 0.15),
            temperature_max=loss.get('temperature_max', 1.2),
            learn_temperature=loss.get('learn_temperature', True),
            temperature_lr=loss.get('temperature_lr', 1.25e-4),
            lambda_len_push=loss.get('lambda_len_push', 3.0),
            push_margin=loss.get('push_margin', 0.20),
            push_tau=loss.get('push_tau', 0.5),
            lambda_coloc=loss.get('lambda_coloc', 1.0),
            tau_calibration=loss.get('tau_calibration', 0.6),
            focal_gamma=loss.get('focal_gamma', 1.0),
            use_debiased=loss.get('use_debiased', True),
            tau_plus=loss.get('tau_plus', 0.08),
            lambda_self=loss.get('lambda_self', 0.0),
            normalize_loss=loss.get('normalize_loss', True),
            # Quality
            quality_metric=quality.get('quality_metric', 'slack'),
            tns_scale_factor=quality.get('tns_scale_factor', 10.0),
            # Clustering
            min_cluster_size=clustering.get('min_cluster_size', 200),
            cluster_selection_epsilon=clustering.get('cluster_selection_epsilon', 0.0),
            leiden_resolution=clustering.get('leiden_resolution', 1.0),
            use_leiden=clustering.get('use_leiden', False),
            # Misc
            seed=misc.get('seed', 42),
            device=misc.get('device', 'cuda'),
            cuda_id=misc.get('cuda_id', 0 if misc.get('device', 'cuda').startswith('cuda') else None),
            naive_weights=misc.get('naive_weights', False),
        )

#
# Backward-compatibility: some modules import TrainConfig from config.py.
# We alias TrainConfig to DualViewConfig so the codebase can import a single source.
#
@dataclass
class TrainConfig(DualViewConfig):
    pass

