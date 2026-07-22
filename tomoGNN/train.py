# train_dualview.py
# Training loop for Dual-View Multi-Run GNN with Subgraph Sampling
from __future__ import annotations
import os
from typing import Optional
import torch
from torch import nn
import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from torch_geometric.data import Data
# Try to use NeighborLoader, but fall back to simple loader if pyg-lib/torch-sparse unavailable
try:
    from torch_geometric.loader import NeighborLoader
    # Test if it actually works (not just imported)
    HAS_PYG_LOADER = True
except (ImportError, Exception):
    HAS_PYG_LOADER = False
# from simple_neighbor_loader import create_simple_neighbor_loader
from config_dualview import DualViewConfig, TrainConfig
from models.models_dualview import TomoGNN
from models.losses import WeightedInfoNCEWithLenPush, EnhancedContrastiveLoss
from utils.sampler import sample_pairs, sample_pairs_per_anchor
from data.colocation_graph import compute_colocation_consistency


def setup_device(cuda_id: Optional[int] = None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, training requires GPU")
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("CUDA reported as available but no devices found")

    if cuda_id is None:
        cuda_id = 0

    if cuda_id < 0 or cuda_id >= device_count:
        raise ValueError(f"Requested CUDA device {cuda_id} but only {device_count} device(s) available")

    device = torch.device(f'cuda:{cuda_id}')
    torch.cuda.set_device(device)
    print("="*60)
    print("GPU Information")
    print("="*60)
    print(f"  Device index: {cuda_id}")
    print(f"  Device: {torch.cuda.get_device_name(cuda_id)}")
    print(f"  Memory: {torch.cuda.get_device_properties(cuda_id).total_memory/1e9:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print("="*60)
    return device


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.wait = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, val: float, epoch: int) -> bool:
        """
        Update early stopping state.
        
        Args:
            val: Current validation/loss value
            epoch: Current epoch number
            
        Returns:
            True if this is a new best model, False otherwise
        """
        improved = False
        if val < self.best - self.min_delta:
            self.best = val
            self.best_epoch = epoch
            self.wait = 0
            improved = True
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return improved


def train_tomognn(
    data_per_run: list,
    run_quality: torch.Tensor,
    cfg: TrainConfig | DualViewConfig,
    device: torch.device = None,
    output_dir: str = None,
    coordinates_all_runs: list = None,
    sigmas_all_runs: list = None,
    pretrained_checkpoint: dict = None,
    freeze_layers: int = 0,
    reset_temperature: bool = False
) -> tuple[TomoGNN, np.ndarray]:
    """
    Train dual-view GNN model.
    
    Args:
        data_per_run: List of dicts with dual-graph data per run
        run_quality: [num_runs] quality scores
        cfg: Training configuration
        device: Torch device
        output_dir: Directory for checkpoints
        coordinates_all_runs: Coordinates for all runs (for calibration loss)
        sigmas_all_runs: Sigma values for all runs (for calibration loss)
        pretrained_checkpoint: Pre-trained checkpoint dict (for fine-tuning)
        freeze_layers: Number of early layers to freeze (0=none, for fine-tuning)
        reset_temperature: Reset temperature to initial value instead of loading from checkpoint
        
    Returns:
        model: Trained model
        embeddings: [N, output_dim] final node embeddings
    """
    if device is None:
        cuda_id = getattr(cfg, 'cuda_id', None)
        device = setup_device(cuda_id)
    elif isinstance(device, str):
        device = torch.device(device)

    # Update config with resolved device info for downstream use/checkpointing
    if hasattr(cfg, 'device'):
        cfg.device = str(device)
    if device.type == 'cuda' and hasattr(cfg, 'cuda_id'):
        if device.index is not None:
            cfg.cuda_id = device.index
        else:
            cfg.cuda_id = torch.cuda.current_device()
    
    # Move data to device (memory-efficient: keep co-location graphs on CPU, move on-demand)
    print("Moving data to device (memory-efficient mode)...")
    
    # Move shared netlist graph G to GPU (small, reused across runs)
    # Keep co-location graphs H on CPU to save memory
    for run_data in data_per_run:
        # Always on GPU: node features and netlist graph
        if 'x' in run_data and isinstance(run_data['x'], torch.Tensor):
            run_data['x'] = run_data['x'].to(device)
        if 'edge_index_G' in run_data:
            run_data['edge_index_G'] = run_data['edge_index_G'].to(device)
        if 'edge_attr_G' in run_data:
            run_data['edge_attr_G'] = run_data['edge_attr_G'].to(device)
        if 'edge_weight_G' in run_data and run_data['edge_weight_G'] is not None:
            run_data['edge_weight_G'] = run_data['edge_weight_G'].to(device)
        # Also move edge_weights_G_full to GPU (needed for sampling)
        if 'edge_weights_G_full' in run_data:
            run_data['edge_weights_G_full'] = run_data['edge_weights_G_full'].to(device)
        
        # Keep on CPU: co-location graphs (moved to GPU only when needed)
        # Explicitly ensure they stay on CPU (they're large: ~520K edges per run)
        if 'edge_index_H' in run_data:
            if isinstance(run_data['edge_index_H'], torch.Tensor):
                run_data['edge_index_H'] = run_data['edge_index_H'].cpu()
        if 'edge_attr_H' in run_data:
            if isinstance(run_data['edge_attr_H'], torch.Tensor):
                run_data['edge_attr_H'] = run_data['edge_attr_H'].cpu()
        if 'edge_weight_H' in run_data:
            if isinstance(run_data['edge_weight_H'], torch.Tensor):
                run_data['edge_weight_H'] = run_data['edge_weight_H'].cpu()
        
        # Mark as CPU tensors - will move to GPU in forward pass
        run_data['_h_on_cpu'] = True  # Flag for memory-efficient loading
    
    run_quality = run_quality.to(device)
    
    # Clear cache after data loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Get dimensions
    num_runs = len(data_per_run)
    num_nodes = data_per_run[0]['x'].size(0)
    in_dim = data_per_run[0]['x'].size(1)
    
    # Setup checkpoint directory
    checkpoint_dir = None
    best_model_path = None
    if output_dir is not None:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_model_path = os.path.join(checkpoint_dir, 'best_model_tomognn.pt')
    
    print("="*60)
    print("Training TomoGNN")
    print("="*60)
    print(f"  Nodes: {num_nodes:,}")
    print(f"  Runs: {num_runs}")
    print(f"  Input features: {in_dim}")
    print(f"  Output dim: {cfg.output_dim}")
    print(f"  Epochs: {cfg.epochs}")
    if output_dir:
        print(f"  Checkpoint dir: {checkpoint_dir}")
    
    # Create model
    model = TomoGNN(
        in_dim=in_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        num_layers=cfg.num_layers,
        num_runs=num_runs,
        edge_feat_dim_G=2,  # [c, L]
        edge_feat_dim_H=1,  # [distance]
        dropout=cfg.dropout
    ).to(device)
    
    # Load pre-trained weights if provided (for fine-tuning)
    if pretrained_checkpoint is not None:
        print()
        print("="*60)
        print("Loading pre-trained weights for fine-tuning")
        print("="*60)
        
        pretrained_state = pretrained_checkpoint['model_state_dict']
        model_state = model.state_dict()
        
        # Load compatible weights
        loaded_keys = []
        skipped_keys = []
        
        for key, value in pretrained_state.items():
            if key in model_state:
                # Check shape compatibility
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {model_state[key].shape} vs {value.shape})")
            else:
                skipped_keys.append(f"{key} (not in target model)")
        
        model.load_state_dict(model_state)
        
        print(f"✓ Loaded {len(loaded_keys)}/{len(pretrained_state)} layers")
        if skipped_keys:
            print(f"⚠ Skipped {len(skipped_keys)} incompatible layers:")
            for key in skipped_keys[:5]:  # Show first 5
                print(f"    - {key}")
            if len(skipped_keys) > 5:
                print(f"    ... and {len(skipped_keys)-5} more")
        
        print("="*60)
        print()
    
    # Loss function - CREATE FIRST (before optimizer, so we can include its parameters)
    use_enhanced_loss = (coordinates_all_runs is not None and sigmas_all_runs is not None)
    
    if use_enhanced_loss:
        print("Using EnhancedContrastiveLoss with tomography calibration")
        print(f"  → Learnable temperature: {getattr(cfg, 'learn_temperature', True)}")
        print(f"  → Initial temp: {getattr(cfg, 'temperature_init', 0.8):.3f}, Range: [{getattr(cfg, 'temperature_min', 0.07):.3f}, {getattr(cfg, 'temperature_max', 1.2):.3f}]")
        print(f"  → Debiased InfoNCE: {getattr(cfg, 'use_debiased', True)}, tau_plus: {getattr(cfg, 'tau_plus', 0.08):.3f}")
        print(f"  → Focal gamma: {getattr(cfg, 'focal_gamma', 1.0):.2f}")
        criterion = EnhancedContrastiveLoss(
            temperature_init=getattr(cfg, 'temperature_init', 0.8),  # Increased from 0.6
            temperature_min=getattr(cfg, 'temperature_min', 0.07),
            temperature_max=getattr(cfg, 'temperature_max', 1.2),    # Decreased from 1.5
            learn_temperature=getattr(cfg, 'learn_temperature', True),
            lambda_push=getattr(cfg, 'lambda_len_push', 1.5),
            push_margin=getattr(cfg, 'push_margin', 0.10),
            push_tau=getattr(cfg, 'push_tau', 0.5),
            lambda_coloc=getattr(cfg, 'lambda_coloc', 0.5),          # Decreased from 1.0
            tau_calibration=getattr(cfg, 'tau_calibration', 0.6),
            focal_gamma=getattr(cfg, 'focal_gamma', 1.0),
            use_debiased=getattr(cfg, 'use_debiased', True),
            tau_plus=getattr(cfg, 'tau_plus', 0.08),
            lambda_self=getattr(cfg, 'lambda_self', 0.0),
            normalize_loss=getattr(cfg, 'normalize_loss', True),
        ).to(device)  # Move criterion to device (for learnable parameters)
        
        # Load temperature from checkpoint if provided and not resetting
        if pretrained_checkpoint is not None and not reset_temperature:
            if 'criterion_state_dict' in pretrained_checkpoint:
                try:
                    criterion.load_state_dict(pretrained_checkpoint['criterion_state_dict'])
                    current_temp = float(torch.exp(criterion.log_T))
                    print(f"  → Loaded temperature from checkpoint: {current_temp:.4f}")
                except Exception as e:
                    print(f"  ⚠ Could not load temperature from checkpoint: {e}")
                    print(f"  → Using initial temperature: {getattr(cfg, 'temperature_init', 0.8):.4f}")
            else:
                print(f"  ⚠ Checkpoint does not contain criterion state")
                print(f"  → Using initial temperature: {getattr(cfg, 'temperature_init', 0.8):.4f}")
        elif reset_temperature:
            print(f"  → Temperature reset to initial: {getattr(cfg, 'temperature_init', 0.8):.4f}")
    else:
        print("Using legacy WeightedInfoNCEWithLenPush (no calibration)")
        criterion = WeightedInfoNCEWithLenPush(
            temperature=getattr(cfg, 'temperature_init', 0.1),
            lambda_len_push=getattr(cfg, 'lambda_len_push', 1.0)
        )
    
    # Optimizer - CREATE AFTER criterion
    # CRITICAL: Use SEPARATE optimizers for model and temperature
    # Temperature should learn much slower (different scale)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Temperature optimizer with much slower learning rate
    temp_optimizer = None
    if use_enhanced_loss and hasattr(criterion, 'parameters') and getattr(cfg, 'learn_temperature', True):
        # Temperature LR should be ~100x slower than model LR
        temp_lr = getattr(cfg, 'temperature_lr', cfg.lr / 80)  # Default: 8e-3 / 80 = 1e-4
        print(f"  → Separate temperature optimizer: LR={temp_lr:.2e} (model LR: {cfg.lr:.2e})")
        temp_optimizer = torch.optim.AdamW(
            criterion.parameters(),  # Just log_T
            lr=temp_lr,
            weight_decay=0.0  # No regularization for temperature
        )
    
    # Freeze layers if requested (for fine-tuning)
    if freeze_layers > 0:
        print()
        print("="*60)
        print("Freezing layers for fine-tuning")
        print("="*60)
        
        # Freeze input projection
        for param in model.input_proj.parameters():
            param.requires_grad = False
        print("✓ Frozen: input_proj")
        
        # Freeze FiLM
        for param in model.film.parameters():
            param.requires_grad = False
        print("✓ Frozen: film (run-conditioned)")
        
        # Freeze specified dual-graph layers
        num_layers_to_freeze = min(freeze_layers, len(model.dual_layers))
        for i in range(num_layers_to_freeze):
            for param in model.dual_layers[i].parameters():
                param.requires_grad = False
            print(f"✓ Frozen: dual_layers[{i}]")
        
        print("✗ Trainable: output_proj")
        print("✗ Trainable: fusion")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = total_params - frozen_params
        
        print()
        print(f"Total: {total_params:,}, Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%), "
              f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
        print("="*60)
        print()
    
    # Learning rate scheduler for adaptive learning rate (gentler)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.7,        # Gentler reduction (was 0.5)
        patience=20,       # More patience (was 10)
        min_lr=1e-4,       # Don't go too low
        verbose=True
    )
    
    # Early stopping
    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    
    print("="*60)
    print("Starting Training...")
    print("="*60)
    
    model.train()
    
    # Memory-efficient: subgraph sampling + run subsampling
    # Strategy: Sample subgraphs per run instead of processing full graph
    print(f"  Memory mode: Subgraph sampling (NeighborLoader)")
    
    # Get subgraph sampling parameters (from TrainConfig or DualViewConfig)
    batch_size_nodes = getattr(cfg, 'batch_size_nodes', 4096)
    fanouts = getattr(cfg, 'fanouts', (15, 10, 5))
    if len(fanouts) < cfg.num_layers:
        # Pad fanouts to match num_layers
        fanouts = list(fanouts) + [fanouts[-1]] * (cfg.num_layers - len(fanouts))
        fanouts = tuple(fanouts[:cfg.num_layers])
    
    print(f"  Batch size (nodes): {batch_size_nodes}")
    print(f"  Fanouts: {fanouts} (for {cfg.num_layers} layers)")
    
    # Create PyG Data objects for NeighborLoader
    # PREBUILD: Attach edge weights directly like gnn_v1 for automatic extraction!
    print("Creating PyG Data objects for subgraph sampling...")
    data_objects = []
    for run_data in data_per_run:
        # Attach edge_weights_G_full to edge_index_G so NeighborLoader extracts it automatically!
        # This eliminates Python loop lookup!
        pyg_data = Data(
            x=run_data['x'],
            edge_index=run_data['edge_index_G'],
            edge_attr=run_data['edge_attr_G'],
            edge_weight_G=run_data['edge_weight_G'],  # Scalar weight for model
            edge_weights=run_data['edge_weights_G_full'],  # [E_G, 2] for sampling - NeighborLoader extracts automatically!
            # Store co-location graph info for subgraph extraction
            edge_index_H=run_data['edge_index_H'].cpu(),  # Keep on CPU until needed
            edge_attr_H=run_data['edge_attr_H'].cpu(),
            edge_weight_H=run_data['edge_weight_H'].cpu(),
        ).to(device)
        data_objects.append(pyg_data)
    print(f"  Created {len(data_objects)} Data objects")
    print("  Edge weights pre-attached for automatic extraction (like gnn_v1)")
    
    # Track training history for plotting
    train_losses = []
    train_epochs = []
    
    # Run subsampling: process only subset of runs per epoch
    # Use config-defined runs_per_epoch, or auto-calculate if None
    if hasattr(cfg, 'runs_per_epoch') and cfg.runs_per_epoch is not None:
        runs_per_epoch = cfg.runs_per_epoch
    else:
        # Default: min(3, num_runs) if num_runs >= 5 else num_runs
        runs_per_epoch = min(3, num_runs) if num_runs >= 5 else num_runs
    runs_per_epoch = min(runs_per_epoch, num_runs)  # Ensure it doesn't exceed total runs
    print(f"  Run subsampling: {runs_per_epoch} runs per epoch (out of {num_runs} total)")
    
    for epoch in range(cfg.epochs):
        # Clear cache at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ========== Temperature Warmup and Freeze ==========
        # Freeze temperature for first 5 epochs, then after epoch 100
        if temp_optimizer is not None:
            if epoch < 5:
                # Warmup: Freeze temperature
                for param in criterion.parameters():
                    param.requires_grad = False
            elif epoch >= 100:
                # After epoch 100: Freeze to prevent hitting minimum
                # InfoNCE is good enough by then, prevent further sharpening
                for param in criterion.parameters():
                    param.requires_grad = False
            else:
                # Epochs 5-100: Allow temperature to learn
                for param in criterion.parameters():
                    param.requires_grad = True
        
        # ========== Temperature-Scaled Push Weight (DELAYED ACTIVATION) ==========
        # As temperature drops, InfoNCE gradients grow exponentially
        # Scale push weight inversely to maintain balance
        # BUT: Only apply after epoch 50 to allow loss to decrease initially
        if use_enhanced_loss and hasattr(criterion, '_get_T') and hasattr(criterion, 'lambda_push'):
            base_push_weight = getattr(cfg, 'lambda_len_push', 3.0)
            
            if epoch >= 50:
                # After epoch 50: Apply temperature scaling to prevent collapse
                T_current = criterion._get_T().item()
                T_init = getattr(cfg, 'temperature_init', 0.8)
                
                # Temperature scale factor (inverse relationship)
                # Cap at 5x to prevent extreme values
                temp_scale = min(T_init / max(T_current, 0.15), 5.0)
                
                # Apply scaled push weight
                criterion.lambda_push = base_push_weight * temp_scale
                
                # Log scaling info every 20 epochs
                if epoch % 20 == 0 or epoch == 50:
                    print(f"  [Dynamic Scaling] T: {T_current:.3f}, Push weight: {criterion.lambda_push:.2f} (base: {base_push_weight}, scale: {temp_scale:.2f}x)")
            else:
                # Before epoch 50: Use fixed weight for normal loss decrease
                criterion.lambda_push = base_push_weight
                if epoch == 0 or epoch == 5:
                    print(f"  [Fixed Push Weight] λ_push: {base_push_weight} (scaling starts at epoch 50)")
        
        # ========== Lambda Annealing for Calibration Loss (IMPROVED) ==========
        if use_enhanced_loss and hasattr(criterion, 'lambda_coloc'):
            # Earlier and gentler annealing
            # Epoch 0-50: Stay low (0.15) - let InfoNCE and push learn
            # Epoch 50-150: Gradual ramp (0.15 → 1.0) - add calibration
            # Epoch 150+: Full weight (1.0) - all losses balanced
            base_lambda = getattr(cfg, 'lambda_coloc', 1.0)  # Balanced for temperature-scaled push
            if epoch < 50:
                criterion.lambda_coloc = base_lambda * 0.3
            elif epoch < 150:
                progress = (epoch - 50) / 100
                criterion.lambda_coloc = base_lambda * (0.3 + 0.7 * progress)
            else:
                criterion.lambda_coloc = base_lambda * 1.0
        
        # Sample subset of runs for this epoch
        selected_run_indices = torch.randperm(num_runs)[:runs_per_epoch].tolist()
        selected_runs_quality = run_quality[selected_run_indices]
        
        # Process selected runs with subgraph sampling
        total_loss = 0.0
        num_batches = 0
        
        # Track loss components (for enhanced loss)
        if use_enhanced_loss:
            total_info_nce = 0.0
            total_len_push = 0.0
            total_calibration = 0.0
            total_normalized_loss = 0.0  # Track normalized loss (fixed weights) for monitoring
        
        for run_idx in selected_run_indices:
            run_pyg_data = data_objects[run_idx]
            
            # NeighborLoader automatically extracts batch.edge_weights from Data.edge_weights
            
            # Create NeighborLoader for this run (subgraph sampling)
            # Use simple loader as fallback (pyg-lib/torch-sparse have GLIBC issues on this system)
            # try:
            train_loader = NeighborLoader(
                run_pyg_data,
                num_neighbors=list(fanouts),
                batch_size=batch_size_nodes,
                shuffle=True
            )
            # except (ImportError, RuntimeError, Exception) as e:
            #     # Fallback to simple loader if NeighborLoader fails (e.g., missing pyg-lib)
            #     print(f"  Using simple neighbor loader (NeighborLoader unavailable: {type(e).__name__})")
            #     train_loader = create_simple_neighbor_loader(
            #         run_pyg_data,
            #         num_neighbors=list(fanouts),
            #         batch_size=batch_size_nodes,
            #         shuffle=True
            #     )
            
            # Process subgraphs for this run
            for batch in train_loader:
                batch = batch.to(device)
                
                # Extract subgraph nodes
                batch_n_id_full = batch.n_id  # Keep on GPU
                batch_nodes_center = batch.n_id[:batch.batch_size]  # Center nodes only
                num_subgraph_nodes = batch.num_nodes  # Total nodes in subgraph
                
                # OPTIMIZED: Co-location subgraph extraction (still needed because NeighborLoader doesn't handle edge_index_H)
                # But we can cache tensors and move co-location graphs to GPU once per run
                # Reuse boolean mask buffer to avoid allocation
                if not hasattr(run_pyg_data, '_nodes_mask_buffer'):
                    run_pyg_data._nodes_mask_buffer = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                    run_pyg_data._node_id_to_local_buffer = torch.zeros(num_nodes, dtype=torch.long, device=device)
                    # Move co-location graphs to GPU once per run (not every batch!)
                    run_pyg_data.edge_index_H = run_pyg_data.edge_index_H.to(device)
                    run_pyg_data.edge_attr_H = run_pyg_data.edge_attr_H.to(device)
                    run_pyg_data.edge_weight_H = run_pyg_data.edge_weight_H.to(device)
                
                # Reuse buffers (no allocation!)
                nodes_in_subgraph = run_pyg_data._nodes_mask_buffer
                nodes_in_subgraph.zero_()  # Clear
                nodes_in_subgraph[batch_n_id_full] = True  # Set nodes in batch
                
                # Vectorized filtering: check if both src and dst are in subgraph
                edge_mask_H = nodes_in_subgraph[run_pyg_data.edge_index_H[0]] & nodes_in_subgraph[run_pyg_data.edge_index_H[1]]
                
                if edge_mask_H.sum() > 0:
                    edge_index_H_batch = run_pyg_data.edge_index_H[:, edge_mask_H]
                    edge_attr_H_batch = run_pyg_data.edge_attr_H[edge_mask_H]
                    edge_weight_H_batch = run_pyg_data.edge_weight_H[edge_mask_H]
                    
                    # Remap to local indices (reuse buffer)
                    node_id_to_local = run_pyg_data._node_id_to_local_buffer
                    node_id_to_local.zero_()  # Clear
                    node_id_to_local[batch_n_id_full] = torch.arange(num_subgraph_nodes, device=device)
                    
                    edge_index_H_batch_remapped = torch.stack([
                        node_id_to_local[edge_index_H_batch[0]],
                        node_id_to_local[edge_index_H_batch[1]]
                    ], dim=0)
                else:
                    # Empty co-location graph (rare)
                    edge_index_H_batch_remapped = torch.zeros((2, 0), dtype=torch.long, device=device)
                    edge_attr_H_batch = torch.zeros((0, 1), dtype=torch.float, device=device)
                    edge_weight_H_batch = torch.zeros((0,), dtype=torch.float, device=device)
                
                # OPTIMIZED: Direct access to edge weights (like gnn_v1)!
                # NeighborLoader automatically extracted batch.edge_weights from Data.edge_weights
                # No Python loop needed - direct access!
                if hasattr(batch, 'edge_weights') and batch.edge_weights is not None:
                    edge_weights_batch = batch.edge_weights  # [E_batch, 2] - already extracted!
                else:
                    # Fallback: create uniform weights (shouldn't happen)
                    edge_weights_batch = torch.full((batch.edge_index.size(1), 2), 0.5, dtype=torch.float32, device=device)
                
                # Forward pass on subgraph
                # Note: batch.x contains features for ALL subgraph nodes (center + neighbors)
                # We only need embeddings for center nodes (batch.batch_size)
                z_subgraph = model.forward_single_run(
                    x=batch.x,
                    run_id=run_idx,
                    edge_index_G=batch.edge_index,
                    edge_attr_G=batch.edge_attr,
                    edge_weight_G=batch.edge_weight_G,
                    edge_index_H=edge_index_H_batch_remapped,
                    edge_attr_H=edge_attr_H_batch,
                    edge_weight_H=edge_weight_H_batch,
                    device=device
                )  # [num_subgraph_nodes, output_dim]
                
                # Extract embeddings only for center nodes (for loss computation)
                z_batch = z_subgraph[:batch.batch_size]  # [batch_size, output_dim]
                
                # Filter edges to only those between center nodes for loss computation
                # batch.edge_index contains edges in the full subgraph (center + neighbors)
                # We only want edges between center nodes (indices < batch.batch_size)
                center_node_mask = (batch.edge_index[0] < batch.batch_size) & (batch.edge_index[1] < batch.batch_size)
                
                if center_node_mask.sum() == 0:
                    # Skip batch if no edges between center nodes (should be rare)
                    continue
                
                edge_index_center = batch.edge_index[:, center_node_mask]  # [2, E_center]
                # Filter edge weights to match center edges (NeighborLoader extracts all edges, we filter to center)
                edge_weights_center = edge_weights_batch[center_node_mask]  # [E_center, 2]
                
                # Sample pairs from center edges only
                n_pos = max(256, batch.batch_size // 5)
                
                # Use per-anchor mode for enhanced loss (better performance)
                # The new loss function supports both modes based on which parameters are passed
                if use_enhanced_loss:
                    # ========== Per-Anchor Sampling ==========
                    num_negatives_per_anchor = 100  # K negatives per anchor
                    
                    anchors, positives, negatives, anchor_pos_weights, _ = sample_pairs_per_anchor(
                        edge_index_center,
                        edge_weights_center,  # [E_center, 2]
                        n_pos,
                        num_negatives_per_anchor
                    )
                    
                    # Convert to new loss function format
                    # pos_idx: [A, P] where P=1 (one positive per anchor)
                    pos_idx = positives.unsqueeze(1)  # [A] -> [A, 1]
                    pos_mask = torch.ones_like(pos_idx, dtype=torch.bool)  # All valid
                    neg_idx = negatives  # [A, K] already in correct format
                    
                    # Compute co-location consistency for anchor-positive pairs
                    anchor_pos_pairs = torch.stack([anchors, positives], dim=1)  # [A, 2]
                    coloc_pos = compute_colocation_consistency(
                        anchor_pos_pairs,
                        coordinates_all_runs,
                        run_quality,
                        sigmas_all_runs
                    )  # [A] - one score per anchor
                    
                    # Criticality weights for positives: [A, 1]
                    crit_pos = anchor_pos_weights.unsqueeze(1) / (anchor_pos_weights.mean() + 1e-9)
                    
                    # Compute per-anchor InfoNCE loss with NEW signature
                    loss_dict = criterion(
                        z_batch,
                        anchors=anchors,          # [A]
                        pos_idx=pos_idx,          # [A, 1] 
                        pos_mask=pos_mask,        # [A, 1] bool
                        neg_idx=neg_idx,          # [A, K]
                        w_anchor=anchor_pos_weights,  # [A]
                        coloc_pos=coloc_pos,      # [A] (will use best positive)
                        crit_pos=crit_pos         # [A, 1]
                    )
                    loss_batch = loss_dict['total']
                    
                    # Track components
                    total_info_nce += loss_dict['info_nce'].item()
                    total_len_push += loss_dict['len_push'].item()
                    total_calibration += loss_dict['calibration'].item()
                    
                    # Compute normalized loss (fixed reference weights for monitoring)
                    # This shows true progress independent of dynamic weight changes
                    normalized_loss_batch = (
                        loss_dict['info_nce'] + 
                        3.0 * loss_dict['len_push'] +      # Fixed reference weight
                        0.5 * loss_dict['calibration']     # Fixed reference weight
                    )
                    total_normalized_loss += normalized_loss_batch.item()
                    
                else:
                    # ========== Legacy Mode (WeightedInfoNCEWithLenPush) ==========
                    # Used only when coordinates_all_runs is not available
                    n_neg = max(256, batch.batch_size // 5)
                    
                    pos_pairs, neg_pairs, pos_w, neg_w = sample_pairs(
                        edge_index_center,
                        edge_weights_center,  # [E_center, 2]
                        n_pos,
                        n_neg
                    )
                    
                    # Legacy loss (no calibration)
                    loss_batch = criterion(z_batch, pos_pairs, neg_pairs, pos_w, neg_w)
                    
                    # Normalize loss by number of pairs for better convergence (optional but recommended)
                    if getattr(cfg, 'normalize_loss', False):
                        num_pairs = len(pos_pairs) + len(neg_pairs)
                        if num_pairs > 0:
                            loss_batch = loss_batch / num_pairs
                
                # Accumulate loss
                total_loss += loss_batch.item()
                num_batches += 1
                
                # Backward pass
                optimizer.zero_grad(set_to_none=True)
                if temp_optimizer is not None:
                    temp_optimizer.zero_grad(set_to_none=True)
                
                loss_batch.backward()
                
                optimizer.step()
                if temp_optimizer is not None:
                    temp_optimizer.step()
                
                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Average losses over all batches
        avg_loss = total_loss / max(num_batches, 1)  # Weighted loss (for logging)
        
        # Compute normalized loss (fixed weights) for monitoring
        if use_enhanced_loss and num_batches > 0:
            avg_normalized_loss = total_normalized_loss / num_batches
        else:
            avg_normalized_loss = avg_loss  # Fallback
        
        # Track normalized loss for training history (this is what we monitor!)
        train_losses.append(avg_normalized_loss)
        train_epochs.append(epoch + 1)
        
        # Print progress
        if use_enhanced_loss and num_batches > 0:
            avg_info_nce = total_info_nce / num_batches
            avg_len_push = total_len_push / num_batches
            avg_calibration = total_calibration / num_batches
            
            # Get temperature if learnable
            current_temp = None
            if hasattr(criterion, '_get_T'):
                current_temp = criterion._get_T().item()
            
            # Get current lambda_coloc if annealed
            current_lambda_coloc = getattr(criterion, 'lambda_coloc', None)
            
            # Build log message
            # Show both weighted (optimization) and normalized (monitoring) losses
            log_msg = f"  Epoch [{epoch+1}/{cfg.epochs}]  Loss: {avg_normalized_loss:.4f}"
            
            # Show weighted loss if different from normalized (when weights are dynamic)
            if abs(avg_loss - avg_normalized_loss) > 0.1:
                log_msg += f" (weighted: {avg_loss:.4f})"
            
            log_msg += f"  [InfoNCE: {avg_info_nce:.4f}, Push: {avg_len_push:.4f}, Calib: {avg_calibration:.4f}]"
            
            if current_temp is not None:
                log_msg += f"  T: {current_temp:.3f}"
            
            # Show dynamic push weight if scaling is active
            if hasattr(criterion, 'lambda_push'):
                current_lambda_push = criterion.lambda_push
                if current_lambda_push > 3.5:  # Show if significantly scaled
                    log_msg += f"  λ_push: {current_lambda_push:.1f}"
            
            if current_lambda_coloc is not None and epoch < 210:  # Show lambda during annealing
                log_msg += f"  λ_cal: {current_lambda_coloc:.2f}"
            
            log_msg += f"  (batches: {num_batches}, runs: {runs_per_epoch})"
            print(log_msg)
        else:
            print(f"  Epoch [{epoch+1}/{cfg.epochs}]  Loss: {avg_loss:.4f}  (batches: {num_batches}, runs: {runs_per_epoch})")
        
        # Learning rate scheduling - use NORMALIZED loss (shows true progress)
        scheduler.step(avg_normalized_loss)
        
        # Early stopping - use NORMALIZED loss (selects best embeddings)
        improved = early.step(avg_normalized_loss, epoch + 1)
        
        # Save best model (based on normalized loss)
        if improved and best_model_path is not None:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_normalized_loss,  # Save normalized (true quality metric)
                'weighted_loss': avg_loss,     # Also save weighted for reference
                'config': cfg,
                'run_quality': run_quality.cpu()
            }
            # Save criterion state if it has learnable parameters
            if use_enhanced_loss and hasattr(criterion, 'state_dict'):
                checkpoint['criterion_state_dict'] = criterion.state_dict()
            torch.save(checkpoint, best_model_path)
            print(f"  → Best model saved (loss: {avg_normalized_loss:.4f})")
        
        if early.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best model from epoch {early.best_epoch} (loss={early.best:.4f})")
            break
    
    # Load best model if checkpoint exists
    if best_model_path is not None and os.path.exists(best_model_path):
        print("\nLoading best model checkpoint...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model from epoch {checkpoint['epoch']} (loss: {checkpoint['loss']:.4f})")
    
    # Save training curve
    if output_dir is not None and len(train_losses) > 0:
        if HAS_MATPLOTLIB:
            curve_path = os.path.join(output_dir, 'training_curve.png')
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(train_epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
                if early.best_epoch > 0:
                    plt.axvline(x=early.best_epoch, color='r', linestyle='--', 
                               linewidth=1, label=f'Best Epoch {early.best_epoch}')
                    plt.axhline(y=early.best, color='r', linestyle='--', 
                               linewidth=1, alpha=0.5)
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel('Loss', fontsize=12)
                plt.title('Dual-View GNN Training Curve', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=10)
                plt.tight_layout()
                plt.savefig(curve_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\nTraining curve saved: {curve_path}")
            except Exception as e:
                print(f"Warning: Could not save training curve: {e}")
        else:
            # Save training data as text if matplotlib not available
            curve_data_path = os.path.join(output_dir, 'training_curve.txt')
            try:
                with open(curve_data_path, 'w') as f:
                    f.write("Epoch\tLoss\n")
                    for epoch, loss in zip(train_epochs, train_losses):
                        f.write(f"{epoch}\t{loss:.6f}\n")
                print(f"\nTraining curve data saved (text): {curve_data_path}")
                print("Note: Install matplotlib to generate PNG plot")
            except Exception as e:
                print(f"Warning: Could not save training curve data: {e}")
    
    print("="*60)
    print("Training Complete!")
    print("="*60)
    
    # Generate final embeddings using subgraph sampling
    embeddings = generate_dualview_embeddings(
        model, 
        data_per_run, 
        run_quality, 
        device,
        cfg
    )
    
    # Save final model
    if output_dir is not None:
        final_model_path = os.path.join(output_dir, 'final_model_tomognn.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'num_nodes': num_nodes,
            'num_runs': num_runs,
            'input_dim': in_dim,
            'best_loss': early.best,
            'best_epoch': early.best_epoch,
            'run_quality': run_quality.cpu()
        }, final_model_path)
        print(f"\nFinal model saved: {final_model_path}")
    
    return model, embeddings


def generate_dualview_embeddings(
    model: TomoGNN,
    data_per_run: list,
    run_quality: torch.Tensor,
    device: torch.device,
    cfg: TrainConfig | DualViewConfig
) -> np.ndarray:
    """
    Generate final embeddings from trained model using subgraph sampling.
    
    Args:
        model: Trained DualViewGNN
        data_per_run: List of run data
        run_quality: Run quality scores
        device: Torch device
        cfg: Training config (for fanouts and batch_size)
        
    Returns:
        embeddings: [N, output_dim] numpy array
    """
    print("Generating embeddings using subgraph sampling...")
    
    model.eval()
    num_nodes = data_per_run[0]['x'].size(0)
    num_runs = len(data_per_run)
    output_dim = model.output_dim
    
    # Get subgraph sampling parameters
    batch_size_nodes = getattr(cfg, 'batch_size_nodes', 4096)
    
    # Initialize embedding storage
    embeddings_all_runs = torch.zeros(num_runs, num_nodes, output_dim, device=device)
    
    with torch.no_grad():
        # Process each run with full neighborhood sampling for inference
        for run_idx, run_data in enumerate(data_per_run):
            # Create PyG Data for this run
            pyg_data = Data(
                x=run_data['x'],
                edge_index=run_data['edge_index_G'],
                edge_attr=run_data['edge_attr_G'],
                edge_weight_G=run_data['edge_weight_G'],
                edge_weights_G_full=run_data['edge_weights_G_full'],
                edge_index_H=run_data['edge_index_H'].cpu(),
                edge_attr_H=run_data['edge_attr_H'].cpu(),
                edge_weight_H=run_data['edge_weight_H'].cpu(),
            ).to(device)
            
            # Use full neighborhood for inference (like gnn_v1)
            # try:
            inference_loader = NeighborLoader(
                pyg_data,
                num_neighbors=[-1] * cfg.num_layers,  # Full neighborhood
                batch_size=batch_size_nodes,
                shuffle=False
            )
            # except (ImportError, RuntimeError, Exception) as e:
            #     # Fallback to simple loader if NeighborLoader fails
            #     inference_loader = create_simple_neighbor_loader(
            #         pyg_data,
            #         num_neighbors=[-1] * cfg.num_layers,  # Full neighborhood
            #         batch_size=batch_size_nodes,
            #         shuffle=False
            #     )
            
            # Accumulate embeddings batch by batch
            run_embeddings = torch.zeros(num_nodes, output_dim, device=device)
            batch_counts = torch.zeros(num_nodes, device=device)
            
            for batch in inference_loader:
                batch = batch.to(device)
                # Use ALL subgraph nodes (center + neighbors) for proper message passing
                batch_n_id_full = batch.n_id.cpu()  # All nodes in subgraph
                batch_nodes_center = batch.n_id[:batch.batch_size].cpu()  # Center nodes only (for storage)
                num_subgraph_nodes = batch.num_nodes
                
                # Extract co-location subgraph (same logic as training)
                edge_index_H_full = pyg_data.edge_index_H.cpu()
                edge_attr_H_full = pyg_data.edge_attr_H.cpu()
                edge_weight_H_full = pyg_data.edge_weight_H.cpu()
                
                # Create mapping for ALL subgraph nodes
                node_id_map = torch.full((num_nodes,), -1, dtype=torch.long)
                node_id_map[batch_n_id_full] = torch.arange(num_subgraph_nodes)
                
                src_mapped = node_id_map[edge_index_H_full[0]]
                dst_mapped = node_id_map[edge_index_H_full[1]]
                edge_mask = (src_mapped >= 0) & (dst_mapped >= 0)
                
                if edge_mask.sum() > 0:
                    edge_index_H_batch = torch.stack([
                        src_mapped[edge_mask],
                        dst_mapped[edge_mask]
                    ], dim=0).to(device)
                    edge_attr_H_batch = edge_attr_H_full[edge_mask].to(device)
                    edge_weight_H_batch = edge_weight_H_full[edge_mask].to(device)
                else:
                    edge_index_H_batch = torch.zeros((2, 0), dtype=torch.long, device=device)
                    edge_attr_H_batch = torch.zeros((0, 1), dtype=torch.float, device=device)
                    edge_weight_H_batch = torch.zeros((0,), dtype=torch.float, device=device)
                
                # Forward pass
                z_batch = model.forward_single_run(
                    x=batch.x,
                    run_id=run_idx,
                    edge_index_G=batch.edge_index,
                    edge_attr_G=batch.edge_attr,
                    edge_weight_G=batch.edge_weight_G,
                    edge_index_H=edge_index_H_batch,
                    edge_attr_H=edge_attr_H_batch,
                    edge_weight_H=edge_weight_H_batch,
                    device=device
                )
                
                # Store embeddings (only for batch center nodes)
                # z_batch contains embeddings for ALL subgraph nodes
                run_embeddings[batch_nodes_center] = z_batch[:batch.batch_size]
                batch_counts[batch_nodes_center] += 1
            
            # Average embeddings if nodes appeared in multiple batches
            # Need to unsqueeze batch_counts to broadcast properly: [N] -> [N, 1]
            run_embeddings = run_embeddings / (batch_counts.unsqueeze(-1) + 1e-9)
            embeddings_all_runs[run_idx] = run_embeddings
        
        # Quality-aware fusion across runs
        z_fused = model.fusion(embeddings_all_runs, run_quality)
    
    embeddings_np = z_fused.cpu().numpy()
    
    print(f"  Embeddings shape: {embeddings_np.shape}")
    print(f"  Range: [{embeddings_np.min():.3f}, {embeddings_np.max():.3f}]")
    print(f"  Mean norm: {np.linalg.norm(embeddings_np, axis=1).mean():.3f}")
    
    return embeddings_np


def load_dualview_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device = None
) -> tuple[TomoGNN, dict]:
    """
    Load a trained dual-view model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        
    Returns:
        model: Loaded model
        info: Checkpoint info dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading TomoGNN model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config and info
    cfg = checkpoint.get('config')
    input_dim = checkpoint.get('input_dim')
    num_runs = checkpoint.get('num_runs')
    
    if cfg is None or input_dim is None or num_runs is None:
        raise ValueError("Checkpoint missing required fields")
    
    # Create model
    model = TomoGNN(
        in_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        num_layers=cfg.num_layers,
        num_runs=num_runs,
        edge_feat_dim_G=2,
        edge_feat_dim_H=1,
        dropout=cfg.dropout
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare info
    info = {
        'best_loss': checkpoint.get('best_loss'),
        'best_epoch': checkpoint.get('best_epoch'),
        'num_nodes': checkpoint.get('num_nodes'),
        'num_runs': num_runs,
        'config': cfg,
        'run_quality': checkpoint.get('run_quality')
    }
    
    print("Model loaded successfully!")
    print(f"  Best epoch: {info['best_epoch']}")
    print(f"  Best loss: {info['best_loss']:.4f}")
    print(f"  Architecture: {cfg.num_layers} layers, {cfg.hidden_dim}→{cfg.output_dim}")
    
    return model, info

