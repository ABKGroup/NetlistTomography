# models_dualview.py
# Dual-View Multi-Run GNN: Run-conditioned FiLM + Dual-Graph Message Passing + Quality-Aware Fusion
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing, SAGEConv
from typing import List, Tuple


# ============================================================================
# 1. Run-Conditioned FiLM (Feature-wise Linear Modulation)
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation conditioned on run ID.
    
    For run r, learns affine transformation parameters:
        gamma_r, beta_r = FiLM_encoder(r)
        h' = gamma_r * h + beta_r
    """
    def __init__(self, hidden_dim: int, num_runs: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_runs = num_runs
        
        # Run embeddings (learnable)
        self.run_embeddings = nn.Embedding(num_runs, hidden_dim // 4)
        
        # Encoder: run_embedding -> (gamma, beta)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)  # Output: [gamma, beta]
        )
        
    def forward(self, h: torch.Tensor, run_id: int) -> torch.Tensor:
        """
        Args:
            h: [N, hidden_dim] node features
            run_id: scalar run index (0 to num_runs-1)
            
        Returns:
            h_modulated: [N, hidden_dim] FiLM-modulated features
        """
        # Get run embedding
        run_emb = self.run_embeddings(
            torch.tensor(run_id, device=h.device)
        )  # [hidden_dim // 4]
        
        # Encode to gamma, beta
        params = self.encoder(run_emb)  # [2 * hidden_dim]
        gamma, beta = torch.chunk(params, 2, dim=-1)  # Each [hidden_dim]
        
        # Apply FiLM
        h_modulated = gamma.unsqueeze(0) * h + beta.unsqueeze(0)
        
        return h_modulated


# ============================================================================
# 2. Dual-Graph Message Passing with GraphSAGE
# ============================================================================

class EdgeWeightedGraphSAGE(MessagePassing):
    """
    Memory-efficient GraphSAGE with edge weight support.
    
    GraphSAGE formula: h_i' = W_self * h_i + W_neigh * AGG({w_ij * h_j : j ∈ N(i)})
    
    Key advantage: Aggregates neighbors first (node-level), then transforms.
    Avoids creating edge-level tensors [E, hidden_dim].
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        edge_feat_dim: int = 0,  # Kept for compatibility
        aggr: str = "mean",
        normalize: bool = True
    ):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        # GraphSAGE: separate weights for self and neighbors
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        GraphSAGE forward pass (memory-efficient).
        
        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity
            edge_attr: [E, edge_feat_dim] edge features (not used, kept for compatibility)
            edge_weight: [E] optional edge weights (applied during aggregation)
            
        Returns:
            out: [N, out_channels] output node features
        """
        # Self-connection
        x_self = self.lin_self(x)  # [N, out_channels]
        
        # Neighbor aggregation: AGG first (node-level), then transform
        # This is memory-efficient: only creates [N, in_channels] during aggregation
        x_neigh_agg = self.propagate(
            edge_index, 
            x=x, 
            edge_weight=edge_weight
        )  # [N, in_channels] - aggregated neighbor features
        
        # Transform aggregated neighbors
        x_neigh = self.lin_neigh(x_neigh_agg)  # [N, out_channels]
        
        # Combine
        out = x_self + x_neigh
        
        # Normalize if requested
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor | None) -> torch.Tensor:
        """
        Compute messages from neighbors (memory-efficient).
        
        Key: Only processes [E, in_channels], not [E, out_channels].
        Aggregation happens first at node-level, then transformation.
        
        Args:
            x_j: [E, in_channels] neighbor node features
            edge_weight: [E] optional edge weights
            
        Returns:
            messages: [E, in_channels] (will be aggregated to [N, in_channels])
        """
        # Pass through neighbor features (with edge weights if provided)
        # No transformation here - that happens after aggregation!
        if edge_weight is not None:
            return x_j * edge_weight.unsqueeze(-1)  # [E, in_channels]
        else:
            return x_j  # [E, in_channels]


class DualGraphLayer(nn.Module):
    """
    Dual-graph message passing layer with GraphSAGE backbone.
    
    Processes both netlist graph G and co-location graph H using GraphSAGE,
    then fuses messages with learnable gating.
    
    Architecture (per layer):
    - Message passing on G: GraphSAGE(h, G)
    - Message passing on H: GraphSAGE(h, H)  
    - Fusion: m_fused = m_G + α * m_H
    - Residual + LayerNorm: h^(l+1) = LN(h^(l) + m_fused)
    """
    def __init__(
        self, 
        hidden_dim: int, 
        edge_feat_dim_G: int,
        edge_feat_dim_H: int
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GraphSAGE on netlist graph G (memory-efficient)
        self.sage_G = EdgeWeightedGraphSAGE(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_feat_dim=edge_feat_dim_G,
            aggr="mean",
            normalize=True
        )
        
        # GraphSAGE on co-location graph H (memory-efficient)
        self.sage_H = EdgeWeightedGraphSAGE(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_feat_dim=edge_feat_dim_H,
            aggr="mean",
            normalize=True
        )
        
        # Learnable gating parameter for H
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Layer normalization (applied after residual)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        h: torch.Tensor,
        edge_index_G: torch.Tensor,
        edge_attr_G: torch.Tensor,
        edge_weight_G: torch.Tensor | None,
        edge_index_H: torch.Tensor,
        edge_attr_H: torch.Tensor,
        edge_weight_H: torch.Tensor
    ) -> torch.Tensor:
        """
        Dual-graph message passing with GraphSAGE + residual + LayerNorm.
        
        Args:
            h: [N, hidden_dim] node features
            edge_index_G: [2, E_G] netlist edges
            edge_attr_G: [E_G, feat_dim_G] netlist edge features
            edge_weight_G: [E_G] optional netlist edge weights
            edge_index_H: [2, E_H] co-location edges
            edge_attr_H: [E_H, feat_dim_H] co-location edge features
            edge_weight_H: [E_H] co-location edge weights
            
        Returns:
            h_next: [N, hidden_dim] updated node features
        """
        # GraphSAGE message passing on netlist graph G
        m_G = self.sage_G(h, edge_index_G, edge_attr_G, edge_weight_G)
        
        # GraphSAGE message passing on co-location graph H
        m_H = self.sage_H(h, edge_index_H, edge_attr_H, edge_weight_H)
        
        # Pre-fusion normalization for balanced contribution
        # Both m_G and m_H are already L2-normalized by GraphSAGE
        # Additional LayerNorm ensures balanced scale before fusion
        m_G_norm = F.layer_norm(m_G, [self.hidden_dim])
        m_H_norm = F.layer_norm(m_H, [self.hidden_dim])
        
        # Gated fusion of normalized messages
        m_fused = m_G_norm + self.alpha * m_H_norm
        
        # Residual connection + LayerNorm
        # h^(l+1) = LN(h^(l) + m_fused)
        h_next = self.layer_norm(h + m_fused)
        
        return h_next


# ============================================================================
# 3. Quality-Aware View Fusion
# ============================================================================

class QualityAwareFusion(nn.Module):
    """
    Attention-based fusion of per-run embeddings weighted by run quality.
    
    a_i^(r) = softmax_r(q^T tanh(W z_i^(r)) + log(π_r))
    ẑ_i = Σ_r a_i^(r) * z_i^(r)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q = nn.Parameter(torch.randn(embed_dim))
        
    def forward(
        self, 
        z_per_run: torch.Tensor, 
        run_quality: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse per-run embeddings with quality-aware attention.
        
        Args:
            z_per_run: [num_runs, N, embed_dim] per-run node embeddings
            run_quality: [num_runs] PPA-derived quality scores (π_r)
            
        Returns:
            z_fused: [N, embed_dim] final node embeddings
        """
        num_runs, num_nodes, embed_dim = z_per_run.shape
        
        # Compute attention logits
        z_transformed = torch.tanh(self.W(z_per_run))  # [R, N, D]
        logits = torch.einsum('rnd,d->rn', z_transformed, self.q)  # [R, N]
        
        # Add log quality scores (broadcasts across nodes)
        log_quality = torch.log(run_quality + 1e-9).unsqueeze(1)  # [R, 1]
        logits = logits + log_quality  # [R, N]
        
        # Compute attention weights (softmax over runs)
        attention = F.softmax(logits, dim=0)  # [R, N]
        
        # Weighted sum of embeddings
        z_fused = torch.einsum('rn,rnd->nd', attention, z_per_run)  # [N, D]
        
        return z_fused


# ============================================================================
# 4. Complete Dual-View Model
# ============================================================================

class TomoGNN(nn.Module):
    """
    Complete dual-view multi-run GNN model.
    
    Pipeline:
    1. Input projection
    2. For each run:
        a. FiLM modulation
        b. K layers of dual-graph message passing
        c. Output per-run embedding
    3. Quality-aware fusion across runs
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 16,
        num_layers: int = 3,
        num_runs: int = 30,
        edge_feat_dim_G: int = 2,  # [slack, length] for netlist
        edge_feat_dim_H: int = 1,  # [exp(-distance^2)] for co-location
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_runs = num_runs
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # FiLM layers (per run)
        self.film = FiLMLayer(hidden_dim, num_runs)
        
        # Dual-graph message passing layers
        self.dual_layers = nn.ModuleList([
            DualGraphLayer(hidden_dim, edge_feat_dim_G, edge_feat_dim_H)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (per run)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Quality-aware fusion
        self.fusion = QualityAwareFusion(output_dim)
        
        print(f"TomoGNN initialized (GraphSAGE backbone):")
        print(f"  Input: {in_dim} -> Hidden: {hidden_dim} -> Output: {output_dim}")
        print(f"  Layers: {num_layers} (GraphSAGE with residual + LayerNorm)")
        print(f"  Runs: {num_runs}")
        print(f"  Edge features: G={edge_feat_dim_G}, H={edge_feat_dim_H}")
        
    def forward_single_run(
        self,
        x: torch.Tensor,
        run_id: int,
        edge_index_G: torch.Tensor,
        edge_attr_G: torch.Tensor,
        edge_weight_G: torch.Tensor | None,
        edge_index_H: torch.Tensor,
        edge_attr_H: torch.Tensor,
        edge_weight_H: torch.Tensor,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Forward pass for a single run.
        
        Args:
            x: [N, in_dim] input node features
            run_id: scalar run index
            edge_index_G: [2, E_G] netlist edges
            edge_attr_G: [E_G, 2] netlist edge features
            edge_weight_G: [E_G] optional netlist edge weights
            edge_index_H: [2, E_H] co-location edges for this run
            edge_attr_H: [E_H, 1] co-location edge features
            edge_weight_H: [E_H] co-location edge weights
            device: Device to use (if None, inferred from x)
            
        Returns:
            z: [N, output_dim] per-run node embeddings
        """
        # Get device from x
        if device is None:
            device = x.device
        
        # Move co-location graph to GPU if on CPU (memory-efficient loading)
        if edge_index_H.device != device:
            edge_index_H = edge_index_H.to(device)
        if edge_attr_H.device != device:
            edge_attr_H = edge_attr_H.to(device)
        if edge_weight_H.device != device:
            edge_weight_H = edge_weight_H.to(device)
        
        # Input projection
        h = self.input_proj(x)  # [N, hidden_dim]
        
        # Run-conditioned FiLM modulation
        h = self.film(h, run_id)
        
        # Dual-graph message passing
        for layer in self.dual_layers:
            h = layer(
                h,
                edge_index_G, edge_attr_G, edge_weight_G,
                edge_index_H, edge_attr_H, edge_weight_H
            )
            h = self.dropout(h)
        
        # Output projection
        z = self.output_proj(h)  # [N, output_dim]
        z = F.normalize(z, p=2, dim=-1)  # L2 normalize
        
        return z
    
    def forward(
        self,
        data_per_run: List[dict],
        run_quality: torch.Tensor,
        batch_size: int = 2  # Process 2 runs at a time (balance memory vs speed)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass across all runs (processed one at a time to save memory).
        
        Args:
            data_per_run: List of dicts, one per run, each containing:
                - x: [N, in_dim] input features
                - edge_index_G: [2, E_G] netlist edges
                - edge_attr_G: [E_G, 2] netlist edge features
                - edge_weight_G: [E_G] optional netlist edge weights
                - edge_index_H: [2, E_H] co-location edges
                - edge_attr_H: [E_H, 1] co-location edge features
                - edge_weight_H: [E_H] co-location edge weights
            run_quality: [num_runs] quality scores
            batch_size: Number of runs to process at once (default: 1 for memory efficiency)
            
        Returns:
            z_per_run: [num_runs, N, output_dim] per-run embeddings
            z_fused: [N, output_dim] final fused embeddings
        """
        num_runs = len(data_per_run)
        num_nodes = data_per_run[0]['x'].size(0)
        
        # Process runs in small batches to balance memory and speed
        # Co-location graphs are huge (~520K edges per run), batch_size=1-2 is safest
        z_per_run = []
        device = data_per_run[0]['x'].device
        
        for batch_start in range(0, num_runs, batch_size):
            batch_end = min(batch_start + batch_size, num_runs)
            batch_z = []
            
            # Process small batch of runs
            for run_id in range(batch_start, batch_end):
                data = data_per_run[run_id]
                
                # Process single run
                z = self.forward_single_run(
                    x=data['x'],
                    run_id=run_id,
                    edge_index_G=data['edge_index_G'],
                    edge_attr_G=data['edge_attr_G'],
                    edge_weight_G=data.get('edge_weight_G', None),
                    edge_index_H=data['edge_index_H'],
                    edge_attr_H=data['edge_attr_H'],
                    edge_weight_H=data['edge_weight_H'],
                    device=device
                )
                batch_z.append(z)
            
            # Stack batch embeddings
            batch_z_stack = torch.stack(batch_z, dim=0)  # [batch_size, N, D]
            z_per_run.append(batch_z_stack)
            
            # Clear cache after each batch to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batch embeddings
        z_per_run = torch.cat(z_per_run, dim=0)  # [R, N, D]
        
        # Quality-aware fusion
        z_fused = self.fusion(z_per_run, run_quality)  # [N, D]
        
        return z_per_run, z_fused

