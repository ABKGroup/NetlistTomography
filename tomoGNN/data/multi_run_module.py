# multi_run_module.py
# Learnable and naive multi-run weight aggregation
from __future__ import annotations
import torch
from torch import nn
import numpy as np
from typing import Tuple, Dict, Union
from config import DualViewConfig, TrainConfig


class MultiRunWeightingModule(nn.Module):
    """
    Multi-run weight aggregation module.

    Supports two modes:
    1. Learnable: Learn weights via softmax(alpha), softmax(beta)
    2. Naive: Use equal weights (1/N for each run)

    The module aggregates omega_plus and omega_minus across runs:
        omega_plus = Σ(w_ai * f_i * g_i)
        omega_minus = Σ(w_ri * (1-f_i) * (1-g_i))

    where f_i = c_i^p_crit, g_i = exp(-k_len * L_i)
    """

    def __init__(self, cfg: Union[TrainConfig, DualViewConfig]):
        super().__init__()
        self.num_runs = cfg.num_runs
        self.p_crit = cfg.p_crit
        self.k_len = cfg.k_len
        self.naive = cfg.naive_weights

        if not self.naive:
            # Learnable weights (trainable parameters)
            self.alpha = nn.Parameter(
                torch.randn(self.num_runs)
            )
            self.beta = nn.Parameter(
                torch.randn(self.num_runs)
            )
            print(f"Multi-run weighting: LEARNABLE mode "
                  f"({self.num_runs} runs)")
        else:
            # Naive mode: equal weights (not trainable)
            equal_weight = 1.0 / self.num_runs
            self.register_buffer(
                'alpha',
                torch.full((self.num_runs,), equal_weight)
            )
            self.register_buffer(
                'beta',
                torch.full((self.num_runs,), equal_weight)
            )
            print(f"Multi-run weighting: NAIVE mode "
                  f"(equal weights, {self.num_runs} runs)")

    def forward(
        self,
        criticality_runs: torch.Tensor,
        length_runs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate weights across runs.

        Args:
            criticality_runs: [num_runs, num_edges]
            length_runs: [num_runs, num_edges]

        Returns:
            Tuple of (omega_plus_base, omega_minus_base)
                Each shape: [num_edges]
        """
        # Get weights
        if self.naive:
            # Equal weights
            w_ai = self.alpha  # [num_runs]
            w_ri = self.beta   # [num_runs]
        else:
            # Learnable weights (softmax normalization)
            w_ai = torch.softmax(self.alpha, dim=0)
            w_ri = torch.softmax(self.beta, dim=0)

        # Compute f and g for each run
        # f = c^p_crit, g = exp(-k_len * L)
        f_runs = torch.pow(
            criticality_runs,
            self.p_crit
        )  # [num_runs, num_edges]

        g_runs = torch.exp(
            -self.k_len * length_runs
        )  # [num_runs, num_edges]

        # Weighted aggregation
        # omega_plus = Σ(w_ai * f_i * g_i)
        # High for: critical + short edges
        # Purpose: Attraction weight - pull these edges together
        omega_plus_base = torch.sum(
            w_ai.unsqueeze(1) * f_runs * g_runs,
            dim=0
        )  # [num_edges]

        # omega_minus = Σ(w_ri * (1-f_i) * (1-g_i))
        # High for: non-critical + long edges
        # Purpose: Repulsion weight - push these edges apart to make room for critical paths
        omega_minus_base = torch.sum(
            w_ri.unsqueeze(1) * (1.0 - f_runs) * (1.0 - g_runs),
            dim=0
        )  # [num_edges]

        return omega_plus_base, omega_minus_base

    def get_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get current weights (for inspection/logging).

        Returns:
            Dict with 'attraction_weights' and 'repulsion_weights'
        """
        if self.naive:
            return {
                'attraction_weights': self.alpha.clone(),
                'repulsion_weights': self.beta.clone()
            }
        else:
            return {
                'attraction_weights': torch.softmax(
                    self.alpha, dim=0
                ),
                'repulsion_weights': torch.softmax(
                    self.beta, dim=0
                )
            }


def prepare_multi_run_tensors(
    runs_data: list,
    cfg: Union[TrainConfig, DualViewConfig]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert per-run data to tensors for weighting module.

    Args:
        runs_data: List of dicts with 'criticality', 'length'
        cfg: Training configuration

    Returns:
        Tuple of (criticality_runs, length_runs)
            Each shape: [num_runs, num_edges]
    """
    if len(runs_data) != cfg.num_runs:
        raise ValueError(
            f"Expected {cfg.num_runs} runs, "
            f"got {len(runs_data)}"
        )

    # Stack criticality and length from all runs
    criticality_runs = torch.stack([
        torch.from_numpy(run['criticality']).float()
        for run in runs_data
    ])  # [num_runs, num_edges]

    length_runs = torch.stack([
        torch.from_numpy(run['length']).float()
        for run in runs_data
    ])  # [num_runs, num_edges]

    return criticality_runs, length_runs
