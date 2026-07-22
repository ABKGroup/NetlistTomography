# losses.py
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

class WeightedInfoNCEWithLenPush(nn.Module):
    """
    LEGACY: Weighted InfoNCE with length push regularization.
    
    Kept for backward compatibility. Use EnhancedContrastiveLoss instead.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_len_push: float = 1.0
    ):
        super().__init__()
        self.T = temperature
        self.lambda_len = lambda_len_push
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        z: torch.Tensor,
        pos_pairs: torch.Tensor,
        neg_pairs: torch.Tensor,
        pos_weights: torch.Tensor,
        neg_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted contrastive loss.

        Args:
            z: [N, D] L2-normalized node embeddings
            pos_pairs: [P, 2] positive pair indices
            neg_pairs: [Q, 2] negative pair indices
            pos_weights: [P] omega_plus values
            neg_weights: [Q] omega_minus values

        Returns:
            Scalar loss value
        """
        eps = 1e-9

        # Positive similarities (cosine, since z is normalized)
        zi_pos = z[pos_pairs[:, 0]]
        zj_pos = z[pos_pairs[:, 1]]
        sim_pos = torch.sum(
            zi_pos * zj_pos, dim=-1
        )  # [P]

        # InfoNCE numerator
        numer = torch.exp(sim_pos / self.T)  # [P]

        # InfoNCE denominator
        pos_contrib = torch.sum(
            torch.exp(sim_pos / self.T)
        )

        if neg_pairs.numel() > 0:
            # Negative similarities
            zi_neg = z[neg_pairs[:, 0]]
            zj_neg = z[neg_pairs[:, 1]]
            sim_neg = torch.sum(
                zi_neg * zj_neg, dim=-1
            )  # [Q]

            neg_contrib = torch.sum(
                torch.exp(sim_neg / self.T)
            )
            denom = pos_contrib + neg_contrib
        else:
            sim_neg = torch.empty(0, device=z.device)
            denom = pos_contrib

        # Weighted InfoNCE loss
        log_prob = torch.log((numer + eps) / (denom + eps))
        info_nce = -torch.sum(pos_weights * log_prob)

        # Length push regularization
        if neg_pairs.numel() > 0:
            len_push = torch.sum(
                neg_weights * self.sigmoid(sim_neg)
            )
        else:
            len_push = torch.tensor(0.0, device=z.device)

        return info_nce + self.lambda_len * len_push

def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean() if x.numel() > 0 else x.new_tensor(0.0)


def _normalize_weights(w: torch.Tensor | None, mask: torch.Tensor | None = None) -> torch.Tensor | None:
    if w is None:
        return None
    if mask is not None:
        mask_f = mask.float()
        valid = mask_f.sum()
        valid_val = float(valid)
        if valid_val <= 0.0:
            return w
        w = w * mask_f
        denom = (w.abs().sum() / (valid + 1e-6)).clamp_min(1e-6)
        return w / denom
    else:
        mean_abs = w.abs().mean()
        mean_abs_val = float(mean_abs)
        if mean_abs_val <= 0.0:
            return w
        denom = mean_abs.clamp_min(1e-6)
        return w / denom


class EnhancedContrastiveLoss(nn.Module):
    """
    Enhanced contrastive loss with tomography calibration.
    
    Total = L_info + λ_push * L_push + λ_coloc * L_cal + λ_self * L_self  (L_self optional)
    
    Two usage modes:
      (A) Per-anchor (recommended):
          forward(
            z, anchors[A], pos_idx[A,P], pos_mask[A,P], neg_idx[A,K],
            w_anchor[A]=None,
            coloc_pos=None or [A] or [A,P],
            calib_neg_idx[A,M]=None,
            crit_pos[A,P]=None, crit_neg[A,M]=None,
            ...
          )
    
      (B) Global pairs (back-compat):
          forward(z, pos_pairs[P,2], neg_pairs[Q,2], pos_weights[P], neg_weights[Q], ...)
    
    Key features:
      - Multi-positive InfoNCE with logsumexp
      - Debiased denominator (optional, tau_plus)
      - Margin softplus push for negatives
      - Calibration on positives + selected hard negatives (with focal weighting)
      - Learnable temperature T
      - Optional run self-consistency regularizer (z_per_run[R,N,D], run_quality[R])
    """
    
    def __init__(
        self,
        # Temperatures
        temperature_init: float = 0.6,    # Increased from 0.5 for better initial exploration
        temperature_min: float = 0.07,
        temperature_max: float = 1.5,
        learn_temperature: bool = True,
        
        # Push term
        lambda_push: float = 1.5,
        push_margin: float = 0.10,    # Reduced from 0.15 (negatives should be <= -margin)
        push_tau: float = 0.5,        # softness of the margin hinge
        
        # Calibration
        lambda_coloc: float = 1.0,    # Can anneal this in the trainer (e.g., 1.5 -> 0.5)
        tau_calibration: float = 0.6,
        focal_gamma: float = 1.0,     # Reduced from 1.5: start conservative
        
        # Debiased contrastive (false negatives correction)
        use_debiased: bool = True,
        tau_plus: float = 0.08,       # Increased from 0.03: netlist graphs have ~5-10% false negatives
        
        # Self-consistency across runs (optional)
        lambda_self: float = 0.0,     # set >0.0 only if you pass z_per_run & run_quality
        
        # General
        normalize_loss: bool = True,
    ):
        super().__init__()
        self.normalize_loss = normalize_loss
        
        # Learnable temperature
        self.learn_temperature = learn_temperature
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        if learn_temperature:
            self.log_T = nn.Parameter(torch.tensor(math.log(temperature_init), dtype=torch.float))
        else:
            self.register_buffer("fixed_T", torch.tensor(temperature_init, dtype=torch.float))
        
        # Push
        self.lambda_push = lambda_push
        self.push_margin = push_margin
        self.push_tau = push_tau
        
        # Calibration
        self.lambda_coloc = lambda_coloc
        self.tau_c = tau_calibration
        self.focal_gamma = focal_gamma
        
        # Debiased
        self.use_debiased = use_debiased
        self.tau_plus = tau_plus
        
        # Self consistency
        self.lambda_self = lambda_self
    
    # ---------- helpers ----------
    
    def _get_T(self) -> torch.Tensor:
        if self.learn_temperature:
            T = self.log_T.exp()
            return T.clamp(min=self.temperature_min, max=self.temperature_max)
        else:
            return self.fixed_T
    
    @staticmethod
    def _gather(z: torch.Tensor, idx_2d: torch.Tensor) -> torch.Tensor:
        """Gather 2D indices: idx_2d [A, K] -> embeddings [A, K, D]."""
        return z[idx_2d]  # PyTorch supports advanced indexing of shape [A,K,D]
    
    # ---------- core pieces ----------
    
    def _info_nce_multi_positive(
        self,
        z: torch.Tensor,
        anchors: torch.Tensor,          # [A]
        pos_idx: torch.Tensor,          # [A, P]
        pos_mask: torch.Tensor,         # [A, P] bool
        neg_idx: torch.Tensor,          # [A, K]
        w_anchor: torch.Tensor | None,  # [A] or None
        T: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-positive NT-Xent with logsumexp and optional debiasing.
        
        Returns:
          L_info (scalar), sim_an [A,K] (for push), valid_mask [A] (anchors with >=1 pos)
        """
        A, P = pos_idx.shape
        K = neg_idx.shape[1]
        
        za = z[anchors]                     # [A,D]
        zp = self._gather(z, pos_idx)       # [A,P,D]
        zn = self._gather(z, neg_idx)       # [A,K,D]
        
        # similarities
        sim_ap = (za.unsqueeze(1) * zp).sum(dim=-1) / T         # [A,P]
        sim_an = (za.unsqueeze(1) * zn).sum(dim=-1) / T         # [A,K]
        
        # mask invalid positives to -inf so they don't contribute to logsumexp
        sim_ap_masked = sim_ap.masked_fill(~pos_mask, float("-inf"))
        
        # anchors with at least one valid positive
        valid_mask = pos_mask.any(dim=1)                         # [A]
        if valid_mask.sum() == 0:
            return z.new_tensor(0.0), sim_an, valid_mask
        
        # log-sum-exp over positives and negatives
        lp = torch.logsumexp(sim_ap_masked, dim=1)               # [A]
        ln = torch.logsumexp(sim_an, dim=1)                      # [A]
        
        # Debiased denominator: denom = lnexp( ln , lp + log(1 - tau_plus) )
        if self.use_debiased:
            log_one_minus_tau = math.log(max(1e-6, 1.0 - self.tau_plus))
            log_denom = torch.logaddexp(ln, lp + log_one_minus_tau)   # [A]
        else:
            # denom = log(exp(lp) + exp(ln)) = logaddexp(lp, ln)
            log_denom = torch.logaddexp(lp, ln)
        
        per_anchor = -(lp - log_denom)                           # [A]
        
        # weight and normalize
        if w_anchor is not None:
            w_anchor = _normalize_weights(w_anchor, mask=valid_mask)
            per_anchor = per_anchor * w_anchor
        
        if self.normalize_loss:
            L_info = per_anchor[valid_mask].mean()
        else:
            L_info = per_anchor[valid_mask].sum()
        
        return L_info, sim_an, valid_mask
    
    def _push_margin_softplus(
        self,
        sim_an: torch.Tensor,                    # [A,K]
        neg_weights: torch.Tensor | None = None  # [A,K] or None
    ) -> torch.Tensor:
        """Smooth hinge on negatives: penalize sim > -margin."""
        x = (sim_an * 1.0 + self.push_margin) / self.push_tau   # [A,K]
        val = F.softplus(x)                                     # ~max(0, x) smoothed
        if neg_weights is not None:
            neg_weights = _normalize_weights(neg_weights)
            val = val * neg_weights
        return _safe_mean(val)
    
    def _calibration_bce(
        self,
        z: torch.Tensor,
        anchors: torch.Tensor,                   # [A]
        pos_idx: torch.Tensor | None,            # [A,P] or None
        pos_mask: torch.Tensor | None,           # [A,P] bool or None
        coloc_pos: torch.Tensor | None,          # [A] or [A,P] in [0,1] or None
        # negatives for calibration:
        neg_idx_all: torch.Tensor | None,        # [A,K] (from sampling)
        calib_neg_idx: torch.Tensor | None,      # [A,M] (subset) or None -> will auto-pick hardest M
        crit_pos: torch.Tensor | None,           # [A,P] or None
        crit_neg: torch.Tensor | None,           # [A,M] or None
        M_calib: int = 16                        # Increased from 8: how many negs to calibrate per anchor (if auto)
    ) -> torch.Tensor:
        """BCE calibration on positives (targets = coloc) and selected hard negatives (targets = 0)."""
        eps = 1e-6
        A = anchors.shape[0]
        
        pairs_sim = []
        pairs_tgt = []
        pairs_w   = []
        
        # --- positives ---
        if pos_idx is not None and pos_mask is not None:
            za = z[anchors]                      # [A,D]
            zp = self._gather(z, pos_idx)        # [A,P,D]
            sim_ap = (za.unsqueeze(1) * zp).sum(dim=-1)         # [A,P]
            pred_ap = torch.sigmoid(sim_ap / self.tau_c)        # [A,P]
            if coloc_pos is None:
                # default target = 1 for all valid positives
                tgt_ap = torch.ones_like(pred_ap)
                mask = pos_mask
            else:
                if coloc_pos.dim() == 1:  # [A]
                    # broadcast to [A,P] but only evaluate on the *best* positive per anchor:
                    # choose the highest sim among valid positives
                    mask = pos_mask
                    sim_masked = sim_ap.masked_fill(~mask, float("-inf"))
                    best_pos = sim_masked.argmax(dim=1)                  # [A]
                    idx = torch.arange(A, device=z.device)
                    pred_best = pred_ap[idx, best_pos]                   # [A]
                    tgt_best  = coloc_pos.float()                        # [A]
                    # weights (criticality) if provided
                    if crit_pos is not None:
                        w_best = crit_pos[idx, best_pos].float()
                    else:
                        w_best = torch.ones_like(pred_best)
                    pairs_sim.append(pred_best)
                    pairs_tgt.append(tgt_best)
                    pairs_w.append(w_best)
                    # skip the multi-positive branch below
                    pred_ap = None
                else:
                    # coloc_pos [A,P] per positive
                    tgt_ap = coloc_pos * 1.0
                    mask = pos_mask
            
            if pred_ap is not None:
                # focal weights
                w = torch.ones_like(pred_ap)
                if crit_pos is not None:
                    w = w * crit_pos
                # apply mask
                w = w * mask.float()
                # BCE (masked)
                bce = F.binary_cross_entropy(
                    pred_ap.clamp(eps, 1 - eps),
                    (tgt_ap * mask).float(),
                    reduction='none'
                )
                pairs_sim.append(bce)     # we'll merge below with correct sign
                pairs_tgt.append(None)    # placeholder
                pairs_w.append(w)
        
        # --- negatives (hard selection) ---
        if neg_idx_all is not None:
            za = z[anchors]                      # [A,D]
            zn_all = self._gather(z, neg_idx_all)  # [A,K,D]
            sim_an_all = (za.unsqueeze(1) * zn_all).sum(dim=-1)          # [A,K]
            
            if calib_neg_idx is None:
                # take M hardest negatives by similarity
                K = sim_an_all.shape[1]
                M = min(M_calib, K)
                hard_idx = torch.topk(sim_an_all, k=M, dim=1).indices     # [A,M]
            else:
                hard_idx = calib_neg_idx
            
            idx_a = torch.arange(A, device=z.device).unsqueeze(1)         # [A,1]
            sim_an = sim_an_all[idx_a, hard_idx]                          # [A,M]
            pred_an = torch.sigmoid(sim_an / self.tau_c)                  # [A,M]
            tgt_an = torch.zeros_like(pred_an)                            # [A,M]
            if crit_neg is not None:
                w_an = crit_neg
            else:
                w_an = torch.ones_like(pred_an)
            
            # focal weighting
            focal_w = (pred_an - tgt_an).abs().pow(self.focal_gamma)
            w_an = w_an * focal_w
            
            # BCE for negatives
            bce_an = F.binary_cross_entropy(
                pred_an.clamp(eps, 1 - eps),
                tgt_an,
                reduction='none'
            ) * w_an
            
            # Accumulate
            pairs_sim.append(bce_an)
        
        # Merge all BCE terms
        if len(pairs_sim) == 0:
            return z.new_tensor(0.0)
        
        # Two types collected:
        #  - already-BCE tensors (positives multi, negatives)
        #  - raw positive best-pair (pred, tgt, w)
        # Merge carefully:
        bces = []
        for s, t, w in zip(pairs_sim, pairs_tgt, pairs_w):
            if t is None:
                # already BCE with per-element weights 'w'
                # normalize weights per-batch
                w = _normalize_weights(w)
                bces.append((s * w).mean() if self.normalize_loss else (s * w).sum())
            else:
                # raw (pred_best, tgt_best, w_best)
                pred, tgt = s, t
                w = _normalize_weights(w)
                bce = F.binary_cross_entropy(pred.clamp(eps, 1 - eps), tgt, reduction='none')
                bces.append((bce * w).mean() if self.normalize_loss else (bce * w).sum())
        
        return sum(bces) / len(bces)
    
    def _self_consistency(
        self,
        z_per_run: torch.Tensor,   # [R, N, D]
        run_quality: torch.Tensor, # [R]
        node_idx: torch.Tensor | None = None  # optional subset (e.g., anchors)
    ) -> torch.Tensor:
        """Quality-weighted variance of per-run embeddings around fused mean."""
        if z_per_run is None or run_quality is None:
            return z_per_run.new_tensor(0.0) if isinstance(z_per_run, torch.Tensor) else torch.tensor(0.0)
        
        q = (run_quality / (run_quality.sum() + 1e-6)).view(-1, 1, 1)     # [R,1,1]
        if node_idx is not None:
            z_r = z_per_run[:, node_idx, :]                               # [R, A, D]
        else:
            z_r = z_per_run                                               # [R, N, D]
        
        z_bar = (q * z_r).sum(dim=0, keepdim=False)                        # [A or N, D]
        var = ((z_r - z_bar.unsqueeze(0)) ** 2).sum(dim=-1)                # [R, A or N]
        loss = (q.squeeze(-1).squeeze(-1) * var).mean()
        return loss
    
    # ---------- public forward (two modes) ----------
    
    def forward(
        self,
        z: torch.Tensor,  # [N, D], L2 normalized
        
        # ---- Per-anchor mode (recommended) ----
        anchors: torch.Tensor | None = None,        # [A]
        pos_idx: torch.Tensor | None = None,        # [A,P]
        pos_mask: torch.Tensor | None = None,       # [A,P] bool
        neg_idx: torch.Tensor | None = None,        # [A,K]
        w_anchor: torch.Tensor | None = None,       # [A] (importance per anchor)
        
        coloc_pos: torch.Tensor | None = None,      # [A] or [A,P] in [0,1]
        calib_neg_idx: torch.Tensor | None = None,  # [A,M] (optional subset for calibration)
        crit_pos: torch.Tensor | None = None,       # [A,P] weighting for positive calibration
        crit_neg: torch.Tensor | None = None,       # [A,M] weighting for negative calibration
        neg_push_weights: torch.Tensor | None = None,  # [A,K] optional weights for push
        
        # ---- Global pairs mode (back-compat) ----
        pos_pairs: torch.Tensor | None = None,      # [P,2]
        neg_pairs: torch.Tensor | None = None,      # [Q,2]
        pos_weights: torch.Tensor | None = None,    # [P]
        neg_weights: torch.Tensor | None = None,    # [Q]
        
        # ---- Optional self-consistency ----
        z_per_run: torch.Tensor | None = None,      # [R,N,D]
        run_quality: torch.Tensor | None = None,    # [R]
        
    ) -> dict[str, torch.Tensor]:
        T = self._get_T()
        
        # ================= Per-anchor mode =================
        if anchors is not None and pos_idx is not None and pos_mask is not None and neg_idx is not None:
            # 1) Multi-positive InfoNCE (with optional debias)
            L_info, sim_an, valid_mask = self._info_nce_multi_positive(
                z=z,
                anchors=anchors,
                pos_idx=pos_idx,
                pos_mask=pos_mask,
                neg_idx=neg_idx,
                w_anchor=w_anchor,
                T=T,
            )
            
            # 2) Push (margin softplus)
            if neg_push_weights is not None:
                # optional mask: only keep valid anchors for push weights if you want
                L_push = self._push_margin_softplus(sim_an, neg_weights=neg_push_weights)
            else:
                L_push = self._push_margin_softplus(sim_an, neg_weights=None)
            
            # 3) Calibration (positives + selected hard negatives)
            L_cal = self._calibration_bce(
                z=z,
                anchors=anchors,
                pos_idx=pos_idx,
                pos_mask=pos_mask,
                coloc_pos=coloc_pos,
                neg_idx_all=neg_idx,
                calib_neg_idx=calib_neg_idx,
                crit_pos=crit_pos,
                crit_neg=crit_neg,
            )
            
            # 4) Self-consistency across runs (optional)
            if self.lambda_self > 0.0 and z_per_run is not None and run_quality is not None:
                L_self = self._self_consistency(z_per_run, run_quality, node_idx=anchors)
            else:
                L_self = z.new_tensor(0.0)
            
            total = L_info + self.lambda_push * L_push + self.lambda_coloc * L_cal + self.lambda_self * L_self
            
            return {
                "total": total,
                "info_nce": L_info.detach(),
                "len_push": L_push.detach(),
                "calibration": L_cal.detach(),
                "self_consistency": L_self.detach(),
                "temperature": T.detach(),
            }
        
        # ================= Global pairs mode (fallback / compatibility) =================
        # (This keeps your current training working; performance is usually worse than per-anchor.)
        eps = 1e-9
        device = z.device
        
        if pos_pairs is None or pos_pairs.numel() == 0:
            zero = z.new_tensor(0.0)
            return {
                "total": zero, "info_nce": zero, "len_push": zero, "calibration": zero,
                "self_consistency": zero, "temperature": self._get_T().detach()
            }
        
        zi_pos = z[pos_pairs[:, 0]]
        zj_pos = z[pos_pairs[:, 1]]
        sim_pos = torch.sum(zi_pos * zj_pos, dim=-1) / T
        
        exp_pos = torch.exp(sim_pos)                       # [P]
        ln_pos = torch.logsumexp(sim_pos, dim=0)           # scalar
        
        if neg_pairs is not None and neg_pairs.numel() > 0:
            zi_neg = z[neg_pairs[:, 0]]
            zj_neg = z[neg_pairs[:, 1]]
            sim_neg = torch.sum(zi_neg * zj_neg, dim=-1) / T
            ln_neg = torch.logsumexp(sim_neg, dim=0)       # scalar
            
            if self.use_debiased:
                log_one_minus_tau = math.log(max(1e-6, 1.0 - self.tau_plus))
                log_denom = torch.logaddexp(ln_neg, ln_pos + log_one_minus_tau)
            else:
                log_denom = torch.logaddexp(ln_pos, ln_neg)
        else:
            sim_neg = z.new_empty(0)
            log_denom = ln_pos
        
        # weighted infoNCE
        if pos_weights is not None:
            pos_weights = _normalize_weights(pos_weights)
            L_info = -_safe_mean(pos_weights * (sim_pos - log_denom))
        else:
            L_info = -(sim_pos - log_denom).mean()
        
        # push with margin
        if sim_neg.numel() > 0:
            x = (sim_neg + self.push_margin) / self.push_tau
            val = F.softplus(x)
            if neg_weights is not None:
                neg_weights = _normalize_weights(neg_weights)
                val = val * neg_weights
            L_push = _safe_mean(val)
        else:
            L_push = z.new_tensor(0.0)
        
        # basic calibration on given pairs if you supply coloc/tgt externally (not provided here)
        L_cal = z.new_tensor(0.0)
        
        # optional self-consistency (global)
        if self.lambda_self > 0.0 and z_per_run is not None and run_quality is not None:
            L_self = self._self_consistency(z_per_run, run_quality, node_idx=None)
        else:
            L_self = z.new_tensor(0.0)
        
        total = L_info + self.lambda_push * L_push + self.lambda_coloc * L_cal + self.lambda_self * L_self
        return {
            "total": total,
            "info_nce": L_info.detach(),
            "len_push": L_push.detach(),
            "calibration": L_cal.detach(),
            "self_consistency": L_self.detach(),
            "temperature": self._get_T().detach(),
        }
