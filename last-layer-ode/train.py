# train.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional
import math
import random
from pathlib import Path
import re
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from scaffolds import SCAFFOLDS
from models import MODELS
from jumps import make_u_to_y_jump


class ODEDataset(Dataset):
    """
    Required npz fields:
      y0    : (N,P_obs)
      u_seq : (N,K,U)
      y_seq : (N,K,P_obs)
      t_obs : (K+1,)
      control_indices : (U,)
      obs_indices     : (P_obs,)
    Optional:
      names_full, control_names, obs_names
      lengths : (N,) — per-sample valid sequence length.
                When present, __getitem__ returns trimmed tensors
                and collate_varlen pads each batch to its own max.
    """

    def __init__(self, npz_path: str | Path):
        d = np.load(str(npz_path), allow_pickle=True)

        self.y0 = d["y0"].astype(np.float32)  # (N,P_obs)
        self.u_seq = d["u_seq"].astype(np.float32)  # (N,K,U)
        self.y_seq = d["y_seq"].astype(np.float32)  # (N,K,P_obs)
        t_obs = d["t_obs"].astype(np.float32)  # (K+1,)
        self.dt = np.diff(t_obs).astype(np.float32)  # (K,)

        if "control_indices" not in d or "obs_indices" not in d:
            raise ValueError(
                f"Dataset {npz_path} missing control_indices/obs_indices. Regenerate dataset with metadata."
            )
        self.control_indices = d["control_indices"].astype(np.int64)
        self.obs_indices = d["obs_indices"].astype(np.int64)

        self.names_full = d["names_full"].astype(str) if "names_full" in d else None
        self.control_names = d["control_names"].astype(str) if "control_names" in d else None
        self.obs_names = d["obs_names"].astype(str) if "obs_names" in d else None
        # Optional mapping from this (possibly filtered) dataset back to the
        # original row indices in the source dataset. When present, callers
        # can remap hardcoded splits that refer to the original indexing.
        self.original_indices = d["original_indices"].astype(np.int64) if "original_indices" in d else None

        # MinMax stats for u (per channel). Mirrors train_R.py so replot works
        # with models trained via either training script.
        self.u_scale_max = d["u_scale_max"].astype(np.float32) if "u_scale_max" in d else None
        if "u_scaled_cols" in d and self.control_names is not None:
            scaled = list(d["u_scaled_cols"].astype(str))
            ctrl_list = list(self.control_names)
            try:
                self.u_scaled_cols_idx = np.array([ctrl_list.index(c) for c in scaled], dtype=np.int64)
            except ValueError:
                self.u_scaled_cols_idx = None
        else:
            self.u_scaled_cols_idx = None

        # Variable-length support
        if "lengths" in d:
            self.lengths = d["lengths"].astype(np.int64)  # (N,)
            self.variable_length = True
        else:
            self.lengths = None
            self.variable_length = False

    def __len__(self) -> int:
        return self.y0.shape[0]

    def __getitem__(self, i: int):
        if self.variable_length:
            L = int(self.lengths[i])
            return (
                torch.from_numpy(self.y0[i]),          # (P_obs,)
                torch.from_numpy(self.u_seq[i, :L]),   # (L,U)
                torch.from_numpy(self.y_seq[i, :L]),   # (L,P_obs)
            )
        return (
            torch.from_numpy(self.y0[i]),  # (P_obs,)
            torch.from_numpy(self.u_seq[i]),  # (K,U)
            torch.from_numpy(self.y_seq[i]),  # (K,P_obs)
        )


def collate(batch):
    y0, u, y = zip(*batch)
    return torch.stack(y0), torch.stack(u), torch.stack(y), None


def collate_varlen(batch):
    """Pad each batch to its own max length; return lengths tensor."""
    y0_list, u_list, y_list = zip(*batch)
    lengths = torch.tensor([u.shape[0] for u in u_list], dtype=torch.long)
    y0 = torch.stack(y0_list)
    u_padded = torch.nn.utils.rnn.pad_sequence(u_list, batch_first=True)   # (B, K_batch, U)
    y_padded = torch.nn.utils.rnn.pad_sequence(y_list, batch_first=True)   # (B, K_batch, P)
    return y0, u_padded, y_padded, lengths


def _build_loss_mask(lengths: torch.Tensor, K: int, device: torch.device) -> torch.Tensor:
    """Build (B, K) boolean mask: True for valid timesteps."""
    return torch.arange(K, device=device).unsqueeze(0) < lengths.unsqueeze(1)


def _load_fixed_split_triples(
    path: str | Path, 
    idx: int, 
    original_indices: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a python file of hardcoded (test, val, train) splits and return triple `idx`.

    If `original_indices` is provided, the hardcoded indices are mapped to their 
    new positions in the filtered dataset, dropping any that were filtered out.
    """
    import re
    src = Path(path).read_text()
    triples: list[dict[str, list[int]]] = []
    current: dict[str, list[int]] = {}
    pat = re.compile(r"^\s*(test|val|train)\s*=\s*(\[[^\]]*\])", re.MULTILINE)
    for m in pat.finditer(src):
        key, lst = m.group(1), m.group(2)
        current[key] = list(eval(lst))  # safe: list literals only
        if {"test", "val", "train"}.issubset(current):
            triples.append(current)
            current = {}
    if not (0 <= idx < len(triples)):
        raise ValueError(f"fixed_split_idx={idx} out of range; file has {len(triples)} triples")
    
    t = triples[idx]
    train_raw, val_raw, test_raw = t["train"], t["val"], t["test"]

    # If we passed the mapping from the .npz file, translate the indices
    if original_indices is not None:
        # Create a dictionary mapping: { original_index : new_array_index }
        mapping = {int(orig): new_i for new_i, orig in enumerate(original_indices)}
        
        # Keep only the indices that survived filtering, mapped to their new positions
        train = [mapping[i] for i in train_raw if i in mapping]
        val   = [mapping[i] for i in val_raw   if i in mapping]
        test  = [mapping[i] for i in test_raw  if i in mapping]
    else:
        train, val, test = train_raw, val_raw, test_raw

    return (np.asarray(train, dtype=np.int64),
            np.asarray(val,   dtype=np.int64),
            np.asarray(test,  dtype=np.int64))


def loss_fn(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    use_log_loss: bool = True,
) -> torch.Tensor:
    """MSE loss, optionally in log1p space. use_log_loss=False when data is pre-normalised."""
    if use_log_loss:
        pred  = torch.log1p(pred)
        y_seq = torch.log1p(y_seq)
    se = (pred - y_seq).pow(2)  # (B,K,P)
    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)  # (B,K)
        se = se * mask.unsqueeze(-1)
        return se.sum() / (mask.sum() * se.shape[-1])
    return se.mean()


def loss_fn_ivtt_mse(
    *,
    pred_full: torch.Tensor,
    theta: torch.Tensor,
    y_seq_full: torch.Tensor,
    lengths: Optional[torch.Tensor],
    mm_state_idx: int = 3,
    pm_state_idx: int = 5,
) -> torch.Tensor:
    """Replicate bob_model/neural_spline.py:NeuralSpline.train_val() loss.

    Assumes last-layer-ode 7-state layout where mm is at index 3 and pm at 5.
    Uses Bob's fixed loss_weight vector and masking/weighting scheme.
    """
    B, K, _ = y_seq_full.shape
    device = y_seq_full.device
    dtype = y_seq_full.dtype

    if lengths is None:
        lengths = torch.full((B,), K, device=device, dtype=torch.long)

    y_mm = y_seq_full[:, :, mm_state_idx:mm_state_idx + 1]
    y_pm = y_seq_full[:, :, pm_state_idx:pm_state_idx + 1]
    p_zero = torch.zeros_like(y_mm)
    y_seq_3 = torch.cat([y_mm, p_zero, y_pm], dim=-1)  # (B,K,3)

    pred_mm = pred_full[:, :, mm_state_idx:mm_state_idx + 1]
    pred_pm = pred_full[:, :, pm_state_idx:pm_state_idx + 1]
    pred_3 = torch.cat([pred_mm, torch.zeros_like(pred_mm), pred_pm], dim=-1)  # (B,K,3)

    y_clamped = y_seq_3.clamp_min(1.0)
    pred_clamped = pred_3.clamp_min(1.0)
    log_y = torch.log1p(y_clamped)
    log_pred = torch.log1p(pred_clamped)

    # theta layout from BobGRUVerbatim: [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO]
    VTX_max = theta[..., 0:1]
    kdm = theta[..., 1:2]
    VTL_max = theta[..., 2:3]
    kmt = theta[..., 3:4]
    kmtm = theta[..., 4:5]
    R = theta[..., 5:6]

    mean_st = y_seq_3.mean(dim=-1, keepdim=True)
    VTX_raw = VTX_max / 0.125
    kdm_raw = kdm / 0.01
    VTL_raw = VTL_max / 0.075
    kmt_raw = kmt / 0.00035
    kmtm_raw = kmtm / 0.0035
    R_raw = R / 1.0

    VTX_state = (VTX_raw * R_raw * mean_st).clamp_min(0.0)
    kdm_state = (kdm_raw * mean_st).clamp_min(0.0)
    VTL_state = (VTL_raw * R_raw * mean_st).clamp_min(0.0)
    kmt_state = (kmt_raw * mean_st).clamp_min(0.0)
    kmtm_state = (kmtm_raw * mean_st).clamp_min(0.0)
    R_state = (R_raw * mean_st).clamp_min(0.0)

    log_VTX = torch.log1p(VTX_state)
    log_kdm = torch.log1p(kdm_state)
    log_VTL = torch.log1p(VTL_state)
    log_kmt = torch.log1p(kmt_state)
    log_kmtkm = torch.log1p(kmtm_state)
    log_R = torch.log1p(R_state)

    log_pred_all = torch.cat([log_pred, log_VTX, log_kdm, log_VTL, log_kmt, log_kmtkm, log_R], dim=-1)  # (B,K,9)
    log_zero = torch.zeros_like(log_VTX)
    log_y_all = torch.cat([log_y, log_zero, log_zero, log_zero, log_zero, log_zero, log_zero], dim=-1)  # (B,K,9)

    t_grid = torch.arange(K, device=device)
    valid_mask = (t_grid[None, :] < lengths[:, None]).unsqueeze(-1)  # (B,K,1)
    cut_idx = (lengths.to(torch.float32) * 0.95).to(torch.long)
    tail_mask = (t_grid[None, :] >= cut_idx[:, None]) & valid_mask.squeeze(-1)  # (B,K)

    # time weighting for p* (pm): second half gets weight N
    N = 3
    half_idx = lengths // 2
    is_second_half = (t_grid[None, :] >= half_idx[:, None]).unsqueeze(-1)  # (B,K,1)
    w_time = torch.where(
        is_second_half,
        torch.full_like(valid_mask, float(N), dtype=log_y_all.dtype),
        torch.ones_like(valid_mask, dtype=log_y_all.dtype),
    ) * valid_mask.to(torch.float32)

    w = torch.zeros_like(log_y_all, dtype=dtype)
    w[..., 0] = valid_mask.squeeze(-1)                     # mRNA
    w[..., 1] = tail_mask.to(dtype)                        # p (ignored by loss_weight but kept verbatim)
    w[..., 2] = (w_time * valid_mask).squeeze(-1)          # p*
    w[..., 3] = tail_mask.to(dtype)                        # VTX
    w[..., 4] = tail_mask.to(dtype)                        # kdm
    w[..., 5] = tail_mask.to(dtype)                        # VTL
    w[..., 6] = tail_mask.to(dtype)                        # kmt
    w[..., 7] = tail_mask.to(dtype)                        # kmtm
    w[..., 8] = tail_mask.to(dtype)                        # R proxy

    last_indices = (lengths - 1).clamp_min(0)
    batch_idx = torch.arange(B, device=device)
    y_last = log_y_all[batch_idx, last_indices, 2]
    pred_last = log_pred_all[batch_idx, last_indices, 2]
    extra_mse_chan = ((pred_last - y_last) ** 2).mean(0, keepdim=True)  # (1,)

    err2 = (log_pred_all - log_y_all).pow(2) * w
    mse_chan = err2.sum((0, 1)) / w.sum((0, 1)).clamp_min(1)
    mse_chan = torch.cat([mse_chan, extra_mse_chan])  # (10,)

    loss_weight = torch.tensor([1.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1], device=device, dtype=mse_chan.dtype)
    weighted = loss_weight * mse_chan
    mask = loss_weight.ne(0)
    den = mask.sum().clamp_min(1).to(weighted.dtype)
    return weighted[mask].sum() / den


def loss_fn_per_species(
    pred: torch.Tensor,
    y_seq: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    use_log_loss: bool = True,
) -> torch.Tensor:
    if use_log_loss:
        pred  = torch.log1p(pred)
        y_seq = torch.log1p(y_seq)
    se = (pred - y_seq).pow(2)
    if lengths is not None:
        mask = _build_loss_mask(lengths, se.shape[1], se.device)  # (B,K)
        se = se * mask.unsqueeze(-1)
        return se.sum(dim=(0, 1)) / mask.sum()
    return se.mean(dim=(0, 1))


# --- Normalization utilities ---
_NORM_STD_EPS = 1e-8


def _compute_zscore_stats(
    y0: np.ndarray, y_seq: np.ndarray,
    train_idx: np.ndarray, lengths: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std from training samples only (valid timesteps only)."""
    train_y0 = y0[train_idx]
    if lengths is not None:
        parts = [train_y0] + [y_seq[i, :lengths[i]] for i in train_idx]
    else:
        parts = [train_y0, y_seq[train_idx].reshape(-1, y0.shape[-1])]
    all_vals = np.concatenate(parts, axis=0)
    mean = all_vals.mean(axis=0).astype(np.float32)
    std  = np.maximum(all_vals.std(axis=0), _NORM_STD_EPS).astype(np.float32)
    return mean, std


def _apply_norm(
    arr: np.ndarray, method: str,
    mean: np.ndarray | None = None,
    std:  np.ndarray | None = None,
) -> np.ndarray:
    if method == "log":
        return np.log1p(np.clip(arr, 0.0, None)).astype(np.float32)
    if method == "sqrt":
        return np.sqrt(np.clip(arr, 0.0, None)).astype(np.float32)
    if method == "zscore":
        return ((arr - mean) / std).astype(np.float32)
    raise ValueError(f"Unknown obs_normalization: {method!r}. Choose from: log, sqrt, zscore")


def _endpoint_values(
    y_seq: np.ndarray,
    lengths: np.ndarray | None,
    target_idx: list[int],
) -> np.ndarray:
    """Per-sample final values for selected species indices: (N, T)."""
    N = y_seq.shape[0]
    if lengths is None:
        return y_seq[:, -1, target_idx].astype(np.float64)
    end_t = np.clip(lengths.astype(np.int64) - 1, 0, y_seq.shape[1] - 1)
    out = np.empty((N, len(target_idx)), dtype=np.float64)
    for j, t_idx in enumerate(target_idx):
        out[:, j] = y_seq[np.arange(N), end_t, t_idx]
    return out


def _quantile_bin_1d(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile binning robust to ties; returns integer labels."""
    if n_bins <= 1:
        return np.zeros(values.shape[0], dtype=np.int64)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros(values.shape[0], dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=True).astype(np.int64)


def _allocate_counts(stratum_sizes: np.ndarray, total: int) -> np.ndarray:
    """Allocate `total` across strata proportionally (largest remainder)."""
    out = np.zeros_like(stratum_sizes, dtype=np.int64)
    if total <= 0 or stratum_sizes.sum() <= 0:
        return out
    raw = total * (stratum_sizes / max(1, int(stratum_sizes.sum())))
    base = np.floor(raw).astype(np.int64)
    base = np.minimum(base, stratum_sizes)
    out[:] = base

    need = int(total - out.sum())
    if need <= 0:
        return out

    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    for i in order:
        if need <= 0:
            break
        if out[i] < stratum_sizes[i]:
            out[i] += 1
            need -= 1

    if need > 0:
        for i in np.argsort(-stratum_sizes):
            if need <= 0:
                break
            take = min(need, int(stratum_sizes[i] - out[i]))
            if take > 0:
                out[i] += take
                need -= take
    return out


def _make_split_indices(
    *,
    N: int,
    y_seq: np.ndarray,
    lengths: np.ndarray | None,
    n_val: int,
    n_test: int,
    split_seed: int,
    stratified_split: bool,
    stratify_bins: int,
    stratify_targets: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create train/val/test indices; optionally stratified by endpoint bins."""
    if n_test + n_val >= N:
        raise ValueError(f"val_n={n_val} + test_n={n_test} >= N={N}")

    rng = np.random.default_rng(split_seed)
    all_idx = np.arange(N, dtype=np.int64)

    if not stratified_split:
        rng.shuffle(all_idx)
        test_idx = all_idx[:n_test]
        val_idx = all_idx[n_test:n_test + n_val]
        train_idx = all_idx[n_test + n_val:]
        return train_idx, val_idx, test_idx

    P = int(y_seq.shape[-1])
    targets = list(stratify_targets) if stratify_targets is not None else list(range(P))
    targets = [int(t) for t in targets if 0 <= int(t) < P]
    if len(targets) == 0:
        raise ValueError("stratified_split=True but stratify_targets is empty/invalid")

    vals = _endpoint_values(y_seq, lengths, targets)
    std = vals.std(axis=0)
    keep = std > 1e-12
    if not np.any(keep):
        rng.shuffle(all_idx)
        test_idx = all_idx[:n_test]
        val_idx = all_idx[n_test:n_test + n_val]
        train_idx = all_idx[n_test + n_val:]
        return train_idx, val_idx, test_idx
    vals = vals[:, keep]

    ranks = np.empty_like(vals, dtype=np.float64)
    for j in range(vals.shape[1]):
        order = np.argsort(vals[:, j], kind="mergesort")
        inv = np.empty_like(order)
        inv[order] = np.arange(vals.shape[0])
        denom = max(1, vals.shape[0] - 1)
        ranks[:, j] = inv / denom
    score = ranks.mean(axis=1)
    labels = _quantile_bin_1d(score, int(stratify_bins)).astype(np.int64)

    uniq = np.unique(labels)
    strata = []
    for u in uniq:
        idx_u = np.where(labels == u)[0]
        rng.shuffle(idx_u)
        strata.append(idx_u)

    sizes = np.array([len(s) for s in strata], dtype=np.int64)
    take_test = _allocate_counts(sizes, int(n_test))

    test_parts: list[np.ndarray] = []
    rem_parts: list[np.ndarray] = []
    for s, k in zip(strata, take_test):
        test_parts.append(s[:k])
        rem_parts.append(s[k:])

    rem_sizes = np.array([len(s) for s in rem_parts], dtype=np.int64)
    take_val = _allocate_counts(rem_sizes, int(n_val))

    val_parts: list[np.ndarray] = []
    train_parts: list[np.ndarray] = []
    for s, k in zip(rem_parts, take_val):
        val_parts.append(s[:k])
        train_parts.append(s[k:])

    test_idx = np.concatenate(test_parts) if len(test_parts) else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if len(val_parts) else np.empty(0, dtype=np.int64)
    train_idx = np.concatenate(train_parts) if len(train_parts) else np.empty(0, dtype=np.int64)

    rng.shuffle(test_idx)
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)
    return train_idx, val_idx, test_idx


@dataclass
class TrainConfig:
    dataset_path: str

    study: str = "adhoc"
    tags: list[str] | None = None
    exp_name: str = "default"
    out_root: str = "experiments"

    save_model_name: str = "model.pt"  # saved in exp_dir/
    save_last_name: str = "model_last.pt"  # saved in exp_dir/
    save_curves_name: str = "loss_curves.npz"  # saved in exp_dir/logs/

    epochs: int = 200
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 0  # linear LR warmup; 0 disables
    cosine_decay: bool = False  # cosine decay from lr to lr*cosine_decay_min after warmup
    val_n: int = 100   # fixed count for validation set
    test_n: int = 100  # fixed count for held-out test set
    # If set, load test indices from this .npy file (overrides random/stratified
    # test selection). Train/val are then drawn from the remaining samples.
    fixed_test_idx_path: str | None = None
    train_n: int = 0   # cap on train samples (0 = use all remaining after val/test)
    # legacy: val_frac still accepted but val_n/test_n take precedence when > 0
    val_frac: float = 0.0
    seed: int = 42        # controls model init + training stochasticity only
    split_seed: int = 42   # controls train/val/test split — keep fixed across seeds

    # Split balancing by endpoint outcomes (final y_seq values).
    # When enabled, bins selected target species by endpoint quantiles and
    # samples train/val/test proportionally from each bin combination.
    stratified_split: bool = False
    stratify_bins: int = 5
    stratify_targets: list[int] | None = None

    num_workers: int = 0
    pin_memory: bool = True

    scaffold: str = "reduced5"
    hidden: int = 128
    lift_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    theta_lo: float = 1e-3
    theta_hi: float = 2.0
    n_substeps: int = 1

    ff_mult: int = 2  # feedforward multiplier inside each transformer layer (2=current, 4=standard)

    # Transformer-specific (ignored by non-Transformer models via **kwargs)
    context_len: int = 64  # sliding window size; set >= sequence length for full context
    tf_group_size: int = 32  # grouped-TF chunk size for ode_transformer_grouped
    ar_gap: int = 4  # autoregressive steps inserted between grouped-TF chunks

    # Mamba-specific (ignored by non-Mamba models via **kwargs)
    d_state: int = 16
    expand: int = 2
    d_conv: int = 4

    # neural_ode_correction / fixed_theta_nn-specific (ignored by other models via **kwargs)
    nn_hidden: int = 256
    nn_layers: int = 2

    forget_bias_init: Optional[float] = None  # None = PyTorch default; 1.0 = Gers/Jozefowicz positive shift
    legacy_forget_bias_bug: bool = False      # reproduce pre-fix fill_(0.0) on both bias_ih and bias_hh

    # Plain MSE on (mm, pm) in raw scale, no normalization, no aux terms.
    # Requires the model to emit an 8-d theta layout
    # [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO] — produced by any model paired
    # with scaffold=ivtt_analytic.
    use_ivtt_mse_loss: bool = False

    # Random shuffle split with a separate seed, used when no fixed_split_file
    # is provided. Test fraction is taken first, then val fraction of the remainder.
    use_fixed_split: bool = False
    fixed_split_seed: int = 57
    fixed_test_frac: float = 0.125
    fixed_val_frac: float = 0.125

    # Use one of the hardcoded (test, val, train) triples from a python file
    # listing them. Takes precedence over use_fixed_split when set.
    fixed_split_file: Optional[str] = None
    fixed_split_idx: int = 0

    use_basal: bool = False
    beta_regularization: bool = False
    lambda_beta: float = 1.0

    theta_bounded: bool = True   # if False, use softplus (unbounded above) instead of gamma

    # OdeRNN encoder knobs — defaults preserve current behaviour for synthetic data.
    lift_skip: bool = False               # drop the lift MLP and feed feat→GRU directly (Bob)
    gru_variant: str = "nn_gru"           # "nn_gru" | "stacked_cell" (Bob's stacked GRUCell + dropout-on-last)
    gru_init: str = "default"             # "default" | "supervisor" (orthogonal_ + xavier_ + zeros)
    head_init: str = "default"            # "default" | "supervisor" (xavier_ + zeros, unconditional)
    theta_head_transform: str = "log_gamma"  # "log_gamma" | "gamma"
    theta_head_tau: float = 1.0           # log_gamma sigmoid temperature (Bob: 2.3)
    u_transform: str = "none"             # forward-time u feature transform ("none" | "sqrt" | "cumsum" | …)
    y_transform: str = "none"             # forward-time y feature transform ("none" | "sqrt_clamp1" | "log1p" | …)

    grad_clip: float = 1.0
    teacher_forcing: bool = True
    tf_every: int = 50
    tf_drop_epoch: int = 10**9

    # checkpointing cadence (0 disables periodic ckpts)
    ckpt_every: int = 10

    # If True, run endpoint R² analysis at end of training and save the plot
    # + cache into exp_dir. Also picked up by replot.py post-hoc.
    endpoint_r2: bool = False

    l1_regularization: bool = False   # smoothness: penalizes mean |theta[t] - theta[t-1]|
    l2_regularization: bool = False   # smoothness: penalizes mean (theta[t] - theta[t-1])^2

    lambda_reg: float = 0.001

    # If set (e.g. [0, 12]), supervise loss/TF only on those species indices.
    # If null/None, supervises all observed species (default behaviour).
    obs_idx: list[int] | None = None

    # If set, restrict GRU encoder input to these species indices (must match obs_idx
    # for a strictly partial-observation experiment). None = all P species (default).
    gru_y_cols: list[int] | None = None

    # If set, restrict GRU encoder input to these control (u) column indices.
    # Used e.g. by the IVTT analytic scaffold to drop the DNA c column from the
    # encoder feature (DNA c is consumed by the scaffold via dna_cum_total instead).
    # None = all U columns (default).
    gru_u_cols: list[int] | None = None

    # If set, slice ds.y0 and ds.y_seq to these column indices after loading.
    # Lets you use a dataset with more species than the scaffold expects by
    # selecting the subset (in scaffold state order) before the P_obs check.
    # Example: [0, 11, 12, 3, 4, 1, 2] picks the 7 kovacs_7 species from the
    # 14-species pulsed dataset.
    dataset_species_subset: list[int] | None = None

    # When True, compute loss only at the first and last time step (start/end
    # supervision). Useful to test whether models generalise from endpoint-only
    # signal. Not supported for variable-length sequences.
    supervise_endpoints_only: bool = False

    wandb_enabled: bool = False
    wandb_project: str = "theta-lab"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "train"
    wandb_name: str | None = None
    wandb_mode: str | None = None

    jit_scripting: bool = False
    torch_compile: bool = False
    autocast_bf16: bool = False

    # 'ode_rnn' (default), 'ode_rnn_2020' (latent ODE-RNN style),
    # or 'neural_ode' (pure MLP baseline)
    model_class: str = "ode_rnn"

    # Pre-normalise y0/y_seq before training. Options: "log", "sqrt", "zscore", null.
    # When set, the internal log1p loss is replaced with plain MSE.
    obs_normalization: str | None = None


def load_cfg(path: str | Path) -> TrainConfig:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return TrainConfig(**d)



def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    text = text.strip("-_.")
    return text or "run"


def build_run_dir(cfg: TrainConfig, now: datetime) -> tuple[Path, str, str]:
    study_slug = slugify(cfg.study)
    run_name = slugify(cfg.exp_name)
    run_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{run_name}"
    run_dir = Path(cfg.out_root) / study_slug / run_id
    return run_dir, run_id, study_slug


def init_wandb(cfg: TrainConfig, cfg_dict: dict, *, run_id: str, exp_dir: Path):
    if not cfg.wandb_enabled:
        return None, None

    try:
        import wandb
    except ImportError:
        print("[wandb] wandb is not installed; continuing without W&B logging.")
        return None, None

    raw_tags = cfg.tags or []
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    else:
        tags = [str(tag) for tag in raw_tags]
    if cfg.study not in tags:
        tags.append(str(cfg.study))

    init_kwargs = {
        "project": cfg.wandb_project,
        "entity": cfg.wandb_entity,
        "group": cfg.wandb_group or cfg.study,
        "job_type": cfg.wandb_job_type,
        "name": cfg.wandb_name or run_id,
        "tags": tags,
        "config": cfg_dict,
        "dir": str(exp_dir),
    }
    if cfg.wandb_mode is not None:
        init_kwargs["mode"] = cfg.wandb_mode

    try:
        run = wandb.init(**{k: v for k, v in init_kwargs.items() if v is not None})
    except Exception as exc:
        print(f"[wandb] init failed: {exc}")
        return None, None

    if run is not None:
        run.config.update(
            {
                "run_id": run_id,
                "study": cfg.study,
                "run_dir": str(exp_dir.resolve()),
            },
            allow_val_change=True,
        )
    return wandb, run


def log_wandb_images(wandb, run, plots_dir: Path) -> None:
    if wandb is None or run is None or not plots_dir.exists():
        return

    single_images = [
        ("plots/loss_curves", plots_dir / "loss_curves.png"),
        ("plots/val_species_heatmap", plots_dir / "val_species_heatmap.png"),
        ("plots/val_species_final", plots_dir / "val_species_final.png"),
        ("plots/pred_overlays", plots_dir / "pred_overlays_sample000.png"),
        ("plots/theta_sample0", plots_dir / "theta_sample0.png"),
    ]

    payload = {}
    for key, path in single_images:
        if path.exists():
            payload[key] = wandb.Image(str(path))

    pred_paths = sorted(plots_dir.glob("pred_vs_true_*.png"))[:3]
    if pred_paths:
        payload["plots/pred_vs_true_examples"] = [
            wandb.Image(str(path), caption=path.stem) for path in pred_paths
        ]

    if payload:
        run.log(payload)


def log_wandb_artifact(wandb, run, *, exp_dir: Path, run_id: str) -> None:
    if wandb is None or run is None:
        return

    artifact = wandb.Artifact(run_id, type="experiment")
    for rel_path in [
        Path("config.yaml"),
        Path("model.pt"),
        Path("model_last.pt"),
        Path("logs") / "loss_curves.npz",
    ]:
        path = exp_dir / rel_path
        if path.exists():
            artifact.add_file(str(path), name=str(rel_path))

    try:
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"[wandb] artifact logging failed: {exc}")


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: TrainConfig, *, no_plot: bool = False, plot_samples: int = 5, plot_sample_idx: int = 0) -> None:
    t0 = time.time()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = device_auto()
    print(f"Using device: {device}")

    now = datetime.now()
    exp_dir, run_id, study_slug = build_run_dir(cfg, now)
    exp_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config for later reconstruction
    cfg_dict = asdict(cfg)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    print(f"Experiment: {exp_dir}")
    print(f"Study: {study_slug} | Run ID: {run_id}")

    wandb, wandb_run = init_wandb(cfg, cfg_dict, run_id=run_id, exp_dir=exp_dir)

    ds = ODEDataset(cfg.dataset_path)

    if cfg.dataset_species_subset is not None:
        idx = np.array(cfg.dataset_species_subset, dtype=np.int64)
        ds.obs_names   = ds.obs_names[idx]   if ds.obs_names   is not None else None
        ds.obs_indices = ds.obs_indices[idx]
        ds.y0          = ds.y0[:, idx]
        ds.y_seq       = ds.y_seq[:, :, idx]
        names_str = list(ds.obs_names) if ds.obs_names is not None else idx.tolist()
        print(f"Species subset: {names_str} ({len(idx)} of original species)")

    N = len(ds)

    if cfg.fixed_split_file:
        train_idx, val_idx, test_idx = _load_fixed_split_triples(
            cfg.fixed_split_file, int(cfg.fixed_split_idx), ds.original_indices
        )
        # safety: clip to dataset size in case the source file references a
        # superset of experiments (e.g. unfiltered indices)
        for name, arr in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
            if arr.size and (arr.max() >= N or arr.min() < 0):
                raise ValueError(
                    f"fixed_split {name}_idx out of range for N={N} "
                    f"(min={arr.min()}, max={arr.max()})"
                )
        print(
            f"Split: fixed_split_file={cfg.fixed_split_file}"
            f" | idx={int(cfg.fixed_split_idx)}"
            f" | n_test={len(test_idx)} n_val={len(val_idx)} n_train={len(train_idx)}"
        )

    elif cfg.use_fixed_split:
        idx = list(range(N))
        rng = random.Random(int(cfg.fixed_split_seed))
        rng.shuffle(idx)
        n_test = int(N * float(cfg.fixed_test_frac))
        test_idx = np.asarray(idx[:n_test], dtype=np.int64)
        rest = idx[n_test:]
        n_val = int(len(rest) * float(cfg.fixed_val_frac))
        val_idx = np.asarray(rest[:n_val], dtype=np.int64)
        train_idx = np.asarray(rest[n_val:], dtype=np.int64)
        print(
            f"Split: fixed_split"
            f" | seed={int(cfg.fixed_split_seed)}"
            f" | test_frac={float(cfg.fixed_test_frac)}"
            f" | val_frac={float(cfg.fixed_val_frac)}"
            f" | n_test={len(test_idx)} n_val={len(val_idx)} n_train={len(train_idx)}"
        )

    elif not cfg.fixed_split_file:
        n_test = int(cfg.test_n) if cfg.test_n > 0 else 0
        n_val  = int(cfg.val_n)  if cfg.val_n  > 0 else max(1, int(N * cfg.val_frac))

    if (not cfg.use_fixed_split) and (not cfg.fixed_split_file) and cfg.fixed_test_idx_path:
        fixed_test = np.load(cfg.fixed_test_idx_path).astype(np.int64)
        if fixed_test.max() >= N or fixed_test.min() < 0:
            raise ValueError(f"fixed_test_idx out of range for dataset size {N}")
        remaining = np.setdiff1d(np.arange(N, dtype=np.int64), fixed_test)
        rng = np.random.default_rng(int(cfg.split_seed))
        rng.shuffle(remaining)
        if n_val > len(remaining):
            raise ValueError(f"val_n={n_val} exceeds remaining {len(remaining)}")
        val_idx = remaining[:n_val]
        train_idx = remaining[n_val:]
        test_idx = fixed_test
        print(f"Split: fixed test from {cfg.fixed_test_idx_path} "
              f"(n_test={len(test_idx)}, n_val={len(val_idx)}, n_train={len(train_idx)})")
    elif (not cfg.use_fixed_split) and (not cfg.fixed_split_file):
        train_idx, val_idx, test_idx = _make_split_indices(
            N=N,
            y_seq=ds.y_seq,
            lengths=ds.lengths if ds.variable_length else None,
            n_val=n_val,
            n_test=n_test,
            split_seed=int(cfg.split_seed),
            stratified_split=bool(cfg.stratified_split),
            stratify_bins=int(cfg.stratify_bins),
            stratify_targets=cfg.stratify_targets,
        )

    if cfg.stratified_split:
        print(
            "Split: stratified"
            f" | bins={int(cfg.stratify_bins)}"
            f" | targets={cfg.stratify_targets if cfg.stratify_targets is not None else 'auto'}"
        )

    if int(cfg.train_n) > 0 and int(cfg.train_n) < len(train_idx):
        train_idx = train_idx[: int(cfg.train_n)]
        print(f"Data-scarce: capped train to train_n={int(cfg.train_n)}")

    # persist split so plotting always uses the correct test indices
    np.savez(exp_dir / "split.npz",
             train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # --- optional pre-normalisation (stats from train split only) ---
    use_log_loss = cfg.obs_normalization is None
    if cfg.obs_normalization:
        method = cfg.obs_normalization
        if method not in ("log", "sqrt", "zscore"):
            raise ValueError(f"obs_normalization must be one of: log, sqrt, zscore (got {method!r})")
        norm_mean = norm_std = None
        if method == "zscore":
            norm_mean, norm_std = _compute_zscore_stats(
                ds.y0, ds.y_seq, train_idx,
                ds.lengths if ds.variable_length else None,
            )
        ds.y0   = _apply_norm(ds.y0,   method, norm_mean, norm_std)
        ds.y_seq = _apply_norm(ds.y_seq, method, norm_mean, norm_std)
        save_kwargs: dict = {}
        if norm_mean is not None:
            save_kwargs["mean"] = norm_mean
            save_kwargs["std"]  = norm_std
        np.savez(exp_dir / "norm_stats.npz", method=np.array(cfg.obs_normalization), **save_kwargs)
        print(f"Normalization: {method}" + (
            f" | per-channel stats from {len(train_idx)} training samples" if method == "zscore" else ""
        ))

    collate_fn = collate_varlen if ds.variable_length else collate

    train_loader = DataLoader(
        torch.utils.data.Subset(ds, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=bool(cfg.pin_memory),
    )

    val_loader = None
    if len(val_idx) > 0:
        val_loader = DataLoader(
            torch.utils.data.Subset(ds, val_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(cfg.pin_memory),
        )

    # infer dims
    y0_ex, u_ex, _ = ds[0]
    P_obs = int(y0_ex.shape[0])
    U = int(u_ex.shape[-1])

    scaffold = None
    if cfg.model_class != "bob_gru_verbatim":
        if cfg.scaffold not in SCAFFOLDS:
            raise ValueError(f"Unknown scaffold '{cfg.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
        scaffold = SCAFFOLDS[cfg.scaffold]

        if scaffold.P != P_obs:
            raise ValueError(f"Scaffold {cfg.scaffold} expects P={scaffold.P}, but dataset has P_obs={P_obs}.")

    u_to_y_jump = make_u_to_y_jump(ds.control_indices, ds.obs_indices, device=device)  # (U,P_obs)

    if cfg.model_class not in MODELS:
        raise ValueError(f"Unknown model_class '{cfg.model_class}'. Available: {list(MODELS.keys())}")
    model = MODELS[cfg.model_class](
        U=U,
        rhs=scaffold,
        u_to_y_jump=u_to_y_jump,
        hidden=cfg.hidden,
        lift_dim=cfg.lift_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        ff_mult=cfg.ff_mult,
        theta_lo=cfg.theta_lo,
        theta_hi=cfg.theta_hi,
        n_substeps=cfg.n_substeps,
        use_basal=cfg.use_basal,
        context_len=cfg.context_len,
        tf_group_size=cfg.tf_group_size,
        ar_gap=cfg.ar_gap,
        theta_bounded=cfg.theta_bounded,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        forget_bias_init=cfg.forget_bias_init,
        legacy_forget_bias_bug=cfg.legacy_forget_bias_bug,
        gru_y_cols=cfg.gru_y_cols,
        gru_u_cols=cfg.gru_u_cols,
        lift_skip=cfg.lift_skip,
        gru_variant=cfg.gru_variant,
        gru_init=cfg.gru_init,
        head_init=cfg.head_init,
        theta_head_transform=cfg.theta_head_transform,
        theta_head_tau=cfg.theta_head_tau,
    ).to(device)

    # DIAGNOSTIC ONLY — safe to remove once confirmed Flash Attention works on your GPU.
    if cfg.autocast_bf16 and cfg.model_class in {"ode_transformer", "ode_transformer_grouped"} and device.type == "cuda":
        _B, _W, _H = 2, 8, model.hidden
        _x = torch.randn(_B, _W, _H, device=device, dtype=torch.bfloat16)
        _mask = nn.Transformer.generate_square_subsequent_mask(_W, device=device).to(torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                try:
                    model.transformer(_x, mask=_mask, is_causal=True)
                    print("Flash Attention: OK")
                except Exception as e:
                    print(f"Flash Attention: NOT available ({e})")
        del _x, _mask

    compile_model = cfg.torch_compile
    jit_scripting = cfg.jit_scripting

    if jit_scripting == True:
        try:
            model = torch.jit.script(model)
            print('The model compiled successfully')
        except Exception as e:
            # Bare `except:` previously swallowed the error — the actual TorchScript
            # diagnostic is what tells you which line/branch is non-scriptable.
            import traceback
            print(f'JIT scripting failed: {type(e).__name__}: {e}')
            traceback.print_exc()
            print('Continuing with eager model.')

    elif compile_model == True:
        try:
            # dynamic=True: generates one symbolic kernel per distinct graph instead of
            # recompiling for every new tensor shape. Critical for the OdeTransformer
            # whose attention window W grows 1..context_len during the forward loop —
            # without this, Dynamo recompiles 64 times (~hours on GPFS clusters).
            model = torch.compile(model, dynamic=True)
            print("torch.compile: OK (dynamic=True)")
        except Exception as e:
            print(f'The model did not compile: {e}')


    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    scheduler = None
    if cfg.warmup_epochs > 0 and cfg.cosine_decay:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs)
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=float(cfg.lr) * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[int(cfg.warmup_epochs)]
        )
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target) + cosine decay to {float(cfg.lr)*0.01:.2e}")
    elif cfg.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6, end_factor=1.0, total_iters=int(cfg.warmup_epochs)
        )
        print(f"LR warmup: {cfg.warmup_epochs} epochs ({cfg.lr:.2e} target)")
    elif cfg.cosine_decay:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=float(cfg.lr) * 0.01
        )
        print(f"LR cosine decay: {cfg.lr:.2e} → {float(cfg.lr)*0.01:.2e} over {cfg.epochs} epochs")

    mech_names = ds.obs_names.tolist() if ds.obs_names is not None else None

    print(f"Data: N={N} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print(f"Dims: P_obs={P_obs} | scaffold={cfg.scaffold} | U={U}")
    if mech_names is not None:
        print("Species:", ", ".join(str(x) for x in mech_names))

    best_val = float("inf")
    best_state = None

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_species_losses: list[np.ndarray] = []

    def _save_ckpt(path: Path, epoch: int, tag: str):
        torch.save(
            {
                "epoch": int(epoch),
                "tag": str(tag),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "opt_state": opt.state_dict(),
                "best_val": float(best_val),
                "cfg": cfg.__dict__,
            },
            path,
        )

    if cfg.obs_idx is not None:
        obs_idx = torch.tensor(cfg.obs_idx, device=device, dtype=torch.long)
        print(f"Supervising only species indices: {cfg.obs_idx}")
    else:
        obs_idx = torch.arange(P_obs, device=device, dtype=torch.long)

    dt_tensor = torch.from_numpy(ds.dt).to(device)
    grouped_model = cfg.model_class == "ode_transformer_grouped"

    def _inject_feat_transforms(mk: dict) -> dict:
        # Only add when set so models whose forward() doesn't accept these kwargs
        # (transformer/mamba forward signatures vary) aren't disturbed.
        if cfg.u_transform and cfg.u_transform != "none":
            mk["u_transform"] = cfg.u_transform
        if cfg.y_transform and cfg.y_transform != "none":
            mk["y_transform"] = cfg.y_transform
        return mk

    for ep in range(1, cfg.epochs + 1):
        ep_t0 = time.time()
        teacher_forcing = bool(cfg.teacher_forcing) and (ep < int(cfg.tf_drop_epoch))

        # ---- train
        model.train()
        tr_total = 0.0
        tr_batches = 0

        for y0, u_seq, y_seq, batch_lengths in train_loader:
            K_batch = u_seq.shape[1]
            dt_seq = dt_tensor[:K_batch][None, :].expand(y0.shape[0], -1)

            y0 = y0.to(device)
            y_seq = y_seq.to(device)
            u_seq = u_seq.to(device)
            dt_seq = dt_seq.to(device)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device)

            opt.zero_grad(set_to_none=True)
            model_kwargs = {
                "teacher_forcing": teacher_forcing,
                "tf_every": int(cfg.tf_every),
            }
            _inject_feat_transforms(model_kwargs)
            if grouped_model and batch_lengths is not None:
                model_kwargs["lengths"] = batch_lengths
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                pred, theta, _ = model(
                    y0,
                    u_seq,
                    dt_seq,
                    obs_idx,
                    y_seq,
                    **model_kwargs,
                )
            if cfg.use_ivtt_mse_loss:
                loss = loss_fn_ivtt_mse(pred_full=pred, theta=theta, y_seq_full=y_seq, lengths=batch_lengths)
                # keep these for any logging/diagnostics below
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]
            else:
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]

                if cfg.supervise_endpoints_only:
                    pred_l  = torch.stack([pred[:, 0, :],  pred[:, -1, :]],  dim=1)
                    y_seq_l = torch.stack([y_seq[:, 0, :], y_seq[:, -1, :]], dim=1)
                    loss = loss_fn(pred_l, y_seq_l, None, use_log_loss=use_log_loss)
                else:
                    loss = loss_fn(pred, y_seq, batch_lengths, use_log_loss=use_log_loss)

            if cfg.l1_regularization:
                reg_loss = torch.mean(torch.abs(theta[:,1:,:] - theta[:,:-1,:]))
                loss += cfg.lambda_reg * reg_loss

            if cfg.l2_regularization:
                reg_loss = torch.mean((theta[:,1:,:] - theta[:,:-1,:]).pow(2))
                loss += cfg.lambda_reg * reg_loss

            loss.backward()

            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))

            opt.step()

            tr_total += float(loss.item())
            tr_batches += 1

        tr_loss = tr_total / max(1, tr_batches)
        train_losses.append(tr_loss)

        if scheduler is not None:
            scheduler.step()

        # ---- val
        va_loss = None
        sp_last = None

        if val_loader is not None:
            model.eval()
            va_total = 0.0
            va_batches = 0
            sp_total = None

            with torch.no_grad():
                for y0, u_seq, y_seq, batch_lengths in val_loader:
                    K_batch = u_seq.shape[1]
                    dt_seq = torch.from_numpy(ds.dt[:K_batch])
                    dt_seq = dt_seq[None, :].expand(y0.shape[0], -1)

                    y0 = y0.to(device)
                    y_seq = y_seq.to(device)
                    u_seq = u_seq.to(device)
                    dt_seq = dt_seq.to(device)
                    if batch_lengths is not None:
                        batch_lengths = batch_lengths.to(device)

                    if cfg.use_ivtt_mse_loss:
                        model_kwargs = {"teacher_forcing": False, "tf_every": int(cfg.tf_every)}
                        _inject_feat_transforms(model_kwargs)
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                            pred_full, theta, _ = model(y0, u_seq, dt_seq, obs_idx, y_seq, **model_kwargs)
                        loss = loss_fn_ivtt_mse(pred_full=pred_full, theta=theta, y_seq_full=y_seq, lengths=batch_lengths)
                        va_total += float(loss.item())
                        va_batches += 1
                        continue

                    model_kwargs = {"y_seq": None, "teacher_forcing": False}
                    _inject_feat_transforms(model_kwargs)
                    if grouped_model and batch_lengths is not None:
                        model_kwargs["lengths"] = batch_lengths
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.autocast_bf16 and device.type == "cuda")):
                        pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)
                    pred = pred[:, :, obs_idx]
                    y_seq = y_seq[:, :, obs_idx]
                    if cfg.supervise_endpoints_only:
                        pred_l  = torch.stack([pred[:, 0, :],  pred[:, -1, :]],  dim=1)
                        y_seq_l = torch.stack([y_seq[:, 0, :], y_seq[:, -1, :]], dim=1)
                        loss = loss_fn(pred_l, y_seq_l, None, use_log_loss=use_log_loss)
                    else:
                        loss = loss_fn(pred, y_seq, batch_lengths, use_log_loss=use_log_loss)
                    va_total += float(loss.item())

                    sp = loss_fn_per_species(pred, y_seq, batch_lengths, use_log_loss=use_log_loss).detach().cpu()
                    sp_total = sp if sp_total is None else sp_total + sp
                    va_batches += 1

            va_loss = va_total / max(1, va_batches)
            val_losses.append(va_loss)

            if sp_total is not None:
                sp_last = (sp_total / max(1, va_batches)).numpy()
                val_species_losses.append(sp_last)

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        ep_time = time.time() - ep_t0

        if va_loss is None:
            print(f"ep {ep:4d} | train {tr_loss:.6f} | tf={int(teacher_forcing)} | {ep_time:.2f}s")
        else:
            sp_str = ""
            if sp_last is not None:
                if mech_names is None:
                    sp_str = "  [" + "  ".join(f"{v:.4f}" for v in sp_last) + "]"
                else:
                    sp_str = "  [" + "  ".join(f"{n}:{v:.4f}" for n, v in zip(mech_names, sp_last)) + "]"
            print(
                f"ep {ep:4d} | train {tr_loss:.6f} | val {va_loss:.6f} | best {best_val:.6f} | tf={int(teacher_forcing)}{sp_str} | {ep_time:.2f}s"
            )

        if math.isnan(tr_loss) or math.isnan(va_loss):
            print(f"NaN detected at epoch {ep} — stopping early.")
            break

        if wandb_run is not None:
            payload = {
                "epoch": int(ep),
                "train/loss": float(tr_loss),
                "train/teacher_forcing": int(teacher_forcing),
                "system/epoch_time_sec": float(ep_time),
                "system/learning_rate": float(opt.param_groups[0]["lr"]),
            }
            if va_loss is not None:
                payload["val/loss"] = float(va_loss)
                payload["val/best_loss"] = float(best_val)
            if sp_last is not None:
                names = mech_names if mech_names is not None else [f"species_{i}" for i in range(len(sp_last))]
                for name, value in zip(names, sp_last):
                    payload[f"val_species/{name}"] = float(value)
            wandb_run.log(payload, step=int(ep))

        # always keep "last" checkpoint
        _save_ckpt(exp_dir / cfg.save_last_name, ep, tag="last")

        # periodic checkpoints for epoch-evolution overlays
        if int(cfg.ckpt_every) > 0 and (ep % int(cfg.ckpt_every) == 0):
            _save_ckpt(ckpt_dir / f"ckpt_ep{ep:04d}.pt", ep, tag="periodic")

        # write curves every epoch (so final plotting can use full history)
        curves_path = logs_dir / cfg.save_curves_name
        np.savez(
            curves_path,
            train_losses=np.array(train_losses, dtype=np.float32),
            val_losses=np.array(val_losses, dtype=np.float32) if len(val_losses) > 0 else None,
            val_species_losses=np.array(val_species_losses, dtype=np.float32) if len(val_species_losses) > 0 else None,
        )

    # restore best weights (if we had validation)
    if best_state is not None:
        model.load_state_dict(best_state)

    # final test evaluation (on held-out test set, using best weights)
    test_loss: float | None = None
    test_species_loss: np.ndarray | None = None
    if len(test_idx) > 0:
        test_loader = DataLoader(
            torch.utils.data.Subset(ds, test_idx.tolist()),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=bool(cfg.pin_memory),
        )
        model.eval()
        te_total = 0.0
        te_batches = 0
        sp_total = None
        # Accumulate per-sample predictions to compute R² across the entire test set
        # in both linear and log1p space. We pad to the global K_max and apply a
        # length-aware boolean mask before computing SS_res / SS_tot.
        all_pred_chunks: list[torch.Tensor] = []
        all_true_chunks: list[torch.Tensor] = []
        all_len_chunks:  list[torch.Tensor] = []
        with torch.no_grad():
            for y0, u_seq, y_seq, batch_lengths in test_loader:
                K_batch = u_seq.shape[1]
                dt_seq = dt_tensor[:K_batch][None, :].expand(y0.shape[0], -1)
                y0 = y0.to(device)
                y_seq = y_seq.to(device)
                u_seq = u_seq.to(device)
                dt_seq = dt_seq.to(device)
                if batch_lengths is not None:
                    batch_lengths = batch_lengths.to(device)
                model_kwargs = {"y_seq": None, "teacher_forcing": False}
                _inject_feat_transforms(model_kwargs)
                if grouped_model and batch_lengths is not None:
                    model_kwargs["lengths"] = batch_lengths
                pred, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)
                pred = pred[:, :, obs_idx]
                y_seq = y_seq[:, :, obs_idx]
                loss = loss_fn(pred, y_seq, batch_lengths, use_log_loss=use_log_loss)
                te_total += float(loss.item())
                sp = loss_fn_per_species(pred, y_seq, batch_lengths, use_log_loss=use_log_loss).detach().cpu()
                sp_total = sp if sp_total is None else sp_total + sp
                te_batches += 1
                all_pred_chunks.append(pred.detach().cpu())
                all_true_chunks.append(y_seq.detach().cpu())
                if batch_lengths is not None:
                    all_len_chunks.append(batch_lengths.detach().cpu())
        test_loss = te_total / max(1, te_batches)
        if sp_total is not None:
            test_species_loss = (sp_total / max(1, te_batches)).numpy()
            sp_str = "  [" + "  ".join(
                f"{n}:{v:.4f}" for n, v in zip(
                    mech_names if mech_names else [f"s{i}" for i in range(len(test_species_loss))],
                    test_species_loss,
                )
            ) + "]"
        else:
            sp_str = ""
        print(f"\nTest loss (best model): {test_loss:.6f}{sp_str}")

        # ---- R² (per-species) on the test set, both linear and log1p space ----
        # Variable-length batches may have different K, so pad each chunk to K_max
        # before stacking; a length-mask filters padded steps out of the regression.
        test_r2_lin: np.ndarray | None = None
        test_r2_log: np.ndarray | None = None
        if all_pred_chunks:
            K_max = max(p.shape[1] for p in all_pred_chunks)
            S = all_pred_chunks[0].shape[-1]

            def _pad_K(t: torch.Tensor, target: int) -> torch.Tensor:
                # Pad along time dim with zeros so concat works; length mask zeros these out anyway.
                if t.shape[1] == target:
                    return t
                pad = torch.zeros(t.shape[0], target - t.shape[1], t.shape[2], dtype=t.dtype)
                return torch.cat([t, pad], dim=1)

            pred_cat = torch.cat([_pad_K(t, K_max) for t in all_pred_chunks], dim=0)  # (N, K_max, S)
            true_cat = torch.cat([_pad_K(t, K_max) for t in all_true_chunks], dim=0)
            if all_len_chunks:
                lengths_cat = torch.cat(all_len_chunks, dim=0)
                mask_2d = torch.arange(K_max)[None, :] < lengths_cat[:, None]   # (N, K_max)
            else:
                mask_2d = torch.ones(pred_cat.shape[0], K_max, dtype=torch.bool)

            def _r2_per_species(p: torch.Tensor, t: torch.Tensor, m: torch.Tensor) -> np.ndarray:
                p_np, t_np, m_np = p.numpy(), t.numpy(), m.numpy()
                out = np.full((S,), np.nan, dtype=np.float32)
                for s in range(S):
                    pp = p_np[..., s][m_np]
                    tt = t_np[..., s][m_np]
                    if tt.size == 0:
                        continue
                    ss_res = float(((pp - tt) ** 2).sum())
                    ss_tot = float(((tt - tt.mean()) ** 2).sum())
                    out[s] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                return out

            test_r2_lin = _r2_per_species(pred_cat, true_cat, mask_2d)
            # log1p R²: clamp first to avoid log of negative spurious preds.
            test_r2_log = _r2_per_species(
                torch.log1p(pred_cat.clamp_min(0.0)),
                torch.log1p(true_cat.clamp_min(0.0)),
                mask_2d,
            )
            obs_names_for_print = (
                [mech_names[i] for i in cfg.obs_idx] if (mech_names is not None and cfg.obs_idx is not None)
                else (mech_names if mech_names is not None else [f"s{i}" for i in range(S)])
            )
            r2_lin_str = "  ".join(f"{n}:{v:.4f}" for n, v in zip(obs_names_for_print, test_r2_lin))
            r2_log_str = "  ".join(f"{n}:{v:.4f}" for n, v in zip(obs_names_for_print, test_r2_log))
            print(f"Test R² (linear)  : [{r2_lin_str}]   mean={float(np.nanmean(test_r2_lin)):.4f}")
            print(f"Test R² (log1p)   : [{r2_log_str}]   mean={float(np.nanmean(test_r2_log)):.4f}")
            if wandb_run is not None:
                wandb_run.summary["test/r2_lin_mean"] = float(np.nanmean(test_r2_lin))
                wandb_run.summary["test/r2_log_mean"] = float(np.nanmean(test_r2_log))
                for n, v_lin, v_log in zip(obs_names_for_print, test_r2_lin, test_r2_log):
                    wandb_run.summary[f"test/r2_lin/{n}"] = float(v_lin)
                    wandb_run.summary[f"test/r2_log/{n}"] = float(v_log)

    # write final loss_curves.npz including test results
    np.savez(
        logs_dir / cfg.save_curves_name,
        train_losses=np.array(train_losses, dtype=np.float32),
        val_losses=np.array(val_losses, dtype=np.float32) if len(val_losses) > 0 else None,
        val_species_losses=np.array(val_species_losses, dtype=np.float32) if len(val_species_losses) > 0 else None,
        test_loss=np.float32(test_loss) if test_loss is not None else None,
        test_species_losses=test_species_loss.astype(np.float32) if test_species_loss is not None else None,
        test_r2_linear=test_r2_lin if test_r2_lin is not None else None,
        test_r2_log1p=test_r2_log if test_r2_log is not None else None,
    )

    # save best model (plot expects exp_dir/model.pt)
    save_path = exp_dir / cfg.save_model_name
    torch.save(
        {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
         "best_val": float(best_val),
         "test_loss": float(test_loss) if test_loss is not None else None},
        save_path,
    )
    print(f"Saved best model to {save_path}")

    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f}m)")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    # plots ONLY at the end (including epoch evolution overlays from checkpoints)
    if not no_plot:
        try:
            from plot_diagnostics import plot_experiment, plot_epoch_prediction_overlays

            plot_experiment(exp_dir, n_samples=int(plot_samples), sample_idx=int(plot_sample_idx))

            # epochs=None => automatically uses available checkpoints and picks up to max_overlays evenly spaced
            plot_epoch_prediction_overlays(
                exp_dir,
                sample_idx=int(plot_sample_idx),
                epochs=None,
                max_overlays=8,
            )
        except ImportError as e:
            import sys
            print(f"[plot] plot_diagnostics.py import failed: {e}; sys.path={sys.path}; skipping plots.")
        except Exception as e:
            import traceback
            print(f"[plot] failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    if cfg.endpoint_r2:
        try:
            from metrics import endpoint_r2
            from plot_diagnostics import device_auto as _dev_auto

            r2_device = _dev_auto()
            print("\nRunning endpoint R² analysis...")
            result = endpoint_r2.collect_endpoints(
                exp_dir, r2_device, split="test", protein_sp="pm", mrna_sp="mm"
            )
            r2_protein = endpoint_r2.r2(result["true_protein_final"], result["pred_protein_final"])
            r2_mrna    = endpoint_r2.r2(result["true_mrna_max"],      result["pred_mrna_max"])
            print(f"  R²(protein final) = {r2_protein:.4f}")
            print(f"  R²(mRNA max)      = {r2_mrna:.4f}")

            out_path = exp_dir / "endpoint_r2.png"
            endpoint_r2.plot_endpoints(
                [result], protein_sp="pm", mrna_sp="mm", split="test", out_path=out_path
            )
            endpoint_r2.save_r2_cache(exp_dir, result, r2_protein, r2_mrna)

            if wandb_run is not None:
                wandb_run.summary["endpoint_r2/protein_final"] = float(r2_protein)
                wandb_run.summary["endpoint_r2/mrna_max"]      = float(r2_mrna)
        except Exception as e:
            print(f"[endpoint_r2] failed: {e}")

    if wandb_run is not None:
        plots_dir = exp_dir / "plots"
        log_wandb_images(wandb, wandb_run, plots_dir)
        log_wandb_artifact(wandb, wandb_run, exp_dir=exp_dir, run_id=run_id)
        wandb_run.summary["run_dir"] = str(exp_dir.resolve())
        wandb_run.summary["study"] = cfg.study
        wandb_run.summary["scaffold"] = cfg.scaffold
        wandb_run.summary["device"] = str(device)
        wandb_run.summary["elapsed_seconds"] = float(elapsed)
        if train_losses:
            wandb_run.summary["final_train_loss"] = float(train_losses[-1])
        if val_losses:
            wandb_run.summary["final_val_loss"] = float(val_losses[-1])
        if best_state is not None:
            wandb_run.summary["best_val_loss"] = float(best_val)
        wandb_run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=5)
    parser.add_argument("--plot-sample-idx", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override any config field, e.g. --set lr=1e-3 --set scaffold=mof_synthesis_8")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    for kv in args.set:
        key, val = kv.split("=", 1)
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config field: {key!r}. Valid fields: {list(vars(cfg).keys())}")
        parsed = yaml.safe_load(val)
        # yaml.safe_load("None") → str "None", not Python None — fix explicitly
        if isinstance(parsed, str) and parsed == "None":
            parsed = None
        orig_type = type(getattr(cfg, key))
        if parsed is not None and orig_type is not type(None) and not isinstance(parsed, orig_type):
            parsed = orig_type(parsed)
        setattr(cfg, key, parsed)

    train(cfg, no_plot=bool(args.no_plot), plot_samples=int(args.plot_samples), plot_sample_idx=int(args.plot_sample_idx))
