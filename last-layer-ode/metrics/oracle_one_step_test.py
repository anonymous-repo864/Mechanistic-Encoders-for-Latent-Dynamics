"""Oracle one-step test: force GRU input from true observations each step.

Runs two modes on a split:
  1) Oracle TF: teacher_forcing=True, tf_every=1, uses obs_idx
  2) Autoregressive: teacher_forcing=False

Reports trajectory MSE/R2 and endpoint R2 for a target species (default: pm).

Usage:
  python last-layer-ode/metrics/oracle_one_step_test.py \
    experiments/txtl_r_depletion_sweep/20260429_223651_lamr_3e-3 \
    --split test --species pm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_diagnostics import rebuild_model_from_experiment, device_auto, load_yaml
from train import collate, collate_varlen


def _load_split_indices(exp_dir: Path, split_name: str, n: int) -> list[int]:
    if split_name == "all":
        return list(range(n))
    split_path = exp_dir / "split.npz"
    if not split_path.exists():
        return list(range(n))
    split = np.load(split_path)
    key = f"{split_name}_idx"
    if key not in split.files:
        return list(range(n))
    idx = split[key].tolist()
    return idx if len(idx) > 0 else list(range(n))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _append_masked(
    store_true: list[np.ndarray],
    store_pred: list[np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lengths: Optional[np.ndarray],
):
    if lengths is None:
        store_true.append(y_true.reshape(-1))
        store_pred.append(y_pred.reshape(-1))
        return
    for i, L in enumerate(lengths):
        store_true.append(y_true[i, :L].reshape(-1))
        store_pred.append(y_pred[i, :L].reshape(-1))


def _max_over_time(values: np.ndarray, lengths: Optional[np.ndarray]) -> np.ndarray:
    if lengths is None:
        return values.max(axis=1)
    out = np.empty(values.shape[0], dtype=values.dtype)
    for i, L in enumerate(lengths):
        out[i] = values[i, :L].max() if L > 0 else values[i, 0]
    return out


def _final_over_time(values: np.ndarray, lengths: Optional[np.ndarray]) -> np.ndarray:
    if lengths is None:
        return values[:, -1]
    end_idx = np.clip(lengths - 1, 0, values.shape[1] - 1)
    return values[np.arange(values.shape[0]), end_idx]


def run_oracle_test(
    exp_dir: Path,
    split: str,
    species: str,
    batch_size: int,
    endpoint_protein: str,
    endpoint_mrna: str,
):
    device = device_auto()
    model, ds, state_names, _ = rebuild_model_from_experiment(exp_dir, device=device)
    cfg = load_yaml(exp_dir / "config.yaml")

    if species not in state_names:
        raise ValueError(f"Species {species!r} not in state_names={state_names}")
    sp_idx = state_names.index(species)

    if endpoint_protein not in state_names:
        raise ValueError(f"Protein species {endpoint_protein!r} not in state_names={state_names}")
    if endpoint_mrna not in state_names:
        raise ValueError(f"mRNA species {endpoint_mrna!r} not in state_names={state_names}")
    protein_idx = state_names.index(endpoint_protein)
    mrna_idx = state_names.index(endpoint_mrna)

    indices = _load_split_indices(exp_dir, split, len(ds))
    if isinstance(indices, list) and len(indices) > 0:
        dataset = torch.utils.data.Subset(ds, indices)
    else:
        dataset = ds

    raw_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    collate_fn = collate_varlen if getattr(raw_ds, "variable_length", False) else collate
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    obs_idx_cfg = cfg.get("obs_idx")
    if obs_idx_cfg:
        obs_idx = torch.tensor(obs_idx_cfg, device=device)
    else:
        obs_idx = torch.arange(len(state_names), device=device)

    u_transform = str(cfg.get("u_transform", "none"))
    y_transform = str(cfg.get("y_transform", "none"))

    oracle_true, oracle_pred = [], []
    ar_true, ar_pred = [], []

    oracle_end_true, oracle_end_pred = [], []
    ar_end_true, ar_end_pred = [], []

    oracle_p_true, oracle_p_pred = [], []
    oracle_m_true, oracle_m_pred = [], []
    ar_p_true, ar_p_pred = [], []
    ar_m_true, ar_m_pred = [], []

    model.eval()
    with torch.no_grad():
        for y0, u_seq, y_seq, lengths in loader:
            B = y0.shape[0]
            K = u_seq.shape[1]
            dt_seq = torch.from_numpy(raw_ds.dt[:K])[None, :].expand(B, -1)

            y0 = y0.to(device)
            u_seq = u_seq.to(device)
            y_seq = y_seq.to(device)
            dt_seq = dt_seq.to(device)
            lengths_np = lengths.cpu().numpy() if lengths is not None else None

            model_kwargs = {
                "y_seq": y_seq,
                "teacher_forcing": True,
                "tf_every": 1,
                "u_transform": u_transform,
                "y_transform": y_transform,
            }
            if model.__class__.__name__ == "OdeTransformerGrouped" and lengths is not None:
                model_kwargs["lengths"] = lengths.to(device)
            pred_oracle, _, _ = model(y0, u_seq, dt_seq, obs_idx, **model_kwargs)

            pred_ar, _, _ = model(
                y0,
                u_seq,
                dt_seq,
                obs_idx,
                y_seq=None,
                teacher_forcing=False,
                u_transform=u_transform,
                y_transform=y_transform,
            )

            y_true_sp = y_seq[:, :, sp_idx].cpu().numpy()
            y_oracle_sp = pred_oracle[:, :, sp_idx].cpu().numpy()
            y_ar_sp = pred_ar[:, :, sp_idx].cpu().numpy()

            y_true_p = y_seq[:, :, protein_idx].cpu().numpy()
            y_oracle_p = pred_oracle[:, :, protein_idx].cpu().numpy()
            y_ar_p = pred_ar[:, :, protein_idx].cpu().numpy()

            y_true_m = y_seq[:, :, mrna_idx].cpu().numpy()
            y_oracle_m = pred_oracle[:, :, mrna_idx].cpu().numpy()
            y_ar_m = pred_ar[:, :, mrna_idx].cpu().numpy()

            _append_masked(oracle_true, oracle_pred, y_true_sp, y_oracle_sp, lengths_np)
            _append_masked(ar_true, ar_pred, y_true_sp, y_ar_sp, lengths_np)

            oracle_end_true.append(_final_over_time(y_true_sp, lengths_np))
            oracle_end_pred.append(_final_over_time(y_oracle_sp, lengths_np))
            ar_end_true.append(_final_over_time(y_true_sp, lengths_np))
            ar_end_pred.append(_final_over_time(y_ar_sp, lengths_np))

            oracle_p_true.append(_final_over_time(y_true_p, lengths_np))
            oracle_p_pred.append(_final_over_time(y_oracle_p, lengths_np))
            ar_p_true.append(_final_over_time(y_true_p, lengths_np))
            ar_p_pred.append(_final_over_time(y_ar_p, lengths_np))

            oracle_m_true.append(_max_over_time(y_true_m, lengths_np))
            oracle_m_pred.append(_max_over_time(y_oracle_m, lengths_np))
            ar_m_true.append(_max_over_time(y_true_m, lengths_np))
            ar_m_pred.append(_max_over_time(y_ar_m, lengths_np))

    oracle_true = np.concatenate(oracle_true)
    oracle_pred = np.concatenate(oracle_pred)
    ar_true = np.concatenate(ar_true)
    ar_pred = np.concatenate(ar_pred)

    oracle_end_true = np.concatenate(oracle_end_true)
    oracle_end_pred = np.concatenate(oracle_end_pred)
    ar_end_true = np.concatenate(ar_end_true)
    ar_end_pred = np.concatenate(ar_end_pred)

    oracle_p_true = np.concatenate(oracle_p_true)
    oracle_p_pred = np.concatenate(oracle_p_pred)
    oracle_m_true = np.concatenate(oracle_m_true)
    oracle_m_pred = np.concatenate(oracle_m_pred)

    ar_p_true = np.concatenate(ar_p_true)
    ar_p_pred = np.concatenate(ar_p_pred)
    ar_m_true = np.concatenate(ar_m_true)
    ar_m_pred = np.concatenate(ar_m_pred)

    out = {
        "oracle_mse": float(np.mean((oracle_true - oracle_pred) ** 2)),
        "oracle_r2": _r2(oracle_true, oracle_pred),
        "oracle_endpoint_r2": _r2(oracle_end_true, oracle_end_pred),
        "ar_mse": float(np.mean((ar_true - ar_pred) ** 2)),
        "ar_r2": _r2(ar_true, ar_pred),
        "ar_endpoint_r2": _r2(ar_end_true, ar_end_pred),
        "oracle_endpoint_r2_protein": _r2(oracle_p_true, oracle_p_pred),
        "oracle_endpoint_r2_mrna": _r2(oracle_m_true, oracle_m_pred),
        "ar_endpoint_r2_protein": _r2(ar_p_true, ar_p_pred),
        "ar_endpoint_r2_mrna": _r2(ar_m_true, ar_m_pred),
        "n_points": int(oracle_true.shape[0]),
        "n_samples": int(oracle_end_true.shape[0]),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir", type=str)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--species", type=str, default="pm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--endpoint-protein", type=str, default="pm")
    parser.add_argument("--endpoint-mrna", type=str, default="mm")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    metrics = run_oracle_test(
        exp_dir,
        split=args.split,
        species=args.species,
        batch_size=args.batch_size,
        endpoint_protein=args.endpoint_protein,
        endpoint_mrna=args.endpoint_mrna,
    )

    print("\n=== Oracle vs AR one-step test ===")
    print(f"Split: {args.split}   Species: {args.species}")
    print(f"Samples: {metrics['n_samples']}   Points: {metrics['n_points']}")
    print(f"Oracle  MSE: {metrics['oracle_mse']:.6g}   R2: {metrics['oracle_r2']:.4f}   Endpoint R2: {metrics['oracle_endpoint_r2']:.4f}")
    print(f"AR      MSE: {metrics['ar_mse']:.6g}   R2: {metrics['ar_r2']:.4f}   Endpoint R2: {metrics['ar_endpoint_r2']:.4f}")
    print("\nEndpoint R2 (protein final + mRNA max):")
    print(f"Oracle  protein: {metrics['oracle_endpoint_r2_protein']:.4f}   mRNA: {metrics['oracle_endpoint_r2_mrna']:.4f}")
    print(f"AR      protein: {metrics['ar_endpoint_r2_protein']:.4f}   mRNA: {metrics['ar_endpoint_r2_mrna']:.4f}")


if __name__ == "__main__":
    main()
