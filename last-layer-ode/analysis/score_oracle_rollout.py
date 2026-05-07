"""Score per-sample oracle-rollout fit quality for TXTL scaffolds.

For each scaffold's stored fit results (results.npz containing y_true and
pred_rollout), compute per-sample metrics on the mature mRNA (mm) and mature
protein (pm) channels, plus dataset-level R^2 on peak-mRNA and endpoint-protein.

Threshold the per-sample rollout NRMSE to identify the "good" subset usable
as a clean train/test split.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


SCAFFOLDS = {
    "maturation": "results/txtl_theta_fit_maturation_dna/results.npz",
    "resource_maturation": "results/txtl_dna_resource_and_maturation_theta_fit/results.npz",
}


def nrmse(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # per-sample NRMSE normalized by range of the true trajectory.
    err = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    rng = true.max(axis=1) - true.min(axis=1)
    rng = np.where(rng < 1e-12, 1e-12, rng)
    return err / rng


def r2_timeseries(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # per-sample R^2 across the time axis.
    ss_res = np.sum((true - pred) ** 2, axis=1)
    mu = true.mean(axis=1, keepdims=True)
    ss_tot = np.sum((true - mu) ** 2, axis=1)
    ss_tot = np.where(ss_tot < 1e-12, 1e-12, ss_tot)
    return 1.0 - ss_res / ss_tot


def r2_scalar(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def score(npz_path: Path, scaffold_name: str):
    d = np.load(npz_path, allow_pickle=True)
    state_names = [str(s) for s in d["state_names"]]
    mm_idx = state_names.index("mm")
    pm_idx = state_names.index("pm")

    y_true = d["y_true"][:, 1:, :]  # align with pred_* which has T-1 steps
    pred = d["pred_rollout"]
    sample_idx = d["sample_indices"]

    true_mm = y_true[:, :, mm_idx]
    pred_mm = pred[:, :, mm_idx]
    true_pm = y_true[:, :, pm_idx]
    pred_pm = pred[:, :, pm_idx]

    nrmse_mm = nrmse(true_mm, pred_mm)
    nrmse_pm = nrmse(true_pm, pred_pm)
    r2_mm_ts = r2_timeseries(true_mm, pred_mm)
    r2_pm_ts = r2_timeseries(true_pm, pred_pm)

    peak_mm_true = true_mm.max(axis=1)
    peak_mm_pred = pred_mm.max(axis=1)
    end_pm_true = true_pm[:, -1]
    end_pm_pred = pred_pm[:, -1]

    columns = {
        "sample_idx": sample_idx,
        "scaffold": np.array([scaffold_name] * len(sample_idx)),
        "nrmse_mm_rollout": nrmse_mm,
        "nrmse_pm_rollout": nrmse_pm,
        "r2_mm_timeseries": r2_mm_ts,
        "r2_pm_timeseries": r2_pm_ts,
        "peak_mm_true": peak_mm_true,
        "peak_mm_pred": peak_mm_pred,
        "endpoint_pm_true": end_pm_true,
        "endpoint_pm_pred": end_pm_pred,
    }

    summary = {
        "scaffold": scaffold_name,
        "n_samples": int(len(sample_idx)),
        "median_nrmse_mm": float(np.median(nrmse_mm)),
        "median_nrmse_pm": float(np.median(nrmse_pm)),
        "median_r2_mm_ts": float(np.median(r2_mm_ts)),
        "median_r2_pm_ts": float(np.median(r2_pm_ts)),
        "r2_peak_mm_acrossSamples": r2_scalar(peak_mm_true, peak_mm_pred),
        "r2_endpoint_pm_acrossSamples": r2_scalar(end_pm_true, end_pm_pred),
    }
    return columns, summary


def write_csv(path: Path, columns: dict):
    keys = list(columns.keys())
    n = len(columns[keys[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([columns[k][i] for k in keys])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/gpfs/home2/overven1/theta-lab")
    p.add_argument("--out", default="results/oracle_rollout_scoring")
    p.add_argument("--nrmse-thresholds", nargs="+", type=float,
                   default=[0.05, 0.1, 0.2, 0.3, 0.5])
    args = p.parse_args()

    root = Path(args.root)
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_cols = []
    summaries = []
    for name, rel in SCAFFOLDS.items():
        path = root / rel
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        cols, summ = score(path, name)
        write_csv(out_dir / f"per_sample_{name}.csv", cols)
        all_cols.append(cols)
        summaries.append(summ)
        print(f"\n=== {name} ===")
        for k, v in summ.items():
            print(f"  {k}: {v}")

        nrmse_mm = cols["nrmse_mm_rollout"]
        nrmse_pm = cols["nrmse_pm_rollout"]
        sidx = cols["sample_idx"]
        for thr in args.nrmse_thresholds:
            mask = (nrmse_mm < thr) & (nrmse_pm < thr)
            kept = sidx[mask]
            frac = mask.mean()
            print(f"  threshold nrmse<{thr}: {mask.sum()}/{len(sidx)} ({frac:.1%}) kept")
            np.save(out_dir / f"good_indices_{name}_thr{thr}.npy", kept)

    if summaries:
        keys = list(summaries[0].keys())
        with open(out_dir / "summary.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for s in summaries:
                w.writerow([s[k] for k in keys])
    if all_cols:
        merged = {k: np.concatenate([c[k] for c in all_cols]) for k in all_cols[0]}
        write_csv(out_dir / "per_sample_all.csv", merged)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
