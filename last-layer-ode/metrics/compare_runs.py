"""Compare all runs in an experiment folder by NRMSE.

Computes NRMSE on the test split of each run (or loads cached results).
Cache is saved to <exp_root>/nrmse_cache.csv.

Usage:
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --recompute
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --csv results/scaffold.csv
    python last-layer-ode/metrics/compare_runs.py experiments/scaffold_size_effect --plot
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.nrmse import load_or_compute


OBS_SPECIES = {"mm", "pm"}  # only observed species in the TXTL dataset


def _load_or_compute_endpoint_r2(
    exp_dir: Path,
    device,
    protein_sp: str,
    mrna_sp: str,
    split: str,
    recompute: bool,
) -> tuple[float, float]:
    """Read <exp_dir>/r2_cache.csv if present and matches species, else roll
    out the model and compute R² on (protein final, mRNA max). Returns
    (r2_protein_final, r2_mrna_max) — NaN if computation fails or species
    aren't in the run's obs_names."""
    cache = exp_dir / "r2_cache.csv"
    if cache.exists() and not recompute:
        with open(cache) as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if row and "r2_protein_final" in row and "r2_mrna_max" in row:
            try:
                return float(row["r2_protein_final"]), float(row["r2_mrna_max"])
            except ValueError:
                pass

    from metrics.endpoint_r2 import collect_endpoints, save_r2_cache, r2  # local import: heavy deps
    try:
        result = collect_endpoints(exp_dir, device, split, protein_sp, mrna_sp)
    except Exception as e:
        print(f"  [endpoint-r2] skip {exp_dir.name}: {e}")
        return float("nan"), float("nan")
    r2_p = r2(result["true_protein_final"], result["pred_protein_final"])
    r2_m = r2(result["true_mrna_max"],      result["pred_mrna_max"])
    save_r2_cache(exp_dir, result, r2_p, r2_m)
    return r2_p, r2_m


def _aggregate(rows: list[dict], stat: str = "median") -> list[dict]:
    """One row per run: mean of mm/pm medians/means as the sort key.

    Args:
        rows: Raw rows from load_or_compute
        stat: Which field to aggregate ("median" or "mean")
    """
    by_run: dict[str, dict] = {}
    for r in rows:
        run = r["run"]
        if run not in by_run:
            by_run[run] = {"run": run, "scaffold": r["scaffold"],
                           "P": r["P"], "species": {}}
        by_run[run]["species"][r["species"]] = r[stat]

    result = []
    for run, d in by_run.items():
        obs_vals = [v for s, v in d["species"].items() if s in OBS_SPECIES]
        if not obs_vals:
            obs_vals = [v for v in d["species"].values() if v is not None and not np.isnan(v)]
        d["primary"] = float(np.mean(obs_vals)) if obs_vals else float("nan")
        result.append(d)
    return sorted(result, key=lambda r: r["primary"])


def print_table(runs: list[dict]) -> None:
    if not runs:
        print("No completed runs found.")
        return

    all_species: list[str] = []
    for r in runs:
        for s in r["species"]:
            if s not in all_species:
                all_species.append(s)

    col_w, sp_w = 55, 10
    header = f"{'run':<{col_w}}  {'NRMSE':>8}"
    for s in all_species:
        header += f"  {s[:sp_w]:>{sp_w}}"
    has_r2 = any("r2_pm_final" in r for r in runs)
    if has_r2:
        header += f"  {'R²(pm_final)':>14}  {'R²(mm_max)':>12}"
    print(header)
    print("-" * len(header))

    for r in runs:
        line = f"{r['run']:<{col_w}}  {r['primary']:>8.4f}"
        for s in all_species:
            val = r["species"].get(s)
            line += f"  {val:>{sp_w}.4f}" if val is not None else f"  {'N/A':>{sp_w}}"
        if has_r2:
            r2p, r2m = r.get("r2_pm_final", float("nan")), r.get("r2_mm_max", float("nan"))
            line += f"  {r2p:>14.4f}  {r2m:>12.4f}"
        print(line)


def save_csv(runs: list[dict], path: Path, stat: str = "median") -> None:
    """Save results CSV.

    Args:
        runs: Aggregated runs from _aggregate
        path: Output path
        stat: Label for the aggregation method ("median" or "mean")
    """
    all_species: list[str] = []
    for r in runs:
        for s in r["species"]:
            if s not in all_species:
                all_species.append(s)

    has_r2 = any("r2_pm_final" in r for r in runs)
    extra_cols = ["r2_pm_final", "r2_mm_max"] if has_r2 else []

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "nrmse"] + all_species + extra_cols)
        for r in runs:
            name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            row = (
                [name, f"{r['primary']:.6f}"]
                + [f"{r['species'].get(s, ''):.6f}" if r["species"].get(s) is not None else ""
                   for s in all_species]
            )
            if has_r2:
                for k in ("r2_pm_final", "r2_mm_max"):
                    v = r.get(k)
                    row.append(f"{v:.6f}" if v is not None and not np.isnan(v) else "")
            writer.writerow(row)
    print(f"Saved CSV to {path} ({stat})")


def plot_comparison(runs: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["run"] for r in runs]
    values = [r["primary"] for r in runs]

    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 0.8), 5))
    bars = ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NRMSE (lower is better)")
    ax.set_title("Run comparison")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Experiment folder, e.g. experiments/scaffold_size_effect")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute NRMSE")
    parser.add_argument("--csv", type=str, default=None, help="Save summary CSVs to this path (generates _median and _mean variants)")
    parser.add_argument("--plot", action="store_true", help="Save a bar chart")
    parser.add_argument("--plot-out", type=str, default=None)
    parser.add_argument("--no-split", action="store_true",
                        help="Ignore saved train/test split; evaluate on full dataset. "
                             "Useful for models trained on a different dataset size.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset path for all runs (e.g. a small eval set). "
                             "Automatically skips cache.")
    parser.add_argument("--endpoint-r2", action="store_true",
                        help="Also compute R² on (protein final, mRNA max) per run "
                             "and add r2_pm_final/r2_mm_max columns. TXTL-only — "
                             "requires species 'mm' and 'pm' in obs_names.")
    parser.add_argument("--protein", default="pm", help="Protein species (with --endpoint-r2)")
    parser.add_argument("--mrna",    default="mm", help="mRNA species (with --endpoint-r2)")
    parser.add_argument("--r2-split", choices=["test", "train", "all"], default="test",
                        help="Split for endpoint R² rollout (default: test)")
    args = parser.parse_args()

    root = Path(args.root)
    rows = load_or_compute(root, recompute=args.recompute,
                           no_split=args.no_split,
                           dataset_override=args.dataset)
    runs_median = _aggregate(rows, stat="median")
    runs_mean = _aggregate(rows, stat="mean")

    if args.endpoint_r2:
        # Compute endpoint R² once per run dir, attach to BOTH median and mean
        # aggregations (the R² values themselves don't depend on aggregation).
        from plot_diagnostics import device_auto
        device = device_auto()
        run_dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and (d / "model.pt").exists())
        r2_by_name = {}
        for d in run_dirs:
            r2_p, r2_m = _load_or_compute_endpoint_r2(
                d, device, args.protein, args.mrna, args.r2_split, args.recompute
            )
            r2_by_name[d.name] = (r2_p, r2_m)
        for runs in (runs_median, runs_mean):
            for r in runs:
                if r["run"] in r2_by_name:
                    r["r2_pm_final"], r["r2_mm_max"] = r2_by_name[r["run"]]

    print_table(runs_median)

    if args.csv:
        csv_path = Path(args.csv)
        # Split path into stem and suffix to insert _median/_mean before extension
        stem = csv_path.stem
        suffix = csv_path.suffix
        parent = csv_path.parent

        median_path = parent / f"{stem}_median{suffix}"
        mean_path = parent / f"{stem}_mean{suffix}"

        save_csv(runs_median, median_path, stat="median")
        save_csv(runs_mean, mean_path, stat="mean")
    if args.plot:
        out = Path(args.plot_out) if args.plot_out else root / "comparison.png"
        plot_comparison(runs_median, out)
