"""Compare endpoint R² across all runs in a study folder.

Computes R²(protein final) and R²(mRNA max) on the test split for each run,
caches results per run (r2_cache.csv), and prints a ranked table.

Usage:
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep --csv results/r2.csv
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep --plot
    python last-layer-ode/metrics/compare_r2.py experiments/txtl_supervisor_combined_sweep --recompute
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.endpoint_r2 import collect_endpoints, r2
from plot_diagnostics import device_auto

CACHE_NAME = "r2_cache.csv"
CACHE_FIELDS = ["run", "n", "r2_protein_final", "r2_mrna_max"]


def _find_exp_dirs(root: Path) -> list[Path]:
    return sorted(
        d for d in root.iterdir()
        if d.is_dir()
        and (d / "config.yaml").exists()
        and (d / "model.pt").exists()
        and (d / "split.npz").exists()
    )


def _load_cache(exp_dir: Path) -> dict | None:
    cache = exp_dir / CACHE_NAME
    if not cache.exists():
        return None
    with open(cache, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[0]
    return {
        "run": row["run"],
        "n": int(row["n"]),
        "r2_protein_final": float(row["r2_protein_final"]),
        "r2_mrna_max": float(row["r2_mrna_max"]),
    }


def _save_cache(exp_dir: Path, result: dict) -> None:
    cache = exp_dir / CACHE_NAME
    with open(cache, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        writer.writerow({k: result[k] for k in CACHE_FIELDS})


def load_or_compute(root: Path, recompute: bool = False) -> list[dict]:
    exp_dirs = _find_exp_dirs(root)
    if not exp_dirs:
        print(f"No completed runs found in {root}")
        return []

    device = device_auto()
    results = []
    for exp_dir in exp_dirs:
        cached = None if recompute else _load_cache(exp_dir)
        if cached is not None:
            results.append(cached)
            continue

        print(f"  computing R² for {exp_dir.name} ...")
        try:
            raw = collect_endpoints(exp_dir, device, split="test", protein_sp="pm", mrna_sp="mm")
            result = {
                "run": exp_dir.name,
                "n": raw["n"],
                "r2_protein_final": r2(raw["true_protein_final"], raw["pred_protein_final"]),
                "r2_mrna_max":      r2(raw["true_mrna_max"],      raw["pred_mrna_max"]),
            }
            _save_cache(exp_dir, result)
            results.append(result)
        except Exception as e:
            print(f"    skip ({exp_dir.name}): {e}")

    return sorted(results, key=lambda r: r["r2_protein_final"], reverse=True)


def print_table(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return
    print(f"\n{'run':<55}  {'n':>4}  {'R²(protein final)':>18}  {'R²(mRNA max)':>13}")
    print("-" * 97)
    for r in results:
        name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
        print(f"{name:<55}  {r['n']:>4}  {r['r2_protein_final']:>18.4f}  {r['r2_mrna_max']:>13.4f}")
    best = results[0]
    best_name = re.sub(r"^\d{8}_\d{6}_", "", best["run"])
    print(f"\nBest: {best_name}  "
          f"R²(protein)={best['r2_protein_final']:.4f}  R²(mRNA)={best['r2_mrna_max']:.4f}")


def save_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "n", "r2_protein_final", "r2_mrna_max"])
        for r in results:
            name = re.sub(r"^\d{8}_\d{6}_", "", r["run"])
            writer.writerow([name, r["n"],
                             f"{r['r2_protein_final']:.6f}",
                             f"{r['r2_mrna_max']:.6f}"])
    print(f"Saved CSV → {path}")


def plot_comparison(results: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [re.sub(r"^\d{8}_\d{6}_", "", r["run"]) for r in results]
    r2_prot = [r["r2_protein_final"] for r in results]
    r2_mrna = [r["r2_mrna_max"]      for r in results]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))
    w = 0.35
    bars1 = ax.bar(x - w / 2, r2_prot, w, label="R²(protein final)", color="steelblue")
    bars2 = ax.bar(x + w / 2, r2_mrna, w, label="R²(mRNA max)",       color="darkorange")

    for bar, val in zip(list(bars1) + list(bars2), r2_prot + r2_mrna):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²  (higher is better)")
    ax.set_title("Endpoint R² comparison")
    ax.legend()
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Study folder, e.g. experiments/txtl_supervisor_combined_sweep")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute all")
    parser.add_argument("--csv",       type=str, default=None, help="Save summary CSV to this path")
    parser.add_argument("--plot",      action="store_true",    help="Save a grouped bar chart")
    parser.add_argument("--plot-out",  type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    results = load_or_compute(root, recompute=args.recompute)
    print_table(results)

    if args.csv:
        save_csv(results, Path(args.csv))
    if args.plot:
        out = Path(args.plot_out) if args.plot_out else root / "r2_comparison.png"
        plot_comparison(results, out)
