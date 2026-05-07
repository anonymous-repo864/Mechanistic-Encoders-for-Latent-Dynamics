"""Plot NRMSE vs number of mechanistic equations (P) across scaffold sizes.

Accepts an experiment folder directly. Computes NRMSE on the test split
of each run (or loads cached results from <exp_root>/nrmse_cache.csv).

Usage:
    python last-layer-ode/metrics/plot_nrmse.py experiments/scaffold_size_effect
    python last-layer-ode/metrics/plot_nrmse.py experiments/scaffold_size_effect --recompute
    python last-layer-ode/metrics/plot_nrmse.py experiments/scaffold_size_effect --stat mean --error-bar std
    python last-layer-ode/metrics/plot_nrmse.py experiments/scaffold_size_effect --out results/scaffold_size.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.nrmse import load_or_compute


def plot(rows: list[dict], stat: str, error_bar: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # aggregate per (scaffold, P, species) — average across runs with same scaffold
    from collections import defaultdict
    data: dict[tuple[str, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["scaffold"], r["P"])
        data[key][r["species"]].append(r)

    # collect unique species and P values
    all_species = sorted({r["species"] for r in rows})
    all_P = sorted({r["P"] for r in rows})

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cmap = plt.get_cmap("tab10")

    for i, sp in enumerate(all_species):
        xs, ys, errs = [], [], []
        for P in all_P:
            # gather all rows for this species at this P
            sp_rows = [r for r in rows if r["species"] == sp and r["P"] == P]
            if not sp_rows:
                continue
            vals = np.array([r[stat] for r in sp_rows])
            y = float(np.mean(vals))  # average across runs at same P
            if error_bar == "iqr":
                q25 = float(np.mean([r["q25"] for r in sp_rows]))
                q75 = float(np.mean([r["q75"] for r in sp_rows]))
                err = np.array([[y - q25], [q75 - y]])
            else:
                err = float(np.mean([r[error_bar] for r in sp_rows]))
            xs.append(P)
            ys.append(y)
            errs.append(err)

        if not xs:
            continue

        yerr = np.hstack(errs) if error_bar == "iqr" else np.array(errs)
        ax.errorbar(xs, ys, yerr=yerr,
                    marker="o", linestyle="--", linewidth=1.8, markersize=5.5,
                    color=cmap(i % 10), alpha=0.85, capsize=4, label=sp)

    ax.set_xlabel("Number of mechanistic equations ($P$)", fontsize=14)
    ax.set_ylabel("NRMSE (lower is better)", fontsize=14)
    ax.set_xticks(all_P)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="Experiment folder, e.g. experiments/scaffold_size_effect")
    parser.add_argument("--recompute", action="store_true", help="Ignore cache and recompute NRMSE")
    parser.add_argument("--stat", choices=["mean", "median"], default="median")
    parser.add_argument("--error-bar", choices=["sem", "std", "iqr"], default="iqr")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--format", choices=["pdf", "png"], default="pdf")
    args = parser.parse_args()

    root = Path(args.root)
    rows = load_or_compute(root, recompute=args.recompute)

    fmt = args.format
    out = Path(args.out) if args.out else root / f"nrmse_vs_P.{fmt}"
    plot(rows, stat=args.stat, error_bar=args.error_bar, out_path=out)
