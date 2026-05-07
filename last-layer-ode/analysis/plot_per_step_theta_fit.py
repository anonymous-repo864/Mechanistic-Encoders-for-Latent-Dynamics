"""
plot_per_step_theta_fit.py

Standalone plotting from CSV exports produced by per_step_theta_fit.py.
No fitting — reads the exports/ directory and regenerates plots.

Produces two figures:
  1. Summary grid : truth vs rollout per species × scaffold
  2. NRMSE vs P   : rollout NRMSE as a function of scaffold size

Usage:
    python analysis/plot_per_step_theta_fit.py \
        --export-dir results/manual_theta_fit_results_hi10_low1e-5/exports \
        --sample-idx 0 \
        --out results/manual_theta_fit_results_hi10_low1e-5/figures/summary
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffolds import SCAFFOLDS

FULL_SPECIES    = list("ABCDEFGHIJKLM")
SCAFFOLD_ALIASES = {"reduced13": "full13", "full": "full13"}


def normalize_scaffold_name(name: str) -> str:
    return SCAFFOLD_ALIASES.get(name.strip(), name.strip())


def nrmse(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return np.nan
    rng = true[mask].max() - true[mask].min()
    if rng < 1e-10:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


def load_scaffold_csv(export_dir: Path, scaffold_name: str, sample_idx: int) -> dict:
    sc_dir    = export_dir / scaffold_name
    pred_file = sc_dir / f"predictions_sample{sample_idx}.csv"
    loss_file = sc_dir / f"losses_sample{sample_idx}.csv"

    if not pred_file.exists():
        return None

    with open(pred_file) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(pred_file, delimiter=",", skiprows=1)

    time_col    = data[:, 0]
    true_cols   = [h for h in header if h.startswith("true_")]
    state_names = [h.replace("true_", "") for h in true_cols]
    col_idx     = {h: i for i, h in enumerate(header)}

    true_data    = np.column_stack([data[:, col_idx[f"true_{s}"]]    for s in state_names])
    onestep_data = np.column_stack([data[:, col_idx[f"onestep_{s}"]] for s in state_names])
    rollout_data = np.column_stack([data[:, col_idx[f"rollout_{s}"]] for s in state_names])

    result = dict(time=time_col, state_names=state_names,
                  true=true_data, onestep=onestep_data, rollout=rollout_data)

    if loss_file.exists():
        loss_data = np.loadtxt(loss_file, delimiter=",", skiprows=1)
        result["loss_onestep"] = loss_data[:, 1]
        result["loss_rollout"] = loss_data[:, 2]

    return result


def discover_scaffolds(export_dir: Path, sample_idx: int):
    return [
        d.name for d in sorted(export_dir.iterdir())
        if d.is_dir() and (d / f"predictions_sample{sample_idx}.csv").exists()
    ]


def make_summary_grid(loaded, scaffold_order, show_species, sample_idx, out_path,
                      max_cols=6, col_w=2.6, row_h=2.2):
    """Compact grid: truth vs rollout only, NRMSE annotated per panel."""
    n_scaffolds = len(scaffold_order)
    n_species   = len(show_species)
    n_cols      = min(n_scaffolds, max_cols)
    n_rows      = ((n_scaffolds + n_cols - 1) // n_cols) * n_species

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(col_w * n_cols + 0.5, row_h * n_rows + 1.0),
        squeeze=False,
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for sc_idx, sn in enumerate(scaffold_order):
        col       = sc_idx % n_cols
        row_group = sc_idx // n_cols
        res       = loaded[sn]
        state_names = res["state_names"]
        tt        = res["time"]

        for sp_idx, sp in enumerate(show_species):
            row = row_group * n_species + sp_idx
            ax  = axes[row][col]
            ax.axis("on")

            if sp not in state_names:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                si = state_names.index(sp)
                gt = res["true"][:, si]
                os = res["onestep"][:, si]
                ro = res["rollout"][:, si]

                ax.plot(tt, gt, lw=1.8, color="tab:blue", label="truth")
                ax.plot(tt, os, lw=1.3, color="tab:green", ls="--", label="one-step", alpha=0.8)
                ax.plot(tt, ro, lw=1.5, color="tab:red", ls=":", label="rollout")

                n_os = nrmse(os, gt)
                n_ro = nrmse(ro, gt)
                ax.text(0.97, 0.95, f"RO {n_ro:.3f}",
                        transform=ax.transAxes, fontsize=7, fontweight="bold",
                        va="top", ha="right",
                        color="tab:red" if n_ro > 0.05 else "tab:green",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
                ax.text(0.97, 0.78, f"OS {n_os:.3f}",
                        transform=ax.transAxes, fontsize=7,
                        va="top", ha="right",
                        color="tab:green",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
                ax.grid(True, alpha=0.15)
                ax.tick_params(labelsize=7)

            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else "?"
            if sp_idx == 0:
                ax.set_title(f"{sn} (P={P_sc})", fontsize=8, pad=3)
            if col == 0:
                ax.set_ylabel(sp, fontsize=10)
            if sp_idx == n_species - 1:
                ax.set_xlabel("time", fontsize=7)
            if sc_idx == 0 and sp_idx == 0:
                ax.legend(fontsize=6, loc="upper left")

    fig.suptitle(
        f"Per-step GD theta fit  (sample {sample_idx})\n"
        f"NRMSE annotated per panel  |  OS = one-step oracle, RO = honest rollout",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary grid  -> {out_path}")


def make_nrmse_vs_P(loaded, scaffold_order, show_species, out_path):
    """Line plot: x = scaffold size P, y = rollout NRMSE, one line per species.
    Scaffolds with the same P are averaged."""
    fig, ax = plt.subplots(figsize=(7, 4))
    _default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _color_map = {"A": "tab:blue", "M": "tab:orange"}

    for _ci, sp in enumerate(show_species):
        # accumulate per-P values then average
        from collections import defaultdict
        p_vals: dict = defaultdict(list)
        for sn in scaffold_order:
            res  = loaded[sn]
            P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else None
            if P_sc is None or sp not in res["state_names"]:
                continue
            si   = res["state_names"].index(sp)
            n_ro = nrmse(res["rollout"][:, si], res["true"][:, si])
            if not np.isnan(n_ro):
                p_vals[P_sc].append(n_ro)

        ps     = sorted(p_vals)
        nrmses = [float(np.mean(p_vals[p])) for p in ps]

        ax.plot(ps, nrmses, "o-", lw=2, markersize=6,
                label=f"species {sp}",
                color=_color_map.get(sp, _default_colors[_ci % len(_default_colors)]))

    ax.set_xlabel("Scaffold size (P)", fontsize=12)
    ax.set_ylabel("NRMSE (honest rollout)", fontsize=12)
    ax.set_title("Trajectory error vs mechanistic scaffold size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted({SCAFFOLDS[sn].P for sn in scaffold_order if sn in SCAFFOLDS}))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"NRMSE vs P    -> {out_path}")


def make_pred_overlays(loaded: dict, scaffold_order: list, sample_idx: int,
                       out_dir: Path, fmt: str) -> None:
    """Per-scaffold prediction overlay: truth / one-step / rollout per species."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sn in scaffold_order:
        res = loaded[sn]
        state_names = res["state_names"]
        tt = res["time"]
        P = len(state_names)

        fig, axes = plt.subplots(P, 1, figsize=(11, max(6, 2.0 * P)), sharex=True)
        if P == 1:
            axes = [axes]

        for i, (ax, sp) in enumerate(zip(axes, state_names)):
            ax.plot(tt, res["true"][:, i],    lw=2.0, color="tab:blue",   label="truth")
            ax.plot(tt, res["onestep"][:, i], lw=1.5, color="tab:green",  ls="--", label="one-step oracle")
            ax.plot(tt, res["rollout"][:, i], lw=1.5, color="tab:red",    ls=":",  label="honest rollout")
            ax.set_ylabel(sp)
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(fontsize=9)

        axes[-1].set_xlabel("Time")
        P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else "?"
        fig.suptitle(f"Oracle theta fit — {sn} (P={P_sc})  sample {sample_idx}")
        fig.tight_layout()
        out_path = out_dir / f"pred_overlay_{sn}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Pred overlay  -> {out_path}")


def make_theta_plots(export_dir: Path, scaffold_order: list, sample_idx: int,
                     out_dir: Path, fmt: str) -> None:
    """Per-scaffold theta trajectory plot (rollout theta over time)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sn in scaffold_order:
        theta_path = export_dir / sn / f"theta_rollout_sample{sample_idx}.csv"
        if not theta_path.exists():
            continue

        data = np.loadtxt(theta_path, delimiter=",", skiprows=1)
        tt = data[:, 0]
        thetas = data[:, 1:]
        D = thetas.shape[1]

        n_cols = 2
        n_rows = (D + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 4.5, n_rows * 2.5),
                                 squeeze=False)
        for j in range(D):
            ax = axes[j // n_cols][j % n_cols]
            ax.plot(tt, thetas[:, j], lw=1.5, color="tab:purple")
            ax.set_ylabel(f"θ{j}", fontsize=10)
            ax.grid(True, alpha=0.25)
        # hide unused panels
        for j in range(D, n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")

        axes[-1][0].set_xlabel("Time")
        P_sc = SCAFFOLDS[sn].P if sn in SCAFFOLDS else "?"
        fig.suptitle(f"Oracle theta trajectories (rollout) — {sn} (P={P_sc})  sample {sample_idx}")
        fig.tight_layout()
        out_path = out_dir / f"theta_{sn}.{fmt}"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Theta plot    -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot per_step_theta_fit results from CSV exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export-dir",   type=str, required=True)
    parser.add_argument("--scaffolds",    type=str, default=None,
                        help="Comma-separated scaffold names (default: auto-discover).")
    parser.add_argument("--show-species", type=str, default="A,M")
    parser.add_argument("--sample-idx",   type=int, default=0)
    parser.add_argument("--max-cols",     type=int, default=6)
    parser.add_argument("--fmt",          type=str, default="pdf")
    parser.add_argument("--out",          type=str, required=True,
                        help="Output stem (no extension). E.g. results/figures/summary")
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    if not export_dir.exists():
        print(f"[error] Export dir not found: {export_dir}")
        sys.exit(1)

    if args.scaffolds:
        raw = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
        scaffold_order = list(dict.fromkeys(normalize_scaffold_name(s) for s in raw))
    else:
        scaffold_order = discover_scaffolds(export_dir, args.sample_idx)
        print(f"Auto-discovered {len(scaffold_order)} scaffolds.")

    loaded = {}
    for sn in scaffold_order:
        res = load_scaffold_csv(export_dir, sn, args.sample_idx)
        if res is None:
            print(f"[warn] No CSV for '{sn}' sample {args.sample_idx} — skipping.")
        else:
            loaded[sn] = res

    scaffold_order = [sn for sn in scaffold_order if sn in loaded]
    if not scaffold_order:
        print("[error] No data loaded.")
        sys.exit(1)

    # sort by scaffold size P (ascending), then alphabetically within same P
    scaffold_order.sort(key=lambda sn: (SCAFFOLDS[sn].P if sn in SCAFFOLDS else 999, sn))
    print(f"Loaded: {', '.join(scaffold_order)}")

    show_species = [s.strip() for s in args.show_species.split(",") if s.strip()]

    fmt  = args.fmt.strip(".").lower()
    stem = Path(args.out)

    out_dir = stem.parent

    make_summary_grid(loaded, scaffold_order, show_species, args.sample_idx,
                      out_dir / f"{stem.name}_grid.{fmt}", max_cols=args.max_cols)

    make_nrmse_vs_P(loaded, scaffold_order, show_species,
                    out_dir / f"{stem.name}_nrmse_vs_P.{fmt}")

    make_pred_overlays(loaded, scaffold_order, args.sample_idx,
                       out_dir / "pred_overlays", fmt)

    make_theta_plots(export_dir, scaffold_order, args.sample_idx,
                     out_dir / "theta", fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
