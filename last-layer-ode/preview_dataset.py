from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return np.load(str(path), allow_pickle=True)


def _as_str_array(arr) -> Optional[list[str]]:
    if arr is None:
        return None
    return np.asarray(arr).astype(str).tolist()


def _print_summary(d) -> None:
    print("Dataset summary")
    print("=" * 60)
    for key in ["y0", "u_seq", "y_seq", "t_obs", "obs_indices", "control_indices"]:
        if key in d:
            print(f"{key:<16} {np.asarray(d[key]).shape}")
    print("=" * 60)


def _nonzero_u_stats(u_seq: np.ndarray, control_names: Optional[list[str]] = None) -> None:
    u_seq = np.asarray(u_seq, dtype=np.float32)
    N, K, U = u_seq.shape
    nonzero = (u_seq > 0).sum(axis=(0, 1))
    totals = u_seq.sum(axis=(0, 1))

    print("Control-channel usage")
    print("-" * 60)
    for j in range(U):
        name = control_names[j] if (control_names is not None and j < len(control_names)) else f"u[{j}]"
        print(f"{name:<12} nonzero_bins={int(nonzero[j]):>6}   total_amt={float(totals[j]):>10.3f}")
    print("-" * 60)


def _value_range_stats(y0: np.ndarray, y_seq: np.ndarray, obs_names: Optional[list[str]]) -> None:
    """Print per-species value ranges to help diagnose scale mismatches."""
    P = y_seq.shape[-1]
    print("Per-species value ranges (across all samples)")
    print("-" * 60)
    for p in range(P):
        name = obs_names[p] if (obs_names is not None and p < len(obs_names)) else f"y[{p}]"
        vals = np.concatenate([y0[:, p].reshape(-1), y_seq[:, :, p].reshape(-1)])
        print(f"  {name:<8s}  min={vals.min():.4f}  max={vals.max():.4f}  mean={vals.mean():.4f}  std={vals.std():.4f}")
    print("-" * 60)


def _plot_per_species_subplots(
    out_dir: Path,
    t_obs: np.ndarray,
    y0: np.ndarray,
    y_seq: np.ndarray,
    u_seq: Optional[np.ndarray],
    obs_names: Optional[list[str]],
    n_plot: int,
) -> None:
    """
    One figure per sample with a subplot per observed species.
    Each species gets its own y-axis scale — no more T squashing concentrations.
    """
    N, K, P = y_seq.shape
    n_plot = min(int(n_plot), N)
    tt = np.asarray(t_obs, dtype=np.float32)[1:]

    for i in range(n_plot):
        fig, axes = plt.subplots(P, 1, figsize=(12, max(6, 2.0 * P)), sharex=True)
        if P == 1:
            axes = [axes]

        for p, ax in enumerate(axes):
            name = obs_names[p] if (obs_names is not None and p < len(obs_names)) else f"y[{p}]"
            ax.plot(tt, y_seq[i, :, p], linewidth=1.8, color=f"C{p}")
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.25)

            # Mark bolus timesteps
            if u_seq is not None:
                u_i = np.asarray(u_seq[i], dtype=np.float32)
                bolus_bins = np.where(u_i.sum(axis=1) > 0)[0]
                for k in bolus_bins:
                    ax.axvline(tt[k], linewidth=0.6, alpha=0.12, color="gray")

        axes[-1].set_xlabel("time")
        fig.suptitle(f"Per-species trajectories — sample {i}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"y_subplots_{i:03d}.png", dpi=170)
        plt.close(fig)


def _plot_overlay_grouped(
    out_dir: Path,
    t_obs: np.ndarray,
    y0: np.ndarray,
    y_seq: np.ndarray,
    u_seq: Optional[np.ndarray],
    obs_names: Optional[list[str]],
    n_plot: int,
) -> None:
    """
    Group species by scale: concentrations on one axis, environment on another.
    Auto-detects 'environment' species by checking if initial values >> typical concentrations.
    """
    N, K, P = y_seq.shape
    n_plot = min(int(n_plot), N)
    tt = np.asarray(t_obs, dtype=np.float32)[1:]

    # Heuristic: species whose mean initial value is > 50 are likely environment (T, etc.)
    mean_y0 = np.abs(y0).mean(axis=0)
    env_mask = mean_y0 > 50.0
    conc_idx = [p for p in range(P) if not env_mask[p]]
    env_idx = [p for p in range(P) if env_mask[p]]

    if len(env_idx) == 0:
        # No environment species — just do a normal overlay
        groups = [("all species", list(range(P)))]
    elif len(conc_idx) == 0:
        groups = [("all species", list(range(P)))]
    else:
        groups = [("concentrations", conc_idx), ("environment", env_idx)]

    for i in range(n_plot):
        n_groups = len(groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(12, 5 * n_groups), sharex=True)
        if n_groups == 1:
            axes = [axes]

        for ax, (group_name, indices) in zip(axes, groups):
            for p in indices:
                name = obs_names[p] if (obs_names is not None and p < len(obs_names)) else f"y[{p}]"
                ax.plot(tt, y_seq[i, :, p], linewidth=1.6, label=name, alpha=0.85)
            ax.set_ylabel(group_name)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.25)

            if u_seq is not None:
                u_i = np.asarray(u_seq[i], dtype=np.float32)
                bolus_bins = np.where(u_i.sum(axis=1) > 0)[0]
                for k in bolus_bins:
                    ax.axvline(tt[k], linewidth=0.6, alpha=0.12, color="gray")

        axes[-1].set_xlabel("time")
        fig.suptitle(f"Grouped overlay — sample {i}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"y_grouped_{i:03d}.png", dpi=170)
        plt.close(fig)


def _plot_u_per_channel(
    out_dir: Path,
    t_obs: np.ndarray,
    u_seq: np.ndarray,
    control_names: Optional[list[str]],
    n_plot: int,
) -> None:
    """Per-channel bolus plots (one subplot per control channel)."""
    N, K, U = u_seq.shape
    n_plot = min(int(n_plot), N)
    tt = np.asarray(t_obs, dtype=np.float32)[1:]
    bar_w = (tt[1] - tt[0]) * 0.85 if K > 1 else 1.0

    for i in range(n_plot):
        fig, axes = plt.subplots(U, 1, figsize=(12, max(4, 2.0 * U)), sharex=True)
        if U == 1:
            axes = [axes]

        for j, ax in enumerate(axes):
            name = control_names[j] if (control_names is not None and j < len(control_names)) else f"u[{j}]"
            ax.bar(tt, u_seq[i, :, j], width=bar_w, alpha=0.8, color=f"C{j}")
            ax.set_ylabel(name)
            ax.grid(True, axis="y", alpha=0.25)

        axes[-1].set_xlabel("time")
        fig.suptitle(f"Bolus inputs per channel — sample {i}", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"u_channels_{i:03d}.png", dpi=170)
        plt.close(fig)


def _plot_u_total(
    out_dir: Path,
    t_obs: np.ndarray,
    u_seq: np.ndarray,
    n_plot: int,
) -> None:
    N, K, U = u_seq.shape
    n_plot = min(int(n_plot), N)
    tt = np.asarray(t_obs, dtype=np.float32)[1:]

    for i in range(n_plot):
        total_u = np.asarray(u_seq[i], dtype=np.float32).sum(axis=1)
        fig, ax = plt.subplots(figsize=(11, 3.8))
        ax.bar(tt, total_u, width=(tt[1] - tt[0]) * 0.85 if K > 1 else 1.0, alpha=0.8)
        ax.set_title(f"Total bolus per timestep — sample {i}")
        ax.set_xlabel("time")
        ax.set_ylabel("Σ bolus (all channels)")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"u_total_{i:03d}.png", dpi=170)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=str, help="Path to .npz dataset")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--n-plot", type=int, default=3, help="How many samples to plot")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    d = _load_npz(ds_path)

    _print_summary(d)

    y0 = np.asarray(d["y0"], dtype=np.float32)
    u_seq = np.asarray(d["u_seq"], dtype=np.float32)
    y_seq = np.asarray(d["y_seq"], dtype=np.float32)
    t_obs = np.asarray(d["t_obs"], dtype=np.float32)

    control_names = _as_str_array(d.get("control_names", None))
    obs_names = _as_str_array(d.get("obs_names", None))

    _nonzero_u_stats(u_seq, control_names=control_names)
    _value_range_stats(y0, y_seq, obs_names)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else ds_path.with_suffix("").with_name(ds_path.stem + "_preview")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Per-species subplots (always readable regardless of scale)
    _plot_per_species_subplots(out_dir, t_obs, y0, y_seq, u_seq, obs_names, n_plot=args.n_plot)

    # 2. Grouped overlay (concentrations vs environment on separate axes)
    _plot_overlay_grouped(out_dir, t_obs, y0, y_seq, u_seq, obs_names, n_plot=args.n_plot)

    # 3. Per-channel bolus inputs
    _plot_u_per_channel(out_dir, t_obs, u_seq, control_names, n_plot=args.n_plot)

    # 4. Total bolus (backward compat)
    _plot_u_total(out_dir, t_obs, u_seq, n_plot=args.n_plot)

    print(f"\nSaved preview plots to: {out_dir}")


if __name__ == "__main__":
    main()