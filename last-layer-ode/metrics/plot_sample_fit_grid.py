"""Plot true vs fit for one sample across all experiments in a folder.

Creates one stacked summary grid with up to N experiment columns (default 4).
Within each experiment tile, requested species are stacked vertically
(e.g., A on top, M below) to match the honest-plot style.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plot_diagnostics import rebuild_model_from_experiment, device_auto, _filter_model_kwargs


def list_experiment_dirs(exp_root: Path) -> list[Path]:
    exp_dirs = [
        d for d in exp_root.iterdir()
        if d.is_dir() and (d / "config.yaml").exists() and (d / "model.pt").exists()
    ]

    def sort_key(path: Path):
        cfg = yaml.safe_load((path / "config.yaml").read_text())
        scaffold = str(cfg.get("scaffold", path.name))

        m = re.search(r"(reduced|full)(\d+)", scaffold.lower())
        if m:
            num = int(m.group(2))
            return (0, num, scaffold, path.name)

        any_num = re.search(r"(\d+)", scaffold)
        if any_num:
            num = int(any_num.group(1))
            return (1, num, scaffold, path.name)

        return (2, float("inf"), scaffold, path.name)

    return sorted(exp_dirs, key=sort_key)


def pick_dataset_index(exp_dir: Path, sample_pos: int) -> int:
    split_path = exp_dir / "split.npz"
    if split_path.exists():
        test_idx = np.load(split_path)["test_idx"]
        if len(test_idx) == 0:
            return 0
        pos = int(np.clip(sample_pos, 0, len(test_idx) - 1))
        return int(test_idx[pos])
    return max(0, int(sample_pos))


def predict_sample(model, ds, dataset_idx: int, device: torch.device, cfg: dict = None):
    y0, u_seq, y_seq = ds[dataset_idx]
    dt = torch.tensor(ds.dt.astype(np.float32)).unsqueeze(0).to(device)

    import inspect
    forward_params = inspect.signature(model.forward).parameters
    needs_obs_idx = "obs_idx" in forward_params

    extra = {}
    if needs_obs_idx:
        if cfg and cfg.get("obs_idx") is not None:
            obs_idx = torch.tensor(cfg["obs_idx"], device=device, dtype=torch.long)
        else:
            obs_idx = torch.arange(y0.shape[-1], device=device, dtype=torch.long)
        extra["obs_idx"] = obs_idx

    base_kwargs = {
        "y_seq": None,
        "teacher_forcing": False,
        "u_transform": str((cfg or {}).get("u_transform", "none")),
        "y_transform": str((cfg or {}).get("y_transform", "none")),
    }

    with torch.no_grad():
        out = model(
            y0.unsqueeze(0).to(device),
            u_seq.unsqueeze(0).to(device),
            dt,
            **extra,
            **_filter_model_kwargs(model, base_kwargs),
        )

    pred = out[0] if isinstance(out, (tuple, list)) else out
    y_true = y_seq.cpu().numpy()
    y_fit = pred[0].detach().cpu().numpy()
    t = np.cumsum(ds.dt.astype(np.float32))
    return t, y_true, y_fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root", type=str, help="Folder containing experiment run directories")
    parser.add_argument("--species", nargs="*", default=["A", "M"], help="Species to plot")
    parser.add_argument("--sample-pos", type=int, default=0,
                        help="Position inside test split (0=first test sample)")
    parser.add_argument("--max-cols", type=int, default=4)
    parser.add_argument("--col-w", type=float, default=2.6,
                        help="Column width in inches (matches honest plot default)")
    parser.add_argument("--row-h", type=float, default=2.2,
                        help="Row height in inches (matches honest plot default)")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: exp_root)")
    args = parser.parse_args()

    exp_root = Path(args.exp_root)
    outdir = Path(args.outdir) if args.outdir else exp_root
    outdir.mkdir(parents=True, exist_ok=True)

    species_list = [s.upper() for s in args.species]
    max_cols = max(1, int(args.max_cols))
    device = device_auto()

    exp_dirs = list_experiment_dirs(exp_root)
    if not exp_dirs:
        raise FileNotFoundError(f"No experiment folders found in {exp_root}")

    panel_data = []
    for exp_dir in exp_dirs:
        cfg = yaml.safe_load((exp_dir / "config.yaml").read_text())
        scaffold = str(cfg.get("scaffold", exp_dir.name))

        model, ds, state_names, _ = rebuild_model_from_experiment(exp_dir, device)
        snames = [s.upper() for s in state_names]
        P = len(state_names)

        dataset_idx = pick_dataset_index(exp_dir, args.sample_pos)
        if dataset_idx >= len(ds):
            continue

        t, y_true, y_fit = predict_sample(model, ds, dataset_idx, device, cfg=cfg)
        panel_data.append({
            "exp_dir": exp_dir,
            "scaffold": scaffold,
            "P": P,
            "dataset_idx": dataset_idx,
            "state_names": snames,
            "t": t,
            "y_true": y_true,
            "y_fit": y_fit,
        })

    if not panel_data:
        raise RuntimeError("No valid experiment predictions were produced.")

    n_panels = len(panel_data)
    n_cols = min(max_cols, n_panels)
    n_exp_rows = int(math.ceil(n_panels / n_cols))
    n_species = len(species_list)
    n_total_rows = n_exp_rows * n_species

    fig, axes = plt.subplots(
        n_total_rows,
        n_cols,
        figsize=(args.col_w * n_cols + 0.5, args.row_h * n_total_rows + 1.0),
        squeeze=False,
    )

    for i, info in enumerate(panel_data):
        exp_row = i // n_cols
        col = i % n_cols

        for s_idx, sp in enumerate(species_list):
            row = exp_row * n_species + s_idx
            ax = axes[row, col]
            ax.grid(True, alpha=0.25)

            if s_idx == 0:
                ax.set_title(f"{info['scaffold']} (P={info['P']})", fontsize=9)

            if sp not in info["state_names"]:
                ax.text(0.5, 0.5, f"{sp} missing", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                j = info["state_names"].index(sp)
                t = info["t"]
                y_true = info["y_true"][:, j]
                y_fit = info["y_fit"][:, j]
                ax.plot(t, y_true, lw=1.8, label="true")
                ax.plot(t, y_fit, lw=1.8, ls="--", label="fit")

            if col == 0:
                ax.set_ylabel(sp)
            if exp_row == n_exp_rows - 1 and s_idx == n_species - 1:
                ax.set_xlabel("time")
            if i == 0 and s_idx == 0:
                ax.legend(fontsize=8)

    total_cells = n_exp_rows * n_cols
    for idx in range(n_panels, total_cells):
        exp_row = idx // n_cols
        col = idx % n_cols
        for s_idx in range(n_species):
            row = exp_row * n_species + s_idx
            axes[row, col].axis("off")

    species_tag = "_".join(species_list)
    fig.suptitle(
        f"True vs fit stacked grid ({', '.join(species_list)}) (sample_pos={args.sample_pos})",
        y=0.995,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = outdir / f"fit_grid_stacked_{species_tag}.{args.format}"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
