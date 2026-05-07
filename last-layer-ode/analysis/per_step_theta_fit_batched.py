"""
per_step_theta_fit_batched.py

Batched GPU oracle per-step theta fitting.

Instead of ProcessPoolExecutor with one CPU worker per sample, all samples are
stacked into a single tensor and optimized in parallel on the GPU.

  ONE-STEP:   raw shape (N, K, theta_dim) — all samples × all timesteps at once.
  ROLLOUT:    raw shape (N, theta_dim) per timestep — N samples in parallel,
              K timesteps sequentially (required: step k depends on step k-1).

Outputs saved to --out:
  results.npz        — thetas + predictions for all samples
  nrmse.csv          — per-sample NRMSE (one row per sample + summary)
  plots/sample_NNN.pdf (or .png) — trajectory figures (with --plot)

Usage:
    python analysis/per_step_theta_fit_batched.py \\
        --dataset datasets/real_ivtt_raw_no_outliers.npz \\
        --scaffold txtl_resource_and_maturation_dna \\
        --loss-species mm,pm \\
        --n-samples -1 \\
        --gd-steps 400 \\
        --out results/txtl_theta_fit_batched \\
        --plot
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump


# ── helpers ────────────────────────────────────────────────────────────────────

def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


def rk4_batch(rhs, y: torch.Tensor, dt: torch.Tensor,
               theta: torch.Tensor, n_sub: int = 4) -> torch.Tensor:
    """RK4 step. y: (B,P), dt: (B,) or scalar, theta: (B,theta_dim)."""
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = rhs(y,                 theta)
        k2 = rhs(y + 0.5 * h * k1, theta)
        k3 = rhs(y + 0.5 * h * k2, theta)
        k4 = rhs(y +       h * k3,  theta)
        y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return torch.clamp_min(y, 0.0)


def _nrmse_np(pred: np.ndarray, true: np.ndarray) -> float:
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return float("nan")
    rng = float(true[mask].max() - true[mask].min())
    if rng < 1e-10:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── core fitting ───────────────────────────────────────────────────────────────

def fit_batched(
    scaffold_name: str,
    dataset_path: str,
    sample_indices: list[int],
    gd_steps: int = 400,
    lr: float = 0.05,
    n_substeps: int = 4,
    loss_species: list[str] | None = None,
    no_rollout: bool = False,
    drop_states: list[str] | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Returns dict with keys:
        y_true        : (N, K+1, P)  ground-truth (y0 prepended)
        t_obs         : (K+1,)
        state_names   : list[str]
        pred_onestep  : (N, K, P)
        thetas_onestep: (N, K, theta_dim)
        pred_rollout  : (N, K, P)    or zeros if no_rollout
        thetas_rollout: (N, K, theta_dim) or zeros if no_rollout
        sample_indices: list[int]
    """
    sc        = SCAFFOLDS[scaffold_name]
    rhs       = sc
    theta_dim = sc.theta_dim
    P         = sc.P

    d = np.load(dataset_path, allow_pickle=True)

    # Optionally project the dataset down to a smaller scaffold by dropping
    # observed states that the scaffold doesn't model (e.g. drop "O" to use
    # txtl_maturation_dna with the 7-state real_ivtt dataset).
    obs_indices_full = d["obs_indices"].astype(np.int64)         # (P_data,)
    control_indices  = d["control_indices"].astype(np.int64)     # (U,)
    obs_names_full   = (
        [str(s) for s in d["obs_names"]] if "obs_names" in d.files else None
    )
    keep_pos = np.arange(len(obs_indices_full))                  # default: keep all
    if drop_states:
        if obs_names_full is None:
            raise ValueError("--drop-states requires obs_names in the dataset.")
        unknown = [s for s in drop_states if s not in obs_names_full]
        if unknown:
            raise ValueError(
                f"--drop-states {unknown} not in dataset obs_names {obs_names_full}"
            )
        keep_pos = np.array(
            [i for i, n in enumerate(obs_names_full) if n not in drop_states],
            dtype=np.int64,
        )
        kept_names = [obs_names_full[i] for i in keep_pos]
        print(f"  Dropping states {drop_states}; keeping {kept_names} (P={len(keep_pos)})")
        if len(keep_pos) != P:
            raise ValueError(
                f"After dropping {drop_states}, dataset has P={len(keep_pos)} states "
                f"but scaffold '{scaffold_name}' expects P={P}. "
                f"Kept order: {kept_names}; scaffold order: {sc.state_names}"
            )
        if list(kept_names) != list(sc.state_names):
            raise ValueError(
                f"State order mismatch after drop. "
                f"Kept order: {kept_names}; scaffold order: {sc.state_names}. "
                f"The two must match position-for-position."
            )

    obs_indices_new = obs_indices_full[keep_pos]                 # (P,)

    # Build per-parameter bounds
    if sc.theta_lo_vec is not None and sc.theta_hi_vec is not None:
        lo_t = torch.tensor(sc.theta_lo_vec, dtype=torch.float32, device=device)
        hi_t = torch.tensor(sc.theta_hi_vec, dtype=torch.float32, device=device)
    else:
        lo_t = torch.full((theta_dim,), 1e-3, device=device)
        hi_t = torch.full((theta_dim,), 2.0,  device=device)

    t_obs = d["t_obs"].astype(np.float32)
    dt_np = np.diff(t_obs).astype(np.float32)
    K     = len(dt_np)

    # Stack selected samples: (N, K+1, P) and (N, K, U)
    N = len(sample_indices)
    y0_np  = d["y0"][sample_indices][:, keep_pos].astype(np.float32)        # (N, P)
    y_np   = d["y_seq"][sample_indices][:, :, keep_pos].astype(np.float32)  # (N, K, P)
    u_np   = d["u_seq"][sample_indices].astype(np.float32)                  # (N, K, U)

    y_true_np = np.concatenate([y0_np[:, None, :], y_np], axis=1)  # (N, K+1, P)

    jump = make_u_to_y_jump(
        torch.from_numpy(control_indices),
        torch.from_numpy(obs_indices_new),
        device=device,
    )

    y_true  = torch.from_numpy(y_true_np).to(device)   # (N, K+1, P)
    u_seq_t = torch.from_numpy(u_np).to(device)        # (N, K, U)
    dt_t    = torch.from_numpy(dt_np).to(device)       # (K,)

    state_names = list(sc.state_names)
    if loss_species:
        missing = [s for s in loss_species if s not in state_names]
        if missing:
            raise ValueError(f"loss_species {missing} not in scaffold states {state_names}")
        loss_idx = torch.tensor([state_names.index(s) for s in loss_species],
                                dtype=torch.long, device=device)
    else:
        loss_idx = torch.arange(P, device=device)

    # ── ONE-STEP ORACLE ────────────────────────────────────────────────────────
    # y_prev: (N, K, P), y_next: (N, K, P)
    y_prev_os     = y_true[:, :-1, :]                        # (N, K, P)
    y_next_os     = y_true[:, 1:,  :]                        # (N, K, P)
    y_after_jump  = y_prev_os + (u_seq_t @ jump)             # (N, K, P)

    # Flatten N×K into batch dim for rk4
    NK = N * K
    y_flat  = y_after_jump.reshape(NK, P)
    dt_flat = dt_t.unsqueeze(0).expand(N, -1).reshape(NK)   # (N*K,)

    raw_os  = torch.zeros(N, K, theta_dim, device=device, requires_grad=True)
    opt_os  = torch.optim.Adam([raw_os], lr=lr)

    print(f"  One-step oracle: N={N}, K={K}, theta_dim={theta_dim}  [{gd_steps} steps]")
    for step in range(gd_steps):
        opt_os.zero_grad()
        theta_flat = log_gamma(raw_os.reshape(NK, theta_dim), lo_t, hi_t)
        y_hat_flat = rk4_batch(rhs, y_flat, dt_flat, theta_flat, n_substeps)
        y_hat      = y_hat_flat.reshape(N, K, P)
        diff       = torch.log1p(y_hat) - torch.log1p(y_next_os)
        loss       = diff.index_select(-1, loss_idx).pow(2).mean()
        loss.backward()
        opt_os.step()
        if (step + 1) % 100 == 0:
            print(f"    step {step+1:>4}/{gd_steps}  loss={loss.item():.6f}")

    with torch.no_grad():
        theta_flat_final = log_gamma(raw_os.reshape(NK, theta_dim), lo_t, hi_t)
        y_hat_flat_final = rk4_batch(rhs, y_flat, dt_flat, theta_flat_final, n_substeps)
        pred_os   = y_hat_flat_final.reshape(N, K, P).cpu().numpy()
        thetas_os = log_gamma(raw_os, lo_t, hi_t).cpu().numpy()   # (N, K, theta_dim)

    # ── ROLLOUT ────────────────────────────────────────────────────────────────
    if no_rollout:
        pred_ro   = np.zeros((N, K, P), dtype=np.float32)
        thetas_ro = np.zeros((N, K, theta_dim), dtype=np.float32)
    else:
        pred_ro_t   = torch.zeros(N, K, P,         device=device)
        thetas_ro_t = torch.zeros(N, K, theta_dim, device=device)

        y_cur = y_true[:, 0, :].clone()  # (N, P)

        print(f"  Rollout: {K} timesteps × {gd_steps} steps each")
        for k in range(K):
            u_k    = u_seq_t[:, k, :]          # (N, U)
            dt_k   = dt_t[k].expand(N)         # (N,)
            y_tgt  = y_true[:, k + 1, :]       # (N, P)
            y_jmp  = y_cur + (u_k @ jump)      # (N, P)

            raw_k = torch.zeros(N, theta_dim, device=device, requires_grad=True)
            opt_k = torch.optim.Adam([raw_k], lr=lr)

            for _ in range(gd_steps):
                opt_k.zero_grad()
                theta_k = log_gamma(raw_k, lo_t, hi_t)
                y_hat_k = rk4_batch(rhs, y_jmp.detach(), dt_k, theta_k, n_substeps)
                diff_k  = torch.log1p(y_hat_k) - torch.log1p(y_tgt)
                loss_k  = diff_k.index_select(-1, loss_idx).pow(2).mean()
                loss_k.backward()
                opt_k.step()

            with torch.no_grad():
                theta_k = log_gamma(raw_k, lo_t, hi_t)
                y_hat_k = rk4_batch(rhs, y_jmp, dt_k, theta_k, n_substeps)

            pred_ro_t[:, k, :]   = y_hat_k
            thetas_ro_t[:, k, :] = theta_k
            y_cur = y_hat_k.detach()

            if (k + 1) % max(1, K // 10) == 0:
                print(f"    rollout step {k+1:>4}/{K}")

        pred_ro   = pred_ro_t.cpu().numpy()
        thetas_ro = thetas_ro_t.cpu().numpy()

    return dict(
        y_true         = y_true_np,         # (N, K+1, P)
        t_obs          = t_obs,             # (K+1,)
        state_names    = state_names,
        sample_indices = sample_indices,
        pred_onestep   = pred_os,           # (N, K, P)
        thetas_onestep = thetas_os,         # (N, K, theta_dim)
        pred_rollout   = pred_ro,           # (N, K, P)
        thetas_rollout = thetas_ro,         # (N, K, theta_dim)
    )


# ── saving ─────────────────────────────────────────────────────────────────────

def save_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np_path = out_dir / "results.npz"
    np.savez_compressed(
        np_path,
        y_true         = results["y_true"],
        t_obs          = results["t_obs"],
        state_names    = np.array(results["state_names"]),
        sample_indices = np.array(results["sample_indices"]),
        pred_onestep   = results["pred_onestep"],
        thetas_onestep = results["thetas_onestep"],
        pred_rollout   = results["pred_rollout"],
        thetas_rollout = results["thetas_rollout"],
    )
    print(f"Saved results -> {np_path}")


def save_nrmse_csv(results: dict, out_dir: Path,
                   eval_species: list[str] | None = None) -> None:
    state_names = results["state_names"]
    species     = eval_species if eval_species else state_names
    tt          = results["t_obs"][1:]   # (K,) — drop t0
    y_true      = results["y_true"][:, 1:, :]   # (N, K, P)
    pred_os     = results["pred_onestep"]
    pred_ro     = results["pred_rollout"]
    sample_idxs = results["sample_indices"]

    rows = []
    agg: dict[str, dict[str, list[float]]] = {"os": {}, "ro": {}}
    for sp in species:
        agg["os"][sp] = []
        agg["ro"][sp] = []

    for ni, si in enumerate(sample_idxs):
        row: dict = {"sample_idx": si}
        for sp in species:
            if sp not in state_names:
                continue
            j = state_names.index(sp)
            n_os = _nrmse_np(pred_os[ni, :, j], y_true[ni, :, j])
            n_ro = _nrmse_np(pred_ro[ni, :, j], y_true[ni, :, j])
            row[f"os_{sp}"] = f"{n_os:.6f}"
            row[f"ro_{sp}"] = f"{n_ro:.6f}"
            agg["os"][sp].append(n_os)
            agg["ro"][sp].append(n_ro)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "nrmse.csv"
    fieldnames = ["sample_idx"] + [f"{m}_{sp}" for sp in species for m in ("os", "ro")]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        summary = {"sample_idx": "MEAN"}
        for sp in species:
            for m in ("os", "ro"):
                vals = [v for v in agg[m].get(sp, []) if not np.isnan(v)]
                summary[f"{m}_{sp}"] = f"{float(np.mean(vals)):.6f}" if vals else ""
        writer.writerow(summary)

    print(f"Saved NRMSE   -> {csv_path}")

    # Print summary table
    print(f"\n{'Species':<8}  {'OS mean':>10}  {'RO mean':>10}")
    print("-" * 34)
    for sp in species:
        os_vals = [v for v in agg["os"].get(sp, []) if not np.isnan(v)]
        ro_vals = [v for v in agg["ro"].get(sp, []) if not np.isnan(v)]
        os_mean = float(np.mean(os_vals)) if os_vals else float("nan")
        ro_mean = float(np.mean(ro_vals)) if ro_vals else float("nan")
        print(f"  {sp:<6}  {os_mean:>10.4f}  {ro_mean:>10.4f}")


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_sample(
    sample_local_idx: int,
    sample_global_idx: int,
    results: dict,
    out_path: Path,
    fmt: str = "pdf",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_names = results["state_names"]
    t_obs  = results["t_obs"]
    tt     = t_obs[1:]                                       # (K,) prediction times
    P      = len(state_names)
    ni     = sample_local_idx

    y_true = results["y_true"][ni, 1:, :]   # (K, P)
    pred_os = results["pred_onestep"][ni]    # (K, P)
    pred_ro = results["pred_rollout"][ni]    # (K, P)
    theta_os = results["thetas_onestep"][ni] # (K, theta_dim)
    theta_ro = results["thetas_rollout"][ni] # (K, theta_dim)
    theta_dim = theta_os.shape[1]

    no_rollout = np.all(pred_ro == 0)

    # One column of species panels + one column of theta panels
    n_theta_rows = (theta_dim + 1) // 2
    n_rows = max(P, n_theta_rows)
    n_cols = 3  # species | theta OS | theta RO

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.5, n_rows * 2.2),
                              squeeze=False)

    colors = {"truth": "tab:blue", "onestep": "tab:green", "rollout": "tab:red"}

    for i, sp in enumerate(state_names):
        ax = axes[i][0]
        ax.plot(tt, y_true[:, i], lw=2.0, color=colors["truth"],   label="truth")
        ax.plot(tt, pred_os[:, i], lw=1.5, color=colors["onestep"], ls="--",
                label=f"one-step  NRMSE={_nrmse_np(pred_os[:, i], y_true[:, i]):.3f}")
        if not no_rollout:
            ax.plot(tt, pred_ro[:, i], lw=1.5, color=colors["rollout"], ls=":",
                    label=f"rollout   NRMSE={_nrmse_np(pred_ro[:, i], y_true[:, i]):.3f}")
        ax.set_ylabel(sp, fontsize=11)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
    for i in range(P, n_rows):
        axes[i][0].axis("off")

    # Theta panels: col 1 = θ0, θ2, θ4, … | col 2 = θ1, θ3, θ5, …
    for j in range(theta_dim):
        row = j // 2
        col = 1 + (j % 2)
        ax  = axes[row][col]
        ax.plot(tt, theta_os[:, j], lw=1.2, color="tab:purple",  label="OS")
        if not no_rollout:
            ax.plot(tt, theta_ro[:, j], lw=1.2, color="tab:orange", ls="--", label="RO")
        ax.set_ylabel(f"θ{j}", fontsize=9)
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(fontsize=7)
    for j in range(theta_dim, n_rows * 2):
        axes[j // 2][1 + (j % 2)].axis("off")

    axes[-1][0].set_xlabel("Time")
    axes[-1][1].set_xlabel("Time")
    axes[-1][2].set_xlabel("Time")

    axes[0][0].set_title("Predictions", fontsize=10)
    axes[0][1].set_title("θ one-step", fontsize=10)
    axes[0][2].set_title("θ rollout",  fontsize=10)

    fig.suptitle(f"Oracle theta fit — sample {sample_global_idx}", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_samples(results: dict, out_dir: Path, fmt: str = "pdf") -> None:
    sample_idxs = results["sample_indices"]
    N = len(sample_idxs)
    print(f"  Plotting {N} samples -> {out_dir}")
    for ni, si in enumerate(sample_idxs):
        out_path = out_dir / f"sample_{si:04d}.{fmt}"
        plot_sample(ni, si, results, out_path, fmt)
        if (ni + 1) % max(1, N // 10) == 0:
            print(f"    {ni+1}/{N} plotted")
    print(f"  Done. Plots in {out_dir}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batched GPU oracle per-step theta fitting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",      required=True, help=".npz dataset path")
    parser.add_argument("--scaffold",     required=True, help="Scaffold name")
    parser.add_argument("--n-samples",    type=int, default=10,
                        help="Number of samples to fit. -1 = all.")
    parser.add_argument("--sample-offset", type=int, default=0,
                        help="Start from this sample index.")
    parser.add_argument("--gd-steps",    type=int,   default=400)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--n-substeps",  type=int,   default=4)
    parser.add_argument("--loss-species", type=str, default=None,
                        help="Comma-separated species for loss (default: all).")
    parser.add_argument("--eval-species", type=str, default=None,
                        help="Comma-separated species for NRMSE table (default: same as --loss-species).")
    parser.add_argument("--no-rollout",  action="store_true",
                        help="Skip rollout (one-step oracle only, much faster).")
    parser.add_argument("--drop-states", type=str, default=None,
                        help="Comma-separated dataset state names to drop before "
                             "fitting (e.g. 'O' to project the 7-state real_ivtt "
                             "dataset onto txtl_maturation_dna's 6-state layout). "
                             "The remaining state order must match the scaffold's "
                             "state_names.")
    parser.add_argument("--out",         default="results/theta_fit_batched",
                        help="Output directory.")
    parser.add_argument("--plot",        action="store_true",
                        help="Save one trajectory figure per sample.")
    parser.add_argument("--plot-fmt",    default="pdf",
                        help="Plot format: pdf | png (default: pdf).")
    args = parser.parse_args()

    device = device_auto()
    print(f"Device: {device}")

    if args.scaffold not in SCAFFOLDS:
        print(f"Unknown scaffold '{args.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
        sys.exit(1)

    d = np.load(args.dataset, allow_pickle=True)
    total_N = d["y0"].shape[0]
    n = total_N if args.n_samples == -1 else min(args.n_samples, total_N - args.sample_offset)
    sample_indices = list(range(args.sample_offset, args.sample_offset + n))

    loss_species = (
        [s.strip() for s in args.loss_species.split(",") if s.strip()]
        if args.loss_species else None
    )
    drop_states = (
        [s.strip() for s in args.drop_states.split(",") if s.strip()]
        if args.drop_states else None
    )
    eval_species = (
        [s.strip() for s in args.eval_species.split(",") if s.strip()]
        if args.eval_species else loss_species
    )

    print(f"Scaffold: {args.scaffold}  |  N={n}  |  gd_steps={args.gd_steps}  |  loss={loss_species}")

    results = fit_batched(
        scaffold_name  = args.scaffold,
        dataset_path   = args.dataset,
        sample_indices = sample_indices,
        gd_steps       = args.gd_steps,
        lr             = args.lr,
        n_substeps     = args.n_substeps,
        loss_species   = loss_species,
        no_rollout     = args.no_rollout,
        drop_states    = drop_states,
        device         = device,
    )

    out_dir = Path(args.out)
    save_results(results, out_dir)
    save_nrmse_csv(results, out_dir, eval_species)

    if args.plot:
        plot_all_samples(results, out_dir / "plots", fmt=args.plot_fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()
