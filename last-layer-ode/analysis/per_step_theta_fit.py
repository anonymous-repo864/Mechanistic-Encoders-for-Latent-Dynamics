"""
per_step_theta_fit.py

Oracle per-step theta fitting via gradient descent.

Instead of learning to predict theta with an NN, we directly optimise theta
at each timestep using GD to minimise the one-step prediction error.

Two modes:

  ONE-STEP (oracle):
    At step k, start from y_true[k], optimise theta, predict y_hat[k+1].
    Errors never compound — best-case local fit.

  HONEST ROLLOUT:
    At step k, start from own previous prediction, optimise theta to hit
    y_true[k+1], advance from the prediction.  Errors compound naturally.

If the scaffold is structurally sufficient, both modes should track truth.
If not, one-step still fits but honest rollout diverges — proving the scaffold
cannot represent the true dynamics regardless of theta(t).

Usage:
    python analysis/per_step_theta_fit.py \
        --dataset-dir datasets/ \
        --scaffolds reduced2,reduced3,reduced5,reduced7,reduced9,full13 \
        --sample-idx 0 \
        --gd-steps 400 \
        --out results/per_step_theta_fit
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump

FULL_SPECIES = list("ABCDEFGHIJKLM")
SCAFFOLD_ALIASES = {"reduced13": "full13", "full": "full13"}


def normalize_scaffold_name(name: str) -> str:
    return SCAFFOLD_ALIASES.get(name.strip(), name.strip())


def device_auto() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_npz_obs_names(path: Path):
    try:
        d = np.load(str(path), allow_pickle=True)
        if "obs_names" not in d:
            return None
        return set(d["obs_names"].astype(str).tolist())
    except Exception:
        return None


def find_matching_dataset(scaffold_name: str, dataset_dir: Path):
    sc = SCAFFOLDS[scaffold_name]
    target = set(sc.state_names)
    for p in sorted(dataset_dir.glob("*.npz")):
        if load_npz_obs_names(p) == target:
            return p
    return None


def build_scaffold_dataset_map(scaffold_names, dataset_dir: Path):
    mapping = {}
    for sn in scaffold_names:
        canonical = normalize_scaffold_name(sn)
        if canonical not in SCAFFOLDS:
            print(f"[warn] Unknown scaffold '{sn}' — skipping.")
            continue
        if canonical != sn:
            print(f"[info] Alias '{sn}' -> '{canonical}'.")
        match = find_matching_dataset(canonical, dataset_dir)
        if match is None:
            print(f"[warn] No matching dataset for '{canonical}' "
                  f"(need obs_names={set(SCAFFOLDS[canonical].state_names)}) — skipping.")
            continue
        mapping[canonical] = match
    return mapping


def gamma(x, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Log-space sigmoid: geometric midpoint sqrt(lo*hi) at init (x=0).
    Use instead of linear gamma when bounds span orders of magnitude."""
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


def rk4(rhs, y, dt, theta, n_sub=4):
    if dt.ndim == 1:
        dt = dt.unsqueeze(1)
    h = dt / float(n_sub)
    for _ in range(n_sub):
        k1 = rhs(y,                 theta)
        k2 = rhs(y + 0.5 * h * k1, theta)
        k3 = rhs(y + 0.5 * h * k2, theta)
        k4 = rhs(y +       h * k3, theta)
        y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return torch.clamp_min(y, 0.0)


def nrmse(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0:
        return np.nan
    rng = true[mask].max() - true[mask].min()
    if rng < 1e-10:
        return np.nan
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)) / rng)


def run(scaffold_name, dataset_path, sample_idx,
        gd_steps=400, lr=0.05,
        theta_lo=1e-3, theta_hi=2.0, n_substeps=4,
        loss_species=None, no_rollout=False,
        device=torch.device("cpu")):
    sc        = SCAFFOLDS[scaffold_name]
    rhs       = sc
    theta_dim = sc.theta_dim
    P         = sc.P

    d      = np.load(str(dataset_path), allow_pickle=True)
    y0     = d["y0"][sample_idx].astype(np.float32)
    y_seq  = d["y_seq"][sample_idx].astype(np.float32)
    u_seq  = d["u_seq"][sample_idx].astype(np.float32)
    t_obs  = d["t_obs"].astype(np.float32)
    dt     = np.diff(t_obs).astype(np.float32)

    y_full = np.concatenate([y0[None], y_seq], axis=0)   # (K+1, P)
    K      = len(dt)

    # Resolve which species contribute to the loss. The dataset may expose
    # latent (zero-padded) channels alongside real measurements; averaging
    # over them forces the ODE to crush latent states to zero, which
    # breaks rollout. Default to all species (legacy behavior); pass
    # loss_species to mask to the truly-observed channels.
    state_names = list(sc.state_names)
    if loss_species:
        missing = [s for s in loss_species if s not in state_names]
        if missing:
            raise ValueError(f"loss_species {missing} not in scaffold states {state_names}")
        loss_idx = [state_names.index(s) for s in loss_species]
    else:
        loss_idx = list(range(P))
    loss_idx_t = torch.tensor(loss_idx, dtype=torch.long, device=device)

    y_true   = torch.from_numpy(y_full).float().to(device)
    u_tensor = torch.from_numpy(u_seq).float().to(device)
    dt_t     = torch.from_numpy(dt).float().to(device)

    jump = make_u_to_y_jump(
        torch.from_numpy(d["control_indices"].astype(np.int64)),
        torch.from_numpy(d["obs_indices"].astype(np.int64)),
        device=device,
    )

    # Per-parameter bounds: use scaffold-defined vectors if available, else scalar fallback
    if sc.theta_lo_vec is not None and sc.theta_hi_vec is not None:
        lo_t = torch.tensor(sc.theta_lo_vec, dtype=torch.float32, device=device)
        hi_t = torch.tensor(sc.theta_hi_vec, dtype=torch.float32, device=device)
    else:
        lo_t = torch.full((theta_dim,), theta_lo, device=device)
        hi_t = torch.full((theta_dim,), theta_hi, device=device)

    # ── 1) One-step oracle (batched) ──────────────────────────────────────────
    y_prev           = y_true[:-1]
    y_next           = y_true[1:]
    y_after_jump_all = y_prev + (u_tensor @ jump)

    raw_os = torch.zeros(K, theta_dim, device=device, requires_grad=True)
    opt_os = torch.optim.Adam([raw_os], lr=lr)

    for _ in range(gd_steps):
        opt_os.zero_grad()
        theta_b = log_gamma(raw_os, lo_t, hi_t)
        y_hat_b = rk4(rhs, y_after_jump_all, dt_t, theta_b, n_substeps)
        diff = torch.log1p(y_hat_b) - torch.log1p(y_next)
        loss = diff.index_select(-1, loss_idx_t).pow(2).mean()
        loss.backward()
        opt_os.step()

    with torch.no_grad():
        theta_os = log_gamma(raw_os, lo_t, hi_t)
        y_hat_os = rk4(rhs, y_after_jump_all, dt_t, theta_os, n_substeps)
        diff_os  = torch.log1p(y_hat_os) - torch.log1p(y_next)
        losses_os = diff_os.index_select(-1, loss_idx_t).pow(2).mean(dim=-1).cpu().numpy()

    pred_onestep   = y_hat_os.detach().cpu().numpy()
    thetas_onestep = theta_os.detach().cpu().numpy()

    # ── 2) Honest rollout (sequential) ────────────────────────────────────────
    if no_rollout:
        pred_rollout   = np.zeros((K, P))
        thetas_rollout = np.zeros((K, theta_dim))
        losses_rollout = np.zeros(K)
    else:
        pred_rollout   = np.zeros((K, P))
        thetas_rollout = np.zeros((K, theta_dim))
        losses_rollout = np.zeros(K)

        y_cur = y_true[0].unsqueeze(0).clone()

        for k in range(K):
            u_k      = u_tensor[k].unsqueeze(0)
            dt_k     = dt_t[k].unsqueeze(0)
            y_target = y_true[k + 1].unsqueeze(0)

            y_after_jump_k = y_cur + (u_k @ jump)

            raw = torch.zeros(1, theta_dim, device=device, requires_grad=True)
            opt = torch.optim.Adam([raw], lr=lr)

            for _ in range(gd_steps):
                opt.zero_grad()
                theta_k = log_gamma(raw, lo_t, hi_t)
                y_hat   = rk4(rhs, y_after_jump_k.detach(), dt_k, theta_k, n_substeps)
                diff    = torch.log1p(y_hat) - torch.log1p(y_target)
                loss    = diff.index_select(-1, loss_idx_t).pow(2).mean()
                loss.backward()
                opt.step()

            with torch.no_grad():
                theta_k = log_gamma(raw, lo_t, hi_t)
                y_hat   = rk4(rhs, y_after_jump_k, dt_k, theta_k, n_substeps)
                diff    = torch.log1p(y_hat) - torch.log1p(y_target)
                final_loss = diff.index_select(-1, loss_idx_t).pow(2).mean()

            pred_rollout[k]   = y_hat.squeeze(0).detach().cpu().numpy()
            thetas_rollout[k] = theta_k.squeeze(0).detach().cpu().numpy()
            losses_rollout[k] = float(final_loss)
            y_cur = y_hat.detach()

    return dict(
        y_full         = y_full,
        t_obs          = t_obs,
        state_names    = list(sc.state_names),
        pred_onestep   = pred_onestep,
        thetas_onestep = thetas_onestep,
        losses_onestep = losses_os,
        pred_rollout   = pred_rollout,
        thetas_rollout = thetas_rollout,
        losses_rollout = losses_rollout,
    )


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    rng  = float(np.max(y_true) - np.min(y_true))
    return rmse / max(rng, 1e-8)


def _run_one_worker(kwargs: dict) -> dict:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    res = run(**kwargs, no_rollout=True, device=torch.device("cpu"))
    state_names = res["state_names"]
    y_true = res["y_full"][1:]          # (K, P)
    pred   = res["pred_onestep"]        # (K, P)
    nrmse_per_species = {
        sp: _nrmse(y_true[:, j], pred[:, j])
        for j, sp in enumerate(state_names)
    }
    return {"sample_idx": kwargs["sample_idx"], "nrmse": nrmse_per_species}


def run_multi_sample(
    scaffold_name: str,
    dataset_path: str,
    sample_indices: list,
    gd_steps: int = 400,
    lr: float = 0.05,
    n_substeps: int = 4,
    loss_species: list = None,
    n_workers: int = None,
    eval_species: list = None,
    out_csv: str = None,
) -> None:
    if n_workers is None:
        n_workers = os.cpu_count()

    jobs = [
        dict(scaffold_name=scaffold_name, dataset_path=dataset_path,
             sample_idx=i, gd_steps=gd_steps, lr=lr, n_substeps=n_substeps,
             loss_species=loss_species)
        for i in sample_indices
    ]

    per_sample: dict[str, list[float]] = {}
    sample_rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_worker, j): j["sample_idx"] for j in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            sample_rows.append({"sample_idx": res["sample_idx"], **res["nrmse"]})
            for sp, v in res["nrmse"].items():
                per_sample.setdefault(sp, []).append(v)
            done += 1
            if done % max(1, len(sample_indices) // 10) == 0:
                print(f"  {done}/{len(sample_indices)} done", flush=True)

    species_to_show = eval_species if eval_species else list(per_sample.keys())
    print(f"\n{'Species':<10}  {'Mean NRMSE':>12}")
    print("-" * 25)
    primary_vals = []
    for sp in species_to_show:
        if sp not in per_sample:
            continue
        vals = np.array(per_sample[sp])
        m = float(np.mean(vals))
        primary_vals.append(m)
        print(f"  {sp:<8}  {m:>12.4f}")
    if primary_vals:
        print(f"  {'PRIMARY':<8}  {float(np.mean(primary_vals)):>12.4f}  (mean over {species_to_show})")

    if out_csv:
        import csv
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rows.sort(key=lambda r: r["sample_idx"])
        all_species = [k for k in sample_rows[0] if k != "sample_idx"]
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_idx"] + all_species)
            writer.writeheader()
            writer.writerows(sample_rows)
        # append a summary row
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_idx"] + all_species)
            summary = {"sample_idx": "MEAN"}
            for sp in all_species:
                summary[sp] = f"{float(np.mean(per_sample[sp])):.6f}" if sp in per_sample else ""
            writer.writerow(summary)
        print(f"Saved per-sample NRMSE -> {out_path}")


def export_csv(all_results, scaffold_order, sample_idx, export_dir):
    export_dir.mkdir(parents=True, exist_ok=True)

    for sn in scaffold_order:
        res = all_results[sn]
        state_names = res["state_names"]
        tt = res["t_obs"][1:]
        y_true = res["y_full"][1:]

        sc_dir = export_dir / sn
        sc_dir.mkdir(parents=True, exist_ok=True)

        cols = [tt[:, None], y_true, res["pred_onestep"], res["pred_rollout"]]
        header = (
            ["time"]
            + [f"true_{s}" for s in state_names]
            + [f"onestep_{s}" for s in state_names]
            + [f"rollout_{s}" for s in state_names]
        )
        np.savetxt(
            sc_dir / f"predictions_sample{sample_idx}.csv",
            np.concatenate(cols, axis=1),
            delimiter=",", header=",".join(header), comments="",
        )

        D = res["thetas_rollout"].shape[1]
        theta_header = ["time"] + [f"theta_{j}" for j in range(D)]
        np.savetxt(
            sc_dir / f"theta_rollout_sample{sample_idx}.csv",
            np.concatenate([tt[:, None], res["thetas_rollout"]], axis=1),
            delimiter=",", header=",".join(theta_header), comments="",
        )
        np.savetxt(
            sc_dir / f"theta_onestep_sample{sample_idx}.csv",
            np.concatenate([tt[:, None], res["thetas_onestep"]], axis=1),
            delimiter=",", header=",".join(theta_header), comments="",
        )

        loss_header = ["time", "loss_onestep", "loss_rollout"]
        np.savetxt(
            sc_dir / f"losses_sample{sample_idx}.csv",
            np.column_stack([tt, res["losses_onestep"], res["losses_rollout"]]),
            delimiter=",", header=",".join(loss_header), comments="",
        )

    print(f"CSV exports -> {export_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Oracle per-step theta fitting via GD (one-step + honest rollout).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",      default=None,
                        help="Single .npz path (single-scaffold mode).")
    parser.add_argument("--scaffold",     default="reduced5")
    parser.add_argument("--dataset-dir",  default=None,
                        help="Directory of .npz files for auto-matching.")
    parser.add_argument("--scaffolds",
                        default="reduced2,reduced3,reduced5,reduced7,reduced9,full13")
    parser.add_argument("--all-scaffolds", action="store_true")
    parser.add_argument("--sample-idx",   type=int,   default=0)
    parser.add_argument("--show-species", type=str,   default="A,M")
    parser.add_argument("--gd-steps",     type=int,   default=400)
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--n-substeps",   type=int,   default=4)
    parser.add_argument("--theta-lo",     type=float, default=1e-3)
    parser.add_argument("--theta-hi",     type=float, default=2.0)
    parser.add_argument("--out",          default="results/per_step_theta_fit",
                        help="Output directory for CSV exports.")
    parser.add_argument("--loss-species", type=str, default=None,
                        help="Comma-separated species names to include in the "
                             "per-step loss (e.g. 'mm,pm'). Default: all species. "
                             "Use this to mask latent / zero-padded channels.")
    parser.add_argument("--n-samples",   type=int, default=None,
                        help="Run on this many samples in parallel and report mean NRMSE. "
                             "-1 = all samples. Overrides --sample-idx.")
    parser.add_argument("--n-workers",   type=int, default=None,
                        help="Number of parallel workers (default: all CPUs).")
    parser.add_argument("--eval-species", type=str, default=None,
                        help="Species for NRMSE table in multi-sample mode (default: same as --loss-species).")
    parser.add_argument("--out-csv",     type=str, default=None,
                        help="Save per-sample NRMSE + summary row to this CSV path (multi-sample mode only).")
    parser.add_argument("--no-rollout",  action="store_true",
                        help="Skip the honest rollout (much faster; one-step oracle only).")
    args = parser.parse_args()

    device = device_auto()

    out_dir = Path(args.out)
    export_dir = out_dir / "exports"

    # ── Multi-sample parallel mode ────────────────────────────────────────────
    if args.n_samples is not None:
        if args.dataset is None:
            print("[error] --n-samples requires --dataset.")
            sys.exit(1)
        sn = normalize_scaffold_name(args.scaffold)
        if sn not in SCAFFOLDS:
            print(f"Unknown scaffold '{args.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
            sys.exit(1)
        d = np.load(args.dataset, allow_pickle=True)
        N = d["y0"].shape[0]
        n = N if args.n_samples == -1 else min(args.n_samples, N)
        indices = list(range(n))
        loss_species = (
            [s.strip() for s in args.loss_species.split(",") if s.strip()]
            if args.loss_species else None
        )
        eval_species = (
            [s.strip() for s in args.eval_species.split(",") if s.strip()]
            if args.eval_species else loss_species
        )
        n_workers = args.n_workers or os.cpu_count()
        print(f"Oracle theta fit: scaffold={sn}  samples={n}  workers={n_workers}  gd_steps={args.gd_steps}")
        run_multi_sample(
            scaffold_name=sn,
            dataset_path=args.dataset,
            sample_indices=indices,
            gd_steps=args.gd_steps,
            lr=args.lr,
            n_substeps=args.n_substeps,
            loss_species=loss_species,
            n_workers=n_workers,
            eval_species=eval_species,
            out_csv=args.out_csv,
        )
        return

    print(f"Device: {device}")

    if args.dataset is not None:
        sn = normalize_scaffold_name(args.scaffold)
        if sn not in SCAFFOLDS:
            print(f"Unknown scaffold '{args.scaffold}'. Available: {list(SCAFFOLDS.keys())}")
            sys.exit(1)
        run_items = [(sn, Path(args.dataset))]
    else:
        if args.dataset_dir is None:
            print("[error] Provide --dataset (single) or --dataset-dir (multi).")
            sys.exit(1)

        if args.all_scaffolds:
            scaffold_names = list(SCAFFOLDS.keys())
        else:
            raw = [s.strip() for s in args.scaffolds.split(",") if s.strip()]
            scaffold_names = list(dict.fromkeys(normalize_scaffold_name(s) for s in raw))

        sd_map = build_scaffold_dataset_map(scaffold_names, Path(args.dataset_dir))
        valid  = [sn for sn in scaffold_names if sn in sd_map]

        if not valid:
            print("[error] No scaffolds matched to a dataset. Aborting.")
            sys.exit(1)

        print(f"\n{'Scaffold':<30}  {'P':>3}  Dataset")
        print("-" * 80)
        for sn in valid:
            print(f"  {sn:<28}  {SCAFFOLDS[sn].P:>3d}  {sd_map[sn].name}")
        print()
        run_items = [(sn, sd_map[sn]) for sn in valid]

    show_species = [s.strip() for s in args.show_species.split(",") if s.strip()]
    loss_species = (
        [s.strip() for s in args.loss_species.split(",") if s.strip()]
        if args.loss_species else None
    )

    all_results    = {}
    scaffold_order = []

    for scaffold_name, dataset_path in run_items:
        sc = SCAFFOLDS[scaffold_name]
        print(f"\n>> {scaffold_name}  (P={sc.P}, θ_dim={sc.theta_dim})")
        print(f"   dataset: {dataset_path}")

        res = run(
            scaffold_name, str(dataset_path), args.sample_idx,
            gd_steps     = args.gd_steps,
            lr           = args.lr,
            theta_lo     = args.theta_lo,
            theta_hi     = args.theta_hi,
            n_substeps   = args.n_substeps,
            loss_species = loss_species,
            no_rollout   = args.no_rollout,
            device       = device,
        )
        all_results[scaffold_name] = res
        scaffold_order.append(scaffold_name)

    # NRMSE table
    print("\n" + "=" * 80)
    header = f"{'Scaffold':<28} {'P':>3}"
    for sp in show_species:
        header += f"  OS-{sp:>1}    RO-{sp:>1}"
    print(header)
    print("=" * 80)
    for sn in scaffold_order:
        res = all_results[sn]
        state_names = res["state_names"]
        row = f"{sn:<28} {SCAFFOLDS[sn].P:>3d}"
        for sp in show_species:
            if sp in state_names:
                sc_idx = state_names.index(sp)
                gt = res["y_full"][1:, sc_idx]
                row += f"  {nrmse(res['pred_onestep'][:, sc_idx], gt):.4f}"
                row += f"  {nrmse(res['pred_rollout'][:, sc_idx], gt):.4f}"
            else:
                row += "    N/A     N/A"
        print(row)
    print("=" * 80)
    print("OS = one-step  |  RO = honest rollout\n")

    export_csv(all_results, scaffold_order, args.sample_idx, export_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
