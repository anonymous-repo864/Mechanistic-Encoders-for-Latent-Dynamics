"""
per_step_theta_fit_analytical.py

Identical to per_step_theta_fit.py but replaces RK4 with the exact
matrix-exponential solution (torch.linalg.matrix_exp).

This isolates whether RK4 discretisation error is the bottleneck in the
oracle theta fit.  If the analytical version gives the same NRMSE as the
RK4 version, the integrator is not the limiting factor.

Usage:
    python analysis/per_step_theta_fit_analytical.py \
        --dataset-dir datasets/ \
        --scaffolds reduced2,reduced3,reduced5,reduced7,reduced9,full13 \
        --sample-idx 0 \
        --gd-steps 400 \
        --out results/per_step_theta_fit_analytical
"""

import argparse
import sys
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


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Log-space sigmoid: geometric midpoint sqrt(lo*hi) at init (x=0)."""
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


# ── Analytical solver (matrix exponential) ────────────────────────────────────

def build_K_from_rhs(rhs, P: int, theta: torch.Tensor) -> torch.Tensor:
    """
    For a LINEAR rhs of the form rhs(y, theta) = K(theta) @ y,
    recover K by probing with basis vectors.
    theta : (B, theta_dim) — returns: (B, P, P)
    """
    B = theta.shape[0]
    cols = []
    for j in range(P):
        e_j = torch.zeros(B, P, device=theta.device, dtype=theta.dtype)
        e_j[:, j] = 1.0
        cols.append(rhs(e_j, theta))
    return torch.stack(cols, dim=-1)


def analytical_step(rhs, P: int, y: torch.Tensor, dt: torch.Tensor,
                    theta: torch.Tensor) -> torch.Tensor:
    """
    Exact solution to dy/dt = K(theta) y over interval dt via matrix exponential.
    y: (B,P), dt: (B,), theta: (B,theta_dim) — returns: (B,P)
    """
    B = y.shape[0]
    K = build_K_from_rhs(rhs, P, theta)
    dt_ = dt.view(B, 1, 1)
    Kdt = K * dt_
    # matrix_exp not implemented on MPS — fall back to CPU for that op
    if Kdt.device.type == "mps":
        expKdt = torch.linalg.matrix_exp(Kdt.cpu()).to(Kdt.device)
    else:
        expKdt = torch.linalg.matrix_exp(Kdt)
    y_new = torch.bmm(expKdt, y.unsqueeze(-1)).squeeze(-1)
    return torch.clamp_min(y_new, 0.0)


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
        theta_lo=1e-3, theta_hi=2.0,
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
        y_hat_b = analytical_step(rhs, P, y_after_jump_all, dt_t, theta_b)
        loss = (torch.log1p(y_hat_b) - torch.log1p(y_next)).pow(2).mean()
        loss.backward()
        opt_os.step()

    with torch.no_grad():
        theta_os = log_gamma(raw_os, lo_t, hi_t)
        y_hat_os = analytical_step(rhs, P, y_after_jump_all, dt_t, theta_os)
        losses_os = (torch.log1p(y_hat_os) - torch.log1p(y_next)).pow(2).mean(dim=-1).cpu().numpy()

    pred_onestep   = y_hat_os.detach().cpu().numpy()
    thetas_onestep = theta_os.detach().cpu().numpy()
    print(f"  one-step done  (mean loss={float(losses_os.mean()):.5f})")

    # ── 2) Honest rollout (sequential) ────────────────────────────────────────
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
            y_hat   = analytical_step(rhs, P, y_after_jump_k.detach(), dt_k, theta_k)
            loss    = (torch.log1p(y_hat) - torch.log1p(y_target)).pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            theta_k = log_gamma(raw, lo_t, hi_t)
            y_hat   = analytical_step(rhs, P, y_after_jump_k, dt_k, theta_k)
            final_loss = (torch.log1p(y_hat) - torch.log1p(y_target)).pow(2).mean()

        pred_rollout[k]   = y_hat.squeeze(0).detach().cpu().numpy()
        thetas_rollout[k] = theta_k.squeeze(0).detach().cpu().numpy()
        losses_rollout[k] = float(final_loss)
        y_cur = y_hat.detach()

        if k % max(1, K // 10) == 0:
            print(f"  rollout step {k:4d}/{K}  loss={float(final_loss):.5f}")

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
        description="Oracle per-step theta fitting via GD — analytical (matrix-exp) solver.",
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
    parser.add_argument("--theta-lo",     type=float, default=1e-3)
    parser.add_argument("--theta-hi",     type=float, default=2.0)
    parser.add_argument("--out",          default="results/per_step_theta_fit_analytical",
                        help="Output directory for CSV exports.")
    args = parser.parse_args()

    device = device_auto()
    print(f"Device: {device}")
    print(f"Solver: analytical (matrix exponential)")

    out_dir = Path(args.out)
    export_dir = out_dir / "exports"

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

    all_results    = {}
    scaffold_order = []

    for scaffold_name, dataset_path in run_items:
        sc = SCAFFOLDS[scaffold_name]
        print(f"\n>> {scaffold_name}  (P={sc.P}, θ_dim={sc.theta_dim})  [analytical]")
        print(f"   dataset: {dataset_path}")

        res = run(
            scaffold_name, str(dataset_path), args.sample_idx,
            gd_steps   = args.gd_steps,
            lr         = args.lr,
            theta_lo   = args.theta_lo,
            theta_hi   = args.theta_hi,
            device     = device,
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
    print("OS = one-step  |  RO = honest rollout  |  solver = matrix_exp\n")

    export_csv(all_results, scaffold_order, args.sample_idx, export_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
