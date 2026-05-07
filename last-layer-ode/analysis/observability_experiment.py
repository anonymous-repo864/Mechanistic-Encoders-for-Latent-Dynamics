"""
Test how the number of observed species affects per-step theta fitting quality.

Uses the full13 scaffold with all 13 equations. Varies which species contribute
to the fitting loss. Hypothesis: more observed species → worse fit per species,
because shared theta must satisfy more competing constraints.

Supports two evaluation modes:
    - one-step: fit each step from true previous state (oracle local interpolation)
    - rollout:  fit each step from model's own previous prediction (honest rollout)

Usage:
    python observability_experiment.py \
        --dataset datasets/N1000_T300_steps600_zeros_knoise0.0_full13_ABCDEFGHIJKLM.npz \
        --n-samples 10 \
        --mode both \
        --out results/observability
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.per_step_theta_fit import load_sample, nrmse, log_gamma, rk4
from scaffolds import SCAFFOLDS
from jumps import make_u_to_y_jump


def fit_with_obs_mask(
    sample: dict,
    obs_species: list[str],
    gd_steps: int = 400,
    lr: float = 0.05,
    theta_lo: float = 1e-3,
    theta_hi: float = 2.0,
    n_substeps: int = 4,
    device=torch.device("cpu"),
) -> dict:
    """
    Fit per-step theta on full13, but only compute loss on obs_species.
    Evaluate NRMSE on ALL species regardless.
    """
    sc = SCAFFOLDS["full13"]
    rhs = sc
    theta_dim = sc.theta_dim
    state_names = sc.state_names
    P = sc.P

    obs_idx = [state_names.index(s) for s in obs_species]

    y_sc = torch.from_numpy(sample["y_full"]).float().to(device)
    u_tensor = torch.from_numpy(sample["u_seq"]).float().to(device)
    dt_tensor = torch.from_numpy(sample["dt"]).float().to(device)
    K = int(dt_tensor.shape[0])

    jump = make_u_to_y_jump(
        torch.from_numpy(sample["control_indices"]).long(),
        torch.from_numpy(sample["obs_indices"]).long(),
        device=device,
    )

    y_prev = y_sc[:-1]
    y_next = y_sc[1:]
    y_after_jump = y_prev + (u_tensor @ jump)

    lo_t = torch.full((theta_dim,), theta_lo, device=device)
    hi_t = torch.full((theta_dim,), theta_hi, device=device)

    raw = torch.zeros(K, theta_dim, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=lr)

    for i in range(gd_steps):
        opt.zero_grad()
        theta_batch = log_gamma(raw, lo_t, hi_t)
        y_hat = rk4(rhs, y_after_jump, dt_tensor, theta_batch, n_substeps)
        loss = (torch.log1p(y_hat[:, obs_idx]) - torch.log1p(y_next[:, obs_idx])).pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        theta_batch = log_gamma(raw, lo_t, hi_t)
        y_hat = rk4(rhs, y_after_jump, dt_tensor, theta_batch, n_substeps)
        final_loss = (torch.log1p(y_hat[:, obs_idx]) - torch.log1p(y_next[:, obs_idx])).pow(2).mean().item()

    y_hat_np = y_hat.cpu().numpy()
    y_next_np = y_next.cpu().numpy()

    species_nrmse = {}
    for j, sp in enumerate(state_names):
        species_nrmse[sp] = nrmse(y_hat_np[:, j], y_next_np[:, j])

    return {
        "obs_species": obs_species,
        "n_obs": len(obs_species),
        "species_nrmse": species_nrmse,
        "final_loss": final_loss,
    }


def fit_rollout_with_obs_mask(
    sample: dict,
    obs_species: list[str],
    gd_steps: int = 400,
    lr: float = 0.05,
    theta_lo: float = 1e-3,
    theta_hi: float = 2.0,
    n_substeps: int = 4,
    device=torch.device("cpu"),
) -> dict:
    """
    Honest rollout variant:
      step k starts from own previous prediction, not from true y[k].
    """
    sc = SCAFFOLDS["full13"]
    rhs = sc
    theta_dim = sc.theta_dim
    state_names = sc.state_names
    P = sc.P

    obs_idx = [state_names.index(s) for s in obs_species]

    y_sc = torch.from_numpy(sample["y_full"]).float().to(device)
    u_tensor = torch.from_numpy(sample["u_seq"]).float().to(device)
    dt_tensor = torch.from_numpy(sample["dt"]).float().to(device)
    K = int(dt_tensor.shape[0])

    jump = make_u_to_y_jump(
        torch.from_numpy(sample["control_indices"]).long(),
        torch.from_numpy(sample["obs_indices"]).long(),
        device=device,
    )

    lo_t = torch.full((theta_dim,), theta_lo, device=device)
    hi_t = torch.full((theta_dim,), theta_hi, device=device)

    y_target = y_sc[1:]
    pred_rollout = torch.zeros(K, P, device=device)
    y_cur = y_sc[0].unsqueeze(0).clone()
    last_loss = 0.0

    for k in range(K):
        u_k = u_tensor[k].unsqueeze(0)
        dt_k = dt_tensor[k].unsqueeze(0)
        y_next_true = y_target[k].unsqueeze(0)
        y_after_jump_k = y_cur + (u_k @ jump)

        raw = torch.zeros(1, theta_dim, device=device, requires_grad=True)
        opt = torch.optim.Adam([raw], lr=lr)

        for _ in range(gd_steps):
            opt.zero_grad()
            theta_k = log_gamma(raw, lo_t, hi_t)
            y_hat = rk4(rhs, y_after_jump_k.detach(), dt_k, theta_k, n_substeps)
            loss = (torch.log1p(y_hat[:, obs_idx]) - torch.log1p(y_next_true[:, obs_idx])).pow(2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            theta_k = log_gamma(raw, lo_t, hi_t)
            y_hat = rk4(rhs, y_after_jump_k, dt_k, theta_k, n_substeps)
            last_loss = float((torch.log1p(y_hat[:, obs_idx]) - torch.log1p(y_next_true[:, obs_idx])).pow(2).mean())

        pred_rollout[k] = y_hat.squeeze(0)
        y_cur = y_hat.detach()

    y_hat_np = pred_rollout.cpu().numpy()
    y_next_np = y_target.cpu().numpy()

    species_nrmse = {}
    for j, sp in enumerate(state_names):
        species_nrmse[sp] = nrmse(y_hat_np[:, j], y_next_np[:, j])

    return {
        "obs_species": obs_species,
        "n_obs": len(obs_species),
        "species_nrmse": species_nrmse,
        "final_loss": last_loss,
    }


def build_obs_sets():
    """A,M → A,B,M → A,B,C,M → ... → A,B,...,L,M (all 13)."""
    all_species = list("ABCDEFGHIJKLM")
    obs_sets = []
    for n_add in range(0, 12):
        intermediates = all_species[1:12][:n_add]
        obs = ["A"] + intermediates + ["M"]
        seen = set()
        obs_dedup = []
        for s in obs:
            if s not in seen:
                seen.add(s)
                obs_dedup.append(s)
        obs_sets.append(obs_dedup)
    return obs_sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Full13 .npz dataset")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of trajectories to average over")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gd-steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--n-substeps", type=int, default=4)
    parser.add_argument("--mode", type=str, default="both",
                        choices=["one-step", "rollout", "both"])
    parser.add_argument("--out", type=str, default="results/observability")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_sets = build_obs_sets()
    modes = ["one-step", "rollout"] if args.mode == "both" else [args.mode]

    # pick sample indices
    d = np.load(args.dataset, allow_pickle=True)
    N = d["y0"].shape[0]
    rng = np.random.default_rng(args.seed)
    sample_indices = rng.choice(N, size=min(args.n_samples, N), replace=False)

    print(f"Running {len(obs_sets)} obs configs × {len(sample_indices)} samples × {len(modes)} modes")
    print(f"Sample indices: {sorted(sample_indices)}")
    print(f"Device: {device}\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")

        fit_fn = fit_with_obs_mask if mode == "one-step" else fit_rollout_with_obs_mask
        rows = []

        for si, sample_idx in enumerate(sample_indices):
            sample = load_sample(Path(args.dataset), int(sample_idx))
            print(f"\n  Sample {si+1}/{len(sample_indices)} (idx={sample_idx})")

            for obs_species in obs_sets:
                n_obs = len(obs_species)
                res = fit_fn(
                    sample, obs_species,
                    gd_steps=args.gd_steps, lr=args.lr,
                    n_substeps=args.n_substeps, device=device,
                )

                for sp, v in res["species_nrmse"].items():
                    rows.append({
                        "mode": mode,
                        "sample_idx": int(sample_idx),
                        "n_obs": n_obs,
                        "obs_species": ",".join(obs_species),
                        "eval_species": sp,
                        "nrmse": v,
                        "in_obs": sp in obs_species,
                    })

                print(f"    n_obs={n_obs:2d}  A={res['species_nrmse']['A']:.4f}  "
                      f"M={res['species_nrmse']['M']:.4f}  loss={res['final_loss']:.6f}")

        # save detailed CSV
        csv_path = out_dir / f"observability_{mode}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "mode", "sample_idx", "n_obs", "obs_species", "eval_species", "nrmse", "in_obs",
            ])
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {csv_path}")

        # plot: mean ± SEM across samples
        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, sp in zip(axes, ["A", "M"]):
            sub = df[df["eval_species"] == sp]
            agg = sub.groupby("n_obs")["nrmse"].agg(["mean", "std", "count"]).reset_index()
            agg["sem"] = agg["std"] / np.sqrt(agg["count"])

            ax.errorbar(agg["n_obs"], agg["mean"], yerr=agg["sem"],
                        marker="o", linewidth=2, markersize=6, capsize=4)
            ax.set_xlabel("Number of observed species", fontsize=13)
            ax.set_ylabel(f"NRMSE (species {sp})", fontsize=13)
            ax.set_title(f"Species {sp} — {mode} fit quality", fontsize=13)
            ax.grid(True, alpha=0.25)
            ax.set_xticks(sorted(agg["n_obs"].unique()))

        n = len(sample_indices)
        fig.suptitle(
            f"Effect of observation set size on theta fit ({mode})\n"
            f"(full13 scaffold, mean ± SEM over {n} sample{'s' if n > 1 else ''})",
            fontsize=13,
        )
        fig.tight_layout()
        fig_path = out_dir / f"observability_{mode}.pdf"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()