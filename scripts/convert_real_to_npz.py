"""datasets/convert_real_to_npz.py

Convert parsed real IVTT data (parquet files) into the .npz format expected by
`last-layer-ode/train.py` (ODEDataset).

Pipeline:
    - Controls (reagents): either MinMax-scaled jointly across all experiments OR
        kept raw (see --reagent-scaling).
    - `DNA c` is kept raw (no scaling) and either:
            * collapsed into a single bolus at t=0 equal to the cumulative total
                per experiment (supervisor-style), or
            * kept as per-step deltas (bob_model-style)
        (see --dna-mode).
    - Pipetting-error anomalies (pm > 5000 RFU) can be dropped (see --outlier-mode).
    - y_seq observations: Broccoli (mm), mCherry/2 (pm) — stored raw.
    - y0 = obs at t0; y_seq[k] = obs at t_{k+1}; u_seq[k] = u during [t_k, t_{k+1}).
        This alignment can be switched to bob_model-style y_seq[k]=obs at t_k
        (see --obs-align).

Usage:
        python datasets/convert_real_to_npz.py
        python datasets/convert_real_to_npz.py --layout mat_dna
        python datasets/convert_real_to_npz.py --reagent-scaling none
        python datasets/convert_real_to_npz.py --output datasets/foo.npz

Layouts:
    full    : P=7 [R, O, m, mm, p, pm, DNA] — for txtl_resource_and_maturation_dna scaffold
    mat_dna : P=6 [R, m, mm, p, pm, DNA]    — for txtl_maturation_dna scaffold (no oxygen)
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TIME_COL = "Time_seconds"
DNA_C_COL = "DNA c"
EXCLUDE_FROM_U = {TIME_COL, "DNA"}
PM_OUTLIER_THRESHOLD = 5000.0  # mCherry/2 RFU above this = pipetting error (per Leonardo)


LAYOUTS = {
    "full": {
        "P": 7,
        "state_names": ["R", "O", "m", "mm", "p", "pm", "DNA"],
        "dna_idx": 6,
        "mm_idx": 3,
        "pm_idx": 5,
        "x0_init": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "mat_dna": {
        "P": 6,
        "state_names": ["R", "m", "mm", "p", "pm", "DNA"],
        "dna_idx": 5,
        "mm_idx": 2,
        "pm_idx": 4,
        "x0_init": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
}


def collect_experiment_pairs(data_dir: str) -> list[tuple[str, str]]:
    """Return (inputs_path, outputs_path) pairs in **manifest.json key order**.

    Bob's `parse_IVTT_data.load_parsed_io` populates `inputs_dict` by iterating
    `manifest_json.items()` and Python dicts preserve JSON insertion order, so
    his per-experiment indexing follows manifest order. The hardcoded splits in
    `bob_model/3 seed splits.py` are aligned to that ordering. Falling back to
    `sorted(glob(...))` produces a different order and silently misaligns the
    splits with the experiments.

    If `manifest.json` is missing, fall back to sorted globbing (with a warning).
    """
    import json
    manifest = os.path.join(data_dir, "manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        pairs: list[tuple[str, str]] = []
        for key_str, entry in entries.items():
            inp_path = os.path.join(data_dir, entry["inputs_path"])
            out_path = os.path.join(data_dir, entry["outputs_path"])
            if os.path.exists(inp_path) and os.path.exists(out_path):
                pairs.append((inp_path, out_path))
            else:
                print(f"WARNING: missing parquet for manifest key {key_str}, skipping")
        return pairs

    print(f"WARNING: no manifest.json in {data_dir} — falling back to sorted glob (will misalign Bob's splits)")
    inp_files = sorted(glob.glob(os.path.join(data_dir, "*__inputs.parquet")))
    pairs = []
    for inp_path in inp_files:
        out_path = inp_path.replace("__inputs.parquet", "__outputs.parquet")
        if os.path.exists(out_path):
            pairs.append((inp_path, out_path))
        else:
            print(f"WARNING: no matching outputs for {inp_path}, skipping")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convert real IVTT parquet data to .npz (MinMax + outlier filter).")
    parser.add_argument("--layout", type=str, default="full", choices=list(LAYOUTS.keys()))
    parser.add_argument("--data-dir", type=str, default="datasets/data_parsed_pruned")
    parser.add_argument(
        "--reagent-scaling",
        type=str,
        default="minmax",
        choices=["minmax", "none"],
        help="Scaling for reagent controls (DNA c is always raw): 'minmax' (default) or 'none' (raw).",
    )
    parser.add_argument(
        "--dna-mode",
        type=str,
        default="bolus",
        choices=["bolus", "raw"],
        help="How to represent DNA c in u_seq: 'bolus' (t=0 total DNA, default) or 'raw' (per-step deltas; bob_model).",
    )
    parser.add_argument(
        "--obs-align",
        type=str,
        default="next",
        choices=["next", "current"],
        help="Observation alignment for y_seq: 'next' => y_seq[k]=obs at t_{k+1} (default); 'current' => y_seq[k]=obs at t_k (bob_model).",
    )
    parser.add_argument(
        "--outlier-mode",
        type=str,
        default="max",
        choices=["max", "last", "none"],
        help=(
            "How to drop pipetting-error experiments based on pm (mCherry/divisor): "
            "'max' drops if any timestep exceeds 5000 (default); "
            "'last' drops if ONLY the final timestep exceeds 5000 (bob_model); "
            "'none' disables dropping."
        ),
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Output .npz path. Defaults to datasets/real_ivtt_<scaling>_<layout>.npz")
    parser.add_argument("--mcherry-divisor", type=float, default=2.0,
                        help="Divide raw mCherry RFU by this to convert to concentration units (default 2.0).")
    args = parser.parse_args()

    layout = LAYOUTS[args.layout]
    P = layout["P"]
    state_names = layout["state_names"]
    dna_idx = layout["dna_idx"]
    mm_idx = layout["mm_idx"]
    pm_idx = layout["pm_idx"]
    x0_init = np.asarray(layout["x0_init"], dtype=np.float32)

    if args.output is not None:
        output = args.output
    else:
        if args.reagent_scaling == "minmax":
            output = f"datasets/real_ivtt_minmax_{args.layout}.npz"
        else:
            output = f"datasets/real_ivtt_raw_{args.layout}.npz"

    pairs = collect_experiment_pairs(args.data_dir)
    if not pairs:
        raise RuntimeError(f"No experiment pairs found in {args.data_dir}")
    print(f"Found {len(pairs)} experiment pairs | layout={args.layout} (P={P})")

    # Sniff control columns from the first file. Sorted to match supervisor's
    # `sorted([c for c in df.columns if c != time_label])` ordering.
    sample_inp = pd.read_parquet(pairs[0][0])
    control_cols = sorted([c for c in sample_inp.columns if c not in EXCLUDE_FROM_U])
    if DNA_C_COL not in control_cols:
        raise RuntimeError(
            f"Expected column '{DNA_C_COL}' in u_seq but it was excluded. "
            f"Columns seen: {list(sample_inp.columns)}"
        )
    dna_c_col_idx = control_cols.index(DNA_C_COL)
    d_in = len(control_cols)
    reagent_mask = np.ones(d_in, dtype=bool)
    reagent_mask[dna_c_col_idx] = False  # DNA c is excluded from MinMax

    print(f"Control columns ({d_in}): {control_cols}")
    print(f"  DNA c at u_seq idx {dna_c_col_idx} → routed to state '{state_names[dna_idx]}' (idx {dna_idx})")

    expected_inp_cols = set(control_cols) | EXCLUDE_FROM_U

    # ---- Pass 1: load raw arrays + drop pipetting-error experiments ----
    u_raw_list: list[np.ndarray] = []
    obs_list: list[np.ndarray] = []
    y0_obs_list: list[np.ndarray] = []
    t_list: list[np.ndarray] = []
    n_dropped = 0

    original_indices: list[int] = []

    for orig_idx, (inp_path, out_path) in enumerate(pairs):
        df_inp = pd.read_parquet(inp_path)
        assert set(df_inp.columns) == expected_inp_cols, (
            f"Column mismatch in {inp_path}: "
            f"extra={set(df_inp.columns) - expected_inp_cols}, "
            f"missing={expected_inp_cols - set(df_inp.columns)}"
        )
        df_out = pd.read_parquet(out_path)

        times = df_out[TIME_COL].to_numpy(dtype=np.float32)
        broccoli = df_out["Broccoli [RFU]"].to_numpy(dtype=np.float32)
        mcherry = df_out["mCherry [RFU]"].to_numpy(dtype=np.float32) / args.mcherry_divisor

        obs_full = np.stack([broccoli, mcherry], axis=1)

        if args.outlier_mode != "none":
            pm = obs_full[:, 1]
            too_high = False
            if args.outlier_mode == "max":
                too_high = float(pm.max()) > PM_OUTLIER_THRESHOLD
            elif args.outlier_mode == "last":
                too_high = float(pm[-1]) > PM_OUTLIER_THRESHOLD
            else:
                raise ValueError(f"Unknown outlier_mode: {args.outlier_mode}")
            if too_high:
                n_dropped += 1
                continue
        
            original_indices.append(orig_idx)

        # Match supervisor: drop last frame; u_seq[k] drives [t_k, t_{k+1}).
        u_full = df_inp[control_cols].to_numpy(dtype=np.float32)[:-1]

        y0_obs_list.append(obs_full[0])
        if args.obs_align == "next":
            obs_list.append(obs_full[1:])
        else:
            obs_list.append(obs_full[:-1])
        u_raw_list.append(u_full)
        t_list.append(times)

    if args.outlier_mode == "none":
        print(f"Kept {len(obs_list)} experiments (outlier dropping disabled)")
    else:
        print(
            f"Kept {len(obs_list)} experiments (dropped {n_dropped} as pm>{PM_OUTLIER_THRESHOLD:.0f} outliers; mode={args.outlier_mode})"
        )

    u_scale_min = None
    u_scale_max = None
    reagent_col_names = np.array(
        [c for c, keep in zip(control_cols, reagent_mask) if keep], dtype="<U32"
    )

    scaler = None
    if args.reagent_scaling == "minmax":
        # ---- Fit MinMax scaler jointly across all reagent columns (excluding DNA c) ----
        all_reagents = np.concatenate([u[:, reagent_mask] for u in u_raw_list], axis=0)
        scaler = MinMaxScaler().fit(all_reagents)
        u_scale_min = scaler.data_min_.astype(np.float32)
        u_scale_max = scaler.data_max_.astype(np.float32)
        print(f"MinMax fit over {all_reagents.shape[0]} timesteps × {all_reagents.shape[1]} reagents")
    else:
        print("Reagent scaling: none (raw controls)")

    # ---- Apply scaling (optional) + DNA handling ----
    u_list: list[np.ndarray] = []
    dna_totals: list[float] = []
    for u_raw in u_raw_list:
        u = u_raw.copy()
        if scaler is not None:
            u[:, reagent_mask] = scaler.transform(u_raw[:, reagent_mask]).astype(np.float32)
        dna_total = float(u_raw[:, dna_c_col_idx].sum())  # raw cumulative DNA
        dna_totals.append(dna_total)
        if args.dna_mode == "bolus":
            u[:, dna_c_col_idx] = 0.0
            u[0, dna_c_col_idx] = dna_total  # raw, kept unscaled (physical concentration)
        else:
            # keep raw per-step deltas as-is
            u[:, dna_c_col_idx] = u_raw[:, dna_c_col_idx]
        u_list.append(u)

    dna_totals_arr = np.asarray(dna_totals, dtype=np.float32)
    print(f"DNA totals (raw): mean={dna_totals_arr.mean():.4g} "
          f"min={dna_totals_arr.min():.4g} max={dna_totals_arr.max():.4g}")

    # ---- Pad to common length ----
    lengths = np.array([o.shape[0] for o in obs_list], dtype=np.int64)
    K_max = int(lengths.max())
    N = len(obs_list)
    print(f"Samples: {N} | K range: [{lengths.min()}, {K_max}]")

    longest_idx = int(np.argmax(lengths))
    t_obs = t_list[longest_idx].astype(np.float32)

    # ---- Build y0, y_seq, u_seq ----
    y0 = np.broadcast_to(x0_init, (N, P)).copy()
    y0[:, mm_idx] = np.asarray([y[0] for y in y0_obs_list], dtype=np.float32)
    y0[:, pm_idx] = np.asarray([y[1] for y in y0_obs_list], dtype=np.float32)

    y_seq = np.zeros((N, K_max, P), dtype=np.float32)
    u_seq = np.zeros((N, K_max, d_in), dtype=np.float32)

    for i in range(N):
        K_i = obs_list[i].shape[0]
        u_seq[i, :K_i] = u_list[i]
        y_seq[i, :K_i, mm_idx] = obs_list[i][:, 0]
        y_seq[i, :K_i, pm_idx] = obs_list[i][:, 1]
        if K_i < K_max:
            y_seq[i, K_i:, mm_idx] = obs_list[i][-1, 0]
            y_seq[i, K_i:, pm_idx] = obs_list[i][-1, 1]

    # ---- u_to_y_jump routing (DNA c column → DNA state, others → no jump) ----
    obs_indices = np.arange(P, dtype=np.int64)
    control_indices = np.arange(P, P + d_in, dtype=np.int64)
    control_indices[dna_c_col_idx] = dna_idx

    matches = [
        (j, p) for j in range(d_in) for p in range(P)
        if int(control_indices[j]) == int(obs_indices[p])
    ]
    assert matches == [(dna_c_col_idx, dna_idx)], (
        f"Expected exactly one routing (j={dna_c_col_idx}, p={dna_idx}), got {matches}"
    )

    control_names = np.array(control_cols, dtype="<U32")
    obs_names = np.array(state_names, dtype="<U32")
    names_full = np.array(state_names + control_cols, dtype="<U32")
    save_kwargs: dict = {}
    if scaler is not None:
        save_kwargs.update(
            dict(
                u_scale_min=u_scale_min,
                u_scale_max=u_scale_max,
                u_scaled_cols=reagent_col_names,
            )
        )

    # ---- y_seq raw stats (regardless of u scaling) — useful for runtime obs_normalization ----
    obs_raw_min = y_seq.reshape(-1, P).min(axis=0).astype(np.float32)
    obs_raw_max = y_seq.reshape(-1, P).max(axis=0).astype(np.float32)
    valid_mask_2d = (np.arange(K_max)[None, :] < lengths[:, None]).astype(np.float32)
    valid = float(valid_mask_2d.sum())
    obs_raw_mean = (y_seq * valid_mask_2d[..., None]).sum(axis=(0, 1)) / max(valid, 1.0)
    obs_raw_var = ((y_seq - obs_raw_mean) ** 2 * valid_mask_2d[..., None]).sum(axis=(0, 1)) / max(valid, 1.0)
    obs_raw_std = np.sqrt(np.maximum(obs_raw_var, 1e-12)).astype(np.float32)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        y0=y0.astype(np.float32),
        u_seq=u_seq,
        y_seq=y_seq,
        t_obs=t_obs,
        control_indices=control_indices,
        obs_indices=obs_indices,
        names_full=names_full,
        control_names=control_names,
        obs_names=obs_names,
        n_states_full=np.int64(P),
        n_params_full=np.int64(0),
        theta_true=np.zeros(0, dtype=np.float32),
        lengths=lengths,
        # provenance
        reagent_scaling=np.array(args.reagent_scaling, dtype="<U16"),
        dna_mode=np.array(args.dna_mode, dtype="<U16"),
        obs_align=np.array(args.obs_align, dtype="<U16"),
        outlier_mode=np.array(args.outlier_mode, dtype="<U16"),
        dna_totals=dna_totals_arr,
        obs_raw_min=obs_raw_min,
        obs_raw_max=obs_raw_max,
        obs_raw_mean=obs_raw_mean.astype(np.float32),
        obs_raw_std=obs_raw_std.astype(np.float32),
        original_indices=np.array(original_indices, dtype=np.int64),
        **save_kwargs,
    )

    print(f"\nSaved to {out_path}")
    print(f"  y0:      {y0.shape}  (P={P})")
    print(f"  u_seq:   {u_seq.shape}  (MinMax-scaled reagents, raw DNA bolus at t=0)")
    if args.reagent_scaling == "none":
        print("  u_seq:   reagents are RAW (no scaling)")
    if args.dna_mode == "raw":
        print("  u_seq:   DNA c is RAW per-step deltas")
    else:
        print("  u_seq:   DNA c is collapsed to a t=0 bolus")
    print(f"  y_seq:   {y_seq.shape}")
    print(f"  t_obs:   {t_obs.shape}")
    print(f"  lengths: {lengths.shape} (min={lengths.min()}, max={lengths.max()})")
    print(f"  state_names  : {list(obs_names)}")
    print(f"  obs_idx hint : [{mm_idx}, {pm_idx}]  (mm → Broccoli, pm → mCherry/{int(args.mcherry_divisor)})")
    print(f"  dna_idx      : {dna_idx}  (t=0 bolus from u_seq col {dna_c_col_idx})")


if __name__ == "__main__":
    main()
