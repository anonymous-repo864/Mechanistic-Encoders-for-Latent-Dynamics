"""
Verbatim translation of bob_model/spline_models.py:GRU_model_latent_decay_simple_fb_matO
into the train.py model API.

Goal: settle whether "doesn't train" is a model/loss bug in train_R or something
about train.py. This file is *deliberately* a faithful port — same gamma bounds,
same sqrt+clamp_min(1) features, same k%200==0 TF, same stacked GRUCell init,
same approximate analytic step (ivtt_step_R_O_mRNA_maturation).

Differences from Bob's file are only the adapter shims so train.py can call it:
  - __init__ signature accepts train.py's kwargs (most ignored).
  - forward() takes (y0, u_seq, dt_seq, obs_idx, y_seq, ...) and returns
    (pred, theta, beta) where pred is (B,K,7) with mm@3, pm@5.
  - DNA c column is split off u_seq inside forward (Bob did this in dataloader).

Hardcoded npz layout (verified May 2026):
  control_names: ['3PGA','AA','DNA c','FB','K-Glut','Lysate 2%PEG','Maltose',
                  'Mg-Glut','NTPs','PEG8000','T7RNAP','water']     -> DNA c idx = 2
  obs_idx:       [3, 5]   (mm, pm in 7-state [R,O,m,mm,p,pm,DNA])
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


DNA_C_COL_IDX = 2          # u_seq column for DNA c
MM_STATE_IDX  = 3          # y_seq column for mm
PM_STATE_IDX  = 5          # y_seq column for pm


# ---------- verbatim from bob_model/spline_models.py ----------
def gamma(raw: torch.Tensor, lo: float, hi: float, tau: float = 2.3, cap: int = 50) -> torch.Tensor:
    s = torch.sigmoid(raw / tau)
    log_lo = math.log(lo); log_hi = math.log(hi)
    return torch.exp(log_lo + (log_hi - log_lo) * s)


def ivtt_step_R_O_mRNA_maturation(
    m_prev:  torch.Tensor,
    mm_prev: torch.Tensor,
    p_prev:  torch.Tensor,
    pm_prev: torch.Tensor,
    R_prev:  torch.Tensor,
    O_prev:  torch.Tensor,
    dt_k:    torch.Tensor,
    dna_cum_total: torch.Tensor,
    lam_k:   torch.Tensor,
    lam_O_k: torch.Tensor,
    VTXmax:  torch.Tensor,
    kdm:     torch.Tensor,
    VTLmax:  torch.Tensor,
    kmt:     torch.Tensor,
    kmatm:   torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rho_R = torch.exp(-lam_k   * dt_k)
    rho_O = torch.exp(-lam_O_k * dt_k)
    R_curr = R_prev * rho_R
    O_curr = O_prev * rho_O

    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax
    O_eff   = O_curr

    S = VTX_eff * dna_cum_total

    alpha = (kdm + kmatm).clamp_min(eps)
    m_inf = S / alpha
    exp_a = torch.exp(-alpha * dt_k)
    m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

    exp_d  = torch.exp(-kdm   * dt_k)
    exp_mr = torch.exp(-kmatm * dt_k)
    term1 = mm_prev * exp_d
    term2 = m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
    term3 = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)
    mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

    M_prev = m_prev + mm_prev
    M_inf  = S / (kdm + eps)
    exp_M  = exp_d

    eta   = torch.exp(-kmt * dt_k)
    delta = kdm - kmt
    same  = torch.abs(delta) < 1e-6
    int_M_eq  = M_inf * (1.0 - eta) / (kmt + eps) + (M_prev - M_inf) * dt_k * eta
    int_M_gen = M_inf * (1.0 - eta) / (kmt + eps) + (M_prev - M_inf) * (eta - exp_M) / (delta + eps)
    int_M_conv = torch.where(same, int_M_eq, int_M_gen)
    p_curr = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

    int_M_total = M_inf * dt_k + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M)
    pm_curr = torch.clamp_min(
        pm_prev + O_eff * (VTL_eff * int_M_total - (p_curr - p_prev)),
        0.0,
    )
    return m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr


class BobGRUVerbatim(nn.Module):
    """
    Direct port of GRU_model_latent_decay_simple_fb_matO with a train.py-shaped
    forward. Architecture, init, gamma bounds and TF policy are identical to Bob.

    Accepts and silently ignores the kwargs train.py passes for other model
    classes (lift_dim, theta_lo/hi, n_substeps, etc.).
    """

    def __init__(
        self,
        *,
        U: int,
        rhs=None,
        u_to_y_jump: Optional[torch.Tensor] = None,
        hidden: int = 400,
        num_layers: int = 2,
        dropout: float = 0.2,
        dna_c_col_idx: int = DNA_C_COL_IDX,
        mm_state_idx: int = MM_STATE_IDX,
        pm_state_idx: int = PM_STATE_IDX,
        **_kwargs,                          # absorb train.py's other knobs
    ):
        super().__init__()
        assert num_layers >= 1
        self.U = int(U)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.dna_c_col_idx = int(dna_c_col_idx)
        self.mm_state_idx = int(mm_state_idx)
        self.pm_state_idx = int(pm_state_idx)

        # Bob: in_dim = in_u + 2  where in_u = U_after_removing_DNAc
        in_u = self.U - 1
        in_dim = in_u + 2

        cells = [nn.GRUCell(in_dim, self.hidden)]
        cells += [nn.GRUCell(self.hidden, self.hidden) for _ in range(self.num_layers - 1)]
        self.cells = nn.ModuleList(cells)
        self.drop = nn.Dropout(float(dropout))

        self.head = nn.Linear(self.hidden, 7)  # [lam, lam_O, VTX, kdm, VTL, kmt, kmatm]

        # Bob's init — orthogonal_ W_hh + xavier_ W_ih + zeros biases; xavier_ head + zeros bias
        for c in self.cells:
            nn.init.orthogonal_(c.weight_hh)
            nn.init.xavier_uniform_(c.weight_ih)
            nn.init.zeros_(c.bias_hh)
            nn.init.zeros_(c.bias_ih)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Stash the (unused) jump matrix so checkpoints carry the right buffer set.
        if u_to_y_jump is not None:
            self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=False)
        # theta_dim/P attrs — train.py inspects these for some downstream code paths
        self.theta_dim = 7
        self.P = 7

    # --------------- helpers (Bob-equivalent feature build) ----------------
    @staticmethod
    def _split_u(u_seq: torch.Tensor, dna_c_col_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (u_only_no_dna, dna_raw) shaped (B,K,U-1) and (B,K,1)."""
        dna_raw = u_seq[:, :, dna_c_col_idx:dna_c_col_idx + 1]
        keep = [i for i in range(u_seq.shape[-1]) if i != dna_c_col_idx]
        u_only = u_seq[:, :, keep]
        return u_only, dna_raw

    # --------------- forward (train.py API) -------------------------------
    def forward(
        self,
        y0:      torch.Tensor,                  # (B, P=7)
        u_seq:   torch.Tensor,                  # (B, K, U=12) — DNA c included
        dt_seq:  torch.Tensor,                  # (B, K)
        obs_idx: torch.Tensor,                  # ignored; we hardcode mm/pm idx
        y_seq:   Optional[torch.Tensor] = None, # (B, K, P=7) for TF
        teacher_forcing: bool = True,
        tf_every: int = 200,                    # Bob hardcodes 200; left as kwarg for train.py
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        dev = u_seq.device
        dtype = u_seq.dtype

        u_only, dna_raw = self._split_u(u_seq, self.dna_c_col_idx)

        # Bob's x0 layout: (B,3) = [mm0, p0, pm0]; p0 not used from x0.
        mm0 = y0[:, self.mm_state_idx:self.mm_state_idx + 1]
        pm0 = y0[:, self.pm_state_idx:self.pm_state_idx + 1]

        # Bob's y_seq layout: (B,K,3) — y_mm = y_seq[:,:,0:1], y_pm = y_seq[:,:,2:3]
        if y_seq is not None:
            y_mm = y_seq[:, :, self.mm_state_idx:self.mm_state_idx + 1]
            y_pm = y_seq[:, :, self.pm_state_idx:self.pm_state_idx + 1]
        else:
            y_mm = None
            y_pm = None

        # outputs
        mm_out = torch.empty(B, K, 1, device=dev, dtype=dtype)
        p_out  = torch.empty_like(mm_out)
        pm_out = torch.empty_like(mm_out)

        # diagnostic params (Bob order + extras)
        VTX_seq   = torch.empty_like(mm_out)
        kdm_seq   = torch.empty_like(mm_out)
        VTL_seq   = torch.empty_like(mm_out)
        kmt_seq   = torch.empty_like(mm_out)
        kmatm_seq = torch.empty_like(mm_out)
        R_seq     = torch.empty_like(mm_out)
        lam_seq   = torch.empty_like(mm_out)
        lamO_seq  = torch.empty_like(mm_out)

        # Bob's seeded init (m=p=0.01; mm/pm get +0.01)
        mm_prev = mm0 + 0.01
        m_prev  = torch.zeros_like(mm_prev) + 0.01
        pm_prev = pm0 + 0.01
        p_prev  = torch.zeros_like(mm_prev) + 0.01
        R_prev  = torch.ones_like(mm_prev)
        O_prev  = torch.ones_like(mm_prev)

        # constant DNA per sequence (sum across K — equivalent to bolus at t=0 + constant)
        dna_cum_total = torch.cumsum(dna_raw, dim=1)[:, -1, :]   # (B,1)

        hs = [torch.zeros(B, self.hidden, device=dev, dtype=dtype) for _ in range(self.num_layers)]

        for k in range(K):
            dt_k = dt_seq[:, k:k + 1]
            u_k  = u_only[:, k]

            # Bob's TF: every `tf_every` (default 200) steps, replace mm_prev/pm_prev
            # with ground truth from the previous timestep. This *also* fires at k=0
            # (with k-1=-1 wrapping to the last frame). Kept verbatim — minor effect.
            mm_gt_prev = mm_prev
            pm_gt_prev = pm_prev
            if teacher_forcing and (y_mm is not None) and (y_pm is not None) and (k % tf_every == 0):
                mm_gt_prev = y_mm[:, k - 1, :]
                pm_gt_prev = y_pm[:, k - 1, :]

            det_mm = mm_gt_prev.detach()
            det_pm = pm_gt_prev.detach()

            # Bob's feature: sqrt of RAW deltas (no MinMax) ; sqrt of mm/pm clamp_min(1)
            feat_k = torch.cat(
                [torch.sqrt(u_k).clamp_min(0.0),
                 torch.sqrt(det_mm).clamp_min(1.0),
                 torch.sqrt(det_pm).clamp_min(1.0)],
                dim=-1,
            )

            x = feat_k
            for li, cell in enumerate(self.cells):
                hs[li] = cell(x, hs[li])
                x = self.drop(hs[li]) if self.training else hs[li]
            h = x

            raw = self.head(h)
            lam_raw, lamO_raw, VTX_mag, kdm_raw, VTL_mag, kmt_raw, kmatm_raw = raw.split(1, dim=-1)

            lam_k_t = gamma(lam_raw,  1e-6, 5e-4)
            lam_O   = gamma(lamO_raw, 1e-6, 5e-4)
            VTXmax  = gamma(VTX_mag,  5e-5, 1e-1)
            VTLmax  = gamma(VTL_mag,  5e-5, 6e-2)
            kdm_k   = gamma(kdm_raw,  1e-5, 1e-2)
            kmt_k   = gamma(kmt_raw,  1e-5, 3.5e-4)
            kmatm_k = gamma(kmatm_raw, 5e-5, 3.5e-3)

            m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr = ivtt_step_R_O_mRNA_maturation(
                m_prev, mm_prev, p_prev, pm_prev, R_prev, O_prev,
                dt_k, dna_cum_total,
                lam_k_t, lam_O, VTXmax, kdm_k, VTLmax, kmt_k, kmatm_k,
            )

            mm_out[:, k:k + 1, :] = mm_curr.unsqueeze(1)
            p_out [:, k:k + 1, :] = p_curr .unsqueeze(1)
            pm_out[:, k:k + 1, :] = pm_curr.unsqueeze(1)

            VTX_seq  [:, k:k + 1, :] = VTXmax .unsqueeze(1)
            kdm_seq  [:, k:k + 1, :] = kdm_k  .unsqueeze(1)
            VTL_seq  [:, k:k + 1, :] = VTLmax .unsqueeze(1)
            kmt_seq  [:, k:k + 1, :] = kmt_k  .unsqueeze(1)
            kmatm_seq[:, k:k + 1, :] = kmatm_k.unsqueeze(1)
            R_seq    [:, k:k + 1, :] = R_curr .unsqueeze(1)
            lam_seq  [:, k:k + 1, :] = lam_k_t.unsqueeze(1)
            lamO_seq [:, k:k + 1, :] = lam_O  .unsqueeze(1)

            m_prev, mm_prev, p_prev, pm_prev, R_prev, O_prev = (
                m_curr, mm_curr, p_curr, pm_curr, R_curr, O_curr
            )

        # ---- inflate (B,K,3) -> (B,K,7) so train.py can slice obs_idx=[mm, pm] ----
        pred = torch.zeros(B, K, 7, device=dev, dtype=dtype)
        pred[:, :, self.mm_state_idx] = mm_out.squeeze(-1)
        pred[:, :, self.pm_state_idx] = pm_out.squeeze(-1)

        # Match bob_model/spline_models.py param layout for GRU_model_latent_decay_simple_fb_matO:
        # params = [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO]
        theta = torch.cat(
            [VTX_seq, kdm_seq, VTL_seq, kmt_seq, kmatm_seq, R_seq, lam_seq, lamO_seq],
            dim=-1,
        )

        beta = torch.zeros(B, K, theta.shape[-1], device=dev, dtype=dtype)
        return pred, theta, beta
