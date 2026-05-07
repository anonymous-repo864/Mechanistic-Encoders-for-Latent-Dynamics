"""
OdeMambaSSM: Mamba encoder using the mamba_ssm package (CUDA required).

Drop-in replacement for OdeRNN. Requires CUDA — use ode_mamba.py for MPS/CPU.
Install: pip install mamba-ssm causal-conv1d

The closed-loop ODE feedback (y_k -> input at k+1) means full parallel scan
is infeasible, so we use Mamba's recurrent mode via InferenceParams:
  - seqlen_offset=0: first step, initialises conv+SSM cache in the kernels
  - seqlen_offset>0: selective_state_update (fast CUDA recurrent step)

SSM states are updated in-place by CUDA kernels (no gradient through state),
equivalent to TBPTT-1. y_prev is also detached each step, same as OdeRNN.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams

from scaffolds import MechanisticScaffold


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


class OdeMambaSSM(nn.Module):
    """
    Mamba SSM encoder for mechanistic ODE parameter inference.

    Architecture per step:
      (u_k, y_{k-1}) -> lift -> MambaBlocks (recurrent via InferenceParams) -> head -> theta_k
      y <- y + u_k @ jump
      y <- RK4(scaffold, theta_k, dt_k)
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        theta_bounded: bool = True,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        gru_u_cols: Optional[list] = None,
        gru_y_cols: Optional[list] = None,
        lift_skip: bool = False,
        head_init: str = "default",  # "default" (normal_(std=0.01) + zeros) | "supervisor" (xavier_ + zeros)
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self.n_substeps = int(n_substeps)
        self.use_basal = bool(use_basal)
        self.theta_bounded = bool(theta_bounded)
        self.hidden = int(hidden)

        # Analytic-scaffold hooks (mirrors OdeRNN/OdeTransformer).
        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self._tf_at_k_zero      = bool(getattr(rhs, "tf_at_k_zero", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))

        # Encoder column filters (drop DNA c, keep mm/pm only for IVTT).
        self.gru_u_cols = list(gru_u_cols) if gru_u_cols is not None else None
        self.gru_y_cols = list(gru_y_cols) if gru_y_cols is not None else None
        u_cols_dim = len(self.gru_u_cols) if self.gru_u_cols is not None else self.U
        y_cols_dim = len(self.gru_y_cols) if self.gru_y_cols is not None else self.P

        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # lift_skip: collapse the 2-layer SiLU MLP to a single Linear(feat_in, hidden),
        # the analogue of GRU/LSTM's intrinsic W_ih projection. Mamba blocks require
        # d_model=hidden so the projection cannot be skipped entirely.
        self.lift_skip = bool(lift_skip)
        feat_in = u_cols_dim + y_cols_dim
        if self.lift_skip:
            self.lift = nn.Linear(feat_in, hidden)
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, lift_dim),
                nn.SiLU(),
                nn.Linear(lift_dim, hidden),
            )

        # layer_idx is required so each block gets its own slot in InferenceParams cache
        n = max(1, num_layers)
        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=hidden, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i)
            for i in range(n)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n)])

        head_out = self.theta_dim + self.P if use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)
        if head_init not in ("default", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'supervisor', got {head_init}")
        if str(head_init) == "supervisor":
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)
        else:
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.zeros_(self.head.bias)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                      # (B, P)
        u_seq: torch.Tensor,                   # (B, K, U)
        dt_seq: torch.Tensor,                  # (B, K)
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,  # (B, K, P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        y_out    = torch.empty(B, K, self.P,                device=device, dtype=dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit,   device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P,                device=device, dtype=dtype)

        # InferenceParams carries conv and SSM state across steps for each layer
        inference_params = InferenceParams(max_seqlen=K, max_batch_size=B)

        use_partial = obs_idx.numel() > 0

        analytic_ctx: dict = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)
            y_prev = self.rhs.initial_state(y0)
        else:
            y_prev = y0

        # u_transform applied once on the full sequence (encoder view); ODE jump
        # always uses raw u_seq.
        if u_transform == "cumsum" or u_transform == "cumsum_sqrt":
            u_enc = u_seq.cumsum(dim=1)
        else:
            u_enc = u_seq
        if u_transform == "sqrt" or u_transform == "cumsum_sqrt":
            u_enc = u_enc.clamp_min(0.0).sqrt()

        for k in range(K):
            u_k     = u_seq[:, k, :]
            u_enc_k = u_enc[:, k, :]
            dt_k    = dt_seq[:, k]

            y_in = y_prev.detach()
            tf_fires = (k % tf_every == 0) if self._tf_at_k_zero else (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            u_feat = u_enc_k[:, self.gru_u_cols] if self.gru_u_cols is not None else u_enc_k
            y_feat = y_in[:, self.gru_y_cols] if self.gru_y_cols is not None else y_in
            if y_transform == "sqrt":
                y_feat = y_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_feat = y_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_feat = torch.log1p(y_feat.clamp_min(0.0))
            x = self.lift(torch.cat([u_feat, y_feat], dim=-1)).unsqueeze(1)  # (B, 1, hidden)

            # seqlen_offset=0: init cache; >0: fast recurrent update via CUDA kernel
            inference_params.seqlen_offset = k
            for norm, layer in zip(self.norms, self.mamba_layers):
                x = x + layer(norm(x), inference_params=inference_params)

            z   = x.squeeze(1)  # (B, hidden)
            raw = self.head(z)

            if self.use_basal:
                raw_theta = raw[:, :self.theta_dim]
                theta_k = (log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                           if self.theta_bounded else F.softplus(raw_theta))
                beta_k = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                theta_k = (log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                           if self.theta_bounded else F.softplus(raw))
                if self._analytic_scaffold:
                    y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                else:
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        rhs   = self.rhs
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta)
            k2 = rhs(y + 0.5 * hdt * k1, theta)
            k3 = rhs(y + 0.5 * hdt * k2, theta)
            k4 = rhs(y +       hdt * k3,  theta)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_substeps_basal(
        self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor, beta: torch.Tensor
    ) -> torch.Tensor:
        rhs   = self.rhs
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + beta
            k2 = rhs(y + 0.5 * hdt * k1, theta) + beta
            k3 = rhs(y + 0.5 * hdt * k2, theta) + beta
            k4 = rhs(y +       hdt * k3,  theta) + beta
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)
