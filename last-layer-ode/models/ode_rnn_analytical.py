from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


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


def analytical_step(rhs, P: int, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Exact solution to dy/dt = K(theta) y over interval dt via matrix exponential.
    y: (B,P), dt: (B,), theta: (B,theta_dim) — returns: (B,P)
    """
    B = y.shape[0]
    K = build_K_from_rhs(rhs, P, theta)
    dt_ = dt.view(B, 1, 1)
    expKdt = torch.linalg.matrix_exp(K * dt_)
    y_new = torch.bmm(expKdt, y.unsqueeze(-1)).squeeze(-1)
    return torch.clamp_min(y_new, 0.0)


class AnalyticalOdeRNN(nn.Module):
    """
    Same as OdeRNN but replaces RK4 with the exact matrix-exponential solution.
    Only valid for linear scaffolds (all current mechanistic scaffolds are linear).
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,   # unused, kept for API compatibility
        use_basal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)

        self.lift = nn.Sequential(
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.gru = nn.GRU(
            input_size=lift_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, self.theta_dim)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,
        u_seq: torch.Tensor,
        dt_seq: torch.Tensor,
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)
        use_partial = obs_idx.numel() > 0
        y_prev = y0

        for k in range(K):
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()
            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            feat = torch.cat([u_k, y_in], dim=-1)
            x    = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw  = self.head(z.squeeze(1))
            theta_k = gamma(raw, self.theta_lo, self.theta_hi)

            y = y_prev + (u_k @ self.u_to_y_jump)
            y = analytical_step(self.rhs, self.P, y, dt_k, theta_k)

            y_out[:, k, :]  = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out
