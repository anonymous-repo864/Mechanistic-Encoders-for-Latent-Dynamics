from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Linear alternative: lo + (hi - lo) * sigmoid(x) — init at arithmetic midpoint.
    # Log-sigmoid inits at geometric midpoint sqrt(lo*hi), stable for wide rate-constant bounds.
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


class OdeSampleTheta(nn.Module):
    """
    Ablation: per-sample constant theta.

    An MLP encodes the initial condition y0 into a single theta vector that is
    held fixed across all timesteps for that sample. Theta varies across samples
    but not across time — sitting between OdeFixedTheta (one global theta) and
    OdeRNN (time-varying theta per sample).

    Architecture:
      y0 -> MLP -> theta (constant for all K timesteps)
      y_k = y_{k-1} + u_k @ jump
      y_k = RK4(scaffold, theta) over dt_k
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,   # (U, P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.U         = int(U)
        self.P         = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs       = rhs
        self.n_substeps = int(n_substeps)
        self.theta_lo   = float(theta_lo)
        self.theta_hi   = float(theta_hi)

        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))

        # Per-parameter bounds — use scaffold-defined if available, else broadcast scalar
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        layers: list[nn.Module] = [nn.Linear(self.P, lift_dim), nn.SiLU()]
        in_dim = lift_dim
        for _ in range(max(1, num_layers)):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, self.theta_dim)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), "
                f"got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                      # (B, P)
        u_seq: torch.Tensor,                   # (B, K, U)
        dt_seq: torch.Tensor,                  # (B, K)
        obs_idx: torch.Tensor,
        y_seq: Optional[torch.Tensor] = None,
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        # Encode y0 into a single theta per sample — fixed for all timesteps
        raw = self.head(self.mlp(y0))
        theta = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec)  # (B, theta_dim)

        y_out  = torch.empty(B, K, self.P,                 device=device, dtype=dtype)
        th_out = torch.empty(B, K, self.theta_dim_emit,    device=device, dtype=dtype)

        analytic_ctx: dict = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)
            y_prev = self.rhs.initial_state(y0)
        else:
            y_prev = y0

        for k in range(K):
            u_k  = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            if self._analytic_scaffold:
                y = self.rhs.analytic_step(y_prev, dt_k, theta, analytic_ctx)
            else:
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta)

            y_out[:, k, :]  = y
            th_out[:, k, :] = self.rhs.emit_theta(theta, y) if self._analytic_scaffold else theta
            y_prev = y

        beta_out = torch.zeros(B, K, self.P, device=device, dtype=dtype)
        return y_out, th_out, beta_out

    def _rk4_substeps(
        self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
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
