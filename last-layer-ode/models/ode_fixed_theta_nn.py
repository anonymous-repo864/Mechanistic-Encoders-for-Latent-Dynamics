from typing import Optional, Tuple

import torch
import torch.nn as nn

from scaffolds import MechanisticScaffold


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


class NeuralOdeCorrection(nn.Module):
    """
    Baseline: single global learnable theta + state-dependent neural correction.

        ẋ(t) = f_mech(x(t), theta) + f_NN(x(t))

    theta is a shared nn.Parameter (no encoder, no GRU).
    f_NN is a small MLP evaluated at every RK4 evaluation point so the
    correction is truly a function of the current state, not a step-constant.

    This mirrors the literature hybrid-ODE baseline: mechanistic scaffold with
    fixed parameters, augmented by a neural residual on the RHS.
    """

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,   # (U, P)
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        nn_hidden: int = 256,
        nn_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.U         = int(U)
        self.P         = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs       = rhs
        self.n_substeps = int(n_substeps)

        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))

        # Per-parameter bounds
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), float(theta_lo))
            hi = torch.full((self.theta_dim,), float(theta_hi))
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # Single global theta — shared across all samples and timesteps
        self.raw_theta = nn.Parameter(torch.zeros(self.theta_dim))

        # Neural correction MLP: f_NN(x) -> (P,)
        # Evaluated at each RK4 substep so correction tracks x(t), not x(t_k).
        layers: list[nn.Module] = [nn.Linear(self.P, nn_hidden), nn.SiLU()]
        for _ in range(nn_layers - 1):
            layers += [nn.Linear(nn_hidden, nn_hidden), nn.SiLU()]
        layers.append(nn.Linear(nn_hidden, self.P))
        # Zero-init output so training starts from pure mechanistic dynamics
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.nn_correction = nn.Sequential(*layers)

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
        y_seq: Optional[torch.Tensor] = None,  # unused
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        theta = log_gamma(self.raw_theta, self.theta_lo_vec, self.theta_hi_vec)
        theta = theta.unsqueeze(0).expand(B, -1)  # (B, theta_dim)

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
                # Operator split: analytic exact step on mechanism, then an
                # RK4 sub-step on the NN correction alone over the same dt.
                y_mech = self.rhs.analytic_step(y_prev, dt_k, theta, analytic_ctx)
                y = self._rk4_correction_only(y_mech, dt_k)
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
        nn_f  = self.nn_correction
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + nn_f(y)
            k2 = rhs(y + 0.5 * hdt * k1, theta) + nn_f(y + 0.5 * hdt * k1)
            k3 = rhs(y + 0.5 * hdt * k2, theta) + nn_f(y + 0.5 * hdt * k2)
            k4 = rhs(y +       hdt * k3,  theta) + nn_f(y +       hdt * k3)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_correction_only(
        self, y: torch.Tensor, dt: torch.Tensor,
    ) -> torch.Tensor:
        # RK4 over the NN correction alone, used after an analytic mechanism
        # step. dy/dt = f_NN(y) is integrated for `dt` (operator-splitting).
        nn_f  = self.nn_correction
        n_sub = self.n_substeps
        dt    = dt.unsqueeze(1)
        hdt   = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = nn_f(y)
            k2 = nn_f(y + 0.5 * hdt * k1)
            k3 = nn_f(y + 0.5 * hdt * k2)
            k4 = nn_f(y +       hdt * k3)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)
