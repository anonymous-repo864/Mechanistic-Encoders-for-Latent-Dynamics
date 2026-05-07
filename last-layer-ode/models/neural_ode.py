from typing import Optional, Tuple

import torch
import torch.nn as nn


class NeuralODE(nn.Module):
    """
    Baseline: plain MLP parameterises the full vector field directly.
    No mechanistic scaffold — the MLP IS the rhs.
    Returns a 3-tuple (y_out, th_out, beta_out) to match the shared interface;
    th_out and beta_out are zero-filled dummies.
    """

    def __init__(
        self,
        *,
        U: int,
        u_to_y_jump: torch.Tensor,   # (U,P)
        hidden: int = 128,
        lift_dim: int = 32,          # unused, kept for API compatibility
        num_layers: int = 1,         # unused, kept for API compatibility
        dropout: float = 0.0,
        theta_lo: float = 1e-3,      # unused, kept for API compatibility
        theta_hi: float = 2.0,       # unused, kept for API compatibility
        n_substeps: int = 1,
        use_basal: bool = False,     # unused, kept for API compatibility
        u_minmax_max: Optional[torch.Tensor] = None,
        u_minmax_cols: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()
        if u_to_y_jump.ndim != 2:
            raise ValueError(f"u_to_y_jump must be 2-D, got shape {tuple(u_to_y_jump.shape)}")
        P = int(u_to_y_jump.shape[1])
        self.U = int(U)
        self.P = P
        self.n_substeps = int(n_substeps)

        self.mlp = nn.Sequential(
            nn.Linear(self.U + self.P, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.P),
        )

        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

        u_max_full = torch.ones(self.U, dtype=torch.float32)
        if u_minmax_max is not None and u_minmax_cols is not None:
            cols = torch.as_tensor(u_minmax_cols, dtype=torch.long)
            u_max_full[cols] = u_minmax_max.float().clamp_min(1e-8)
            self._has_u_minmax = True
        else:
            self._has_u_minmax = False
        self.register_buffer("u_minmax_max_full", u_max_full, persistent=False)

    def _y_feat(self, y: torch.Tensor, y_transform: str) -> torch.Tensor:
        if y_transform == "sqrt":
            return y.clamp_min(0.0).sqrt()
        if y_transform == "sqrt_clamp1":
            return y.clamp_min(0.0).sqrt().clamp_min(1.0)
        if y_transform == "log1p":
            return torch.log1p(y.clamp_min(0.0))
        return y

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        u_k: torch.Tensor,
        y_transform: str = "none",
    ) -> torch.Tensor:
        n_sub = self.n_substeps
        hdt = dt.unsqueeze(1) / float(n_sub)
        for _ in range(n_sub):
            k1 = self.mlp(torch.cat([self._y_feat(y,                   y_transform), u_k], dim=-1))
            k2 = self.mlp(torch.cat([self._y_feat(y + 0.5 * hdt * k1, y_transform), u_k], dim=-1))
            k3 = self.mlp(torch.cat([self._y_feat(y + 0.5 * hdt * k2, y_transform), u_k], dim=-1))
            k4 = self.mlp(torch.cat([self._y_feat(y +       hdt * k3, y_transform), u_k], dim=-1))
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def forward(
        self,
        y0: torch.Tensor,                     # (B,P)
        u_seq: torch.Tensor,                  # (B,K,U)
        dt_seq: torch.Tensor,                 # (B,K)
        obs_idx: torch.Tensor,                # (num_obs,) — pass torch.arange(P) for full TF
        y_seq: Optional[torch.Tensor] = None, # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",
        y_transform: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.zeros(B, K, 1,      device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        # Pre-compute transformed u for MLP features (raw u still used for ODE jumps).
        u_feat = u_seq
        if u_transform in ("minmax", "minmax_sqrt"):
            u_feat = u_feat / self.u_minmax_max_full.view(1, 1, -1)
        if u_transform in ("sqrt", "cumsum_sqrt", "minmax_sqrt"):
            u_feat = u_feat.clamp_min(0.0).sqrt()

        use_partial = obs_idx.numel() > 0
        has_y_seq   = y_seq is not None

        if u_transform in ("cumsum", "cumsum_sqrt"):
            u_mlp = u_seq.cumsum(dim=1)
        else:
            u_mlp = u_seq
        if u_transform in ("minmax", "minmax_sqrt"):
            if not self._has_u_minmax:
                raise ValueError(
                    "u_transform=" + str(u_transform) + " requires u_minmax_max/u_minmax_cols at model init."
                )
            u_mlp = u_mlp / self.u_minmax_max_full.view(1, 1, -1)
        if u_transform in ("sqrt", "cumsum_sqrt", "minmax_sqrt"):
            u_mlp = u_mlp.clamp_min(0.0).sqrt()

        y_prev = y0
        for k in range(K):
            u_k     = u_seq[:, k, :]   # raw — used for the bolus jump
            u_mlp_k = u_mlp[:, k, :]   # transformed — used as MLP feature
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()

            if teacher_forcing and k > 0 and (k % tf_every == 0) and has_y_seq:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()  # type: ignore[index]
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()  # type: ignore[index]

            y = y_in + (u_k @ self.u_to_y_jump)
            y = self._rk4_substeps(y, dt_k, u_mlp_k, y_transform)

            y_out[:, k, :] = y
            y_prev = y

        return y_out, th_out, beta_out
