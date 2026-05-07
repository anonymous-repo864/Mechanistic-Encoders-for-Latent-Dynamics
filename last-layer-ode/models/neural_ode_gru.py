from typing import Optional, Tuple

import torch
import torch.nn as nn


class NeuralOdeGRU(nn.Module):
    """
    Black-box GRU baseline: GRU predicts dy/dt directly, integrated with Euler.

    No mechanistic scaffold — the GRU IS the vector field. Uses Euler (not RK4)
    because the GRU hidden state cannot be rolled back; calling the RHS multiple
    times per step (as RK4 requires) would incorrectly advance the hidden state.

    Architecture:
      (u_k, y_{k-1}) -> lift (SiLU) -> GRU -> head -> dy/dt
      y_k = y_{k-1} + u_k @ jump + dt_k * dy/dt

    Returns a 3-tuple (y_out, th_out, beta_out) to match the shared interface;
    th_out and beta_out are zero-filled dummies.
    """

    def __init__(
        self,
        *,
        U: int,
        u_to_y_jump: torch.Tensor,   # (U, P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        self.head = nn.Linear(hidden, self.P)

        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

        u_max_full = torch.ones(self.U, dtype=torch.float32)
        if u_minmax_max is not None and u_minmax_cols is not None:
            cols = torch.as_tensor(u_minmax_cols, dtype=torch.long)
            u_max_full[cols] = u_minmax_max.float().clamp_min(1e-8)
            self._has_u_minmax = True
        else:
            self._has_u_minmax = False
        self.register_buffer("u_minmax_max_full", u_max_full, persistent=False)

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

        y_out    = torch.empty(B, K, self.P, device=device, dtype=dtype)
        th_out   = torch.zeros(B, K, 1,      device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P, device=device, dtype=dtype)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=device, dtype=dtype)

        # Pre-compute transformed u for GRU features (raw u still used for ODE jumps).
        u_gru = u_seq
        if u_transform in ("minmax", "minmax_sqrt"):
            u_gru = u_gru / self.u_minmax_max_full.view(1, 1, -1)
        if u_transform in ("sqrt", "cumsum_sqrt", "minmax_sqrt"):
            u_gru = u_gru.clamp_min(0.0).sqrt()

        use_partial = obs_idx.numel() > 0

        if u_transform in ("cumsum", "cumsum_sqrt"):
            u_gru = u_seq.cumsum(dim=1)
        else:
            u_gru = u_seq
        if u_transform in ("minmax", "minmax_sqrt"):
            if not self._has_u_minmax:
                raise ValueError(
                    "u_transform=" + str(u_transform) + " requires u_minmax_max/u_minmax_cols at model init."
                )
            u_gru = u_gru / self.u_minmax_max_full.view(1, 1, -1)
        if u_transform in ("sqrt", "cumsum_sqrt", "minmax_sqrt"):
            u_gru = u_gru.clamp_min(0.0).sqrt()

        y_prev = y0
        for k in range(K):
            u_k     = u_seq[:, k, :]   # raw — used for the bolus jump
            u_gru_k = u_gru[:, k, :]   # transformed — used for GRU feature
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()
            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            if y_transform == "sqrt":
                y_in_feat = y_in.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y_in_feat = y_in.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_in_feat = torch.log1p(y_in.clamp_min(0.0))
            else:
                y_in_feat = y_in

            feat = torch.cat([u_gru_k, y_in_feat], dim=-1)
            x    = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            dydt = self.head(z.squeeze(1))  # (B, P)

            # Euler integration — Euler only; GRU state cannot be rolled back for RK4
            y = y_in + (u_k @ self.u_to_y_jump) + dt_k.unsqueeze(1) * dydt
            y = torch.clamp_min(y, 0.0)

            y_out[:, k, :] = y
            y_prev = y

        return y_out, th_out, beta_out
