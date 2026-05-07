"""
OdeMambaSSM: Mamba encoder using mambapy (pure-PyTorch, CPU/MPS/CUDA compatible).

Drop-in replacement for OdeRNN. No CUDA kernels required.
Install: pip install mambapy

Uses MambaBlock.step() for a fully differentiable recurrent step:
  - cache = (h, inputs): h is (B, ED, N), inputs is (B, ED, d_conv-1)
  - h starts as None (zero-initialised on first step)
  - gradients flow through h across steps
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mambapy.mamba import MambaBlock, MambaConfig

from scaffolds import MechanisticScaffold


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


class OdeMambapySSM(nn.Module):
    """
    Mamba SSM encoder for mechanistic ODE parameter inference.

    Architecture per step:
      (u_k, y_{k-1}) -> lift -> MambaBlocks (recurrent step) -> head -> theta_k
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
        n = max(1, num_layers)

        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        self.lift = nn.Sequential(
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
            nn.Linear(lift_dim, hidden),
        )

        cfg = MambaConfig(
            d_model=hidden,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
            expand_factor=expand,
            pscan=False,
            use_cuda=False,
        )
        self.mamba_layers = nn.ModuleList([MambaBlock(cfg) for _ in range(n)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n)])

        head_out = self.theta_dim + self.P if use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        y_out    = torch.empty(B, K, self.P,         device=device, dtype=dtype)
        th_out   = torch.empty(B, K, self.theta_dim, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P,         device=device, dtype=dtype)

        caches = self._init_caches(B, device, dtype)
        use_partial = obs_idx.numel() > 0
        y_prev = y0

        for k in range(K):
            u_k  = u_seq[:, k, :]  # (B, U)
            dt_k = dt_seq[:, k]    # (B,)

            y_in = y_prev.detach()
            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx  = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            x = self.lift(torch.cat([u_k, y_in], dim=-1))  # (B, hidden)

            for i, (norm, layer) in enumerate(zip(self.norms, self.mamba_layers)):
                x_out, caches[i] = layer.step(norm(x), caches[i])
                x = x + x_out

            raw = self.head(x)  # (B, head_out)

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
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
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

    def _init_caches(self, B, device, dtype):
        caches = []
        cfg = self.mamba_layers[0].config
        for _ in range(len(self.mamba_layers)):
            h = None  # zero-initialised on first step by ssm_step
            inputs = torch.zeros(B, cfg.d_inner, cfg.d_conv - 1, device=device, dtype=dtype)
            caches.append((h, inputs))
        return caches
