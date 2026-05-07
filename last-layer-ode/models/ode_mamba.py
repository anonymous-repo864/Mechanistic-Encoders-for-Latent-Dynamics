"""
ODE-Mamba: Selective State Space Model encoder for mechanistic ODE parameter inference.

Drop-in replacement for OdeRNN. Replaces the GRU with a minimal selective SSM
(S4D/Mamba-style) implemented in pure PyTorch — no external dependencies, no
CUDA kernels required. Compatible with torch.jit.script() and torch.compile().

Architecture per step:
  (u_k, y_{k-1}) -> lift -> MambaBlock -> head -> theta_k
  y <- y + u_k @ jump
  y <- RK4(scaffold, theta_k, dt_k)

The selective SSM replaces the GRU's gating mechanism with a continuous-time
state-space recurrence where B, C, and Δ (discretization step) are input-dependent.
This gives the model a natural inductive bias for continuous-time dynamical systems.

Key differences from GRU:
  - State update is a discretized linear ODE: x_k = exp(Δ_k·A)·x_{k-1} + Δ_k·B_k·u_k
  - A is a fixed diagonal matrix (initialized with HiPPO-inspired negative reals)
  - B_k, C_k, Δ_k are projected from input (selectivity)
  - Short Conv1d provides local context before SSM (like Mamba's architecture)
  - Gated residual path (SiLU gate) controls information flow
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x))


# ---------------------------------------------------------------------------
# Minimal Selective SSM (diagonal, pure PyTorch, scriptable)
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    Single-layer selective SSM operating on a 1-D feature channel.

    For an input sequence processed one step at a time (recurrent mode):
      1. Project input to B_k, C_k, Δ_k (selectivity)
      2. Discretize: Ā_k = exp(Δ_k · A), B̄_k = Δ_k · B_k
      3. Recurrence: x_k = Ā_k * x_{k-1} + B̄_k * z_k
      4. Output: y_k = C_k · x_k

    A is diagonal (N values), initialized as -exp(linspace(log(1), log(N), N))
    which approximates HiPPO initialization for long-range memory.

    Args:
        d_inner: Dimension of the input features (after expansion)
        d_state: SSM state expansion factor N (default 16)
        dt_min:  Minimum discretization step (default 0.001)
        dt_max:  Maximum discretization step (default 0.1)
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A: fixed diagonal, HiPPO-inspired initialization
        # Shape: (d_inner, d_state) — one SSM per channel
        A = -torch.exp(torch.linspace(math.log(1.0), math.log(float(d_state)), d_state))
        self.A_log = nn.Parameter(torch.log(-A).unsqueeze(0).expand(d_inner, -1).clone())
        # A_log shape: (d_inner, d_state)

        # Input-dependent projections for selectivity
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)
        self.proj_dt = nn.Linear(d_inner, d_inner, bias=True)

        # Initialize dt bias so softplus(bias) ∈ [dt_min, dt_max]
        dt_init = torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        dt_init = dt_init.exp().clamp(min=dt_min, max=dt_max)
        inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
        self.proj_dt.bias.data = inv_softplus

        # D: skip connection (like Mamba's D parameter)
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward_step(
        self,
        z: torch.Tensor,       # (B, d_inner) — input at this step
        ssm_state: torch.Tensor,  # (B, d_inner, d_state) — previous SSM state
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single recurrent step. Returns (output, new_state)."""
        B_batch = z.shape[0]

        # Selectivity: compute input-dependent B, C, Δ
        B_k = self.proj_B(z)               # (B, d_state)
        C_k = self.proj_C(z)               # (B, d_state)
        dt_k = F.softplus(self.proj_dt(z))  # (B, d_inner), positive

        # Reconstruct A (negative, diagonal)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize per-channel, per-state:
        # Ā_k = exp(dt_k · A) — element-wise since A is diagonal
        # dt_k: (B, d_inner) -> (B, d_inner, 1) for broadcasting
        # A:    (d_inner, d_state) -> (1, d_inner, d_state)
        dt_A = dt_k.unsqueeze(-1) * A.unsqueeze(0)  # (B, d_inner, d_state)
        A_bar = torch.exp(dt_A)                       # (B, d_inner, d_state)

        # B̄_k = dt_k * B_k (simplified discretization, standard in Mamba)
        # dt_k: (B, d_inner, 1), B_k: (B, 1, d_state) -> (B, d_inner, d_state)
        B_bar = dt_k.unsqueeze(-1) * B_k.unsqueeze(1)  # (B, d_inner, d_state)

        # Recurrence: x_k = Ā_k ⊙ x_{k-1} + B̄_k ⊙ z_k
        # z: (B, d_inner) -> (B, d_inner, 1) for broadcasting into state dim
        new_state = A_bar * ssm_state + B_bar * z.unsqueeze(-1)  # (B, d_inner, d_state)

        # Output: y_k = (C_k · x_k) + D · z_k
        # C_k: (B, d_state) -> (B, 1, d_state)
        # new_state: (B, d_inner, d_state)
        y = (new_state * C_k.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        y = y + self.D * z  # skip connection

        return y, new_state


class MambaBlock(nn.Module):
    """
    One Mamba block: expand -> conv1d -> SSM -> gate -> contract.

    Mirrors the Mamba paper architecture:
      input ──┬── Linear(expand) ── Conv1d ── SiLU ── SSM ──┐
              │                                               × (element-wise)
              └── Linear(expand) ── SiLU ───────────────────┘
              └── Linear(contract) ── output

    For step-by-step recurrence, the Conv1d is implemented as a
    rolling buffer of the last (kernel_size - 1) inputs.

    Args:
        d_model:  Input/output dimension
        d_state:  SSM state expansion factor (default 16)
        expand:   Expansion factor for inner dimension (default 2)
        d_conv:   Conv1d kernel size (default 4)
        dt_min:   Min discretization step for SSM
        dt_max:   Max discretization step for SSM
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Expansion projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv (will be applied step-by-step via rolling buffer)
        self.conv_weight = nn.Parameter(torch.randn(self.d_inner, 1, d_conv) * 0.02)
        self.conv_bias = nn.Parameter(torch.zeros(self.d_inner))

        # Selective SSM
        self.ssm = SelectiveSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            dt_min=dt_min,
            dt_max=dt_max,
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm (pre-norm style)
        self.norm = nn.LayerNorm(d_model)

    def forward_step(
        self,
        x: torch.Tensor,           # (B, d_model)
        ssm_state: torch.Tensor,    # (B, d_inner, d_state)
        conv_buf: torch.Tensor,     # (B, d_inner, d_conv - 1)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single recurrent step. Returns (output, new_ssm_state, new_conv_buf).
        """
        residual = x
        x = self.norm(x)

        # Expand to 2 * d_inner, split into SSM path and gate path
        xz = self.in_proj(x)                          # (B, 2 * d_inner)
        x_ssm = xz[:, :self.d_inner]                  # (B, d_inner)
        z_gate = xz[:, self.d_inner:]                  # (B, d_inner)

        # Conv1d via rolling buffer:
        # Append current input to buffer, apply conv kernel
        # conv_buf: (B, d_inner, d_conv - 1)
        # x_ssm:    (B, d_inner) -> (B, d_inner, 1)
        conv_input = torch.cat([conv_buf, x_ssm.unsqueeze(-1)], dim=-1)  # (B, d_inner, d_conv)
        new_conv_buf = conv_input[:, :, 1:]  # shift: drop oldest, keep last (d_conv-1)

        # Depthwise conv: sum over kernel dim for each channel
        # conv_weight: (d_inner, 1, d_conv) -> squeeze to (d_inner, d_conv)
        # conv_input:  (B, d_inner, d_conv)
        conv_out = (conv_input * self.conv_weight.squeeze(1).unsqueeze(0)).sum(dim=-1)  # (B, d_inner)
        conv_out = conv_out + self.conv_bias

        # Activation after conv
        conv_out = F.silu(conv_out)

        # Selective SSM
        ssm_out, new_ssm_state = self.ssm.forward_step(conv_out, ssm_state)

        # Gated output
        y = ssm_out * F.silu(z_gate)  # (B, d_inner)

        # Contract back to d_model + residual
        out = self.out_proj(y) + residual  # (B, d_model)

        return out, new_ssm_state, new_conv_buf


class MambaEncoder(nn.Module):
    """
    Stack of MambaBlocks used as a sequence encoder, processing one step at a time.

    Args:
        d_model:    Hidden dimension (matches 'hidden' in OdeRNN)
        n_layers:   Number of stacked MambaBlocks
        d_state:    SSM state expansion factor
        expand:     Expansion factor per block
        d_conv:     Conv kernel size per block
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dt_min=dt_min,
                dt_max=dt_max,
            )
            for _ in range(n_layers)
        ])

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[list, list]:
        """Initialize SSM states and conv buffers for all layers."""
        ssm_states = []
        conv_bufs = []
        for _ in range(self.n_layers):
            ssm_states.append(
                torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
            )
            conv_bufs.append(
                torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
            )
        return ssm_states, conv_bufs

    def step(
        self,
        x: torch.Tensor,                    # (B, d_model)
        ssm_states: list,     # list of (B, d_inner, d_state)
        conv_bufs: list,      # list of (B, d_inner, d_conv - 1)
    ) -> Tuple[torch.Tensor, list, list]:
        """Process one timestep through all layers."""
        new_ssm_states = []
        new_conv_bufs = []
        for i, layer in enumerate(self.layers):
            x, new_ssm, new_conv = layer.forward_step(x, ssm_states[i], conv_bufs[i])
            new_ssm_states.append(new_ssm)
            new_conv_bufs.append(new_conv)
        return x, new_ssm_states, new_conv_bufs


# ---------------------------------------------------------------------------
# OdeMamba: Full model with scaffold integration
# ---------------------------------------------------------------------------

class OdeMamba(nn.Module):
    """ 
    Selective SSM encoder for mechanistic ODE parameter inference.

    Drop-in replacement for OdeRNN. Same forward signature:
      forward(y0, u_seq, dt_seq, obs_idx, y_seq, teacher_forcing, tf_every)
        -> (y_out, th_out, beta_out)

    Architecture:
      lift: (U+P) -> lift_dim -> SiLU -> hidden
      encoder: MambaEncoder (stack of MambaBlocks)
      head: hidden -> theta_dim [+ P if use_basal]
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
        # Mamba-specific
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
        self.theta_lo = float(theta_lo)
        self.theta_hi = float(theta_hi)
        self.theta_bounded = bool(theta_bounded)
        self.hidden = int(hidden)

        # Per-parameter bounds
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # Lift: same as OdeRNN but project up to hidden dim
        self.lift = nn.Sequential(
            nn.Linear(self.U + self.P, lift_dim),
            nn.SiLU(),
            nn.Linear(lift_dim, hidden),
        )

        # Mamba encoder
        self.encoder = MambaEncoder(
            d_model=hidden,
            n_layers=max(1, num_layers),
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
        )

        # Head: hidden -> theta (+ basal)
        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        self.head = nn.Linear(hidden, head_out)
        # Small-init head (same reasoning as OdeTransformer)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(
                f"u_to_y_jump must be (U,P)=({self.U},{self.P}), "
                f"got {tuple(u_to_y_jump.shape)}"
            )
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

    def forward(
        self,
        y0: torch.Tensor,                      # (B, P)
        u_seq: torch.Tensor,                    # (B, K, U)
        dt_seq: torch.Tensor,                   # (B, K)
        obs_idx: torch.Tensor,                  # (num_obs,)
        y_seq: Optional[torch.Tensor] = None,   # (B, K, P)
        teacher_forcing: bool = True,
        tf_every: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        device, dtype = y0.device, y0.dtype

        y_out = torch.empty(B, K, self.P, device=device, dtype=dtype)
        th_out = torch.empty(B, K, self.theta_dim, device=device, dtype=dtype)
        beta_out = torch.zeros(B, K, self.P, device=device, dtype=dtype)

        # Initialize Mamba recurrent state
        ssm_states, conv_bufs = self.encoder.init_state(B, device, dtype)

        use_partial = obs_idx.numel() > 0
        y_prev = y0

        for k in range(K):
            u_k = u_seq[:, k, :]
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach()

            if teacher_forcing and k > 0 and (k % tf_every == 0) and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            # Lift to hidden dim
            feat = self.lift(torch.cat([u_k, y_in], dim=-1))  # (B, hidden)

            # Mamba encoder step (detach states to prevent graph accumulation over K steps)
            ssm_states = [s.detach() for s in ssm_states]
            conv_bufs = [c.detach() for c in conv_bufs]
            z, ssm_states, conv_bufs = self.encoder.step(feat, ssm_states, conv_bufs)

            # Decode theta
            raw = self.head(z)  # (B, theta_dim [+ P])

            if self.use_basal:
                raw_theta = raw[:, :self.theta_dim]
                if self.theta_bounded:
                    theta_k = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = F.softplus(raw_theta)
                beta_k = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                if self.theta_bounded:
                    theta_k = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                else:
                    theta_k = F.softplus(raw)
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            th_out[:, k, :] = theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(
        self, y: torch.Tensor, dt: torch.Tensor, theta: torch.Tensor,
    ) -> torch.Tensor:
        rhs = self.rhs
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta)
            k2 = rhs(y + 0.5 * hdt * k1, theta)
            k3 = rhs(y + 0.5 * hdt * k2, theta)
            k4 = rhs(y +       hdt * k3,  theta)
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)

    def _rk4_substeps_basal(
        self, y: torch.Tensor, dt: torch.Tensor,
        theta: torch.Tensor, beta: torch.Tensor,
    ) -> torch.Tensor:
        rhs = self.rhs
        n_sub = self.n_substeps
        dt = dt.unsqueeze(1)
        hdt = dt / float(n_sub)
        for _ in range(n_sub):
            k1 = rhs(y,                   theta) + beta
            k2 = rhs(y + 0.5 * hdt * k1, theta) + beta
            k3 = rhs(y + 0.5 * hdt * k2, theta) + beta
            k4 = rhs(y +       hdt * k3,  theta) + beta
            y  = y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return torch.clamp_min(y, 0.0)