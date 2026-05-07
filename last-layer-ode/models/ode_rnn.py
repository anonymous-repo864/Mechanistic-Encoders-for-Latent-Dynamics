from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scaffolds import MechanisticScaffold


def gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    # Linear-sigmoid: arithmetic midpoint at x=0. Matches supervisor reference.
    return lo + (hi - lo) * torch.sigmoid(x)


def log_gamma(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    # Sigmoid in log-space: bounded in [lo, hi], geometric midpoint at x=0.
    # tau > 1 flattens the sigmoid (supervisor uses tau=2.3), preventing head saturation.
    return lo * torch.exp(torch.log(hi / lo) * torch.sigmoid(x / tau))


class StackedGRUCellBlock(nn.Module):
    """Drop-in replacement for nn.GRU that mirrors the supervisor's stacked GRUCell.

    Differs from nn.GRU: dropout is applied to EVERY layer's output (including
    the last) when training, so the head sees a dropped activation.
    Default init: orthogonal_(W_hh) + xavier_uniform_(W_ih) + zeros for biases.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        cells = [nn.GRUCell(self.input_size, self.hidden_size)]
        cells += [nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.num_layers - 1)]
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout(float(dropout))
        for c in self.cells:
            nn.init.orthogonal_(c.weight_hh)
            nn.init.xavier_uniform_(c.weight_ih)
            nn.init.zeros_(c.bias_hh)
            nn.init.zeros_(c.bias_ih)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        # x: (B, 1, input_size); h: (num_layers, B, hidden_size)
        out = x.squeeze(1)
        new_h = []
        for li, cell in enumerate(self.cells):
            state = cell(out, h[li])
            new_h.append(state)
            out = self.dropout(state) if self.training else state
        h_new = torch.stack(new_h, dim=0)
        return out.unsqueeze(1), h_new


class OdeRNN(nn.Module):
    """
    Closed-loop mechanistic model:
      (u_k, y_{k-1}) -> GRU -> theta_k
      y <- y + u_k @ jump
      y <- integrate ODE(scaffold, theta_k) over dt_k
    """

    # Compile-time flags for TorchScript so the analytic-scaffold branch is DCE'd
    # for non-analytic scaffolds, and so the TF-at-k=0 branch evaluates statically.
    # `_has_gru_*_cols` flags route the encoder feature build to either index_select
    # (TorchScript-friendly) or full-passthrough — selected once at script time.
    __constants__ = [
        "_analytic_scaffold", "_tf_at_k_zero", "theta_dim_emit",
        "_has_gru_u_cols", "_has_gru_y_cols",
    ]

    def __init__(
        self,
        *,
        U: int,
        rhs: MechanisticScaffold,
        u_to_y_jump: torch.Tensor,   # (U,P)
        hidden: int = 128,
        lift_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        theta_lo: float = 1e-3,
        theta_hi: float = 2.0,
        n_substeps: int = 1,
        use_basal: bool = False,
        theta_bounded: bool = True,
        gru_u_cols: Optional[list] = None,  # which u columns enter the GRU (None = all)
        gru_y_cols: Optional[list] = None,  # which y columns enter the GRU (None = all P).
                                            # Restricting to obs_idx prevents the GRU from being
                                            # confused by unobservable latent states.
        head_bias_init: float = 0.0,        # init all head biases to this value (<0 starts theta near lo)
        head_bias_init_vec: Optional[list] = None,  # per-theta-component bias init vector of length theta_dim;
                                                    # when provided, overrides `head_bias_init` on the theta slice
                                                    # of head.bias. Computed offline from per-step fit medians,
                                                    # inverse-transformed through the active theta_head_transform.
        head_weight_gain: float = 1.0,      # Xavier gain for head weights (>1 amplifies per-experiment variation)
        detach_y_prev: bool = True,         # if False, allow gradients to flow through y_prev in GRU
        u_minmax_max: Optional[torch.Tensor] = None,  # per-channel max for "minmax" / "minmax_sqrt" u_transform
        u_minmax_cols: Optional[list] = None,         # indices in u_seq[:,:,U] that u_minmax_max corresponds to
        theta_head_transform: str = "log_gamma",      # "log_gamma" (default) | "gamma" (linear-sigmoid; supervisor)
        theta_head_tau: float = 1.0,                  # temperature for log_gamma sigmoid; supervisor uses 2.3 (gentler slope)
        head_bottle: bool = False,                    # if True, insert hidden→…→SiLU stack before head (see head_bottle_dims)
        head_bottle_dims: Optional[list] = None,      # bottle layer widths; default [120, 40]; supervisor: [128, 64]
        lift_skip: bool = False,                      # if True, drop the lift MLP and feed feat→GRU directly (supervisor)
        gru_variant: str = "nn_gru",                  # "nn_gru" (default) | "stacked_cell" (supervisor's stacked GRUCell + dropout-on-last)
        gru_init: str = "default",                    # "default" (PyTorch defaults) | "supervisor" (orthogonal_ W_hh + xavier_ W_ih + zeros)
        head_init: str = "default",                   # "default" (PyTorch defaults; respect head_bias_init/head_weight_gain) | "supervisor" (xavier_ + zeros, unconditional)
        y0_theta_init: bool = False,                  # if True, add an MLP(y0) logit bias so the GRU starts from a
                                                      # per-sample theta prior rather than from the population mean
        **kwargs,
    ):
        super().__init__()
        self.U = int(U)
        self.P = int(rhs.P)
        self.theta_dim = int(rhs.theta_dim)
        self.rhs = rhs
        # Analytic-scaffold hooks: when the scaffold defines a closed-form step
        # (e.g. IvttAnalyticScaffold), bypass the u-jump + RK4 path and let the
        # scaffold own integration. Default-False keeps every other scaffold's
        # codepath unchanged.
        self._analytic_scaffold = bool(getattr(rhs, "has_analytic_step", False))
        self._tf_at_k_zero      = bool(getattr(rhs, "tf_at_k_zero", False))
        self.theta_dim_emit     = int(getattr(rhs, "theta_dim_emit", self.theta_dim))
        self.n_substeps   = int(n_substeps)
        self.use_basal    = bool(use_basal)
        self.theta_lo     = float(theta_lo)
        self.theta_hi     = float(theta_hi)
        self.theta_bounded = bool(theta_bounded)
        # gru_u_cols / gru_y_cols storage — keep the original `Optional[List[int]]`
        # shadow attrs for back-compat (some external code may inspect them) but
        # the forward loop uses TorchScript-friendly buffer + bool flag below.
        self.gru_u_cols   = list(gru_u_cols) if gru_u_cols is not None else None
        self._has_gru_u_cols: bool = gru_u_cols is not None
        self.register_buffer(
            "gru_u_idx",
            torch.tensor(list(gru_u_cols) if gru_u_cols is not None else [], dtype=torch.long),
            persistent=False,
        )
        self.detach_y_prev = bool(detach_y_prev)
        if theta_head_transform not in ("log_gamma", "gamma"):
            raise ValueError(f"theta_head_transform must be 'log_gamma' or 'gamma', got {theta_head_transform}")
        self.theta_head_transform = str(theta_head_transform)
        self.theta_head_tau = float(theta_head_tau)
        self.head_bottle_enabled = bool(head_bottle)
        self.head_bottle_dims = list(head_bottle_dims) if head_bottle_dims is not None else [120, 40]
        self.lift_skip = bool(lift_skip)
        if gru_variant not in ("nn_gru", "stacked_cell"):
            raise ValueError(f"gru_variant must be 'nn_gru' or 'stacked_cell', got {gru_variant}")
        if gru_init not in ("default", "supervisor"):
            raise ValueError(f"gru_init must be 'default' or 'supervisor', got {gru_init}")
        if head_init not in ("default", "supervisor"):
            raise ValueError(f"head_init must be 'default' or 'supervisor', got {head_init}")
        self.gru_variant = str(gru_variant)
        self.gru_init = str(gru_init)
        self.head_init = str(head_init)
        self.y0_theta_init = bool(y0_theta_init)

        # Per-parameter bounds — use scaffold-defined if available, else broadcast scalar
        if rhs.theta_lo_vec is not None and rhs.theta_hi_vec is not None:
            lo = torch.tensor(rhs.theta_lo_vec, dtype=torch.float32)
            hi = torch.tensor(rhs.theta_hi_vec, dtype=torch.float32)
        else:
            lo = torch.full((self.theta_dim,), theta_lo)
            hi = torch.full((self.theta_dim,), theta_hi)
        self.register_buffer("theta_lo_vec", lo)
        self.register_buffer("theta_hi_vec", hi)

        # `gru_y_cols`: indices of y to feed into the GRU. None = all P (legacy).
        # When restricted (e.g. to obs only), the GRU avoids being dominated by
        # unobservable latent states whose values are model fabrications.
        self.gru_y_cols = list(gru_y_cols) if gru_y_cols is not None else None
        self._has_gru_y_cols: bool = gru_y_cols is not None
        self.register_buffer(
            "gru_y_idx",
            torch.tensor(list(gru_y_cols) if gru_y_cols is not None else [], dtype=torch.long),
            persistent=False,
        )
        gru_y_dim = len(self.gru_y_cols) if self.gru_y_cols is not None else self.P

        gru_feat_dim = len(self.gru_u_cols) if self.gru_u_cols is not None else self.U
        feat_in = gru_feat_dim + gru_y_dim
        if self.lift_skip:
            self.lift = nn.Identity()
            gru_input_dim = feat_in
        else:
            self.lift = nn.Sequential(
                nn.Linear(feat_in, lift_dim),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            gru_input_dim = lift_dim
        if self.gru_variant == "stacked_cell":
            self.gru = StackedGRUCellBlock(gru_input_dim, hidden, num_layers, dropout)
        else:
            self.gru = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        head_out = self.theta_dim + self.P if self.use_basal else self.theta_dim
        if self.head_bottle_enabled:
            dims = self.head_bottle_dims
            layers: list[nn.Module] = []
            prev = hidden
            for d in dims:
                layers += [nn.Linear(prev, int(d)), nn.SiLU()]
                prev = int(d)
            self.head_bottle = nn.Sequential(*layers)
            head_in = prev
        else:
            self.head_bottle = nn.Identity()
            head_in = hidden
        self.head = nn.Linear(head_in, head_out)

        # Head init: "supervisor" applies xavier_uniform + zeros UNCONDITIONALLY.
        # "default" preserves the legacy guard (only override on non-default config values).
        if self.head_init == "supervisor":
            nn.init.xavier_uniform_(self.head.weight, gain=1.0)
            nn.init.zeros_(self.head.bias)
        else:
            if head_bias_init != 0.0:
                nn.init.constant_(self.head.bias, float(head_bias_init))
            if head_weight_gain != 1.0:
                nn.init.xavier_uniform_(self.head.weight, gain=float(head_weight_gain))
            if head_bias_init_vec is not None:
                vec = torch.as_tensor(list(head_bias_init_vec), dtype=self.head.bias.dtype)
                if vec.numel() != self.theta_dim:
                    raise ValueError(
                        f"head_bias_init_vec has length {vec.numel()}, expected theta_dim={self.theta_dim}"
                    )
                with torch.no_grad():
                    self.head.bias[:self.theta_dim].copy_(vec)

        # GRU init: "supervisor" forces orthogonal_(W_hh) + xavier_uniform_(W_ih) + zeros(biases)
        # on whichever variant is in use. nn.GRU stores params under names like
        # "weight_ih_l{k}", "weight_hh_l{k}", "bias_ih_l{k}", "bias_hh_l{k}".
        # StackedGRUCellBlock already applies this init at construction; redo to honor seed ordering.
        if self.gru_init == "supervisor":
            if self.gru_variant == "stacked_cell":
                for c in self.gru.cells:
                    nn.init.orthogonal_(c.weight_hh)
                    nn.init.xavier_uniform_(c.weight_ih)
                    nn.init.zeros_(c.bias_hh)
                    nn.init.zeros_(c.bias_ih)
            else:
                for name, p in self.gru.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(p)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)

        # y0 MLP: encodes the initial observation y0 into a per-sample logit bias added
        # to the GRU head output at every timestep.  Zero-initialized output layer so the
        # model starts as a standard GRU (no regression to the population mean at init).
        # Input uses the same y-column subset the GRU sees (gru_y_cols if set, else all P).
        if self.y0_theta_init:
            self.y0_mlp = nn.Sequential(
                nn.Linear(gru_y_dim, lift_dim),
                nn.SiLU(),
                nn.Linear(lift_dim, self.theta_dim),
            )
            nn.init.zeros_(self.y0_mlp[-1].weight)
            nn.init.zeros_(self.y0_mlp[-1].bias)
        else:
            self.y0_mlp = None

        if u_to_y_jump.shape != (self.U, self.P):
            raise ValueError(f"u_to_y_jump must be (U,P)=({self.U},{self.P}), got {tuple(u_to_y_jump.shape)}")
        self.register_buffer("u_to_y_jump", u_to_y_jump.float(), persistent=True)

        # Per-channel MinMax max (used by "minmax" / "minmax_sqrt" u_transform).
        # Built into a (U,) vector with 1.0 in non-scaled columns (e.g. DNA c)
        # so applying scale = 1/u_max_full leaves them unchanged.
        # Always register the buffer so TorchScript can resolve the attribute.
        # When unused, it stays at all-ones (a no-op divisor).
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
        y0: torch.Tensor,                     # (B,P)
        u_seq: torch.Tensor,                  # (B,K,U)
        dt_seq: torch.Tensor,                 # (B,K)
        obs_idx: torch.Tensor,                # (num_obs,) — pass torch.arange(P) for full TF
        y_seq: Optional[torch.Tensor] = None, # (B,K,P) for teacher forcing
        teacher_forcing: bool = True,
        tf_every: int = 50,
        u_transform: str = "none",            # GRU input transform:
                                              #   "none" | "cumsum" | "sqrt" | "cumsum_sqrt"
                                              #   "minmax" | "minmax_sqrt"  (require u_minmax_max at init)
        y_transform: str = "none",            # y_in transform: "none" | "sqrt" | "log1p"
                                              # Without a transform, large protein values (~10^3) dominate
                                              # the lift layer over reagent values (~1) and the GRU stops
                                              # responding to inputs. sqrt or log1p compresses the scale.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, K, _ = u_seq.shape
        y_out    = torch.empty(B, K, self.P, device=y0.device, dtype=y0.dtype)
        th_out   = torch.empty(B, K, self.theta_dim_emit, device=y0.device, dtype=y0.dtype)
        beta_out = torch.zeros(B, K, self.P, device=y0.device, dtype=y0.dtype)

        # Analytic scaffolds (e.g. IVTT) compute per-batch context (e.g. cumulative
        # DNA) once, and re-seed y_prev to a hidden-state-aware initial vector.
        # Typed as `Dict[str, Tensor]` (not `Optional[Dict]`) so TorchScript can
        # pass it into `analytic_step` without a None-narrow.
        analytic_ctx: Dict[str, torch.Tensor] = {}
        if self._analytic_scaffold:
            analytic_ctx = self.rhs.precompute_batch(y0, u_seq)

        h = torch.zeros(self.gru.num_layers, B, self.gru.hidden_size, device=y0.device, dtype=y0.dtype)

        use_partial = obs_idx.numel() > 0

        # Pre-compute the GRU's view of u_seq (separate from the raw delta used for ODE jumps).
        # cumsum: after a bolus at step t, the GRU sees it at ALL subsequent steps — no long-range
        # memory required. The ODE jump always uses the raw delta u_seq.
        # Replaced `in (...)` membership tests with explicit `==` chains —
        # TorchScript handles plain string equality but not always the tuple membership form.
        if u_transform == "cumsum" or u_transform == "cumsum_sqrt":
            u_gru = u_seq.cumsum(dim=1)
        else:
            u_gru = u_seq
        if u_transform == "minmax" or u_transform == "minmax_sqrt":
            if not self._has_u_minmax:
                raise ValueError(
                    "u_transform=" + str(u_transform) + " requires u_minmax_max/u_minmax_cols at model init."
                )
            u_gru = u_gru / self.u_minmax_max_full.view(1, 1, -1)
        if u_transform == "sqrt" or u_transform == "cumsum_sqrt" or u_transform == "minmax_sqrt":
            u_gru = u_gru.clamp_min(0.0).sqrt()

        # Per-sample logit bias from y0 MLP (None when y0_theta_init=False).
        if self.y0_mlp is not None:
            y0_feat = torch.index_select(y0, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y0
            if y_transform == "sqrt":
                y0_feat = y0_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                y0_feat = y0_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y0_feat = torch.log1p(y0_feat.clamp_min(0.0))
            raw_y0_bias: Optional[torch.Tensor] = self.y0_mlp(y0_feat)  # (B, theta_dim)
        else:
            raw_y0_bias = None

        y_prev = self.rhs.initial_state(y0) if self._analytic_scaffold else y0
        for k in range(K):
            u_k     = u_seq[:, k, :]   # raw delta — used only for ODE jumps
            u_gru_k = u_gru[:, k, :]   # transformed — used for GRU features
            dt_k = dt_seq[:, k]

            y_in = y_prev.detach() if self.detach_y_prev else y_prev

            # Bob's verbatim policy: TF fires at k=0 too (with k-1=-1 wrapping to the
            # last frame). Default OdeRNN behaviour skips k=0.
            tf_fires = (k % tf_every == 0) if self._tf_at_k_zero else (k > 0 and k % tf_every == 0)
            if teacher_forcing and tf_fires and y_seq is not None:
                if use_partial:
                    y_in = y_prev.clone()
                    idx = obs_idx.to(device=y_in.device, dtype=torch.long)
                    y_in[:, idx] = y_seq[:, k - 1, idx].to(dtype=y_in.dtype).detach()
                else:
                    y_in = y_seq[:, k - 1, :].to(dtype=y_prev.dtype).detach()

            # Use index_select on the precomputed long buffer instead of advanced
            # indexing with a Python list (TorchScript can't reliably script `t[:, list]`).
            u_gru_k_feat = torch.index_select(u_gru_k, dim=1, index=self.gru_u_idx) if self._has_gru_u_cols else u_gru_k

            y_in_feat = torch.index_select(y_in, dim=1, index=self.gru_y_idx) if self._has_gru_y_cols else y_in
            if y_transform == "sqrt":
                y_in_feat = y_in_feat.clamp_min(0.0).sqrt()
            elif y_transform == "sqrt_clamp1":
                # Supervisor's transform: sqrt first, then clamp at 1.
                # sqrt(0)=0 → clamped to 1, so failing experiments (pm≈0) still
                # give the GRU a non-zero, non-degenerate signal of 1.0 instead of 0.
                y_in_feat = y_in_feat.clamp_min(0.0).sqrt().clamp_min(1.0)
            elif y_transform == "log1p":
                y_in_feat = torch.log1p(y_in_feat.clamp_min(0.0))
            feat = torch.cat([u_gru_k_feat, y_in_feat], dim=-1)
            x = self.lift(feat).unsqueeze(1)
            z, h = self.gru(x, h)
            raw = self.head(self.head_bottle(z.squeeze(1)))
            if raw_y0_bias is not None:
                raw = raw + raw_y0_bias if not self.use_basal else torch.cat(
                    [raw[:, :self.theta_dim] + raw_y0_bias, raw[:, self.theta_dim:]], dim=-1
                )

            if self.use_basal:
                raw_theta = raw[:, :self.theta_dim]
                if self.theta_bounded:
                    if self.theta_head_transform == "gamma":
                        theta_k = gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec)
                    else:
                        theta_k = log_gamma(raw_theta, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                else:
                    theta_k = F.softplus(raw_theta)
                beta_k = raw[:, self.theta_dim:] * (y_prev / (y_prev + 1.0))
                beta_out[:, k, :] = beta_k
                y = y_prev + (u_k @ self.u_to_y_jump)
                y = self._rk4_substeps_basal(y, dt_k, theta_k, beta_k)
            else:
                if self.theta_bounded:
                    if self.theta_head_transform == "gamma":
                        theta_k = gamma(raw, self.theta_lo_vec, self.theta_hi_vec)
                    else:
                        theta_k = log_gamma(raw, self.theta_lo_vec, self.theta_hi_vec, tau=self.theta_head_tau)
                else:
                    theta_k = F.softplus(raw)
                if self._analytic_scaffold:
                    # Closed-form integrator owns the step (no u-jump, no RK4).
                    y = self.rhs.analytic_step(y_prev, dt_k, theta_k, analytic_ctx)
                else:
                    y = y_prev + (u_k @ self.u_to_y_jump)
                    y = self._rk4_substeps(y, dt_k, theta_k)

            y_out[:, k, :] = y
            # When the scaffold defines a wider loss-facing theta layout (e.g. Bob's
            # 8-d [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO]), let it repack here.
            th_out[:, k, :] = self.rhs.emit_theta(theta_k, y) if self._analytic_scaffold else theta_k
            y_prev = y

        return y_out, th_out, beta_out

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
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
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
        beta: torch.Tensor,
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
