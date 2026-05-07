"""
Analytical integrator for TXTLResourceandMaturationDNAScaffold.

The ODE system is cascade-linear: R and O decay independently as exponentials,
DNA is constant between jumps, and m/mm/p/pm are linear ODEs driven by those
known signals. This gives a closed-form solution via convolution of exponentials.

States: [R, O, m, mm, p, pm, DNA]
Params: [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]
"""
import torch

from models.ode_rnn import OdeRNN


def _safe(x: torch.Tensor) -> torch.Tensor:
    """Replace near-zero with 1 so masked branches in torch.where stay finite."""
    return torch.where(x.abs() < 1e-7, torch.ones_like(x), x)


def _conv(A: torch.Tensor,
          r_in: torch.Tensor, r_out: torch.Tensor,
          e_in: torch.Tensor, e_out: torch.Tensor,
          dt: torch.Tensor) -> torch.Tensor:
    """
    A * integral_0^dt  exp(-r_out*(dt-s)) * exp(-r_in*s) ds
      = A * (e_in - e_out) / (r_out - r_in)   [generic]
      = A * dt * e_out                          [r_in == r_out]
    """
    diff = r_out - r_in
    is_deg = diff.abs() < 1e-7
    generic = A * (e_in - e_out) / _safe(diff)
    degen   = A * dt * e_out
    return torch.where(is_deg, degen, generic)


def _coeff(A: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
    """A / diff with near-zero diff mapped to 0 (degenerate-case coefficient)."""
    return torch.where(diff.abs() < 1e-7, torch.zeros_like(A), A / _safe(diff))


def _integ(C: torch.Tensor, r_x: torch.Tensor,
           lam_O: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """C * (1 - exp(-(lam_O + r_x)*dt)) / (lam_O + r_x)  — integral of C*exp(-(lam_O+r_x)*t)."""
    r_tot = (lam_O + r_x).clamp_min(1e-7)
    return C * (1.0 - torch.exp(-r_tot * dt)) / r_tot


def _integ_bleach(C: torch.Tensor, r_x: torch.Tensor,
                  lam_O: torch.Tensor, kbleach: torch.Tensor,
                  e_b: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    C * integral_0^dt exp(-kbleach*(dt-s)) * exp(-(lam_O + r_x)*s) ds

    Generic:    C * (exp(-(lam_O+r_x)*dt) - exp(-kbleach*dt)) / (kbleach - (lam_O+r_x))
    Degenerate: C * dt * exp(-kbleach*dt)             [when kbleach == lam_O+r_x]

    Reduces to _integ when kbleach == 0 (then exp(-kbleach*dt)=1 and the diff
    cancels into (1 - exp(-r_tot*dt))/r_tot).
    """
    rtot = lam_O + r_x
    diff = kbleach - rtot
    e_tot = torch.exp(-rtot * dt)
    is_deg = diff.abs() < 1e-7
    generic = C * (e_tot - e_b) / _safe(diff)
    degen = C * dt * e_b
    return torch.where(is_deg, degen, generic)


def txtl_step(y0: torch.Tensor, theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for TXTLResourceandMaturationDNAScaffold over one interval dt.

    y0    : (B, 7) — [R, O, m, mm, p, pm, DNA]
    theta : (B, 7) — [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]
    dt    : (B,)
    Returns y_new (B, 7), clamped >= 0.
    """
    R0, O0, m0, mm0, p0, pm0, D0 = y0.clamp_min(0.0).unbind(-1)
    lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)

    beta = kdm + kmatm          # total m-decay rate

    # Precomputed exponentials for each decay rate
    e_lam  = torch.exp(-lam   * dt)
    e_lamO = torch.exp(-lam_O * dt)
    e_beta = torch.exp(-beta  * dt)
    e_kdm  = torch.exp(-kdm   * dt)
    e_kmt  = torch.exp(-kmt   * dt)
    # Combined rates needed for forcing of p
    r_2lam  = lam + lam
    r_lbeta = lam + beta
    r_lkdm  = lam + kdm
    e_2lam  = e_lam * e_lam     # exp(-(2*lam)*dt)
    e_lbeta = e_lam * e_beta    # exp(-(lam+beta)*dt)
    e_lkdm  = e_lam * e_kdm    # exp(-(lam+kdm)*dt)

    # -------------------------------------------------------------------------
    # R, O, DNA: trivial
    # -------------------------------------------------------------------------
    R_new   = R0 * e_lam
    O_new   = O0 * e_lamO
    DNA_new = D0

    # -------------------------------------------------------------------------
    # m(t) = m0*exp(-beta*t) + VTXmax*D0*R0 * conv(lam→beta)
    # -------------------------------------------------------------------------
    VDR   = VTXmax * D0 * R0
    m_new = m0 * e_beta + _conv(VDR, lam, beta, e_lam, e_beta, dt)

    # Decompose m(t) = m_A*exp(-lam*t) + m_B*exp(-beta*t)
    # so downstream forcing can be tracked per-mode.
    m_A = _coeff(VDR, beta - lam)          # coefficient on exp(-lam*t)
    m_B = m0 - m_A                         # coefficient on exp(-beta*t)
    # (In the degen case lam≈beta, _coeff → 0 and m_B → m0; _conv handles m_new correctly.)

    # -------------------------------------------------------------------------
    # mm(t) = mm0*exp(-kdm*t) + kmatm*m_A*conv(lam→kdm) + kmatm*m_B*conv(beta→kdm)
    # -------------------------------------------------------------------------
    mm_new = (mm0 * e_kdm
              + _conv(kmatm * m_A, lam,  kdm, e_lam,  e_kdm, dt)
              + _conv(kmatm * m_B, beta, kdm, e_beta, e_kdm, dt))

    # Decompose mm(t) = mm_A*exp(-lam*t) + mm_B*exp(-beta*t) + mm_C*exp(-kdm*t)
    mm_A = _coeff(kmatm * m_A, kdm - lam)
    mm_B = _coeff(kmatm * m_B, kdm - beta)
    mm_C = mm0 - mm_A - mm_B

    # -------------------------------------------------------------------------
    # p(t): dp/dt = VTLmax*R(t)*(m+mm) - kmt*p
    # Forcing = VTLmax*R0*exp(-lam*t) * [(m_A+mm_A)*exp(-lam*t)
    #                                    + (m_B+mm_B)*exp(-beta*t)
    #                                    + mm_C*exp(-kdm*t)]
    # Three exponential forcing terms with combined rates 2lam, lam+beta, lam+kdm.
    # -------------------------------------------------------------------------
    VR0    = VTLmax * R0
    sA_mA  = m_A + mm_A           # amplitude of exp(-lam*t) mode in (m+mm)
    sA_mB  = m_B + mm_B           # amplitude of exp(-beta*t) mode
    sA_mmC = mm_C                  # amplitude of exp(-kdm*t) mode

    p_new = (p0 * e_kmt
             + _conv(VR0 * sA_mA,  r_2lam,  kmt, e_2lam,  e_kmt, dt)
             + _conv(VR0 * sA_mB,  r_lbeta, kmt, e_lbeta, e_kmt, dt)
             + _conv(VR0 * sA_mmC, r_lkdm,  kmt, e_lkdm,  e_kmt, dt))

    # Decompose p(t) for pm integral:
    # p(t) = p_A*exp(-2lam*t) + p_B*exp(-(lam+beta)*t)
    #       + p_C*exp(-(lam+kdm)*t) + p_D*exp(-kmt*t)
    p_A = _coeff(VR0 * sA_mA,  kmt - r_2lam)
    p_B = _coeff(VR0 * sA_mB,  kmt - r_lbeta)
    p_C = _coeff(VR0 * sA_mmC, kmt - r_lkdm)
    p_D = p0 - p_A - p_B - p_C

    # -------------------------------------------------------------------------
    # pm(t) = pm0 + kmt*O0 * integral_0^dt exp(-lam_O*s) * p(s) ds
    # Each exponential mode p_X*exp(-r_X*t) contributes:
    #   p_X * (1 - exp(-(lam_O+r_X)*dt)) / (lam_O + r_X)
    # All combined rates (lam_O + r_X) are strictly positive so no degeneracy.
    # -------------------------------------------------------------------------
    pm_new = (pm0
              + kmt * O0 * (
                  _integ(p_A, r_2lam,  lam_O, dt)
                + _integ(p_B, r_lbeta, lam_O, dt)
                + _integ(p_C, r_lkdm,  lam_O, dt)
                + _integ(p_D, kmt,     lam_O, dt)
              ))

    return torch.stack(
        [R_new, O_new, m_new, mm_new, p_new, pm_new, DNA_new], dim=-1
    ).clamp_min(0.0)


class TXTLAnalyticalOdeRNN(OdeRNN):
    """
    OdeRNN with the TXTL cascade solved exactly instead of via RK4.

    Inherits the full GRU → log_gamma(theta) → integrate pipeline from OdeRNN;
    only the integration step is replaced. n_substeps is ignored.

    Use with scaffold: TXTLResourceandMaturationDNAScaffold
    Config key:        ode_rnn_txtl
    """

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        # dt arrives as (B,) from OdeRNN.forward (before the unsqueeze in the base method)
        return txtl_step(y, theta, dt)


def txtl_step_bleach(y0: torch.Tensor, theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Exact solution for TXTLResourceandMaturationDNABleachScaffold over one interval dt.
    Same as txtl_step but with dpm/dt = O*kmt*p - kbleach*pm.

    y0    : (B, 7) — [R, O, m, mm, p, pm, DNA]
    theta : (B, 8) — [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm, kbleach]
    dt    : (B,)
    """
    R0, O0, m0, mm0, p0, pm0, D0 = y0.clamp_min(0.0).unbind(-1)
    lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm, kbleach = theta.unbind(-1)

    beta = kdm + kmatm

    e_lam   = torch.exp(-lam     * dt)
    e_lamO  = torch.exp(-lam_O   * dt)
    e_beta  = torch.exp(-beta    * dt)
    e_kdm   = torch.exp(-kdm     * dt)
    e_kmt   = torch.exp(-kmt     * dt)
    e_kbl   = torch.exp(-kbleach * dt)

    r_2lam  = lam + lam
    r_lbeta = lam + beta
    r_lkdm  = lam + kdm
    e_2lam  = e_lam * e_lam
    e_lbeta = e_lam * e_beta
    e_lkdm  = e_lam * e_kdm

    # R, O, DNA: trivial
    R_new   = R0 * e_lam
    O_new   = O0 * e_lamO
    DNA_new = D0

    # m(t)
    VDR   = VTXmax * D0 * R0
    m_new = m0 * e_beta + _conv(VDR, lam, beta, e_lam, e_beta, dt)
    m_A = _coeff(VDR, beta - lam)
    m_B = m0 - m_A

    # mm(t)
    mm_new = (mm0 * e_kdm
              + _conv(kmatm * m_A, lam,  kdm, e_lam,  e_kdm, dt)
              + _conv(kmatm * m_B, beta, kdm, e_beta, e_kdm, dt))
    mm_A = _coeff(kmatm * m_A, kdm - lam)
    mm_B = _coeff(kmatm * m_B, kdm - beta)
    mm_C = mm0 - mm_A - mm_B

    # p(t)
    VR0    = VTLmax * R0
    sA_mA  = m_A + mm_A
    sA_mB  = m_B + mm_B
    sA_mmC = mm_C
    p_new = (p0 * e_kmt
             + _conv(VR0 * sA_mA,  r_2lam,  kmt, e_2lam,  e_kmt, dt)
             + _conv(VR0 * sA_mB,  r_lbeta, kmt, e_lbeta, e_kmt, dt)
             + _conv(VR0 * sA_mmC, r_lkdm,  kmt, e_lkdm,  e_kmt, dt))
    p_A = _coeff(VR0 * sA_mA,  kmt - r_2lam)
    p_B = _coeff(VR0 * sA_mB,  kmt - r_lbeta)
    p_C = _coeff(VR0 * sA_mmC, kmt - r_lkdm)
    p_D = p0 - p_A - p_B - p_C

    # pm(t) with bleaching:
    #   pm(dt) = pm0*exp(-kbleach*dt)
    #          + kmt*O0 * Σ_X p_X * conv_bleach(lam_O + r_X, kbleach)
    pm_new = (pm0 * e_kbl
              + kmt * O0 * (
                  _integ_bleach(p_A, r_2lam,  lam_O, kbleach, e_kbl, dt)
                + _integ_bleach(p_B, r_lbeta, lam_O, kbleach, e_kbl, dt)
                + _integ_bleach(p_C, r_lkdm,  lam_O, kbleach, e_kbl, dt)
                + _integ_bleach(p_D, kmt,     lam_O, kbleach, e_kbl, dt)
              ))

    return torch.stack(
        [R_new, O_new, m_new, mm_new, p_new, pm_new, DNA_new], dim=-1
    ).clamp_min(0.0)


def ivtt_step_approx(y0: torch.Tensor, theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Bob's approximate step for TXTLResourceandMaturationDNAScaffold.

    Key approximations vs txtl_step:
    - Transcription uses R at END of step (R_curr) as constant rather than
      integrating R(t) over the interval.
    - m solved via pseudo-equilibrium m_inf = S/alpha with R_curr.
    - p integration lumps m+mm into M with M_inf = S/kdm.
    - pm uses a shortcut: pm += O_curr * (VTL_eff*integral_M - delta_p).

    y0    : (B, 7) — [R, O, m, mm, p, pm, DNA]
    theta : (B, 7) — [lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm]
    dt    : (B,)
    """
    R0, O0, m0, mm0, p0, pm0, D0 = y0.clamp_min(0.0).unbind(-1)
    lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)
    eps = 1e-9

    rho_R = torch.exp(-lam   * dt)
    rho_O = torch.exp(-lam_O * dt)
    R_curr = R0 * rho_R
    O_curr = O0 * rho_O

    # Transcription / translation use end-of-step R (Bob's approx)
    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax

    S = VTX_eff * D0
    alpha = (kdm + kmatm).clamp_min(eps)
    m_inf = S / alpha
    exp_a = torch.exp(-alpha * dt)
    m_curr = torch.clamp_min(m_inf + (m0 - m_inf) * exp_a, 0.0)

    exp_d  = torch.exp(-kdm   * dt)
    exp_mr = torch.exp(-kmatm * dt)
    mm_curr = torch.clamp_min(
        mm0 * exp_d
        + m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
        + (m0 - m_inf) * exp_d * (1.0 - exp_mr),
        0.0,
    )

    M_prev = m0  + mm0
    M_inf  = S   / (kdm + eps)
    exp_M  = exp_d

    eta   = torch.exp(-kmt * dt)
    delta = kdm - kmt
    same  = delta.abs() < 1e-6
    int_M_eq  = M_inf * (1.0 - eta) / (kmt + eps) + (M_prev - M_inf) * dt * eta
    int_M_gen = M_inf * (1.0 - eta) / (kmt + eps) + (M_prev - M_inf) * (eta - exp_M) / (delta + eps)
    int_M_conv = torch.where(same, int_M_eq, int_M_gen)
    p_curr = torch.clamp_min(p0 * eta + VTL_eff * int_M_conv, 0.0)

    int_M_total = M_inf * dt + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M)
    pm_curr = torch.clamp_min(
        pm0 + O_curr * (VTL_eff * int_M_total - (p_curr - p0)),
        0.0,
    )

    return torch.stack([R_curr, O_curr, m_curr, mm_curr, p_curr, pm_curr, D0], dim=-1)


def ivtt_maturation_step_approx(y0: torch.Tensor, theta: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Bob's approximate step adapted for TXTLMaturationOnly7Scaffold.

    Identical to ivtt_step_approx except:
      - theta has 6 params (no lam_O): [lam, VTXmax, kdm, VTLmax, kmt, kmatm]
      - O is a frozen dummy state (dO=0): O_curr = O0
      - pm uses dpm = kmt*p  (no O multiplier): pm += VTL_eff*int_M_total - delta_p

    y0    : (B, 7) — [R, O, m, mm, p, pm, DNA]
    theta : (B, 6) — [lam, VTXmax, kdm, VTLmax, kmt, kmatm]
    dt    : (B,)
    """
    R0, O0, m0, mm0, p0, pm0, D0 = y0.clamp_min(0.0).unbind(-1)
    lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)
    eps = 1e-9

    R_curr = R0 * torch.exp(-lam * dt)
    O_curr = O0  # dummy, dO=0

    VTX_eff = R_curr * VTXmax
    VTL_eff = R_curr * VTLmax

    S     = VTX_eff * D0
    alpha = (kdm + kmatm).clamp_min(eps)
    m_inf = S / alpha
    exp_a = torch.exp(-alpha * dt)
    m_curr = torch.clamp_min(m_inf + (m0 - m_inf) * exp_a, 0.0)

    exp_d  = torch.exp(-kdm   * dt)
    exp_mr = torch.exp(-kmatm * dt)
    mm_curr = torch.clamp_min(
        mm0 * exp_d
        + m_inf * (kmatm / (kdm + eps)) * (1.0 - exp_d)
        + (m0 - m_inf) * exp_d * (1.0 - exp_mr),
        0.0,
    )

    M_prev = m0 + mm0
    M_inf  = S / (kdm + eps)
    exp_M  = exp_d

    eta   = torch.exp(-kmt * dt)
    delta = kdm - kmt
    same  = delta.abs() < 1e-6
    int_M_eq  = M_inf * (1.0 - eta) / (kmt + eps) + (M_prev - M_inf) * dt * eta
    int_M_gen = M_inf * (1.0 - eta) / (kmt + eps) + (M_prev - M_inf) * (eta - exp_M) / (delta + eps)
    int_M_conv = torch.where(same, int_M_eq, int_M_gen)
    p_curr = torch.clamp_min(p0 * eta + VTL_eff * int_M_conv, 0.0)

    int_M_total = M_inf * dt + (M_prev - M_inf) / (kdm + eps) * (1.0 - exp_M)
    pm_curr = torch.clamp_min(
        pm0 + (VTL_eff * int_M_total - (p_curr - p0)),  # no O_curr factor — dpm = kmt*p
        0.0,
    )

    return torch.stack([R_curr, O_curr, m_curr, mm_curr, p_curr, pm_curr, D0], dim=-1)


class TXTLMaturationApproxOdeRNN(OdeRNN):
    """
    OdeRNN using Bob's approximate step adapted for TXTLMaturationOnly7Scaffold.

    Same approximations as TXTLApproxOdeRNN but O is a frozen dummy (no lam_O
    in theta) and pm does not depend on O.

    Use with scaffold: txtl_maturation_only_dna
    Config key:        ode_rnn_txtl_mat_approx
    """

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return ivtt_maturation_step_approx(y, theta, dt)


class TXTLApproxOdeRNN(OdeRNN):
    """
    OdeRNN using Bob's approximate TXTL integrator instead of the exact analytical step.

    Use with scaffold: TXTLResourceandMaturationDNAScaffold
    Config key:        ode_rnn_txtl_approx
    """

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return ivtt_step_approx(y, theta, dt)


class TXTLAnalyticalBleachOdeRNN(OdeRNN):
    """
    OdeRNN with TXTL+bleaching cascade solved exactly.

    Use with scaffold: TXTLResourceandMaturationDNABleachScaffold
    Config key:        ode_rnn_txtl_bleach
    """

    def _rk4_substeps(
        self,
        y: torch.Tensor,
        dt: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        return txtl_step_bleach(y, theta, dt)
