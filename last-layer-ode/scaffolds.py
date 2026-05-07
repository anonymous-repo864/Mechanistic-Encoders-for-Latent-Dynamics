import torch
import torch.nn as nn
import math
from typing import Dict, List

from sim.glycolysis import (
    GlycolysisOracle22Scaffold,
    GlycolysisReduced12Scaffold,
    GlycolysisReduced8Scaffold,
    GlycolysisReduced4Scaffold,
)

class MechanisticScaffold(nn.Module):
    """Base class for mechanistic scaffolds.

    Carries optional hooks (precompute_batch / initial_state / analytic_step /
    emit_theta) used by models that support analytic-step integration. The base
    implementations are scriptable no-ops so non-analytic scaffolds compile fine
    even when the model code references the hooks behind a constant gate.
    """

    # `__constants__` lets TorchScript treat these as compile-time so models
    # that gate `if scaffold.has_analytic_step:` can DCE the dead branch.
    __constants__ = ["has_analytic_step", "tf_at_k_zero", "theta_dim_emit"]

    def __init__(self, P: int, theta_dim: int):
        super().__init__()
        self.P = int(P)
        self.theta_dim = int(theta_dim)
        # Encoder emits theta_dim params; some scaffolds repack to a different
        # width via emit_theta() before the loss sees it (e.g. Bob's IVTT loss).
        self.theta_dim_emit = int(theta_dim)
        # Set as instance attrs (so TorchScript tracks them); subclasses overwrite.
        self.has_analytic_step: bool = False
        self.tf_at_k_zero: bool = False
        self.state_names: List[str] = []
        # Per-parameter bounds — set by subclasses. None means use scalar fallback.
        self.theta_lo_vec: "list[float] | None" = None
        self.theta_hi_vec: "list[float] | None" = None

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ---------- analytic-scaffold hooks (used iff has_analytic_step) ----------
    # All four are scriptable no-ops on the base class so TorchScript can compile
    # any model that calls `self.rhs.<hook>(...)` regardless of which concrete
    # scaffold is used. Subclasses override with real bodies.

    def precompute_batch(
        self, y0: torch.Tensor, u_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        return out

    def initial_state(self, y0: torch.Tensor) -> torch.Tensor:
        return y0

    def analytic_step(
        self,
        y_prev: torch.Tensor,
        dt_k: torch.Tensor,
        theta_k: torch.Tensor,
        ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # No-op default — subclasses with has_analytic_step=True must override.
        return y_prev

    def emit_theta(
        self, theta_enc: torch.Tensor, y_state: torch.Tensor,
    ) -> torch.Tensor:
        return theta_enc

class MOFSynthesis12Scaffold(MechanisticScaffold):
    """
    Full 12-state MOF synthesis scaffold. Preserves all mechanistic structure
    from MOF_model.py; all 16 kinetic constants are learned as θ(t).

    States (12): Met, LigH, Lig_minus, H_plus, Base, Mod,
                 SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 4), Mod (idx 5)

    Parameters θ (16):
      0  k_deprot  : LigH + Base -> Lig_minus deprotonation rate
      1  k_prot    : Lig_minus + H+ -> LigH reprotonation rate
      2  k_oli     : Met^a * Lig_minus^b -> SBU oligomerization rate
      3  k_cap     : SBU + Mod -> SBU_capped capping rate
      4  k_uncap   : SBU_capped -> SBU + Mod uncapping rate
      5  K_I       : modulator inhibition constant for crystalline growth
      6  knuc_A    : amorphous nucleation prefactor
      7  kgro_A    : amorphous growth rate
      8  kagg_A    : amorphous aggregation rate
      9  n_A       : SBU exponent for amorphous nucleation
      10 knuc_C    : crystalline nucleation prefactor
      11 kgro_C    : crystalline growth rate
      12 kagg_C    : crystalline aggregation rate
      13 n_C       : SBU exponent for crystalline nucleation
      14 a         : Met exponent in oligomerization
      15 b         : Lig_minus exponent in oligomerization
    """
    def __init__(self):
        super().__init__(P=12, theta_dim=16)
        self.state_names = [
            "Met", "LigH", "Lig_minus", "H_plus",
            "Base", "Mod", "SBU", "SBU_capped",
            "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        # Per-parameter bounds (true values: k_deprot=5, k_prot=1, k_oli=3, k_cap=2,
        # k_uncap=0.5, K_I=0.1, knuc_A=10, kgro_A=1, kagg_A=1, n_A=3,
        # knuc_C=0.5, kgro_C=4, kagg_C=1, n_C=1.5, a=1, b=1)
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.01, 0.001, 0.001,
                             0.1,  0.01, 0.01, 0.5,
                             0.001, 0.01, 0.01, 0.5,
                             0.1, 0.1]
        self.theta_hi_vec = [50.0, 20.0, 30.0, 20.0, 10.0, 2.0,
                             100.0, 20.0, 20.0, 10.0,
                             20.0, 50.0, 20.0, 8.0,
                             5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        (
            Met, LigH, Lig_minus, H_plus,
            Base, Mod, SBU, SBU_capped,
            Nuc_A, Am, Nuc_C, MOF_C,
        ) = y.unbind(dim=-1)
        (
            k_deprot, k_prot, k_oli, k_cap, k_uncap, K_I,
            knuc_A, kgro_A, kagg_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
            a, b,
        ) = theta.unbind(dim=-1)

        Met_p         = torch.clamp_min(Met, 0.0)
        LigH_p        = torch.clamp_min(LigH, 0.0)
        Lig_minus_p   = torch.clamp_min(Lig_minus, 0.0)
        H_plus_p      = torch.clamp_min(H_plus, 0.0)
        Base_p        = torch.clamp_min(Base, 0.0)
        Mod_p         = torch.clamp_min(Mod, 0.0)
        SBU_p         = torch.clamp_min(SBU, 0.0)
        SBU_capped_p  = torch.clamp_min(SBU_capped, 0.0)
        Nuc_A_p       = torch.clamp_min(Nuc_A, 0.0)
        Am_p          = torch.clamp_min(Am, 0.0)
        Nuc_C_p       = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p       = torch.clamp_min(MOF_C, 0.0)

        k_deprot = torch.clamp_min(k_deprot, 0.0)
        k_prot   = torch.clamp_min(k_prot,   0.0)
        k_oli    = torch.clamp_min(k_oli,    0.0)
        k_cap    = torch.clamp_min(k_cap,    0.0)
        k_uncap  = torch.clamp_min(k_uncap,  0.0)
        K_I      = torch.clamp_min(K_I,      1e-6)
        knuc_A   = torch.clamp_min(knuc_A,   0.0)
        kgro_A   = torch.clamp_min(kgro_A,   0.0)
        kagg_A   = torch.clamp_min(kagg_A,   0.0)
        n_A      = torch.clamp_min(n_A,      1e-6)
        knuc_C   = torch.clamp_min(knuc_C,   0.0)
        kgro_C   = torch.clamp_min(kgro_C,   0.0)
        kagg_C   = torch.clamp_min(kagg_C,   0.0)
        n_C      = torch.clamp_min(n_C,      1e-6)
        a        = torch.clamp_min(a,        1e-6)
        b        = torch.clamp_min(b,        1e-6)

        r_deprot = k_deprot * LigH_p * Base_p
        r_prot   = k_prot * Lig_minus_p * H_plus_p
        r_oli    = k_oli * (Met_p + 1e-8).pow(a) * (Lig_minus_p + 1e-8).pow(b)
        r_cap    = k_cap * SBU_p * Mod_p
        r_uncap  = k_uncap * SBU_capped_p
        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        r_agg_A  = kagg_A * Nuc_A_p.pow(2.0)
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dMet        = -r_oli
        dLigH       = -r_deprot + r_prot
        dLig_minus  =  r_deprot - r_prot - r_oli
        dH_plus     =  r_deprot - r_prot + r_oli
        dBase       = -r_deprot
        dMod        = -r_cap + r_uncap
        dSBU        =  r_oli - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dSBU_capped =  r_cap - r_uncap
        dNuc_A      =  r_nuc_A - r_agg_A
        dAm         =  r_agg_A + r_gro_A
        dNuc_C      =  r_nuc_C - r_agg_C
        dMOF_C      =  r_agg_C + r_gro_C

        return torch.stack((
            dMet, dLigH, dLig_minus, dH_plus,
            dBase, dMod, dSBU, dSBU_capped,
            dNuc_A, dAm, dNuc_C, dMOF_C,
        ), dim=-1)


class MOFSynthesis8Scaffold(MechanisticScaffold):
    """
    8-state MOF synthesis scaffold. Collapses the four deprotonation species
    (Met, LigH, Lig_minus, H_plus) into an effective SBU production term driven
    by Base. Retains SBU_capped explicitly so Mod dynamics are exact. Includes
    cooperative nucleation exponents n_A, n_C as learned θ parameters.

    States (8): Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (13):
      0  k_base_decay : effective Base consumption rate
      1  k_oli_eff    : effective SBU production rate from Base
      2  k_cap        : SBU + Mod -> SBU_capped capping rate
      3  k_uncap      : SBU_capped -> SBU + Mod uncapping rate
      4  K_I          : modulator inhibition constant
      5  knuc_A       : amorphous nucleation prefactor
      6  kgro_A       : amorphous growth rate
      7  kagg_A       : amorphous aggregation rate
      8  n_A          : SBU exponent for amorphous nucleation
      9  knuc_C       : crystalline nucleation prefactor
      10 kgro_C       : crystalline growth rate
      11 kagg_C       : crystalline aggregation rate
      12 n_C          : SBU exponent for crystalline nucleation
    """
    def __init__(self):
        super().__init__(P=8, theta_dim=13)
        self.state_names = [
            "Base", "Mod", "SBU", "SBU_capped",
            "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        # Per-parameter bounds (k_base_decay, k_oli_eff, k_cap, k_uncap, K_I,
        # knuc_A, kgro_A, kagg_A, n_A, knuc_C, kgro_C, kagg_C, n_C)
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.001, 0.001,
                             0.1,  0.01, 0.01, 0.5,
                             0.001, 0.01, 0.01, 0.5]
        self.theta_hi_vec = [50.0, 30.0, 20.0, 10.0,  2.0,
                             100.0, 20.0, 20.0, 10.0,
                             20.0,  50.0, 20.0, 8.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C = y.unbind(dim=-1)
        (
            k_base_decay, k_oli_eff, k_cap, k_uncap, K_I,
            knuc_A, kgro_A, kagg_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
        ) = theta.unbind(dim=-1)

        Base_p        = torch.clamp_min(Base, 0.0)
        Mod_p         = torch.clamp_min(Mod, 0.0)
        SBU_p         = torch.clamp_min(SBU, 0.0)
        SBU_capped_p  = torch.clamp_min(SBU_capped, 0.0)
        Nuc_A_p       = torch.clamp_min(Nuc_A, 0.0)
        Am_p          = torch.clamp_min(Am, 0.0)
        Nuc_C_p       = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p       = torch.clamp_min(MOF_C, 0.0)

        k_base_decay = torch.clamp_min(k_base_decay, 0.0)
        k_oli_eff    = torch.clamp_min(k_oli_eff,    0.0)
        k_cap        = torch.clamp_min(k_cap,        0.0)
        k_uncap      = torch.clamp_min(k_uncap,      0.0)
        K_I          = torch.clamp_min(K_I,          1e-6)
        knuc_A       = torch.clamp_min(knuc_A,       0.0)
        kgro_A       = torch.clamp_min(kgro_A,       0.0)
        kagg_A       = torch.clamp_min(kagg_A,       0.0)
        n_A          = torch.clamp_min(n_A,          1e-6)
        knuc_C       = torch.clamp_min(knuc_C,       0.0)
        kgro_C       = torch.clamp_min(kgro_C,       0.0)
        kagg_C       = torch.clamp_min(kagg_C,       0.0)
        n_C          = torch.clamp_min(n_C,          1e-6)

        r_cap    = k_cap * SBU_p * Mod_p
        r_uncap  = k_uncap * SBU_capped_p
        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        r_agg_A  = kagg_A * Nuc_A_p.pow(2.0)
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dBase       = -k_base_decay * Base_p
        dMod        = -r_cap + r_uncap
        dSBU        =  k_oli_eff * Base_p - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dSBU_capped =  r_cap - r_uncap
        dNuc_A      =  r_nuc_A - r_agg_A
        dAm         =  r_agg_A + r_gro_A
        dNuc_C      =  r_nuc_C - r_agg_C
        dMOF_C      =  r_agg_C + r_gro_C

        return torch.stack((
            dBase, dMod, dSBU, dSBU_capped,
            dNuc_A, dAm, dNuc_C, dMOF_C,
        ), dim=-1)


class MOFSynthesis6Scaffold(MechanisticScaffold):
    """
    6-state MOF synthesis scaffold. Applies two further reductions on top of
    MOFSynthesis8Scaffold: (1) quasi-steady-state on SBU_capped so dMod = 0
    between boluses (net capping flux is zero); (2) fast-nucleation collapse of
    Nuc_A directly into Am. Retains cooperative nucleation exponents n_A, n_C
    as learned θ parameters (advisor recommendation: option b).

    States (6): Base, Mod, SBU, Am, Nuc_C, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (10):
      0  k_base_decay : effective Base consumption rate
      1  k_oli_eff    : effective SBU production rate from Base
      2  knuc_A       : amorphous nucleation prefactor (feeds Am directly)
      3  kgro_A       : amorphous growth rate
      4  n_A          : SBU exponent for amorphous nucleation
      5  knuc_C       : crystalline nucleation prefactor
      6  kgro_C       : crystalline growth rate
      7  kagg_C       : crystalline aggregation rate
      8  n_C          : SBU exponent for crystalline nucleation
      9  K_I          : modulator inhibition constant
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=10)
        self.state_names = ["Base", "Mod", "SBU", "Am", "Nuc_C", "MOF_C"]
        # Per-parameter bounds (k_base_decay, k_oli_eff, knuc_A, kgro_A, n_A,
        # knuc_C, kgro_C, kagg_C, n_C, K_I)
        self.theta_lo_vec = [0.1,  0.01, 0.1,  0.01, 0.5,  0.001, 0.01, 0.01, 0.5,  0.001]
        self.theta_hi_vec = [50.0, 30.0, 100.0, 20.0, 10.0, 20.0, 50.0, 20.0, 8.0,  2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, SBU, Am, Nuc_C, MOF_C = y.unbind(dim=-1)
        (
            k_base_decay, k_oli_eff,
            knuc_A, kgro_A, n_A,
            knuc_C, kgro_C, kagg_C, n_C,
            K_I,
        ) = theta.unbind(dim=-1)

        Base_p  = torch.clamp_min(Base,  0.0)
        Mod_p   = torch.clamp_min(Mod,   0.0)
        SBU_p   = torch.clamp_min(SBU,   0.0)
        Am_p    = torch.clamp_min(Am,    0.0)
        Nuc_C_p = torch.clamp_min(Nuc_C, 0.0)
        MOF_C_p = torch.clamp_min(MOF_C, 0.0)

        k_base_decay = torch.clamp_min(k_base_decay, 0.0)
        k_oli_eff    = torch.clamp_min(k_oli_eff,    0.0)
        knuc_A       = torch.clamp_min(knuc_A,       0.0)
        kgro_A       = torch.clamp_min(kgro_A,       0.0)
        n_A          = torch.clamp_min(n_A,          1e-6)
        knuc_C       = torch.clamp_min(knuc_C,       0.0)
        kgro_C       = torch.clamp_min(kgro_C,       0.0)
        kagg_C       = torch.clamp_min(kagg_C,       0.0)
        n_C          = torch.clamp_min(n_C,          1e-6)
        K_I          = torch.clamp_min(K_I,          1e-6)

        r_nuc_A  = knuc_A * (SBU_p + 1e-8).pow(n_A)
        r_nuc_C  = knuc_C * (SBU_p + 1e-8).pow(n_C)
        r_gro_A  = kgro_A * SBU_p * Am_p
        inhib    = K_I / (K_I + Mod_p + 1e-6)
        r_gro_C  = kgro_C * SBU_p * MOF_C_p * inhib
        r_agg_C  = kagg_C * Nuc_C_p.pow(2.0)

        dBase  = -k_base_decay * Base_p
        dMod   = torch.zeros_like(Base)   # QSS: r_cap == r_uncap between boluses
        dSBU   =  k_oli_eff * Base_p - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
        dAm    =  r_nuc_A + r_gro_A       # Nuc_A fast: collapses directly into Am
        dNuc_C =  r_nuc_C - r_agg_C
        dMOF_C =  r_agg_C + r_gro_C

        return torch.stack((dBase, dMod, dSBU, dAm, dNuc_C, dMOF_C), dim=-1)


class MOFSynthesis4Scaffold(MechanisticScaffold):
    """
    4-state MOF synthesis scaffold. Most aggressively reduced: no SBU tracked.
    Base acts as proxy for SBU availability; nucleation is linear (no cooperative
    exponent) since SBU is not an explicit state. Mod decays via a slow first-order
    approximation (GRU compensates for the full capping dynamics).

    States (4): Base, Mod, Am, MOF_C
    Control inputs (bolused): Base (idx 0), Mod (idx 1)

    Parameters θ (7):
      0  k_base   : effective Base decay rate
      1  k_mod    : effective Mod decay rate (first-order approximation)
      2  k_nuc_A  : amorphous nucleation rate (linear in Base)
      3  k_gro_A  : amorphous growth rate (Base * Am)
      4  k_nuc_C  : crystalline nucleation rate (linear in Base)
      5  k_gro_C  : crystalline growth rate (Base * MOF_C * inhibition)
      6  K_I      : modulator inhibition constant
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=7)
        self.state_names = ["Base", "Mod", "Am", "MOF_C"]
        # Per-parameter bounds (k_base, k_mod, k_nuc_A, k_gro_A, k_nuc_C, k_gro_C, K_I)
        self.theta_lo_vec = [0.1,  0.001, 0.1,  0.01, 0.001, 0.01, 0.001]
        self.theta_hi_vec = [50.0, 10.0, 100.0, 20.0, 20.0,  50.0, 2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Base, Mod, Am, MOF_C = y.unbind(dim=-1)
        k_base, k_mod, k_nuc_A, k_gro_A, k_nuc_C, k_gro_C, K_I = theta.unbind(dim=-1)

        Base_p  = torch.clamp_min(Base,  0.0)
        Mod_p   = torch.clamp_min(Mod,   0.0)
        Am_p    = torch.clamp_min(Am,    0.0)
        MOF_C_p = torch.clamp_min(MOF_C, 0.0)

        k_base  = torch.clamp_min(k_base,  0.0)
        k_mod   = torch.clamp_min(k_mod,   0.0)
        k_nuc_A = torch.clamp_min(k_nuc_A, 0.0)
        k_gro_A = torch.clamp_min(k_gro_A, 0.0)
        k_nuc_C = torch.clamp_min(k_nuc_C, 0.0)
        k_gro_C = torch.clamp_min(k_gro_C, 0.0)
        K_I     = torch.clamp_min(K_I,     1e-6)

        inhib  = K_I / (K_I + Mod_p + 1e-6)

        dBase  = -k_base * Base_p
        dMod   = -k_mod * Mod_p
        dAm    =  k_nuc_A * Base_p + k_gro_A * Base_p * Am_p
        dMOF_C =  k_nuc_C * Base_p + k_gro_C * Base_p * MOF_C_p * inhib

        return torch.stack((dBase, dMod, dAm, dMOF_C), dim=-1)


class SingleEnzymeLumpedScaffold(MechanisticScaffold):
    """
    2-state reduced scaffold for the Single Enzyme scenario.

    The full 6-state system is simulated but only A (substrate, idx 0) and
    C (product, idx 2) are observed. The scaffold approximates the dynamics
    with a simple first-order reversible reaction:

        dA_approx = -kf * A + kr * C
        dC_approx =  kf * A - kr * C

    This is structurally wrong in two ways:
      1. The true reaction is bimolecular (rate ∝ A·B); B is hidden
      2. There is no saturation / denominator term

    The neural network must learn time-varying kf(t) and kr(t) to compensate
    for the missing B dependence and the wrong kinetics.

    States (2): S ↔ A (observed substrate), P ↔ C (observed product)
    Control: A-bolus maps to S; B-bolus is a hidden input (seen by the GRU
             via u_seq but not directly reflected in the observed state)
    Parameters θ (2): kf (effective forward rate), kr (effective reverse rate)

    Use with: datasets/single_enzyme_lumped.npz  (--obs-indices 0,2)
    """
    def __init__(self):
        super().__init__(P=2, theta_dim=2)
        self.state_names = ["S", "P"]
        # True rates are kcat_f·E ≈ 10 and kcat_r·E ≈ 2, but with the denominator
        # the effective observed rate is much lower; use wide bounds.
        self.theta_lo_vec = [0.001, 0.001]
        self.theta_hi_vec = [100.0,  50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        S, P = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)

        S_p = torch.clamp_min(S, 0.0)
        P_p = torch.clamp_min(P, 0.0)
        kf  = torch.clamp_min(kf, 0.0)
        kr  = torch.clamp_min(kr, 0.0)

        v = kf * S_p - kr * P_p

        dS = -v
        dP =  v

        return torch.stack((dS, dP), dim=-1)


class SingleEnzymeReduced4Scaffold(MechanisticScaffold):
    """
    Reduced 4-state mass-action scaffold for the Single Enzyme scenario.

    The true system uses Reversible Bi-Bi (Michaelis-Menten) kinetics with a
    nonlinear denominator. This scaffold intentionally simplifies to plain
    mass-action, dropping the inert states E and I (which are constant in the
    data: E=1, I=0) and removing the denominator entirely:

        v  = kf * A * B  −  kr * C * D

    The scaffold structure (A+B → C+D reversibly) is topologically correct,
    but the kinetics are wrong. The neural network must learn time-varying
    kf(t) and kr(t) to compensate for the missing saturation terms.

    States (4): A, B, C, D
    Control inputs (bolused): A (idx 0), B (idx 1)
    Parameters θ (2): kf (effective forward rate), kr (effective reverse rate)

    Use with: datasets/single_enzyme_4.npz  (--obs-indices 0,1,2,3)
    Ground-truth Bi-Bi values for reference: kcat_f·E=10, kcat_r·E=2
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=2)
        self.state_names = ["A", "B", "C", "D"]
        # Bounds: true effective forward rate ≈ 10, reverse ≈ 2
        self.theta_lo_vec = [0.01, 0.001]
        self.theta_hi_vec = [200.0, 100.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D = y.unbind(dim=-1)
        kf, kr = theta.unbind(dim=-1)

        A_p = torch.clamp_min(A, 0.0)
        B_p = torch.clamp_min(B, 0.0)
        C_p = torch.clamp_min(C, 0.0)
        D_p = torch.clamp_min(D, 0.0)

        kf = torch.clamp_min(kf, 0.0)
        kr = torch.clamp_min(kr, 0.0)

        v = kf * A_p * B_p - kr * C_p * D_p

        dA = -v
        dB = -v
        dC =  v
        dD =  v

        return torch.stack((dA, dB, dC, dD), dim=-1)


class SingleEnzymeScaffold(MechanisticScaffold):
    """
    6-state Reversible Bi-Bi enzyme kinetics scaffold.

    Reaction: A + B <-> C + D  (catalysed by enzyme E, inhibitor I inert)

    States (6): A, B, C, D, E, I
    Control inputs (bolused): A (idx 0), B (idx 1)

    Parameters θ (6):
      0  kcat_f : forward catalytic rate constant
      1  kcat_r : reverse catalytic rate constant
      2  Ka     : Michaelis constant for substrate A
      3  Kb     : Michaelis constant for substrate B
      4  Kc     : Michaelis constant for product C
      5  Kd     : Michaelis constant for product D

    Ground-truth values: kcat_f=10.0, kcat_r=2.0, Ka=2.0, Kb=2.0, Kc=5.0, Kd=5.0
    Dataset: datasets/single_enzyme_6.npz  (--t-span 10 --n-steps 200)
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=6)
        self.state_names = ["A", "B", "C", "D", "E", "I"]
        # Per-parameter bounds: wide enough to contain the true values with room to search
        self.theta_lo_vec = [0.1,  0.01, 0.01, 0.01, 0.01, 0.01]
        self.theta_hi_vec = [100.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        A, B, C, D, E, I = y.unbind(dim=-1)
        kcat_f, kcat_r, Ka, Kb, Kc, Kd = theta.unbind(dim=-1)

        eps: float = 1e-12

        A_p = torch.clamp_min(A, 0.0)
        B_p = torch.clamp_min(B, 0.0)
        C_p = torch.clamp_min(C, 0.0)
        D_p = torch.clamp_min(D, 0.0)
        E_p = torch.clamp_min(E, 0.0)

        Ka = torch.clamp_min(Ka, eps)
        Kb = torch.clamp_min(Kb, eps)
        Kc = torch.clamp_min(Kc, eps)
        Kd = torch.clamp_min(Kd, eps)

        Vf = kcat_f * E_p
        Vr = kcat_r * E_p

        D0 = Ka * Kb
        denom = (
            D0 * (1.0 + C_p / Kc + D_p / Kd + (C_p * D_p) / (Kc * Kd))
            + (Kb * A_p) * (1.0 + D_p / Kd)
            + (Ka * B_p) * (1.0 + C_p / Kc)
            + (A_p * B_p)
            + eps
        )

        v = (Vf * A_p * B_p - Vr * C_p * D_p) / denom

        dA = -v
        dB = -v
        dC =  v
        dD =  v
        dE = E * 0.0   # conserved: always zero
        dI = I * 0.0   # inert: always zero

        return torch.stack((dA, dB, dC, dD, dE, dI), dim=-1)

# -----------------------------------------------------------------------------
# 3) JIT‐scripted analytic ODE: This is the simplest model This has to be integrated into same format as the rest of the scaffolds here. 
# -----------------------------------------------------------------------------
# @torch.jit.script
# def _step_integration(
#     m0: torch.Tensor, p0: torch.Tensor,
#     dt: torch.Tensor,
#     VTX: torch.Tensor, KTX: torch.Tensor,
#     dna: torch.Tensor, kdm: torch.Tensor,
#     VTL: torch.Tensor,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     eps = 1e-8
#     A     = VTX * dna / (KTX + dna + eps)
#     m_inf = A / (kdm + eps)
#     expT  = torch.exp(-kdm * dt)
#     m1    = m_inf + (m0 - m_inf) * expT
#     int_m = m_inf * dt + (m0 - m_inf) * (1.0 - expT) / (kdm + eps)
#     p1    = p0 + VTL * int_m
#     return m1.clamp(min=0.0), p1.clamp(min=0.0), int_m


# class TXTL_mRNAMaturation(MechanisticScaffold):
#     # states:    [R, m, mm, p, pm]  (optionally +DNA as 6th state)
#     # theta:     [lam, VTXmax, kdm, VTLmax, kmt, kmatm]
#     def __init__(self):
#         super().__init__(P=5, theta_dim=6)
#         self.state_names = ["R", "m", "mm", "p", "pm"]
#         self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
#         self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

#     def forward(self, y, theta, dna):  # or embed DNA as y[:,5]
#         R, m, mm, p, pm = y.unbind(-1)
#         lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(-1)
#         dR  = -lam * R
#         dm  = R * VTXmax * dna - (kdm + kmatm) * m
#         dmm = kmatm * m - kdm * mm
#         dp  = R * VTLmax * (m + mm) - kmt * p
#         dpm = kmt * p
#         return torch.stack([dR, dm, dmm, dp, dpm], dim=-1)

class TXTLMaturationDNAScaffold(MechanisticScaffold):
    """
    6-state TXTL scaffold with DNA as an explicit, bolus-driven state.

    The mechanism is the supervisor's `TXTL_mRNAMaturation`, with DNA promoted
    from an exogenous scalar to a latent state so no scaffold-API change is
    needed: the dataset's `u_to_y_jump` routes the "DNA c" (dilution-corrected
    concentration delta) column of u_seq onto state idx 5, and dDNA/dt = 0
    between jumps — so y[..., 5] at step k is exactly cumsum("DNA c") up to k.

    States (6): R (resource pool), m (immature mRNA), mm (mature mRNA,
                observed as Broccoli), p (immature protein),
                pm (mature protein, observed as mCherry / 2), DNA

    Parameters θ (6):
      0  lam    : resource decay rate
      1  VTXmax : transcription rate (per DNA per R)
      2  kdm    : mRNA degradation rate (applies to both m and mm)
      3  VTLmax : translation rate (per total mRNA per R)
      4  kmt    : protein maturation rate (p → pm)
      5  kmatm  : mRNA maturation rate (m → mm)

    Observed indices within P: [2, 4]  (mm=Broccoli, pm=mCherry/2)
    Use with: datasets/real_ivtt_full.npz (layout='full')
    """
    def __init__(self):
        super().__init__(P=6, theta_dim=6)
        self.state_names = ["R", "m", "mm", "p", "pm", "DNA"]
        # Supervisor's log-uniform bounds for TXTL_mRNAMaturation
        self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dm, dmm, dp, dpm, dDNA), dim=-1)

class TXTLResourceandMaturationDNAScaffold(MechanisticScaffold):
    """
    6-state TXTL scaffold with DNA as an explicit, bolus-driven state.

    The mechanism is the supervisor's `TXTL_mRNAMaturation`, with DNA promoted
    from an exogenous scalar to a latent state so no scaffold-API change is
    needed: the dataset's `u_to_y_jump` routes the "DNA c" (dilution-corrected
    concentration delta) column of u_seq onto state idx 5, and dDNA/dt = 0
    between jumps — so y[..., 5] at step k is exactly cumsum("DNA c") up to k.

    States (6): R (resource pool), m (immature mRNA), mm (mature mRNA,
                observed as Broccoli), p (immature protein),
                pm (mature protein, observed as mCherry / 2), DNA

    Parameters θ (6):
      0  lam    : resource decay rate
      1  VTXmax : transcription rate (per DNA per R)
      2  kdm    : mRNA degradation rate (applies to both m and mm)
      3  VTLmax : translation rate (per total mRNA per R)
      4  kmt    : protein maturation rate (p → pm)
      5  kmatm  : mRNA maturation rate (m → mm)

    Observed indices within P: [2, 4]  (mm=Broccoli, pm=mCherry/2)
    Use with: datasets/real_ivtt_full.npz (layout='full')
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=7)
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "DNA"]
        # Supervisor's log-uniform bounds for TXTL_mRNAMaturation
        self.theta_lo_vec = [1e-6, 1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]
    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam,lam_O, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        O_p   = torch.clamp_min(O,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dO   = -lam_O * O_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = O_p * kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dO, dm, dmm, dp, dpm, dDNA), dim=-1)


class TXTLMaturationOnly7Scaffold(MechanisticScaffold):
    """
    7-state TXTL scaffold (same layout as TXTLResourceandMaturationDNAScaffold)
    but with O decoupled from dpm — i.e., pure mRNA-maturation kinetics only.

    This is the ablation counterpart of txtl_resource_and_maturation_dna: same
    obs_idx=[3,5], same dataset layout, same theta bounds except lam_O is
    dropped (theta_dim=6 vs 7).  The only mechanistic difference is:
        dpm = kmt * p      (resource scaffold has dpm = O * kmt * p)
    O is retained as a dummy state (dO = 0) so the dataset's 7-column y_seq
    can be used directly without any dataset changes.

    States (7): R, O (dummy), m, mm, p, pm, DNA
    Parameters θ (6): lam, VTXmax, kdm, VTLmax, kmt, kmatm
    Observed indices: [3, 5]  (mm, pm) — identical to resource variant
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=6)
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "DNA"]
        self.theta_lo_vec = [1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, VTXmax, kdm, VTLmax, kmt, kmatm = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dO   = torch.zeros_like(O)
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = kmt * p_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dO, dm, dmm, dp, dpm, dDNA), dim=-1)


class TXTLResourceandMaturationDNABleachScaffold(MechanisticScaffold):
    """
    Extension of TXTLResourceandMaturationDNAScaffold with a pm bleaching term:
        dpm/dt = O * kmt * p - kbleach * pm

    Motivation: real IVTT pm trajectories on failure runs *decrease* over time
    (mCherry photobleaching during long readouts). The base scaffold's dpm is
    non-negative, so it cannot represent this — best it can do is plateau pm.
    Adding kbleach as an 8th learned parameter lets the encoder pull pm down on
    samples where the truth declines, without affecting samples where it rises.

    States (7): R, O, m, mm, p, pm, DNA  (same as base)

    Parameters θ (8):
      0  lam     : resource decay rate
      1  lam_O   : oxygen decay rate
      2  VTXmax  : transcription rate
      3  kdm     : mRNA degradation rate
      4  VTLmax  : translation rate
      5  kmt     : protein maturation rate (p → pm)
      6  kmatm   : mRNA maturation rate (m → mm)
      7  kbleach : pm decay rate (photobleaching / measurement decay)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "DNA"]
        self.theta_lo_vec = [1e-6, 1e-6, 3e-5, 1e-5, 3e-5, 1e-5, 5e-5, 1e-7]
        self.theta_hi_vec = [5e-4, 5e-4, 1.2e-1, 1e-2, 8e-2, 3.5e-4, 3.5e-3, 1e-4]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, O, m, mm, p, pm, DNA = y.unbind(dim=-1)
        lam, lam_O, VTXmax, kdm, VTLmax, kmt, kmatm, kbleach = theta.unbind(dim=-1)

        R_p   = torch.clamp_min(R,   0.0)
        O_p   = torch.clamp_min(O,   0.0)
        m_p   = torch.clamp_min(m,   0.0)
        mm_p  = torch.clamp_min(mm,  0.0)
        p_p   = torch.clamp_min(p,   0.0)
        pm_p  = torch.clamp_min(pm,  0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dR   = -lam * R_p
        dO   = -lam_O * O_p
        dm   = R_p * VTXmax * DNA_p - (kdm + kmatm) * m_p
        dmm  = kmatm * m_p - kdm * mm_p
        dp   = R_p * VTLmax * (m_p + mm_p) - kmt * p_p
        dpm  = O_p * kmt * p_p - kbleach * pm_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dR, dO, dm, dmm, dp, dpm, dDNA), dim=-1)


class TXTLSimpleDNAScaffold(MechanisticScaffold):
    """
    3-state minimal TXTL scaffold with DNA as an explicit, bolus-driven state.

    The simplest cascade DNA → mm → pm with first-order kinetics. No resource
    pool, no mRNA maturation, no protein maturation — the network must learn
    time-varying θ(t) to compensate for the missing structure.

    States (3): mm (Broccoli), pm (mCherry / 2), DNA

    Parameters θ (3):
      0  k_tx : transcription rate (DNA → mm)
      1  k_tl : translation rate (mm → pm)
      2  kdm  : mRNA degradation rate

    Observed indices within P: [0, 1]  (mm, pm)
    Use with: datasets/real_ivtt_simple.npz (layout='simple')
    """
    def __init__(self):
        super().__init__(P=3, theta_dim=3)
        self.state_names = ["mm", "pm", "DNA"]
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5]
        self.theta_hi_vec = [1e-1, 1e-1, 1e-2]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        mm, pm, DNA = y.unbind(dim=-1)
        k_tx, k_tl, kdm = theta.unbind(dim=-1)

        mm_p  = torch.clamp_min(mm,  0.0)
        DNA_p = torch.clamp_min(DNA, 0.0)

        dmm  = k_tx * DNA_p - kdm * mm_p
        dpm  = k_tl * mm_p
        dDNA = torch.zeros_like(DNA)

        return torch.stack((dmm, dpm, dDNA), dim=-1)

class MethaneGlobal4Step_NO_Scaffold(MechanisticScaffold):
    """
    A physically grounded 4-step macroscopic scaffold for Methane oxidation.
    Instead of a 49-parameter black box, we only learn 4 kinetic parameters 
    representing the main branches of combustion.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (4):
      0: k_methane_ox : CH4 -> CO + H2O (Partial oxidation)
      1: k_co_ox      : CO -> CO2       (CO burnout)
      2: k_oh_prod    : O2 + H2O -> OH  (Radical pool generation)
      3: k_thermal_no : O2 -> NO        (Thermal NO formation proxy)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=4) # Dropped from 49 to 4!
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounding the 4 reaction rates
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [10.0, 10.0, 10.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        k_methane, k_co, k_oh, k_no = theta.unbind(dim=-1)

        # Clamp to prevent negative concentrations causing runaway physics
        CH4_p = torch.clamp_min(CH4, 0.0)
        O2_p  = torch.clamp_min(O2,  0.0)
        CO_p  = torch.clamp_min(CO,  0.0)
        H2O_p = torch.clamp_min(H2O, 0.0)

        # Ensure rates are positive
        k_methane = torch.clamp_min(k_methane, 0.0)
        k_co      = torch.clamp_min(k_co, 0.0)
        k_oh      = torch.clamp_min(k_oh, 0.0)
        k_no      = torch.clamp_min(k_no, 0.0)

        # Calculate fluxes for the 4 macroscopic steps (using simple mass action / linear rates)
        # 1. CH4 + 1.5 O2 -> CO + 2 H2O
        r1 = k_methane * CH4_p * O2_p 
        
        # 2. CO + 0.5 O2 -> CO2
        r2 = k_co * CO_p * O2_p
        
        # 3. O2 -> 2 OH (Conceptual radical formation)
        r3 = k_oh * O2_p
        
        # 4. N2 + O2 -> 2 NO (N2 is assumed constant in air, so rate just depends on O2)
        r4 = k_no * O2_p

        # Apply stoichiometry to state derivatives
        dCH4 = -r1
        dO2  = -1.5 * r1 - 0.5 * r2 - r3 - r4
        dCO  =  r1 - r2
        dCO2 =  r2
        dH2O =  2.0 * r1
        dOH  =  2.0 * r3
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)

class MethaneGlobal4Step_CH2O_Scaffold(MechanisticScaffold):
    """
    A physically grounded 4-step macroscopic scaffold for the Smooke methane model.
    Routes carbon explicitly through the CH2O intermediate.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, CH2O
    Parameters (4):
      0: k_methane : CH4 + O2 -> CH2O + H2O      (Methane to Formaldehyde)
      1: k_ch2o    : CH2O + 0.5 O2 -> CO + H2O   (Formaldehyde to CO)
      2: k_co      : CO + 0.5 O2 -> CO2          (CO burnout)
      3: k_oh      : O2 -> 2 OH                  (Radical pool proxy)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=4)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "CH2O"]
        
        self.theta_lo_vec = [1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [10.0, 10.0, 10.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, CH2O = y.unbind(dim=-1)
        k_methane, k_ch2o, k_co, k_oh = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations
        CH4_p  = torch.clamp_min(CH4, 0.0)
        O2_p   = torch.clamp_min(O2,  0.0)
        CO_p   = torch.clamp_min(CO,  0.0)
        CH2O_p = torch.clamp_min(CH2O, 0.0)

        # Clamp rates to prevent reverse physics
        k_methane = torch.clamp_min(k_methane, 0.0)
        k_ch2o    = torch.clamp_min(k_ch2o, 0.0)
        k_co      = torch.clamp_min(k_co, 0.0)
        k_oh      = torch.clamp_min(k_oh, 0.0)

        # 1. CH4 -> CH2O
        r1 = k_methane * CH4_p * O2_p 
        
        # 2. CH2O -> CO
        r2 = k_ch2o * CH2O_p * O2_p
        
        # 3. CO -> CO2
        r3 = k_co * CO_p * O2_p
        
        # 4. OH generation proxy
        r4 = k_oh * O2_p

        # Apply mass-balanced stoichiometry to the derivatives
        dCH4  = -r1
        dO2   = -r1 - 0.5 * r2 - 0.5 * r3 - r4
        dCO   =  r2 - r3
        dCO2  =  r3
        dH2O  =  r1 + r2
        dOH   =  2.0 * r4
        dCH2O =  r1 - r2

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dCH2O), dim=-1)


class MethaneDomainInformedCH2O_OHGate4Step_Scaffold(MechanisticScaffold):
        """
        Domain-informed 4-step CH2O scaffold with OH-gated CH2O oxidation.

        States (7): CH4, O2, CO, CO2, H2O, OH, CH2O
        Parameters (4):
            0: k_methane : CH4 + O2 -> CH2O + H2O
            1: k_ch2o    : CH2O + OH -> CO + H2O + H (OH-gated)
            2: k_co      : CO + 0.5 O2 -> CO2
            3: k_oh      : O2 -> 2 OH
        """
        def __init__(self):
                super().__init__(P=7, theta_dim=4)
                self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "CH2O"]

                self.theta_lo_vec = [1e-5, 1e-5, 1e-5, 1e-6]
                self.theta_hi_vec = [10.0, 10.0, 10.0, 1.0]

        def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
                CH4, O2, CO, CO2, H2O, OH, CH2O = y.unbind(dim=-1)
                k_methane, k_ch2o, k_co, k_oh = theta.unbind(dim=-1)

                CH4_p  = torch.clamp_min(CH4, 0.0)
                O2_p   = torch.clamp_min(O2,  0.0)
                CO_p   = torch.clamp_min(CO,  0.0)
                CH2O_p = torch.clamp_min(CH2O, 0.0)
                OH_p   = torch.clamp_min(OH,  0.0)

                k_methane = torch.clamp_min(k_methane, 0.0)
                k_ch2o    = torch.clamp_min(k_ch2o, 0.0)
                k_co      = torch.clamp_min(k_co, 0.0)
                k_oh      = torch.clamp_min(k_oh, 0.0)

                r1 = k_methane * CH4_p * O2_p
                r2 = k_ch2o * CH2O_p * (OH_p / (OH_p + 1e-4))
                r3 = k_co * CO_p * O2_p
                r4 = k_oh * O2_p

                dCH4  = -r1
                dO2   = -r1 - 0.5 * r3 - r4
                dCO   = r2 - r3
                dCO2  = r3
                dH2O  = r1 + r2
                dOH   = 2.0 * r4
                dCH2O = r1 - r2

                return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dCH2O), dim=-1)
    
class MethaneDomainInformedOHGate4Step_NO_Scaffold(MechanisticScaffold):
    """
    Domain-informed macroscopic scaffold for Methane oxidation.
    Incorporates reversibility, water-assisted CO oxidation, and fractional exponents.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (8):
      0: k_methane_ox : CH4 forward oxidation
      1: n_o2_methane : Fractional order of O2 in CH4 oxidation
      2: k_co_f       : CO -> CO2 forward rate
      3: k_co_r       : CO2 -> CO reverse rate (Equilibrium bottleneck)
      4: k_wgs        : Water-gas shift proxy (CO + H2O -> CO2)
      5: k_oh_prod    : Radical pool generation
      6: k_thermal_no : NO formation
      7: n_o2_no      : Fractional order of O2 in NO formation (Thermal NO is highly non-linear)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounds: rates can be wide, exponents bounded between ~0.1 and 2.0
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 0.1]
        self.theta_hi_vec = [10.0, 2.0, 10.0, 10.0, 10.0, 10.0, 1.0,  2.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        (k_methane, n_o2_methane, k_co_f, k_co_r, 
         k_wgs, k_oh, k_no, n_o2_no) = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations and NaN powers
        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        CO_p  = torch.clamp_min(CO,  eps)
        CO2_p = torch.clamp_min(CO2, eps)
        H2O_p = torch.clamp_min(H2O, eps)
        OH_p  = torch.clamp_min(OH,  0.0)

        # Clamp parameters to bounds
        k_methane    = torch.clamp_min(k_methane, 0.0)
        n_o2_methane = torch.clamp(n_o2_methane, min=0.1, max=2.0)
        k_co_f       = torch.clamp_min(k_co_f, 0.0)
        k_co_r       = torch.clamp_min(k_co_r, 0.0)
        k_wgs        = torch.clamp_min(k_wgs, 0.0)
        k_oh         = torch.clamp_min(k_oh, 0.0)
        k_no         = torch.clamp_min(k_no, 0.0)
        n_o2_no      = torch.clamp(n_o2_no, min=0.1, max=2.0)

        # 1. CH4 Oxidation (with learned fractional O2 dependence + OH gating)
        oh_gate = OH_p / (OH_p + 1e-3)
        r1 = k_methane * CH4_p * (O2_p ** n_o2_methane) * oh_gate
        
        # 2. Reversible CO Burnout + Water Gas Shift Proxy
        # r2_f: CO + 0.5 O2 -> CO2
        # r2_r: CO2 -> CO + 0.5 O2
        # r_wgs: CO + H2O -> CO2 + (hidden)
        r2_f  = k_co_f * CO_p * (O2_p ** 0.5)
        r2_r  = k_co_r * CO2_p
        r_wgs = k_wgs * CO_p * H2O_p
        
        # 3. OH Generation
        r3 = k_oh * O2_p
        
        # 4. Thermal NO formation (highly sensitive to O2)
        r4 = k_no * (O2_p ** n_o2_no)

        # Apply stoichiometry
        dCH4 = -r1
        dO2  = -1.5 * r1 - 0.5 * r2_f + 0.5 * r2_r - r3 - r4
        dCO  =  r1 - r2_f + r2_r - r_wgs
        dCO2 =  r2_f - r2_r + r_wgs
        dH2O =  2.0 * r1 - r_wgs
        dOH  =  2.0 * r3
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)
    
class MethaneRevWGS_OHGate4Step_NO_Scaffold(MechanisticScaffold):
    """
    Advanced Domain-informed macroscopic scaffold.
    Fixes the CO2 overshoot problem by replacing O2-driven CO burnout 
    with OH-gated CO burnout, mirroring the true CO + OH <-> CO2 + H reaction.
    Also introduces a reversible Water-Gas Shift proxy.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (8):
      0: k_methane_ox : CH4 forward oxidation
      1: n_o2_methane : Fractional order of O2 in CH4 oxidation
      2: k_co_oh      : CO -> CO2 forward rate (GATED BY OH)
      3: k_co_r       : CO2 -> CO reverse rate 
      4: k_wgs_f      : Water-gas shift forward (CO + H2O -> CO2 + ...)
      5: k_wgs_r      : Water-gas shift reverse (CO2 -> CO + H2O proxy)
      6: k_oh_prod    : Radical pool generation
      7: k_thermal_no : NO formation
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounds optimized for the new kinetic formulation
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [10.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        (k_methane, n_o2_methane, k_co_oh, k_co_r, 
         k_wgs_f, k_wgs_r, k_oh, k_no) = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations and NaN powers
        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        CO_p  = torch.clamp_min(CO,  eps)
        CO2_p = torch.clamp_min(CO2, eps)
        H2O_p = torch.clamp_min(H2O, eps)
        OH_p  = torch.clamp_min(OH,  eps)

        # Clamp parameters to bounds
        k_methane    = torch.clamp_min(k_methane, 0.0)
        n_o2_methane = torch.clamp(n_o2_methane, min=0.1, max=2.0)
        k_co_oh      = torch.clamp_min(k_co_oh, 0.0)
        k_co_r       = torch.clamp_min(k_co_r, 0.0)
        k_wgs_f      = torch.clamp_min(k_wgs_f, 0.0)
        k_wgs_r      = torch.clamp_min(k_wgs_r, 0.0)
        k_oh         = torch.clamp_min(k_oh, 0.0)
        k_no         = torch.clamp_min(k_no, 0.0)

        # 1. CH4 Oxidation (Requires OH pool to truly kick off - induction proxy)
        # We add a mild OH saturation term to force the ignition delay
        r1 = k_methane * CH4_p * (O2_p ** n_o2_methane) * (OH_p / (OH_p + 1e-3))
        
        # 2. Reversible CO Burnout (GATED BY OH instead of O2)
        # r2_f: CO + OH -> CO2 + H (proxy)
        r2_f = k_co_oh * CO_p * OH_p
        r2_r = k_co_r * CO2_p
        
        # 3. Fully Reversible Water-Gas Shift
        r_wgs_f = k_wgs_f * CO_p * H2O_p
        r_wgs_r = k_wgs_r * CO2_p  # We don't track H2, so we approximate the reverse rate linearly
        r_wgs_net = r_wgs_f - r_wgs_r
        
        # 4. OH Generation (Fuel inhibition proxy: early CH4 suppresses OH accumulation)
        r3 = k_oh * O2_p / (1.0 + 10.0 * CH4_p)
        
        # 5. NO formation
        r4 = k_no * O2_p

        # Apply stoichiometry
        dCH4 = -r1
        dO2  = -1.5 * r1 - r3 - r4
        dCO  =  r1 - r2_f + r2_r - r_wgs_net
        dCO2 =  r2_f - r2_r + r_wgs_net
        dH2O =  2.0 * r1 - r_wgs_net
        dOH  =  2.0 * r3 - r2_f  # OH is consumed during CO burnout
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)
    

class methaneHydrogenBalance_Scaffold(MechanisticScaffold):
    """
    Advanced Scaffold (V3) with Virtual Hydrogen Balance.
    - Uses Hydrogen atom conservation to calculate a virtual H2 state.
    - Corrects CO2 accumulation by using H2 in the reverse WGS equilibrium.
    - Refines OH stoichiometry to prevent radical starvation.
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        # Initial Hydrogen reservoir (adjust based on your dataset inlet)
        # For stoichiometric CH4/Air: 4 * CH4_init + 2 * H2O_init
        self.H_total_init = 4.0 
        
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [50.0, 2.0, 50.0, 50.0, 50.0, 50.0, 50.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        (k_methane, n_o2_methane, k_co_oh, k_co_r, 
         k_wgs_f, k_wgs_r, k_oh, k_no) = theta.unbind(dim=-1)

        eps = 1e-8
        CH4_p, O2_p, CO_p, CO2_p, H2O_p, OH_p = [torch.clamp_min(s, eps) for s in [CH4, O2, CO, CO2, H2O, OH]]

        # --- 1. VIRTUAL HYDROGEN BALANCE ---
        # Calculate how much Hydrogen is "missing" from the explicit states
        # H_balance = Total_H - (4*CH4 + 2*H2O + 1*OH)
        H_missing = self.H_total_init - (4.0 * CH4_p + 2.0 * H2O_p + OH_p)
        H2_virtual = torch.clamp_min(0.5 * H_missing, eps)

        # --- 2. KINETIC RATES ---
        # CH4 Oxidation (Partial oxidation to CO + H2/H2O)
        r1 = k_methane * CH4_p * (O2_p ** n_o2_methane) * (OH_p / (OH_p + 1e-3))
        
        # Reversible CO Burnout: CO + OH <-> CO2 + H (proxy)
        # The reverse rate MUST depend on H2 (as a proxy for H) to prevent CO2 accumulation [1]
        r2_f = k_co_oh * CO_p * OH_p
        r2_r = k_co_r * CO2_p * (H2_virtual / (H2O_p + eps)) 
        
        # Reversible Water-Gas Shift: CO + H2O <-> CO2 + H2
        r_wgs_f = k_wgs_f * CO_p * H2O_p
        r_wgs_r = k_wgs_r * CO2_p * H2_virtual
        r_wgs_net = r_wgs_f - r_wgs_r
        
        # OH Generation with exponential auto-ignition switch
        r3 = k_oh * O2_p * torch.exp(-10.0 * CH4_p)
        
        # NO formation
        r4 = k_no * O2_p

        # --- 3. STOICHIOMETRY ---
        dCH4 = -r1
        dO2  = -1.5 * r1 - r3 - r4
        dCO  =  r1 - r2_f + r2_r - r_wgs_net
        dCO2 =  r2_f - r2_r + r_wgs_net
        # H2O is only produced by fuel oxidation; WGS consumes/produces it
        dH2O =  2.0 * r1 - r_wgs_net 
        # OH is produced by O2 and recycled/consumed by CO burnout
        # In reality, CO + OH -> CO2 + H, and H + O2 -> OH + O (chain branching)
        # We model this by making the OH loss for CO burnout very small (0.1)
        dOH  =  2.0 * r3 - 0.1 * r2_f 
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)

class methaneRevWGS_fixed(MechanisticScaffold):
    """
    Advanced Domain-informed macroscopic scaffold (V2).
    - Incorporates proxy non-linearities for missing H/H2 reducing agents.
    - Replaces linear fuel inhibition with an exponential auto-ignition switch.
    
    States (7): CH4, O2, CO, CO2, H2O, OH, NO
    Parameters (8):
      0: k_methane_ox : CH4 forward oxidation
      1: n_o2_methane : Fractional order of O2 in CH4 oxidation
      2: k_co_oh      : CO -> CO2 forward rate (GATED BY OH)
      3: k_co_r       : CO2 -> CO reverse rate (GATED BY CO proxy for H2)
      4: k_wgs_f      : Water-gas shift forward (CO + H2O -> CO2 + ...)
      5: k_wgs_r      : Water-gas shift reverse (CO2 -> CO + H2O proxy)
      6: k_oh_prod    : Radical pool generation (EXPONENTIAL INHIBITION)
      7: k_thermal_no : NO formation
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O", "OH", "NO"]
        
        # Bounds expanded to allow GRU to capture rapid ignition transients 
        # now that the structural decomposition loops are fixed.
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6]
        self.theta_hi_vec = [50.0, 2.0, 50.0, 50.0, 50.0, 50.0, 50.0, 1.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O, OH, NO = y.unbind(dim=-1)
        (k_methane, n_o2_methane, k_co_oh, k_co_r, 
         k_wgs_f, k_wgs_r, k_oh, k_no) = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations and NaN powers
        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        CO_p  = torch.clamp_min(CO,  eps)
        CO2_p = torch.clamp_min(CO2, eps)
        H2O_p = torch.clamp_min(H2O, eps)
        OH_p  = torch.clamp_min(OH,  eps)

        # Clamp parameters to bounds
        k_methane    = torch.clamp_min(k_methane, 0.0)
        n_o2_methane = torch.clamp(n_o2_methane, min=0.1, max=2.0)
        k_co_oh      = torch.clamp_min(k_co_oh, 0.0)
        k_co_r       = torch.clamp_min(k_co_r, 0.0)
        k_wgs_f      = torch.clamp_min(k_wgs_f, 0.0)
        k_wgs_r      = torch.clamp_min(k_wgs_r, 0.0)
        k_oh         = torch.clamp_min(k_oh, 0.0)
        k_no         = torch.clamp_min(k_no, 0.0)

        # 1. CH4 Oxidation
        r1 = k_methane * CH4_p * (O2_p ** n_o2_methane) * (OH_p / (OH_p + 1e-3))
        
        # 2. Reversible CO Burnout 
        # FIX: Reverse rate uses (CO / (CO + eps)) proxy to mimic the presence 
        # of H/H2 reducing agents, preventing spontaneous CO2 decomposition.
        r2_f = k_co_oh * CO_p * OH_p
        r2_r = k_co_r * CO2_p * (CO_p / (CO_p + eps))
        
        # 3. Fully Reversible Water-Gas Shift
        r_wgs_f = k_wgs_f * CO_p * H2O_p
        # FIX: Same proxy applied to WGS reverse rate
        r_wgs_r = k_wgs_r * CO2_p * (CO_p / (CO_p + eps)) 
        r_wgs_net = r_wgs_f - r_wgs_r
        
        # 4. OH Generation 
        # FIX: Exponential fuel inhibition creates a realistic, steep auto-ignition delay
        r3 = k_oh * O2_p * torch.exp(-10.0 * CH4_p)
        
        # 5. NO formation
        r4 = k_no * O2_p

        # Apply stoichiometry
        dCH4 = -r1
        dO2  = -1.5 * r1 - r3 - r4
        dCO  =  r1 - r2_f + r2_r - r_wgs_net
        dCO2 =  r2_f - r2_r + r_wgs_net
        dH2O =  2.0 * r1 - r_wgs_net
        dOH  =  2.0 * r3 - r2_f  # OH is consumed during CO burnout
        dNO  =  2.0 * r4

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O, dOH, dNO), dim=-1)
    
    import torch

class Methane5State_Global_Scaffold(MechanisticScaffold):
    """
    Ultra-simplified 5-State Macroscopic Scaffold.
    Relies on the GRU's hidden state to implicitly model the radical pool 
    (auto-ignition delay) rather than explicitly tracking OH.
    
    States (5): CH4, O2, CO, CO2, H2O
    Parameters (4):
      0: k_methane_ox : Global CH4 -> CO forward rate
      1: n_o2         : O2 reaction order
      2: k_co_f       : Global CO -> CO2 forward rate
      3: k_co_r       : Global CO2 -> CO reverse rate
    """
    def __init__(self):
        super().__init__(P=5, theta_dim=4)
        self.state_names = ["CH4", "O2", "CO", "CO2", "H2O"]
        
        # Kept bounds wide since global mechanisms need massive rate 
        # spikes to compensate for missing intermediate steps.
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5]
        self.theta_hi_vec = [50.0, 2.0, 50.0, 50.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        CH4, O2, CO, CO2, H2O = y.unbind(dim=-1)
        k_methane, n_o2, k_co_f, k_co_r = theta.unbind(dim=-1)

        # Clamp states to prevent negative concentrations and NaN powers
        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        CO_p  = torch.clamp_min(CO,  eps)
        CO2_p = torch.clamp_min(CO2, eps)

        # Clamp parameters to bounds
        k_methane = torch.clamp_min(k_methane, 0.0)
        n_o2      = torch.clamp(n_o2, min=0.1, max=2.0)
        k_co_f    = torch.clamp_min(k_co_f, 0.0)
        k_co_r    = torch.clamp_min(k_co_r, 0.0)

        # 1. Global Partial Oxidation: CH4 + 1.5 O2 -> CO + 2 H2O
        # The GRU must learn to keep k_methane near 0 until ignition time
        r1 = k_methane * CH4_p * (O2_p ** n_o2)
        
        # 2. Global Reversible CO Burnout: CO + 0.5 O2 <-> CO2
        # Forward rate depends on O2.
        r2_f = k_co_f * CO_p * (O2_p ** 0.5) 
        
        # Reverse rate proxy. We still use the (CO / CO+eps) trick to prevent 
        # spontaneous CO2 decay when no fuel/intermediates are present.
        r2_r = k_co_r * CO2_p * (CO_p / (CO_p + eps))

        # Apply stoichiometry
        dCH4 = -r1
        dO2  = -1.5 * r1 - 0.5 * r2_f + 0.5 * r2_r
        dCO  =  r1 - r2_f + r2_r
        dCO2 =  r2_f - r2_r
        dH2O =  2.0 * r1

        return torch.stack((dCH4, dO2, dCO, dCO2, dH2O), dim=-1)

class WestbrookDryer2Step(MechanisticScaffold):
    """
    Classical 2-step reduced mechanism (Westbrook & Dryer, 1981).
    States: CH4, O2, CO, CO2   (H2O recovered by H-atom balance externally)

    R1:  CH4 + 1.5 O2 -> CO + 2 H2O      (irreversible)
    R2:  CO  + 0.5 O2 <-> CO2            (reversible — captures CO/CO2 equilibrium)

    No radical states, no OH gating. Ignition transient is delegated entirely
    to theta(t) from the GRU, which is precisely what your framework is good at.
    """
    def __init__(self):
        super().__init__(P=4, theta_dim=5)
        self.state_names = ["CH4", "O2", "CO", "CO2"]
        # [k1, n_o2_1, k2_f, k2_r, n_o2_2]
        self.theta_lo_vec = [1e-5, 0.1, 1e-5, 1e-5, 0.1]
        self.theta_hi_vec = [50.0, 2.0, 50.0, 50.0, 2.0]

    def forward(self, y, theta):
        CH4, O2, CO, CO2 = y.unbind(dim=-1)
        k1, n1, k2f, k2r, n2 = theta.unbind(dim=-1)

        eps = 1e-8
        CH4_p, O2_p, CO_p, CO2_p = [torch.clamp_min(s, eps) for s in (CH4, O2, CO, CO2)]
        n1 = torch.clamp(n1, 0.1, 2.0)
        n2 = torch.clamp(n2, 0.1, 2.0)
        k1, k2f, k2r = [torch.clamp_min(k, 0.0) for k in (k1, k2f, k2r)]

        r1 = k1  * CH4_p * (O2_p ** n1)
        r2 = k2f * CO_p  * (O2_p ** n2) - k2r * CO2_p

        dCH4 = -r1
        dO2  = -1.5 * r1 - 0.5 * r2
        dCO  =  r1 - r2
        dCO2 =  r2
        return torch.stack((dCH4, dO2, dCO, dCO2), dim=-1)


class GlobalOneStep(MechanisticScaffold):
    """
    Single global reaction:   CH4 + 2 O2 -> CO2 + 2 H2O
    States: CH4, O2, CO2.     H2O via H-balance, CO is treated as algebraically zero
    (i.e. the scaffold does not resolve CO/CO2 equilibrium — that lives in theta(t)).

    All transient richness is in theta(t) = (k, a, b) — a fractional-order Arrhenius
    proxy where the GRU is free to drive k from ~0 (induction) to large (ignition)
    to small (depletion).
    """
    def __init__(self):
        super().__init__(P=3, theta_dim=3)
        self.state_names = ["CH4", "O2", "CO2"]
        self.theta_lo_vec = [1e-5, 0.1, 0.1]
        self.theta_hi_vec = [50.0, 2.0, 2.0]

    def forward(self, y, theta):
        CH4, O2, CO2 = y.unbind(dim=-1)
        k, a, b = theta.unbind(dim=-1)

        eps = 1e-8
        CH4_p = torch.clamp_min(CH4, eps)
        O2_p  = torch.clamp_min(O2,  eps)
        a = torch.clamp(a, 0.1, 2.0)
        b = torch.clamp(b, 0.1, 2.0)
        k = torch.clamp_min(k, 0.0)

        r = k * (CH4_p ** a) * (O2_p ** b)
        return torch.stack((-r, -2.0 * r, r), dim=-1)
    
class Kovacs54Scaffold(MechanisticScaffold):
    """
    Kovacs virtual-species 14-state methane mechanism (54 reactions).
    30 virtual fuel-breakdown reactions + 24 core H2/CO reactions (HCO removed).

    States (14): CH4, O2, H2O, CO, CO2, H2, H, O, OH, HO2, H2O2, CH3(R), CH2O(IO), N2
    Parameters θ (54): one learned rate constant per reaction.

    Dataset obs-indices: 15,5,7,12,13,3,4,6,8,11,10,16,28,1  (AramcoMech 3.0)
    """
    def __init__(self):
        super().__init__(P=14, theta_dim=54)
        self.state_names = [
            'CH4', 'O2', 'H2O', 'CO', 'CO2', 'H2', 'H', 'O',
            'OH', 'HO2', 'H2O2', 'CH3', 'CH2O', 'N2'
        ]
        # No per-reaction override — falls back to scalar [theta_lo, theta_hi]
        # from config (defaults [1e-6, 100]). This range is what was used
        # before and trained stably; the wider bounds I tried earlier
        # interact badly with the mole-fraction correction and NaN out RK4.

    def forward(self, y, k):
        """
        y: Tensor of shape (batch, 14)
        k: Tensor of shape (batch, 54)
        """
        # No ReLU on y — non-smooth gate kills gradient flow when species
        # cross zero during integration. Mass-action terms naturally damp
        # toward zero anyway; clamping happens once for the dilution weight.
        FUEL = y[:, 0]
        O2   = y[:, 1]
        H2O  = y[:, 2]
        CO   = y[:, 3]
        CO2  = y[:, 4]
        H2   = y[:, 5]
        H    = y[:, 6]
        O    = y[:, 7]
        OH   = y[:, 8]
        HO2  = y[:, 9]
        H2O2 = y[:, 10]
        R    = y[:, 11] 
        IO   = y[:, 12] 
        N2   = y[:, 13]

        # 2. Third-body efficiency pool (Simplified sum)
        M = FUEL + O2 + H2O + CO + CO2 + H2 + H + O + OH + HO2 + H2O2 + R + IO + N2

        # 3. Unpack Rates
        k_vals = [k[:, i] for i in range(self.theta_dim)]

        # =====================================================================
        # 30 VIRTUAL REACTIONS 
        # =====================================================================
        r1  = k_vals[0]  * FUEL * H       
        r2  = k_vals[1]  * FUEL * OH      
        r3  = k_vals[2]  * FUEL * O       
        r4  = k_vals[3]  * FUEL * HO2     
        r5  = k_vals[4]  * FUEL * OH      
        r6  = k_vals[5]  * FUEL * O       
        r7  = k_vals[6]  * FUEL * HO2     
        r8  = k_vals[7]  * FUEL * OH      
        r9  = k_vals[8]  * FUEL * O       
        r10 = k_vals[9]  * FUEL * HO2     
        r11 = k_vals[10] * R * OH         
        r12 = k_vals[11] * R * O          
        r13 = k_vals[12] * R * HO2        
        r14 = k_vals[13] * R * O2         
        r15 = k_vals[14] * R * OH         
        r16 = k_vals[15] * R * O          
        r17 = k_vals[16] * R * HO2        
        r18 = k_vals[17] * R * H * M      
        r19 = k_vals[18] * IO * OH        
        r20 = k_vals[19] * IO * O         
        r21 = k_vals[20] * IO * HO2       
        r22 = k_vals[21] * IO * HO2       
        r23 = k_vals[22] * IO * O2        
        r24 = k_vals[23] * IO * OH        
        r25 = k_vals[24] * IO * O         
        r26 = k_vals[25] * IO * HO2       
        r27 = k_vals[26] * IO * HO2       
        r28 = k_vals[27] * IO * H * H     
        r29 = k_vals[28] * IO * R         
        r30 = k_vals[29] * CO * H * H * M 

        # =====================================================================
        # 24 CORE H2/CO REACTIONS (HCO safely removed)
        # =====================================================================
        r31 = k_vals[30] * H * O2         
        r32 = k_vals[31] * O * H2         
        r33 = k_vals[32] * OH * H2        
        r34 = k_vals[33] * O * H2O        
        r35 = k_vals[34] * H * H * M      
        r36 = k_vals[35] * O * O * M      
        r37 = k_vals[36] * O * H * M      
        r38 = k_vals[37] * H * OH * M     
        r39 = k_vals[38] * H * O2 * M     
        r40 = k_vals[39] * HO2 * H        
        r41 = k_vals[40] * HO2 * H        
        r42 = k_vals[41] * HO2 * H        
        r43 = k_vals[42] * HO2 * O        
        r44 = k_vals[43] * HO2 * OH       
        r45 = k_vals[44] * HO2 * HO2      
        r46 = k_vals[45] * H2O2 * M       
        r47 = k_vals[46] * H2O2 * H       
        r48 = k_vals[47] * H2O2 * H       
        r49 = k_vals[48] * H2O2 * O       
        r50 = k_vals[49] * H2O2 * OH      
        r51 = k_vals[50] * CO * O * M     
        r52 = k_vals[51] * CO * O2        
        r53 = k_vals[52] * CO * OH        
        r54 = k_vals[53] * CO * HO2       

        # =====================================================================
        # 4. Construct the Species Derivatives (dy/dt)
        # =====================================================================
        
        d_FUEL = -r1 - r2 - r3 - r4 - r5 - r6 - r7 - r8 - r9 - r10 + r18 + r28 + r29
        
        d_R    = r1 + r2 + r3 + r4 - r11 - r12 - r13 - r14 - r15 - r16 - r17 - r18 - r29
        
        d_IO   = r5 + r6 + r7 + r11 + r12 + r13 + r14 - r19 - r20 - r21 - r22 - r23 \
                 - r24 - r25 - r26 - r27 - r28 - r29 + r30
                 
        # HCO reactions (r55-r57) are fully removed from CO and others
        d_CO   = r8 + r9 + r10 + r15 + r16 + r17 + r19 + r20 + r21 + r22 + r23 \
                 + r29 - r30 - r51 - r52 - r53 - r54 
                 
        d_CO2  = r24 + r25 + r26 + r27 + r51 + r52 + r53 + r54
        
        d_H2   = r1 + r5 + r6 + r7 + 2*r8 + 2*r9 + 2*r10 + r11 + 2*r15 + r16 + r17 \
                 + r24 + r25 + r26 - r32 - r33 + r35 + r41 + r47
                 
        d_H    = -r1 + r5 + r8 + r12 + r16 - r18 + r19 + r20 + r21 + r27 - 2*r28 \
                 + r29 - 2*r30 - r31 + r32 + r33 - 2*r35 - r37 - r38 - r39 - r40 \
                 - r41 - r42 - r47 - r48 + r53 
                 
        d_O    = -r3 - r6 - r9 - r12 - r16 - r20 - r25 + r28 + r31 - r32 - r34 \
                 - 2*r36 - r37 + r42 - r43 + r44 - r49 - r51 + r52
                 
        d_OH   = -r2 + r3 - r5 - r8 - r11 + r14 - r15 + r17 - r19 + r20 + r22 \
                 - r24 + r26 + r31 + r32 - r33 + 2*r34 + r37 - r38 + 2*r40 \
                 + r43 - r44 + 2*r46 + r48 + r49 - r50 - r53 + r54 
                 
        d_O2   = -r14 - r23 - r31 + r36 - r39 + r41 + r43 + r44 + r45 - r52 
                 
        d_H2O  = r2 + r13 + r22 + r27 + r33 - r34 + r38 + r42 + r44 + r48 + r50 
        
        d_HO2  = -r4 - r7 - r10 - r13 - r17 - r21 - r22 - r26 - r27 + r39 - r40 \
                 - r41 - r42 - r43 - r44 - 2*r45 + r47 + r49 + r50 - r54 
                 
        d_H2O2 = r4 + r21 + r23 + r45 - r46 - r47 - r48 - r49 - r50
        
        d_N2   = torch.zeros_like(FUEL)

        dy_dt = torch.stack([
            d_FUEL, d_O2, d_H2O, d_CO, d_CO2, d_H2, d_H, d_O,
            d_OH, d_HO2, d_H2O2, d_R, d_IO, d_N2
        ], dim=-1)

        # Mole-fraction correction (Gibbs-Duhem-like): when total moles change
        # due to net reaction, dx_i/dt = R_i - x_i * sum(R_j). This makes inert
        # species (N2) shift via dilution and keeps sum(x) normalized.
        # Clamp y to [0,1] so RK4 overshoots don't blow up the dilution term.
        x = torch.clamp(y, 0.0, 1.0)
        dy_dt = dy_dt - x * dy_dt.sum(dim=-1, keepdim=True)

        return dy_dt

class KovacsMethaneSRPScaffold(MechanisticScaffold):
    """
    7-state scaffold for CH4 combustion following Kovács et al. (2026) SRP.

    States (P=7): [FUEL, R, IO, CO, CO2, O2, H2O]
    Mapping to AramcoMech 3.0: CH4, CH3, CH2O, CO, CO2, O2, H2O

    Radical species (H, O, OH, HO2, H2O2, H2) are NOT tracked.
    They appear only via θ(t): the neural encoder learns the time-varying
    effective pseudo-first-order rates k_eff(t) = k_Arrhenius * [X](t),
    implicitly encoding the radical pool dynamics.

    Effective fluxes (pseudo-first-order in virtual species):
      v1 = k1 * FUEL    FUEL → R       (H-abstraction, dominant)
      v2 = k2 * FUEL    FUEL → IO      (direct shortcut)
      v3 = k3 * FUEL    FUEL → CO      (direct shortcut)
      v4 = k4 * R       R → IO         (radical oxidation)
      v5 = k5 * R       R → CO         (direct)
      v6 = k6 * IO      IO → CO        (aldehyde oxidation, dominant)
      v7 = k7 * IO      IO → CO2       (direct)
      v8 = k8 * CO      CO → CO2       (H2/CO core, CO oxidation)

    O2 and H2O follow approximate global stoichiometry:
      O2 consumed ~1 mol per mol FUEL C-atom converted (rough average)
      H2O produced ~0.5 mol per mol FUEL consumed at each H-abstraction step

    θ_dim = 8: [k1, k2, k3, k4, k5, k6, k7, k8]
    Observed: all 7 states (fitting to AramcoMech simulation data)
    """
    def __init__(self):
        super().__init__(P=7, theta_dim=8)
        self.state_names = ["FUEL", "R", "IO", "CO", "CO2", "O2", "H2O"]
        # Pseudo-first-order effective rate constants [normalized_conc^-1 * s^-1 effectively]
        # Wide bounds: radical pool concentration spans ~1e-6 to 1e-3,
        # Arrhenius A factors ~1e8–1e14, so k_eff = k*[X] ~ 1e2–1e8 s^-1
        # In your normalized concentration space, scale down proportionally
        self.theta_lo_vec = [1e-3] * 8
        self.theta_hi_vec = [1e3]  * 8

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        FUEL, R, IO, CO, CO2, O2, H2O = y.unbind(dim=-1)
        k1, k2, k3, k4, k5, k6, k7, k8 = theta.unbind(dim=-1)

        # Physical non-negativity
        FUEL_p = torch.clamp_min(FUEL, 0.0)
        R_p    = torch.clamp_min(R,    0.0)
        IO_p   = torch.clamp_min(IO,   0.0)
        CO_p   = torch.clamp_min(CO,   0.0)
        O2_p   = torch.clamp_min(O2,   0.0)

        # Effective fluxes (all pseudo-first-order; radical [X] absorbed into k_i(t))
        v1 = k1 * FUEL_p   # FUEL → R
        v2 = k2 * FUEL_p   # FUEL → IO  (shortcut)
        v3 = k3 * FUEL_p   # FUEL → CO  (shortcut)
        v4 = k4 * R_p      # R → IO
        v5 = k5 * R_p      # R → CO
        v6 = k6 * IO_p     # IO → CO
        v7 = k7 * IO_p     # IO → CO2
        v8 = k8 * CO_p     # CO → CO2

        dFUEL = -(v1 + v2 + v3)
        dR    =   v1 - (v4 + v5)
        dIO   =   v2 + v4 - (v6 + v7)
        dCO   =   v3 + v5 + v6 - v8
        dCO2  =   v7 + v8

        # Global O2 stoichiometry:
        # CH4 → CO2 + 2H2O consumes 2 O2 total.
        # Distribute: ~1 O2 from FUEL abstraction, ~1 O2 from CO oxidation.
        # Rough split: v_O2 = (v1+v2+v3) * 1.0 + v8 * 1.0
        # Using simpler: total O2 ≈ 2 × FUEL consumption rate
        total_fuel_flux = v1 + v2 + v3
        dO2  = -(total_fuel_flux + v8)  # one O2 for C-H bond abstraction, one for CO→CO2

        # H2O stoichiometry: produced at H-abstraction steps
        # Each FUEL+OH→R+H2O or R+OH→IO+H2 etc.
        # Per Kovács Table 2: mainly at v1 (FUEL→R via OH gives H2O) and v6 (IO→CO via OH gives H2O)
        dH2O =  v1 + v6 + v2  # rough; dominant sources are H-abstraction from FUEL and IO

        dy_dt = torch.stack((dFUEL, dR, dIO, dCO, dCO2, dO2, dH2O), dim=-1)

        # Mole-fraction correction: dx_i/dt = R_i - x_i * sum(R_j), so the
        # subset's mole-fractions stay self-consistent even when species
        # outside the 7-state subset (radicals) carry net mole change.
        x = torch.clamp(y, 0.0, 1.0)
        dy_dt = dy_dt - x * dy_dt.sum(dim=-1, keepdim=True)

        return dy_dt

# ============================================================================
# IvttAnalyticScaffold — verbatim port of Bob's 7-state IVTT closed-form step.
#
# Replaces RK4 integration with the analytic update from
#   bob_model/spline_models.py:ivtt_step_R_O_mRNA_maturation
# (also mirrored in last-layer-ode/models/bob_gru_verbatim.py). The scaffold owns:
#   - the 7-state layout [R, O, m, mm, p, pm, _]
#   - per-parameter bounds for the 7 encoder outputs [lam, lam_O, VTX, kdm, VTL, kmt, kmatm]
#   - the seeded init (R=O=1, m=p=0.01, mm=mm0+0.01, pm=pm0+0.01)
#   - the per-batch DNA cumulative total (constant per sequence)
#   - the loss-facing theta repack [VTX, kdm, VTL, kmt, kmatm, R, lam, lam_O]
#
# Encoder-side concerns (DNA c column dropped from u, sqrt features on u and
# sqrt+clamp_min(1) on mm/pm) are handled by OdeRNN via its existing
# u_transform/y_transform/gru_u_cols/gru_y_cols knobs.
# ============================================================================
class IvttAnalyticScaffold(MechanisticScaffold):
    # Layout indices into the 7-d state and the U=12 control vector — declared as
    # class constants so TorchScript treats them as compile-time integers.
    __constants__ = [
        "DNA_C_COL_IDX", "R_IDX", "O_IDX", "M_IDX", "MM_IDX", "P_IDX", "PM_IDX",
    ]
    DNA_C_COL_IDX: int = 2     # column of u_seq that carries DNA c (per sequence)
    R_IDX:  int = 0
    O_IDX:  int = 1
    M_IDX:  int = 2
    MM_IDX: int = 3
    P_IDX:  int = 4
    PM_IDX: int = 5

    def __init__(self):
        # P=7 (state width matching the dataset y); theta_dim=7 (encoder output).
        # theta_dim_emit=8 because Bob's loss reads [VTX, kdm, VTL, kmt, kmatm, R, lam, lam_O].
        super().__init__(P=7, theta_dim=7)
        self.theta_dim_emit = 8
        # Override base defaults — TorchScript reads these from the instance.
        self.has_analytic_step = True
        self.tf_at_k_zero = True   # Bob fires TF at k=0 (with k-1=-1 wrap to last frame)
        self.state_names = ["R", "O", "m", "mm", "p", "pm", "_"]
        # Bound order matches encoder output: [lam, lam_O, VTX, kdm, VTL, kmt, kmatm]
        self.theta_lo_vec = [1e-6, 1e-6, 5e-5, 1e-5, 5e-5, 1e-5, 5e-5]
        self.theta_hi_vec = [5e-4, 5e-4, 1e-1, 1e-2, 6e-2, 3.5e-4, 3.5e-3]

    # Standard scaffold.forward (RHS for RK4) is unused — keep a stub so the
    # base contract is still satisfied if anyone calls it accidentally.
    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "IvttAnalyticScaffold has no RHS; integrate via analytic_step() instead."
        )

    def precompute_batch(
        self, y0: torch.Tensor, u_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # u_seq: (B, K, U). Bob's "DNA cumulative" is just sum of the DNA c column
        # along time, equivalent to a bolus at t=0 plus a constant; reused at every step.
        dna_raw = u_seq[:, :, self.DNA_C_COL_IDX:self.DNA_C_COL_IDX + 1]
        dna_cum_total = dna_raw.cumsum(dim=1)[:, -1, :]   # (B, 1)
        out: Dict[str, torch.Tensor] = {"dna_cum_total": dna_cum_total}
        return out

    def initial_state(self, y0: torch.Tensor) -> torch.Tensor:
        # Seeded hidden states — exact values from bob_gru_verbatim.forward().
        # Built via concatenation of (B, 1) columns so TorchScript stays happy
        # (slice-assignment of scalars also works, but concat is bulletproof).
        mm0   = y0[:, self.MM_IDX:self.MM_IDX + 1]
        pm0   = y0[:, self.PM_IDX:self.PM_IDX + 1]
        ones  = torch.ones_like(mm0)             # (B, 1) — for R, O
        cents = torch.full_like(mm0, 0.01)       # (B, 1) — for m, p
        tail  = torch.zeros_like(mm0)            # (B, 1) — unused slot 6
        return torch.cat(
            [ones, ones, cents, mm0 + 0.01, cents, pm0 + 0.01, tail], dim=-1,
        )

    def analytic_step(
        self,
        y_prev: torch.Tensor,
        dt_k: torch.Tensor,
        theta_k: torch.Tensor,
        ctx: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # y_prev: (B, 7); dt_k: (B,) or (B, 1); theta_k: (B, 7)
        if dt_k.dim() == 1:
            dt_k = dt_k.unsqueeze(-1)
        R_prev  = y_prev[:, self.R_IDX:self.R_IDX + 1]
        O_prev  = y_prev[:, self.O_IDX:self.O_IDX + 1]
        m_prev  = y_prev[:, self.M_IDX:self.M_IDX + 1]
        mm_prev = y_prev[:, self.MM_IDX:self.MM_IDX + 1]
        p_prev  = y_prev[:, self.P_IDX:self.P_IDX + 1]
        pm_prev = y_prev[:, self.PM_IDX:self.PM_IDX + 1]

        lam_k   = theta_k[:, 0:1]
        lamO_k  = theta_k[:, 1:2]
        VTXmax  = theta_k[:, 2:3]
        kdm_k   = theta_k[:, 3:4]
        VTLmax  = theta_k[:, 4:5]
        kmt_k   = theta_k[:, 5:6]
        kmatm_k = theta_k[:, 6:7]

        eps = 1e-9
        rho_R = torch.exp(-lam_k  * dt_k)
        rho_O = torch.exp(-lamO_k * dt_k)
        R_curr = R_prev * rho_R
        O_curr = O_prev * rho_O

        VTX_eff = R_curr * VTXmax
        VTL_eff = R_curr * VTLmax
        O_eff   = O_curr

        S = VTX_eff * ctx["dna_cum_total"]

        alpha = (kdm_k + kmatm_k).clamp_min(eps)
        m_inf = S / alpha
        exp_a = torch.exp(-alpha * dt_k)
        m_curr = torch.clamp_min(m_inf + (m_prev - m_inf) * exp_a, 0.0)

        exp_d  = torch.exp(-kdm_k   * dt_k)
        exp_mr = torch.exp(-kmatm_k * dt_k)
        term1 = mm_prev * exp_d
        term2 = m_inf * (kmatm_k / (kdm_k + eps)) * (1.0 - exp_d)
        term3 = (m_prev - m_inf) * exp_d * (1.0 - exp_mr)
        mm_curr = torch.clamp_min(term1 + term2 + term3, 0.0)

        M_prev = m_prev + mm_prev
        M_inf  = S / (kdm_k + eps)
        exp_M  = exp_d

        eta   = torch.exp(-kmt_k * dt_k)
        delta = kdm_k - kmt_k
        same  = torch.abs(delta) < 1e-6
        int_M_eq  = M_inf * (1.0 - eta) / (kmt_k + eps) + (M_prev - M_inf) * dt_k * eta
        int_M_gen = M_inf * (1.0 - eta) / (kmt_k + eps) + (M_prev - M_inf) * (eta - exp_M) / (delta + eps)
        int_M_conv = torch.where(same, int_M_eq, int_M_gen)
        p_curr = torch.clamp_min(p_prev * eta + VTL_eff * int_M_conv, 0.0)

        int_M_total = M_inf * dt_k + (M_prev - M_inf) / (kdm_k + eps) * (1.0 - exp_M)
        pm_curr = torch.clamp_min(
            pm_prev + O_eff * (VTL_eff * int_M_total - (p_curr - p_prev)),
            0.0,
        )

        # Build y_new via concatenation rather than zero-init + index assignment;
        # TorchScript handles cat() of slice views more reliably than mutating
        # an empty tensor with `[:, slice] = …`.
        tail = torch.zeros_like(R_curr)  # (B, 1) — placeholder for state idx 6
        y_new = torch.cat(
            [R_curr, O_curr, m_curr, mm_curr, p_curr, pm_curr, tail], dim=-1,
        )
        return y_new

    def emit_theta(
        self, theta_enc: torch.Tensor, y_state: torch.Tensor,
    ) -> torch.Tensor:
        # theta_enc: (B, 7) [lam, lamO, VTX, kdm, VTL, kmt, kmatm]
        # y_state:   (B, 7) — post-step state, R at idx 0
        # emit:      (B, 8) [VTX, kdm, VTL, kmt, kmatm, R, lam, lamO]  (Bob's loss layout)
        lam   = theta_enc[:, 0:1]
        lamO  = theta_enc[:, 1:2]
        VTX   = theta_enc[:, 2:3]
        kdm   = theta_enc[:, 3:4]
        VTL   = theta_enc[:, 4:5]
        kmt   = theta_enc[:, 5:6]
        kmatm = theta_enc[:, 6:7]
        R     = y_state[:, self.R_IDX:self.R_IDX + 1]
        return torch.cat([VTX, kdm, VTL, kmt, kmatm, R, lam, lamO], dim=-1)


SCAFFOLDS: dict[str, MechanisticScaffold] = {
    "mof_synthesis_12":  MOFSynthesis12Scaffold(),
    "mof_synthesis_8":   MOFSynthesis8Scaffold(),
    "mof_synthesis_6":   MOFSynthesis6Scaffold(),
    "mof_synthesis_4":   MOFSynthesis4Scaffold(),
    "single_enzyme_6":   SingleEnzymeScaffold(),
    "single_enzyme_4":   SingleEnzymeReduced4Scaffold(),
    "single_enzyme_lumped": SingleEnzymeLumpedScaffold(),
    "txtl_maturation_dna": TXTLMaturationDNAScaffold(),
    "txtl_simple_dna":     TXTLSimpleDNAScaffold(),
    "txtl_resource_and_maturation_dna": TXTLResourceandMaturationDNAScaffold(),
    "methane_global4_no":    MethaneGlobal4Step_NO_Scaffold(),
    "methane_global4_ch2o":  MethaneGlobal4Step_CH2O_Scaffold(),
    "methane_domain4_ch2o_ohgate": MethaneDomainInformedCH2O_OHGate4Step_Scaffold(),
    "methane_domain4_no_ohgate": MethaneDomainInformedOHGate4Step_NO_Scaffold(),
    "methane_revWGS_ohgate_no": MethaneRevWGS_OHGate4Step_NO_Scaffold(),
    "methane_hydrogen_balance": methaneHydrogenBalance_Scaffold(),
    "methane_revWGS_fixed": methaneRevWGS_fixed(),
    "methane_obs5": Methane5State_Global_Scaffold(),
    "westbrook_dryer_2step": WestbrookDryer2Step(),
    "global_one_step": GlobalOneStep(),
    "kovacs_54":  Kovacs54Scaffold(),
    "kovacs_7": KovacsMethaneSRPScaffold(),
    "txtl_resource_and_maturation_dna_bleach": TXTLResourceandMaturationDNABleachScaffold(),
    "txtl_maturation_only_dna": TXTLMaturationOnly7Scaffold(),
    # Glycolysis scaffolds (oracle + 3 reduced models)
    "glycolysis_oracle22":  GlycolysisOracle22Scaffold(),
    "glycolysis_reduced12": GlycolysisReduced12Scaffold(),
    "glycolysis_reduced8":  GlycolysisReduced8Scaffold(),
    "glycolysis_reduced4":  GlycolysisReduced4Scaffold(),
    "ivtt_analytic":        IvttAnalyticScaffold(),
}
