# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# glycolysis_oracle_and_reduced_scaffolds.py

import torch
import torch.nn as nn
from typing import Dict, List


class MechanisticScaffold(nn.Module):
    """
    Base class following the scaffold style used in the main codebase.

    Each scaffold defines:
      - P: state dimension
      - theta_dim: effective parameter dimension
      - state_names: names of state variables
      - theta_lo_vec / theta_hi_vec: optional per-parameter bounds
      - forward(y, theta): vector field dy/dt
    """

    def __init__(self, P: int, theta_dim: int):
        super().__init__()
        self.P = int(P)
        self.theta_dim = int(theta_dim)
        self.state_names: List[str] = []
        self.theta_lo_vec: List[float] | None = None
        self.theta_hi_vec: List[float] | None = None

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# =============================================================================
# 1) FULL ORACLE: 22-state inhibited glycolysis scaffold
# =============================================================================

class GlycolysisOracle22Scaffold(MechanisticScaffold):
    """
    Rich non-oscillatory glycolysis oracle.

    This is a synthetic controlled benchmark inspired by EMP glycolysis. It is not a
    calibrated published yeast glycolysis model. The scaffold keeps many pathway
    intermediates, cofactors, a lactate branch, and hidden inhibitor pools.

    States (22):
      0  Glc       glucose
      1  G6P       glucose-6-phosphate
      2  F6P       fructose-6-phosphate
      3  FBP       fructose-1,6-bisphosphate
      4  GAP       glyceraldehyde-3-phosphate
      5  DHAP      dihydroxyacetone phosphate
      6  BPG13     1,3-bisphosphoglycerate
      7  PG3       3-phosphoglycerate
      8  PG2       2-phosphoglycerate
      9  PEP       phosphoenolpyruvate
      10 Pyr       pyruvate
      11 Lac       lactate
      12 ATP
      13 ADP
      14 NAD
      15 NADH
      16 I_HK      inhibitor pool for Glc -> G6P
      17 I_PFK     inhibitor pool for F6P -> FBP
      18 I_GAPDH   inhibitor pool for GAP -> BPG13
      19 I_PGK     inhibitor pool for BPG13 -> PG3
      20 I_ENO     inhibitor pool for PG2 -> PEP
      21 I_PK      inhibitor pool for PEP -> Pyr

    Inputs / boluses:
      glucose_bolus -> Glc
      atp_bolus     -> ATP
      adp_bolus     -> ADP
      nad_bolus     -> NAD
      nadh_bolus    -> NADH

      hk_inhib_bolus    -> I_HK
      pfk_inhib_bolus   -> I_PFK
      gapdh_inhib_bolus -> I_GAPDH
      pgk_inhib_bolus   -> I_PGK
      eno_inhib_bolus   -> I_ENO
      pk_inhib_bolus    -> I_PK

    Primary observed species:
      Glc, Pyr, ATP, NADH

    Primary observed indices:
      [0, 10, 12, 15]

    Parameters theta (33):
      0  k_hk
      1  k_pgi_f
      2  k_pgi_r
      3  k_pfk
      4  k_ald_f
      5  k_ald_r
      6  k_tpi_f
      7  k_tpi_r
      8  k_gapdh
      9  k_pgk
      10 k_pgm_f
      11 k_pgm_r
      12 k_eno
      13 k_pk
      14 k_ldh
      15 k_pyr_sink
      16 k_atpase
      17 k_nadh_ox
      18 k_leak_hex
      19 k_leak_tri
      20 k_leak_lower
      21 K_I_HK
      22 K_I_PFK
      23 K_I_GAPDH
      24 K_I_PGK
      25 K_I_ENO
      26 K_I_PK
      27 k_I_HK_decay
      28 k_I_PFK_decay
      29 k_I_GAPDH_decay
      30 k_I_PGK_decay
      31 k_I_ENO_decay
      32 k_I_PK_decay

    Inhibition model:
      activity_j = K_I_j / (K_I_j + I_j)

    This is deliberately mass-action / coarse-grained effective kinetics, not
    Michaelis--Menten kinetics. This keeps the oracle relaxation-like and avoids
    feedback-driven oscillatory behavior.
    """

    def __init__(self):
        super().__init__(P=22, theta_dim=33)
        self.state_names = [
            "Glc", "G6P", "F6P", "FBP", "GAP", "DHAP",
            "BPG13", "PG3", "PG2", "PEP", "Pyr", "Lac",
            "ATP", "ADP", "NAD", "NADH",
            "I_HK", "I_PFK", "I_GAPDH", "I_PGK", "I_ENO", "I_PK",
        ]

        self.theta_lo_vec = [
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5,
            1e-4, 1e-5, 1e-4, 1e-4, 1e-4, 1e-5,
            1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5,
            1e-6, 1e-6, 1e-6,
            1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4,
            1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5,
        ]

        self.theta_hi_vec = [
            50.0, 50.0, 50.0, 50.0, 50.0, 20.0,
            50.0, 20.0, 50.0, 50.0, 50.0, 20.0,
            50.0, 50.0, 20.0, 10.0, 10.0, 10.0,
            5.0, 5.0, 5.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
        ]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        (
            Glc, G6P, F6P, FBP, GAP, DHAP,
            BPG13, PG3, PG2, PEP, Pyr, Lac,
            ATP, ADP, NAD, NADH,
            I_HK, I_PFK, I_GAPDH, I_PGK, I_ENO, I_PK,
        ) = y.unbind(dim=-1)

        (
            k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r,
            k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r,
            k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox,
            k_leak_hex, k_leak_tri, k_leak_lower,
            K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK,
            k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay,
            k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay,
        ) = theta.unbind(dim=-1)

        eps = 1e-8

        # Nonnegative states.
        Glc_p = torch.clamp_min(Glc, 0.0)
        G6P_p = torch.clamp_min(G6P, 0.0)
        F6P_p = torch.clamp_min(F6P, 0.0)
        FBP_p = torch.clamp_min(FBP, 0.0)
        GAP_p = torch.clamp_min(GAP, 0.0)
        DHAP_p = torch.clamp_min(DHAP, 0.0)
        BPG13_p = torch.clamp_min(BPG13, 0.0)
        PG3_p = torch.clamp_min(PG3, 0.0)
        PG2_p = torch.clamp_min(PG2, 0.0)
        PEP_p = torch.clamp_min(PEP, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        Lac_p = torch.clamp_min(Lac, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        ADP_p = torch.clamp_min(ADP, 0.0)
        NAD_p = torch.clamp_min(NAD, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        I_HK_p = torch.clamp_min(I_HK, 0.0)
        I_PFK_p = torch.clamp_min(I_PFK, 0.0)
        I_GAPDH_p = torch.clamp_min(I_GAPDH, 0.0)
        I_PGK_p = torch.clamp_min(I_PGK, 0.0)
        I_ENO_p = torch.clamp_min(I_ENO, 0.0)
        I_PK_p = torch.clamp_min(I_PK, 0.0)

        # Nonnegative parameters.
        params_nonnegative = [
            k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r,
            k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r,
            k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox,
            k_leak_hex, k_leak_tri, k_leak_lower,
            K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK,
            k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay,
            k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay,
        ]
        (
            k_hk, k_pgi_f, k_pgi_r, k_pfk, k_ald_f, k_ald_r,
            k_tpi_f, k_tpi_r, k_gapdh, k_pgk, k_pgm_f, k_pgm_r,
            k_eno, k_pk, k_ldh, k_pyr_sink, k_atpase, k_nadh_ox,
            k_leak_hex, k_leak_tri, k_leak_lower,
            K_I_HK, K_I_PFK, K_I_GAPDH, K_I_PGK, K_I_ENO, K_I_PK,
            k_I_HK_decay, k_I_PFK_decay, k_I_GAPDH_decay,
            k_I_PGK_decay, k_I_ENO_decay, k_I_PK_decay,
        ) = [torch.clamp_min(p, eps) for p in params_nonnegative]

        # Inhibitor-dependent activities.
        a_hk = K_I_HK / (K_I_HK + I_HK_p + eps)
        a_pfk = K_I_PFK / (K_I_PFK + I_PFK_p + eps)
        a_gapdh = K_I_GAPDH / (K_I_GAPDH + I_GAPDH_p + eps)
        a_pgk = K_I_PGK / (K_I_PGK + I_PGK_p + eps)
        a_eno = K_I_ENO / (K_I_ENO + I_ENO_p + eps)
        a_pk = K_I_PK / (K_I_PK + I_PK_p + eps)

        # Glycolytic reaction rates.
        r_hk = a_hk * k_hk * Glc_p * ATP_p
        r_pgi = k_pgi_f * G6P_p - k_pgi_r * F6P_p
        r_pfk = a_pfk * k_pfk * F6P_p * ATP_p
        r_ald = k_ald_f * FBP_p - k_ald_r * GAP_p * DHAP_p
        r_tpi = k_tpi_f * DHAP_p - k_tpi_r * GAP_p
        r_gapdh = a_gapdh * k_gapdh * GAP_p * NAD_p
        r_pgk = a_pgk * k_pgk * BPG13_p * ADP_p
        r_pgm = k_pgm_f * PG3_p - k_pgm_r * PG2_p
        r_eno = a_eno * k_eno * PG2_p
        r_pk = a_pk * k_pk * PEP_p * ADP_p

        # Branches and dissipative terms.
        r_ldh = k_ldh * Pyr_p * NADH_p
        r_pyr_sink = k_pyr_sink * Pyr_p
        r_atpase = k_atpase * ATP_p
        r_nadh_ox = k_nadh_ox * NADH_p

        l_G6P = k_leak_hex * G6P_p
        l_F6P = k_leak_hex * F6P_p
        l_FBP = k_leak_hex * FBP_p
        l_GAP = k_leak_tri * GAP_p
        l_DHAP = k_leak_tri * DHAP_p
        l_BPG13 = k_leak_lower * BPG13_p
        l_PG3 = k_leak_lower * PG3_p
        l_PG2 = k_leak_lower * PG2_p
        l_PEP = k_leak_lower * PEP_p

        # State equations.
        dGlc = -r_hk
        dG6P = r_hk - r_pgi - l_G6P
        dF6P = r_pgi - r_pfk - l_F6P
        dFBP = r_pfk - r_ald - l_FBP
        dGAP = r_ald + r_tpi - r_gapdh - l_GAP
        dDHAP = r_ald - r_tpi - l_DHAP
        dBPG13 = r_gapdh - r_pgk - l_BPG13
        dPG3 = r_pgk - r_pgm - l_PG3
        dPG2 = r_pgm - r_eno - l_PG2
        dPEP = r_eno - r_pk - l_PEP
        dPyr = r_pk - r_ldh - r_pyr_sink
        dLac = r_ldh

        dATP = -r_hk - r_pfk + r_pgk + r_pk - r_atpase
        dADP = r_hk + r_pfk - r_pgk - r_pk + r_atpase
        dNAD = -r_gapdh + r_ldh + r_nadh_ox
        dNADH = r_gapdh - r_ldh - r_nadh_ox

        dI_HK = -k_I_HK_decay * I_HK_p
        dI_PFK = -k_I_PFK_decay * I_PFK_p
        dI_GAPDH = -k_I_GAPDH_decay * I_GAPDH_p
        dI_PGK = -k_I_PGK_decay * I_PGK_p
        dI_ENO = -k_I_ENO_decay * I_ENO_p
        dI_PK = -k_I_PK_decay * I_PK_p

        return torch.stack(
            (
                dGlc, dG6P, dF6P, dFBP, dGAP, dDHAP,
                dBPG13, dPG3, dPG2, dPEP, dPyr, dLac,
                dATP, dADP, dNAD, dNADH,
                dI_HK, dI_PFK, dI_GAPDH, dI_PGK, dI_ENO, dI_PK,
            ),
            dim=-1,
        )


# =============================================================================
# 2) REDUCED 12-state scaffold
# =============================================================================

class GlycolysisReduced12Scaffold(MechanisticScaffold):
    """
    Reduced glycolysis scaffold with coarse pathway pools.

    States (12):
      0  Glc
      1  HexP   coarse G6P/F6P/FBP pool
      2  TriP   coarse GAP/DHAP pool
      3  BPG13
      4  PG3
      5  PEP
      6  Pyr
      7  Lac
      8  ATP
      9  ADP
      10 NAD
      11 NADH

    Observed species:
      Glc, Pyr, ATP, NADH

    Observed indices:
      [0, 6, 8, 11]

    Inhibitor boluses are not states in this reduced model. They should be passed to the
    encoder input history so that theta(t) can absorb their hidden effect.
    """

    def __init__(self):
        super().__init__(P=12, theta_dim=12)
        self.state_names = [
            "Glc", "HexP", "TriP", "BPG13", "PG3", "PEP",
            "Pyr", "Lac", "ATP", "ADP", "NAD", "NADH",
        ]
        self.theta_lo_vec = [1e-5] * 12
        self.theta_hi_vec = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Glc, HexP, TriP, BPG13, PG3, PEP, Pyr, Lac, ATP, ADP, NAD, NADH = y.unbind(dim=-1)

        (
            k_hk_eff,
            k_commit_eff,
            k_gapdh_eff,
            k_pgk_eff,
            k_lower_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_hex_leak,
            k_tri_leak,
        ) = theta.unbind(dim=-1)

        Glc_p = torch.clamp_min(Glc, 0.0)
        HexP_p = torch.clamp_min(HexP, 0.0)
        TriP_p = torch.clamp_min(TriP, 0.0)
        BPG13_p = torch.clamp_min(BPG13, 0.0)
        PG3_p = torch.clamp_min(PG3, 0.0)
        PEP_p = torch.clamp_min(PEP, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        ADP_p = torch.clamp_min(ADP, 0.0)
        NAD_p = torch.clamp_min(NAD, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        (
            k_hk_eff,
            k_commit_eff,
            k_gapdh_eff,
            k_pgk_eff,
            k_lower_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_hex_leak,
            k_tri_leak,
        ) = [torch.clamp_min(p, 0.0) for p in (
            k_hk_eff, k_commit_eff, k_gapdh_eff, k_pgk_eff,
            k_lower_eff, k_pk_eff, k_ldh_eff, k_pyr_sink,
            k_atpase, k_nadh_ox, k_hex_leak, k_tri_leak,
        )]

        r_hk = k_hk_eff * Glc_p * ATP_p
        r_commit = k_commit_eff * HexP_p * ATP_p
        r_gapdh = k_gapdh_eff * TriP_p * NAD_p
        r_pgk = k_pgk_eff * BPG13_p * ADP_p
        r_lower = k_lower_eff * PG3_p
        r_pk = k_pk_eff * PEP_p * ADP_p
        r_ldh = k_ldh_eff * Pyr_p * NADH_p

        r_pyr_sink = k_pyr_sink * Pyr_p
        r_atpase = k_atpase * ATP_p
        r_nadh_ox = k_nadh_ox * NADH_p
        r_hex_leak = k_hex_leak * HexP_p
        r_tri_leak = k_tri_leak * TriP_p

        dGlc = -r_hk
        dHexP = r_hk - r_commit - r_hex_leak
        dTriP = 2.0 * r_commit - r_gapdh - r_tri_leak
        dBPG13 = r_gapdh - r_pgk
        dPG3 = r_pgk - r_lower
        dPEP = r_lower - r_pk
        dPyr = r_pk - r_ldh - r_pyr_sink
        dLac = r_ldh

        dATP = -r_hk - r_commit + r_pgk + r_pk - r_atpase
        dADP = r_hk + r_commit - r_pgk - r_pk + r_atpase
        dNAD = -r_gapdh + r_ldh + r_nadh_ox
        dNADH = r_gapdh - r_ldh - r_nadh_ox

        return torch.stack(
            (dGlc, dHexP, dTriP, dBPG13, dPG3, dPEP, dPyr, dLac, dATP, dADP, dNAD, dNADH),
            dim=-1,
        )


# =============================================================================
# 3) REDUCED 8-state scaffold
# =============================================================================

class GlycolysisReduced8Scaffold(MechanisticScaffold):
    """
    More compressed glycolysis scaffold.

    States (8):
      0 Glc
      1 SugarP   coarse phosphorylated sugar/triose pool
      2 PEP
      3 Pyr
      4 Lac
      5 ATP
      6 NAD
      7 NADH

    Observed species:
      Glc, Pyr, ATP, NADH

    Observed indices:
      [0, 3, 5, 7]
    """

    def __init__(self):
        super().__init__(P=8, theta_dim=9)
        self.state_names = ["Glc", "SugarP", "PEP", "Pyr", "Lac", "ATP", "NAD", "NADH"]
        self.theta_lo_vec = [1e-5] * 9
        self.theta_hi_vec = [50.0, 50.0, 50.0, 20.0, 10.0, 10.0, 10.0, 5.0, 5.0]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Glc, SugarP, PEP, Pyr, Lac, ATP, NAD, NADH = y.unbind(dim=-1)

        (
            k_hk_eff,
            k_path_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_sugar_leak,
            k_pep_leak,
        ) = theta.unbind(dim=-1)

        Glc_p = torch.clamp_min(Glc, 0.0)
        SugarP_p = torch.clamp_min(SugarP, 0.0)
        PEP_p = torch.clamp_min(PEP, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        NAD_p = torch.clamp_min(NAD, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        (
            k_hk_eff,
            k_path_eff,
            k_pk_eff,
            k_ldh_eff,
            k_pyr_sink,
            k_atpase,
            k_nadh_ox,
            k_sugar_leak,
            k_pep_leak,
        ) = [torch.clamp_min(p, 0.0) for p in (
            k_hk_eff, k_path_eff, k_pk_eff, k_ldh_eff,
            k_pyr_sink, k_atpase, k_nadh_ox, k_sugar_leak, k_pep_leak,
        )]

        r_hk = k_hk_eff * Glc_p * ATP_p
        r_path = k_path_eff * SugarP_p * NAD_p
        r_pk = k_pk_eff * PEP_p
        r_ldh = k_ldh_eff * Pyr_p * NADH_p

        r_pyr_sink = k_pyr_sink * Pyr_p
        r_atpase = k_atpase * ATP_p
        r_nadh_ox = k_nadh_ox * NADH_p
        r_sugar_leak = k_sugar_leak * SugarP_p
        r_pep_leak = k_pep_leak * PEP_p

        dGlc = -r_hk
        dSugarP = r_hk - r_path - r_sugar_leak
        dPEP = r_path - r_pk - r_pep_leak
        dPyr = r_pk - r_ldh - r_pyr_sink
        dLac = r_ldh

        dATP = -r_hk + r_pk - r_atpase
        dNAD = -r_path + r_ldh + r_nadh_ox
        dNADH = r_path - r_ldh - r_nadh_ox

        return torch.stack((dGlc, dSugarP, dPEP, dPyr, dLac, dATP, dNAD, dNADH), dim=-1)


# =============================================================================
# 4) REDUCED 4-state scaffold
# =============================================================================

class GlycolysisReduced4Scaffold(MechanisticScaffold):
    """
    Observed-state-only reduced glycolysis scaffold.

    States (4):
      0 Glc
      1 Pyr
      2 ATP
      3 NADH

    Observed species:
      Glc, Pyr, ATP, NADH

    Observed indices:
      [0, 1, 2, 3]

    All pathway intermediates, cofactors other than ATP/NADH, lactate, and inhibitor pools
    are omitted. Inhibitor additions should be passed only to the causal encoder.
    """

    def __init__(self):
        super().__init__(P=4, theta_dim=7)
        self.state_names = ["Glc", "Pyr", "ATP", "NADH"]

        # All bounds positive: ATP and NADH are net *produced* from glucose
        # breakdown (2 ATP, 2 NADH per glucose), so k_atp_eff and k_nadh_eff
        # are sign-fixed positive. (Earlier signed bounds broke `log_gamma`,
        # which requires lo and hi to share sign.)
        self.theta_lo_vec = [
            1e-5,   # k_glc_use
            1e-5,   # k_pyr_prod
            1e-5,   # k_pyr_sink
            1e-5,   # k_atp_eff
            1e-5,   # k_atp_loss
            1e-5,   # k_nadh_eff
            1e-5,   # k_nadh_loss
        ]

        self.theta_hi_vec = [
            50.0,  # k_glc_use
            50.0,  # k_pyr_prod
            20.0,  # k_pyr_sink
            10.0,  # k_atp_eff
            10.0,  # k_atp_loss
            10.0,  # k_nadh_eff
            10.0,  # k_nadh_loss
        ]

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        Glc, Pyr, ATP, NADH = y.unbind(dim=-1)

        (
            k_glc_use,
            k_pyr_prod,
            k_pyr_sink,
            k_atp_eff,
            k_atp_loss,
            k_nadh_eff,
            k_nadh_loss,
        ) = theta.unbind(dim=-1)

        Glc_p = torch.clamp_min(Glc, 0.0)
        Pyr_p = torch.clamp_min(Pyr, 0.0)
        ATP_p = torch.clamp_min(ATP, 0.0)
        NADH_p = torch.clamp_min(NADH, 0.0)

        k_glc_use = torch.clamp_min(k_glc_use, 0.0)
        k_pyr_prod = torch.clamp_min(k_pyr_prod, 0.0)
        k_pyr_sink = torch.clamp_min(k_pyr_sink, 0.0)
        k_atp_loss = torch.clamp_min(k_atp_loss, 0.0)
        k_nadh_loss = torch.clamp_min(k_nadh_loss, 0.0)

        r_glc = k_glc_use * Glc_p
        r_pyr = k_pyr_prod * Glc_p
        r_sink = k_pyr_sink * Pyr_p

        dGlc = -r_glc
        dPyr = r_pyr - r_sink

        # Signed effective closure terms.
        dATP = k_atp_eff * Glc_p - k_atp_loss * ATP_p
        dNADH = k_nadh_eff * Glc_p - k_nadh_loss * NADH_p

        return torch.stack((dGlc, dPyr, dATP, dNADH), dim=-1)


# =============================================================================
# Registry and metadata
# =============================================================================

SCAFFOLDS: Dict[str, MechanisticScaffold] = {
    "glycolysis_oracle22": GlycolysisOracle22Scaffold(),
    "glycolysis_reduced12": GlycolysisReduced12Scaffold(),
    "glycolysis_reduced8": GlycolysisReduced8Scaffold(),
    "glycolysis_reduced4": GlycolysisReduced4Scaffold(),
}


GLYCOLYSIS_ORACLE22_INPUT_MAP: Dict[str, int] = {
    "glucose_bolus": 0,
    "atp_bolus": 12,
    "adp_bolus": 13,
    "nad_bolus": 14,
    "nadh_bolus": 15,
    "hk_inhib_bolus": 16,
    "pfk_inhib_bolus": 17,
    "gapdh_inhib_bolus": 18,
    "pgk_inhib_bolus": 19,
    "eno_inhib_bolus": 20,
    "pk_inhib_bolus": 21,
}

GLYCOLYSIS_REDUCED12_INPUT_MAP: Dict[str, int] = {
    "glucose_bolus": 0,
    "atp_bolus": 8,
    "adp_bolus": 9,
    "nad_bolus": 10,
    "nadh_bolus": 11,
    # inhibitor boluses are encoder inputs only
}

GLYCOLYSIS_REDUCED8_INPUT_MAP: Dict[str, int] = {
    "glucose_bolus": 0,
    "atp_bolus": 5,
    "nad_bolus": 6,
    "nadh_bolus": 7,
    # inhibitor boluses are encoder inputs only
}

GLYCOLYSIS_REDUCED4_INPUT_MAP: Dict[str, int] = {
    "glucose_bolus": 0,
    "atp_bolus": 2,
    "nadh_bolus": 3,
    # NAD, ADP, lactate, pathway intermediates, and inhibitors are encoder inputs only
}


GLYCOLYSIS_INPUT_NAMES: List[str] = [
    "glucose_bolus",
    "atp_bolus",
    "adp_bolus",
    "nad_bolus",
    "nadh_bolus",
    "hk_inhib_bolus",
    "pfk_inhib_bolus",
    "gapdh_inhib_bolus",
    "pgk_inhib_bolus",
    "eno_inhib_bolus",
    "pk_inhib_bolus",
]


GLYCOLYSIS_ORACLE22_OBS_INDICES: List[int] = [0, 10, 12, 15]  # Glc, Pyr, ATP, NADH
GLYCOLYSIS_REDUCED12_OBS_INDICES: List[int] = [0, 6, 8, 11]   # Glc, Pyr, ATP, NADH
GLYCOLYSIS_REDUCED8_OBS_INDICES: List[int] = [0, 3, 5, 7]     # Glc, Pyr, ATP, NADH
GLYCOLYSIS_REDUCED4_OBS_INDICES: List[int] = [0, 1, 2, 3]     # Glc, Pyr, ATP, NADH

GLYCOLYSIS_OBS_NAMES: List[str] = ["Glc", "Pyr", "ATP", "NADH"]


def get_glycolysis_scaffold(name: str) -> MechanisticScaffold:
    if name not in SCAFFOLDS:
        valid = ", ".join(sorted(SCAFFOLDS.keys()))
        raise KeyError(f"Unknown scaffold '{name}'. Valid options are: {valid}")
    return SCAFFOLDS[name]


if __name__ == "__main__":
    batch_size = 4

    for name, scaffold in SCAFFOLDS.items():
        y = torch.rand(batch_size, scaffold.P)
        theta = torch.rand(batch_size, scaffold.theta_dim)
        dy = scaffold(y, theta)
        print(name, "state:", y.shape, "theta:", theta.shape, "dy:", dy.shape)