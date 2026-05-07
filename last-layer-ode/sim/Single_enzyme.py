
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 12:43:11 2026

@author: bob-van-sluijs
"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def SingleEnzyme(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """
    6-state Reduced Reversible Bi-Bi enzyme kinetics model.

    Reaction: A + B <-> C + D  (catalysed by enzyme E, inhibitor I inert)

    States (6):
      0  A  : substrate A
      1  B  : substrate B
      2  C  : product C
      3  D  : product D
      4  E  : enzyme (conserved, inert dynamics)
      5  I  : inhibitor / inert species

    Parameters (6):
      0  kcat_f : forward catalytic rate constant
      1  kcat_r : reverse catalytic rate constant
      2  Ka     : Michaelis constant for substrate A
      3  Kb     : Michaelis constant for substrate B
      4  Kc     : Michaelis constant for product C
      5  Kd     : Michaelis constant for product D

    Default values:
      kcat_f=50.0, kcat_r=5.0, Ka=20.0, Kb=20.0, Kc=20.0, Kd=20.0

    Control channels: A (idx 0) and B (idx 1) receive substrate bolus additions.
    Recommended: --t-span 10 --n-steps 200
    """
    if dim:
        states = 6
        parameters = 6
        names = ["A", "B", "C", "D", "E", "I"]
        return states, parameters, names

    y = np.maximum(0.0, y)
    A, B, C, D, E, I = y
    kcat_f, kcat_r, Ka, Kb, Kc, Kd = k

    eps = 1e-12
    Ka  = max(Ka,  eps)
    Kb  = max(Kb,  eps)
    Kc  = max(Kc,  eps)
    Kd  = max(Kd,  eps)

    Vf = kcat_f * E
    Vr = kcat_r * E

    D0    = Ka * Kb
    denom = (
        D0 * (1.0 + C / Kc + D / Kd + (C * D) / (Kc * Kd))
        + Kb * A * (1.0 + D / Kd)
        + Ka * B * (1.0 + C / Kc)
        + A * B
        + eps
    )

    v = (Vf * A * B - Vr * C * D) / denom

    dA = -v
    dB = -v
    dC =  v
    dD =  v
    dE =  0.0
    dI =  0.0

    return np.array([dA, dB, dC, dD, dE, dI], dtype=float)

# DOWN HERE IS THE RAW FILE, AS GIVEN BY THE SUPERVISOR:
# class SingleEnzymeODE:
#     """Standardized wrapper for the ODE model."""
    
#     # 1. Define metadata once here
#     NAME = "Reduced Reversible Bi-Bi"
#     STATES = ["A", "B", "C", "D", "E", "I"]
#     PARAMS = ["kcat_f", "kcat_r", "Ka", "Kb", "Kc", "Kd"]
    
#     # 2. Define the (low, high) bounds for each parameter in the same order
#     PARAM_RANGES = [
#         (1.0, 200.0), # kcat_f
#         (1.0, 200.0), # kcat_r
#         (1.0, 200.0), # Ka
#         (1.0, 200.0), # Kb
#         (1.0, 200.0), # Kc
#         (1.0, 200.0), # Kd
#         ]
    
#     # 2. The JIT compiled math (Zero overhead, runs at C++ speed)
#     @staticmethod
#     @torch.jit.script
#     def rhs(y: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
#         # Unpack using the exact same order as above
#         A, B, C, D, E, I = y.unbind(dim=-1)
#         kcat_f, kcat_r, Ka, Kb, Kc, Kd = k.unbind(dim=-1)
        
#         eps: float = 1e-12

#         Apos = torch.clamp_min(A, 0.0)
#         Bpos = torch.clamp_min(B, 0.0)
#         Cpos = torch.clamp_min(C, 0.0)
#         Dpos = torch.clamp_min(D, 0.0)
#         Epos = torch.clamp_min(E, 0.0)

#         Ka = torch.clamp_min(Ka, eps)
#         Kb = torch.clamp_min(Kb, eps)
#         Kc = torch.clamp_min(Kc, eps)
#         Kd = torch.clamp_min(Kd, eps)

#         Vf = kcat_f * Epos
#         Vr = kcat_r * Epos

#         D0 = Ka * Kb
#         denom = (
#             D0 * (1.0 + Cpos / Kc + Dpos / Kd + (Cpos * Dpos) / (Kc * Kd))
#             + (Kb * Apos) * (1.0 + Dpos / Kd)
#             + (Ka * Bpos) * (1.0 + Cpos / Kc)
#             + (Apos * Bpos)
#             + eps
#         )

#         v = (Vf * Apos * Bpos - Vr * Cpos * Dpos) / denom

#         dA = -v 
#         dB = -v 
#         dC =  v 
#         dD =  v 
#         dE =  v-v 
#         dI =  v-v    
        
#         return torch.stack([dA, dB, dC, dD, dE, dI], dim=-1)
