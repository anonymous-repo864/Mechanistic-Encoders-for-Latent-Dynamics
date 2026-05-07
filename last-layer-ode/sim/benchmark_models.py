# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 23:44:23 2026

@author: bobva
"""
import numpy as np

def FullModel(t: float, y: np.ndarray, k: np.ndarray, dim = False) -> np.ndarray:
    """
    Full 13-species system (A-M) with species and rates unpacked at the start.
    If you look at the reduced model, you will find that that description
    of this chain reaction skips 2 states for compared to the full model.
    These states can hide the 
    """        
    if dim == True:
        states = 13
        parameters = 19
        names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'] 
        return states, parameters, names
    
    # Unpack species
    A, B, C, D, E, F, G, H, I, J, K, L, M = y
    
    # Unpack  rates
    kf1, kf2, kf3, kf4, kf5, kf6, kf7, kf8, kf9, kf10, kf11, kf12, kr1, kr3, kr5, kr7, kr9, kr11, kr12 = k
    
  
    # ODE Equations
    dA = -kf1*A + kr1*B
    dB =  kf1*A - kr1*B - kf2*B # some of A is converted to B (kf1), some of B converts back to A (-kr1), some of B converts to C (-kf2)
    dC =  kf2*B - kf3*C + kr3*D # etc. etc.
    dD =  kf3*C - kr3*D - kf4*D
    dE =  kf4*D - kf5*E + kr5*F
    dF =  kf5*E - kr5*F - kf6*F
    dG =  kf6*F - kf7*G + kr7*H
    dH =  kf7*G - kr7*H - kf8*H
    dI =  kf8*H - kf9*I + kr9*J
    dJ =  kf9*I - kr9*J - kf10*J
    dK =  kf10*J - kf11*K + kr11*L
    dL =  kf11*K - kr11*L - kf12*L + kr12*M 
    dM =  kf12*L - kr12*M


    return np.array([dA, dB, dC, dD, dE, dF, dG, dH, dI, dJ, dK, dL, dM], dtype=float)


def ReducedModel(t, y, k, dim = False):
    """
    y: [A, D, G, J, M] (5 species)
    Kf: [Kf1, Kf2, Kf3, Kf4] forward lumped rates
    Kr: [Kr1, Kr2, Kr3] reverse lumped rates
    """       
    if dim == True:
        states = 5
        parameters = 8
        return states, parameters
    
    
    A, D, G, J, M = y
    kf1, kf2, kf3, kf4, kr1, kr2, kr3,kr4 = k
    
    # A <-> D <-> G <-> J -> M
    # last step assumed irreversible

    # A <-> D
    dA = -kf1 * A + kr1 * D
    
    # D <-> G
    dD =  kf1 * A - kr1 * D - kf2 * D + kr2 * G
    
    # G <-> J
    dG =  kf2 * D - kr2 * G - kf3 * G + kr3 * J
    
    # J -> M (Assuming terminal step is still irreversible)
    # dJ =  kf3 * G - kr3 * J - kf4 * J + - kr4 * M # original, extra minus sign?
    dJ =  kf3 * G - kr3 * J - kf4 * J + kr4 * M
    # but then if we assume irreversible: then there should be no backward rate from M to J right? 

    dM =  kf4 * J  - kr4 * M

    return np.array([dA, dD, dG, dJ, dM], dtype=float)