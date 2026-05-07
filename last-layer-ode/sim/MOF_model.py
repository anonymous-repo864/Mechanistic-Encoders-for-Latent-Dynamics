import numpy as np


def MOF_Synthesis(t: float, y: np.ndarray, k: np.ndarray, dim: bool = False):
    """
    12-state MOF synthesis model.

    Models the competing amorphous vs. crystalline MOF formation pathways,
    with modulator-controlled inhibition of crystalline growth.

    States (12):
      0  Met        : reactive metal in solution
      1  LigH       : protonated ligand (acid form)
      2  Lig_minus  : deprotonated ligand (binding-competent)
      3  H_plus     : proton concentration
      4  Base       : base (control input — drives deprotonation)
      5  Mod        : modulator (control input — caps SBUs, inhibits crystalline growth)
      6  SBU        : secondary building unit (free)
      7  SBU_capped : modulator-capped SBU (inactive)
      8  Nuc_A      : amorphous nuclei
      9  Am         : amorphous product
      10 Nuc_C      : crystalline nuclei
      11 MOF_C      : target crystalline MOF

    Parameters (16):
      0  k_deprot  : deprotonation rate (LigH + Base -> Lig- + ...)
      1  k_prot    : reprotonation rate
      2  k_oli     : oligomerization rate (Met + Lig- -> SBU)
      3  k_cap     : capping rate (SBU + Mod -> SBU_capped)
      4  k_uncap   : uncapping rate
      5  K_I       : modulator inhibition constant for crystalline growth
      6  knuc_A    : amorphous nucleation prefactor
      7  kgro_A    : amorphous growth rate
      8  kagg_A    : amorphous aggregation rate
      9  n_A       : amorphous nucleation exponent on SBU
      10 knuc_C    : crystalline nucleation prefactor
      11 kgro_C    : crystalline growth rate
      12 kagg_C    : crystalline aggregation rate
      13 n_C       : crystalline nucleation exponent on SBU
      14 a         : metal exponent in oligomerization
      15 b         : ligand exponent in oligomerization

    Supervisor default values (from MOF_synthesis.py):
      k_deprot=5.0, k_prot=1.0, k_oli=3.0, k_cap=2.0, k_uncap=0.5,
      K_I=0.1, knuc_A=10.0, kgro_A=1.0, kagg_A=1.0, n_A=3.0,
      knuc_C=0.5, kgro_C=4.0, kagg_C=1.0, n_C=1.5, a=1.0, b=1.0
    """
    if dim:
        states = 12
        parameters = 16
        names = [
            "Met", "LigH", "Lig_minus", "H_plus", "Base", "Mod",
            "SBU", "SBU_capped", "Nuc_A", "Am", "Nuc_C", "MOF_C",
        ]
        return states, parameters, names

    y = np.maximum(0.0, y)
    Met, LigH, Lig_minus, H_plus, Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C = y
    k_deprot, k_prot, k_oli, k_cap, k_uncap, K_I, knuc_A, kgro_A, kagg_A, n_A, knuc_C, kgro_C, kagg_C, n_C, a, b = k

    r_deprot = k_deprot * LigH * Base
    r_prot   = k_prot * Lig_minus * H_plus
    r_oli    = k_oli * (Met ** a) * (Lig_minus ** b)
    r_cap    = k_cap * SBU * Mod
    r_uncap  = k_uncap * SBU_capped
    r_nuc_A  = knuc_A * (SBU ** n_A)
    r_nuc_C  = knuc_C * (SBU ** n_C)
    r_gro_A  = kgro_A * SBU * Am
    r_agg_A  = kagg_A * (Nuc_A ** 2)
    inhibition_factor = K_I / (K_I + Mod + 1e-6)
    r_gro_C  = kgro_C * SBU * MOF_C * inhibition_factor
    r_agg_C  = kagg_C * (Nuc_C ** 2)

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

    return np.array([
        dMet, dLigH, dLig_minus, dH_plus, dBase, dMod,
        dSBU, dSBU_capped, dNuc_A, dAm, dNuc_C, dMOF_C,
    ], dtype=float)

# DOWN HERE IS THE RAW FILE, AS GIVEN BY THE SUPERVISOR:
# # -*- coding: utf-8 -*-
# """
# Created on Tue Mar 24 14:56:08 2026

# @author: bobva
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp

# # 1. Complex Kinetic Parameters
# params = {
#     'k_deprot': 5.0, 'k_prot': 1.0, 'k_oli': 3.0, 
#     'k_cap': 2.0, 'k_uncap': 0.5, 'K_I': 0.1, 
#     'knuc_A': 10.0, 'kgro_A': 1.0, 'kagg_A': 1.0, 'n_A': 3.0, 
#     'knuc_C': 0.5, 'kgro_C': 4.0, 'kagg_C': 1.0, 'n_C': 1.5, 
#     'a': 1.0, 'b': 1.0
# }

# # 2. Guardrailed ODE System
# def complex_mof_odes(t, y, p):
#     # Guardrail: prevent negative concentrations to avoid NaN in fractional powers
#     y = np.maximum(0.0, y)
    
#     Met, LigH, Lig_minus, H_plus, Base, Mod, SBU, SBU_capped, Nuc_A, Am, Nuc_C, MOF_C = y
    
#     r_deprot = p['k_deprot'] * LigH * Base
#     r_prot   = p['k_prot'] * Lig_minus * H_plus
#     r_oli = p['k_oli'] * (Met**p['a']) * (Lig_minus**p['b'])
#     r_cap   = p['k_cap'] * SBU * Mod
#     r_uncap = p['k_uncap'] * SBU_capped
#     r_nuc_A = p['knuc_A'] * (SBU**p['n_A'])
#     r_nuc_C = p['knuc_C'] * (SBU**p['n_C'])
#     r_gro_A = p['kgro_A'] * SBU * Am
#     r_agg_A = p['kagg_A'] * (Nuc_A**2)
#     inhibition_factor = p['K_I'] / (p['K_I'] + Mod + 1e-6)
#     r_gro_C = p['kgro_C'] * SBU * MOF_C * inhibition_factor
#     r_agg_C = p['kagg_C'] * (Nuc_C**2)
    
#     dMet = -r_oli
#     dLigH = -r_deprot + r_prot
#     dLig_minus = r_deprot - r_prot - r_oli
#     dH_plus = r_deprot - r_prot + r_oli
#     dBase = -r_deprot
#     dMod = -r_cap + r_uncap
#     dSBU = r_oli - r_cap + r_uncap - r_nuc_A - r_gro_A - r_nuc_C - r_gro_C
#     dSBU_capped = r_cap - r_uncap
#     dNuc_A = r_nuc_A - r_agg_A
#     dAm = r_agg_A + r_gro_A
#     dNuc_C = r_nuc_C - r_agg_C
#     dMOF_C = r_agg_C + r_gro_C
    
#     return [dMet, dLigH, dLig_minus, dH_plus, dBase, dMod, dSBU, dSBU_capped, dNuc_A, dAm, dNuc_C, dMOF_C]

# # 3. Simulation Engine
# def simulate_schedule(schedule, t_end=30):
#     initial_state = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
#     current_state = np.array(initial_state)
#     schedule = sorted(schedule, key=lambda x: x[0])
    
#     t_eval_all, y_out_all = [], []
#     start_t = 0.0
    
#     for event_t, base_dose, mod_dose in schedule:
#         if event_t > start_t:
#             sol = solve_ivp(complex_mof_odes, [start_t, event_t], current_state, args=(params,), method='BDF')
#             t_eval_all.append(sol.t[:-1])
#             y_out_all.append(sol.y[:, :-1])
#             current_state = np.maximum(0.0, sol.y[:, -1])
        
#         current_state[4] += base_dose 
#         current_state[5] += mod_dose  
#         start_t = event_t
    
#     if start_t < t_end:
#         sol = solve_ivp(complex_mof_odes, [start_t, t_end], current_state, args=(params,), method='BDF')
#         t_eval_all.append(sol.t)
#         y_out_all.append(sol.y)
        
#     return np.concatenate(t_eval_all), np.concatenate(y_out_all, axis=1)

# # --- CALCULATE THE BASELINE (ALL AT T=0) ---
# baseline_schedule = [(0, 2.0, 1.0)]
# t_base, y_base = simulate_schedule(baseline_schedule)

# # 4. Define the Time-Varying Strategies
# schedules = {
#     "1. Static Batch (The Baseline)": [(0, 2.0, 1.0)],
#     "2. Fast Drip Base, No Modulator": [(0, 0.5, 0), (2, 0.5, 0), (4, 0.5, 0), (6, 0.5, 0)],
#     "3. Slow Drip Base, Late Modulator": [(0, 0.5, 0), (5, 0.5, 0), (10, 0.5, 0), (15, 0.5, 1.0)],
#     "4. Slow Drip Base, Early Modulator": [(0, 0.5, 1.0), (5, 0.5, 0), (10, 0.5, 0), (15, 0.5, 0)],
#     "5. Two Big Halves": [(0, 1.0, 0.5), (10, 1.0, 0.5)],
#     "6. Alternating Spikes": [(0, 1.0, 0), (5, 0, 1.0), (10, 1.0, 0)]
# }

# # 5. Execute and Plot
# fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
# axes = axes.flatten()

# for idx, (title, schedule) in enumerate(schedules.items()):
#     t, y = simulate_schedule(schedule)
#     ax = axes[idx]
    
#     # Plot Baseline (Static Batch at t=0) as dashed lines
#     ax.plot(t_base, y_base[9], color='salmon', linestyle='--', linewidth=2, label='Amorphous (Static Batch Baseline)' if idx==1 else "")
#     ax.plot(t_base, y_base[11], color='lightgreen', linestyle='--', linewidth=2, label='Target MOF (Static Batch Baseline)' if idx==1 else "")

#     # Plot Current Strategy as solid lines
#     ax.plot(t, y[9], label='Amorphous (Time-Programmed)', color='red', linewidth=3)
#     ax.plot(t, y[11], label='Target MOF (Time-Programmed)', color='green', linewidth=3)
    
#     # Mark input events visually
#     for event_t, b, m in schedule:
#         if b > 0: ax.axvline(event_t, color='blue', alpha=0.3, linestyle=':', label='Base added' if event_t==schedule[0][0] and idx==1 else "")
#         if m > 0: ax.axvline(event_t, color='purple', alpha=0.3, linestyle='-.', label='Mod added' if event_t==schedule[0][0] and idx==1 else "")
    
#     ax.set_title(title, fontsize=12, fontweight='bold')
#     ax.set_xlabel('Time')
#     if idx % 3 == 0: ax.set_ylabel('Concentration')
#     ax.grid(True, alpha=0.3)
    
#     # Add legend only to the second plot to keep it clean, but visible
#     if idx == 1:
#         handles, labels = ax.get_legend_handles_labels()
#         by_label = dict(zip(labels, handles))
#         ax.legend(by_label.values(), by_label.keys(), loc='center right', fontsize=8)

# plt.tight_layout()
# plt.show()