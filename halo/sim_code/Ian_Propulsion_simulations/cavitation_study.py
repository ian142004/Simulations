#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cavitation / flashing risk study for a self-pressurizing N2O feedline.

What it does
------------
Given: tank T (thus Psat), optional supercharge margin, mdot targets,
tube size, length, and a set of minor-loss components,
it computes:
  - major/minor pressure losses to the control valve
  - upstream-of-valve pressure P_up
  - "flashing margin" M = P_up - Psat(T)
  - cavitation number in the pipe: sigma_pipe = (P_up - Psat) / (0.5*rho*v^2)
  - NPSH_available = (P_up - Psat) / (rho*g)
  - minimum supercharge needed to keep single-phase at the valve: dP_supercharge_req = max(0, ΔP_line + margin)

Notes
-----
• In a purely self-pressurized system P_tank ≈ Psat(T). Any upstream pressure loss
  makes P_up < Psat → two-phase upstream of the valve. To keep single phase,
  most teams add supercharge (He/Ar/N2) above Psat; this tool tells you how much.

• Uses single-phase liquid hydraulics for the line estimate; once flashing begins,
  the real pressure drop will differ, but this gives a conservative first look and
  the **required supercharge** to avoid it.

• CoolProp is used if available; otherwise a compact Airgas-style saturation table is used
  together with an empirical μ_l(T) curve.

Author: ChatGPT for Ian (UCSB)
"""

import math
import numpy as np

# =============================
# N2O properties (same style as your model)
# =============================
class N2OProps:
    def __init__(self):
        self.has_coolprop = False
        try:
            import CoolProp.CoolProp as CP
            self.CP = CP
            self.has_coolprop = True
        except Exception:
            self.CP = None
            self.has_coolprop = False

        if not self.has_coolprop:
            # Airgas-like fallback table (saturation)
            F = np.array([0, 10, 20, 32, 40, 50, 60, 70, 80, 97], float)
            P_psia = np.array([283.0, 335.0, 387.0, 460.0, 520.0, 590.0, 675.0, 760.0, 865.0, 1069.0])
            rho_l  = np.array([63.1, 61.2, 59.2, 57.0, 54.7, 52.3, 49.2, 46.5, 40.0, 26.5])
            v_v    = np.array([0.303, 0.262, 0.217, 0.1785, 0.160, 0.119, 0.119, 0.106, 0.080, 0.0377])
            h_fg   = np.array([123.0,118.5,113.4,106.8,101.5,93.7,86.2,77.7,69.0,0.0])

            self.TK_grid  = (F - 32.0) * 5.0/9.0 + 273.15
            self.P_grid   = P_psia * 6894.757293168
            self.rhoL_grid= rho_l  * 16.01846337396
            v_m3kg        = v_v * 0.028316846592 / 0.45359237
            self.rhoV_grid= 1.0 / v_m3kg
            self.hfg_grid = h_fg * 2326.0

            self.T_min = float(self.TK_grid[0])
            self.T_max = float(self.TK_grid[-1])
        else:
            self.Pc = self.CP.PropsSI('Pcrit','',0,'',0,'N2O')
            self.Tc = self.CP.PropsSI('Tcrit','',0,'',0,'N2O')
            self.T_min = 250.0
            self.T_max = float(self.Tc - 0.5)

    def _interp(self, T, xgrid):
        return float(np.interp(T, self.TK_grid, xgrid))

    def Psat(self, T):
        if self.has_coolprop:
            return float(self.CP.PropsSI('P','T',T,'Q',0,'N2O'))
        T = float(np.clip(T, self.T_min, self.T_max))
        return self._interp(T, self.P_grid)

    def rho_l(self, T):
        if self.has_coolprop:
            return float(self.CP.PropsSI('D','T',T,'Q',0,'N2O'))
        T = float(np.clip(T, self.T_min, self.T_max))
        return self._interp(T, self.rhoL_grid)

    def mu_l(self, T):
        # Try CoolProp with (T,P) because (T,Q) viscosity is missing in some builds
        if self.has_coolprop:
            try:
                Ps = self.CP.PropsSI('P','T',T,'Q',0,'N2O')
                try:
                    return float(self.CP.PropsSI('V','T',T,'P',Ps,'N2O'))
                except Exception:
                    return float(self.CP.PropsSI('V','T',T,'P',Ps,'NitrousOxide'))
            except Exception:
                pass
        # fallback ~5.8e-4 Pa·s @ 20 °C with mild T dependence
        return 5.8e-4 * math.exp(-0.02 * (T - 293.15))

# =============================
# Standard tube IDs (316SS, drawn)
# =============================
def get_tube_spec(name: str):
    eps = 1.5e-6  # m, drawn SS roughness
    table = {
        "1/4_OD_0.035_wall": {"ID_m": (0.25 - 2*0.035)*0.0254},  # ~4.57 mm
        "3/8_OD_0.035_wall": {"ID_m": (0.375 - 2*0.035)*0.0254}, # ~7.75 mm
        "1/2_OD_0.035_wall": {"ID_m": (0.5 - 2*0.035)*0.0254},   # ~10.92 mm
        "1/2_OD_0.049_wall": {"ID_m": (0.5 - 2*0.049)*0.0254},   # ~10.21 mm
        "5/8_OD_0.049_wall": {"ID_m": (0.625 - 2*0.049)*0.0254}, # ~13.18 mm
    }
    if name not in table:
        raise ValueError(f"Unknown tube spec '{name}'. Options: {list(table.keys())}")
    return float(table[name]["ID_m"]), eps

# =============================
# Minor loss coefficients (typical)
# =============================
def K_ball_valve_full_open(): return 0.05
def K_globe_valve_full_open(): return 10.0
def K_check_valve_swing():    return 2.0
def K_elbow_90_long():        return 0.3
def K_elbow_45():             return 0.2
def K_tee_run_through():      return 0.2
def K_tee_branch():           return 1.0
def K_inlet_sharp():          return 0.5
def K_exit_free():            return 1.0
def K_sudden_enlargement():   return 0.5
def K_sudden_contraction():   return 0.5

def K_sum(counts: dict):
    total = 0.0
    total += counts.get("inlet", 0)*K_inlet_sharp()
    total += counts.get("ball_valve", 0)*K_ball_valve_full_open()
    total += counts.get("globe_valve", 0)*K_globe_valve_full_open()
    total += counts.get("check_valve", 0)*K_check_valve_swing()
    total += counts.get("elbow_90", 0)*K_elbow_90_long()
    total += counts.get("elbow_45", 0)*K_elbow_45()
    total += counts.get("tee_run", 0)*K_tee_run_through()
    total += counts.get("tee_branch", 0)*K_tee_branch()
    total += counts.get("sudden_enlargement", 0)*K_sudden_enlargement()
    total += counts.get("sudden_contraction", 0)*K_sudden_contraction()
    total += counts.get("exit", 0)*K_exit_free()
    return total

# =============================
# Friction factor (Churchill)
# =============================
def friction_factor_churchill(Re, eps, D):
    if Re <= 0.0: return 0.0
    A = (-2.457*math.log((7.0/Re)**0.9 + 0.27*(eps/D)))**16
    B = (37530.0/Re)**16
    return float(8.0*((8.0/Re)**12 + 1.0/(A+B)**1.5)**(1.0/12.0))

# =============================
# Hydraulics
# =============================
def dp_major_darcy(mdot, rho, mu, D, L, eps):
    A = math.pi*D**2/4.0
    v = mdot / max(rho*A, 1e-12)
    Re = rho*v*D / max(mu, 1e-12)
    if Re < 2300.0:
        f = 64.0/max(Re, 1e-6)
    else:
        f = friction_factor_churchill(Re, eps, D)
    dp = f * (L/max(D,1e-12)) * 0.5*rho*v*v
    return float(dp), float(Re), float(v), float(f)

def dp_minor_K(mdot, rho, D, K_total):
    A = math.pi*D**2/4.0
    v = mdot / max(rho*A, 1e-12)
    return float(K_total * 0.5*rho*v*v)

# =============================
# Study core
# =============================
def cavitation_report(
    T=293.15,                          # K (≈20 °C)
    mdot_list=(0.80, 0.60, 0.40),      # kg/s
    tube_specs=("3/8_OD_0.035_wall", "1/2_OD_0.035_wall", "1/2_OD_0.049_wall"),
    L_pipe=1.5,                        # m
    components={"inlet":1, "elbow_45":2, "ball_valve":1},
    P_back=101325.0,                   # Pa
    supercharge_overpressure=0.0,      # Pa above Psat(T)
    single_phase_margin=0.0            # Pa extra margin desired above Psat at the valve
):
    g = 9.80665
    props = N2OProps()
    Ps = props.Psat(T)
    rho = props.rho_l(T)
    mu  = props.mu_l(T)
    P_tank = Ps + supercharge_overpressure
    Ktot = K_sum(components)

    rows = []
    for spec in tube_specs:
        D, eps = get_tube_spec(spec)
        for mdot in mdot_list:
            # Line losses
            dp_maj, Re, v, f = dp_major_darcy(mdot, rho, mu, D, L_pipe, eps)
            dp_min = dp_minor_K(mdot, rho, D, Ktot)
            dP_line = dp_maj + dp_min

            # Upstream of valve pressure and margins
            P_up = max(P_tank - dP_line, 1.0)
            flashing_margin = P_up - Ps               # < 0 means flashing/two-phase upstream
            NPSH_avail = flashing_margin / (rho * g)  # meters of liquid head
            sigma_pipe = flashing_margin / max(0.5*rho*v*v, 1e-9)

            # If you want single-phase to the valve with 'single_phase_margin' above Psat:
            dP_supercharge_req = max(0.0, dP_line + single_phase_margin - supercharge_overpressure)

            rows.append({
                "tube_spec": spec,
                "ID_mm": D*1e3,
                "L_m": L_pipe,
                "mdot_kg_s": mdot,
                "rho_kg_m3": rho,
                "mu_mPa_s": mu*1e3,
                "Re": Re,
                "vel_m_s": v,
                "f": f,
                "dP_major_MPa": dp_maj*1e-6,
                "dP_minor_MPa": dp_min*1e-6,
                "dP_line_MPa": dP_line*1e-6,
                "P_tank_MPa": P_tank*1e-6,
                "Psat_MPa": Ps*1e-6,
                "P_up_MPa": P_up*1e-6,
                "flash_margin_MPa": flashing_margin*1e-6,
                "NPSH_m": NPSH_avail,
                "sigma_pipe": sigma_pipe,
                "supercharge_req_MPa": dP_supercharge_req*1e-6
            })
    return rows

def pretty_print(rows):
    # Group by spec
    by_spec = {}
    for r in rows:
        by_spec.setdefault(r["tube_spec"], []).append(r)

    print("\n=== Cavitation / Flashing Risk Summary ===")
    for spec, arr in by_spec.items():
        arr = sorted(arr, key=lambda x: -x["mdot_kg_s"])
        Dmm = arr[0]["ID_mm"]
        print(f"\n--- {spec}  (ID ≈ {Dmm:0.2f} mm) ---")
        for r in arr:
            print(f"mdot={r['mdot_kg_s']:0.2f} kg/s | Re={r['Re']:.0f} | v={r['vel_m_s']:0.1f} m/s | "
                  f"ΔP_line={r['dP_line_MPa']:0.3f} MPa (maj {r['dP_major_MPa']:0.3f}, min {r['dP_minor_MPa']:0.3f})")
            print(f"  P_up={r['P_up_MPa']:0.3f} MPa | Psat={r['Psat_MPa']:0.3f} MPa | "
                  f"flash_margin={r['flash_margin_MPa']:0.3f} MPa | NPSH={r['NPSH_m']:0.2f} m | σ_pipe={r['sigma_pipe']:0.2f}")
            print(f"  Supercharge needed for single-phase to valve (margin=0): {r['supercharge_req_MPa']:0.3f} MPa")
    print("\nInterpretation:")
    print("• flash_margin < 0 → liquid would flash in the line (two-phase upstream of valve).")
    print("• σ_pipe ≲ 1–2 and/or NPSH small → high cavitation susceptibility in valve/geometry.")
    print("• Add supercharge (He/Ar/N2) so P_tank = Psat + ΔP_supercharge ≥ Psat + ΔP_line + margin.")
    print("• Or increase line ID and reduce fittings to push ΔP_line down.")

if __name__ == "__main__":
    # Example: your numbers (20 °C tank, no supercharge yet, 1.5 m, 2x 45° + ball valve + sharp inlet)
    rows = cavitation_report(
        T=293.15,
        mdot_list=(0.80, 0.60, 0.40),
        tube_specs=("3/8_OD_0.035_wall", "1/2_OD_0.035_wall", "1/2_OD_0.049_wall"),
        L_pipe=1.5,
        components={"inlet":1, "elbow_45":2, "ball_valve":1},
        P_back=101325.0,
        supercharge_overpressure=0.0,      # try e.g. 2.0e6 (≈20 bar) to see the effect
        single_phase_margin=0.0
    )
    pretty_print(rows)

    # Quick pick heuristic: for mdot=0.8 kg/s, choose the smallest tube that needs <= 0.3 MPa supercharge
    target_mdot = 0.80
    cap = 0.3  # MPa
    candidates = [r for r in rows if abs(r["mdot_kg_s"]-target_mdot)<1e-9 and r["supercharge_req_MPa"] <= cap]
    if candidates:
        best = sorted(candidates, key=lambda r: r["ID_mm"])[0]
        print(f"\n>>> Suggested line (heuristic): {best['tube_spec']} (ID≈{best['ID_mm']:.2f} mm) "
              f"meets single-phase at {target_mdot:.2f} kg/s with ≤{cap:.1f} bar supercharge.")
    else:
        need = min([r["supercharge_req_MPa"] for r in rows if abs(r["mdot_kg_s"]-target_mdot)<1e-9])
        print(f"\n>>> No listed tube keeps single-phase at {target_mdot:.2f} kg/s without more supercharge. "
              f"Minimum needed ≈ {need:.2f} MPa. Consider bigger line or add pressurization.")
