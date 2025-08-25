
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2O self-pressurizing tank with closed-loop throttle valve (mdot tracking)
+ Two-node wall heat transfer (liquid-wet & vapor-wet wall)
+ Upstream feedline pressure losses (Darcy–Weisbach, minor K-losses, standard tubes)
+ Helium supercharge system with proportional (PI) regulator + metering orifice  <-- UPDATED
+ No-flashing controller: maintains P_up >= Psat + margin using He (starts with zero He)
+ Startup geometry/hydraulics summary printout

Updated: Injector drop modeled as a fraction of chamber pressure:
         ΔP_injector ≈ inj_dp_frac_of_Pc * Pc  (default 0.30 ≈ 30% of Pc)
         This ΔP is placed BETWEEN valve outlet and chamber (downstream of valve).

New (critical control updates):
- He loop is now CLOSED-LOOP: PI on P_tank -> regulator outlet pressure P_reg, with supply-aware clamps.
- mdot loop is now CLOSED-LOOP: PI trim on valve area around feedforward command.
- No-flash setpoint uses ACTUAL Pc (Pc_prev) and last-step ACTUAL line ΔP to build P_tank_sp.
- Supply-aware throttle soft clamp when He supply/regulator saturates.

Notes:
- Pc is computed from c*: Pc = (mdot_total * CSTAR) / At.
- mdot_total = mdot_ox / ox_mass_fraction. Default ox_mass_fraction=1.0 (backward compatible).
  If you later add a fuel line, set ox_mass_fraction to your OX/(OX+FUEL) mass fraction or compute mdot_total directly.

Author: ChatGPT for Ian (UCSB)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# =============================
# Thermophysical properties
# =============================
class N2OProps:
    """ Saturation properties of N2O (SI). Uses CoolProp if available; otherwise a compact table.
        Also provides a simple mu_l(T) fallback for liquid viscosity.
    """
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
            # Fallback table (Airgas-like): F, P[psia], rho_l[lb/ft^3], v_v[ft^3/lb], h_fg[BTU/lb]
            data_F = np.array([0, 10, 20, 32, 40, 50, 60, 70, 80, 97], float)
            P_psia = np.array([283.0, 335.0, 387.0, 460.0, 520.0, 590.0, 675.0, 760.0, 865.0, 1069.0])
            rho_l  = np.array([63.1, 61.2, 59.2, 57.0, 54.7, 52.3, 49.2, 46.5, 40.0, 26.5])
            v_v    = np.array([0.303, 0.262, 0.217, 0.1785, 0.160, 0.119, 0.119, 0.106, 0.080, 0.0377])
            h_fg   = np.array([123.0,118.5,113.4,106.8,101.5,93.7,86.2,77.7,69.0,0.0])

            self.TK_grid  = (data_F - 32.0) * (5.0/9.0) + 273.15
            self.P_grid   = P_psia * 6894.757293168
            self.rhoL_grid= rho_l * 16.01846337396
            v_m3kg = v_v * 0.028316846592 / 0.45359237
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

    def rho_v(self, T):
        if self.has_coolprop:
            return float(self.CP.PropsSI('D','T',T,'Q',1,'N2O'))
        T = float(np.clip(T, self.T_min, self.T_max))
        return self._interp(T, self.rhoV_grid)

    def h_fg(self, T):
        if self.has_coolprop:
            h_v = self.CP.PropsSI('H','T',T,'Q',1,'N2O')
            h_l = self.CP.PropsSI('H','T',T,'Q',0,'N2O')
            return float(h_v - h_l)
        T = float(np.clip(T, self.T_min, self.T_max))
        return max(0.0, self._interp(T, self.hfg_grid))

    def mu_l(self, T):
        """Liquid dynamic viscosity [Pa·s]."""
        if self.has_coolprop:
            try:
                Ps = self.CP.PropsSI('P', 'T', T, 'Q', 0, 'N2O')
                try:
                    return float(self.CP.PropsSI('V', 'T', T, 'P', Ps, 'N2O'))
                except Exception:
                    return float(self.CP.PropsSI('V', 'T', T, 'P', Ps, 'NitrousOxide'))
            except Exception:
                pass
        # Empirical fallback: ~5.8e-4 Pa·s at 20 °C with mild T dependence
        return 5.8e-4 * math.exp(-0.02 * (T - 293.15))

# Helium constants
R_HE   = 2077.1     # J/kg-K
GAM_HE = 1.66

# =============================
# Simple PI controller (for He reg and mdot trim)
# =============================
class PI:
    def __init__(self, kp, ki, umin=-1e9, umax=1e9):
        self.kp = kp; self.ki = ki
        self.umin = umin; self.umax = umax
        self.I = 0.0
        self.u = 0.0
    def update(self, e, dt, freeze_I=False):
        if not freeze_I:
            self.I += self.ki * e * dt
        u = self.kp * e + self.I
        # anti-windup clamp with back-calc
        if u > self.umax:
            u = self.umax
            if not freeze_I: self.I = u - self.kp * e
        elif u < self.umin:
            u = self.umin
            if not freeze_I: self.I = u - self.kp * e
        self.u = u
        return u

# =============================
# Setpoint schedules
# =============================
def mdot_setpoint(t):
    # Example schedule; replace with your own command profile
    if t < 5.0:
        return t*0.1 + 0.30   # ramp 0.30->0.80
    elif t < 10.0:
        return 0.80           # hold
    elif t < 15.0:
        return 0.80 - (t-10)*0.05  # ramp to 0.55
    else:
        return 0.50

def he_overpressure_setpoint(t, he_set_MPa=0.60):
    """Base He overpressure above Psat (MPa). Used in addition to no-flash demand if larger."""
    return he_set_MPa

# =============================
# Geometry helpers (upright cylinder)
# =============================
def cylinder_from_volume(V_tank, aspect_ratio=3.0):
    D = (4.0 * V_tank / (math.pi * aspect_ratio)) ** (1.0/3.0)
    H = aspect_ratio * D
    return D, H

def internal_areas(D, H):
    A_side = math.pi * D * H
    A_cap  = math.pi * (0.5*D)**2
    A_int_total = A_side + 2.0 * A_cap
    return A_side, A_cap, A_int_total

def wetted_areas_from_fill(V_tank, D, H, A_side, A_cap, m_l, rhoL):
    if rhoL <= 0.0:
        return 0.0, (A_side + 2.0*A_cap)
    V_l = max(min(m_l / rhoL, V_tank), 0.0)
    h = max(min((V_l / V_tank) * H, H), 0.0)
    A_side_liq = math.pi * D * h
    A_side_vap = max(A_side - A_side_liq, 0.0)
    A_bottom_liq = A_cap if V_l > 0.0 else 0.0
    A_top_liq    = A_cap if h >= H else 0.0
    A_liq = A_side_liq + A_bottom_liq + A_top_liq
    A_vap = (A_side + 2.0*A_cap) - A_liq
    return A_liq, max(A_vap, 0.0)

# =============================
# Standard tube specs (316SS)
# =============================
def get_tube_spec(name: str):
    eps = 1.5e-6  # drawn stainless roughness
    table = {
        "1/4_OD_0.035_wall": {"ID_m": (0.25 - 2*0.035)*0.0254, "eps_m": eps},  # ID ≈ 4.57 mm
        "3/8_OD_0.035_wall": {"ID_m": (0.375 - 2*0.035)*0.0254, "eps_m": eps}, # ID ≈ 7.75 mm
        "1/2_OD_0.035_wall": {"ID_m": (0.5 - 2*0.035)*0.0254, "eps_m": eps},   # ID ≈ 10.92 mm
        "1/2_OD_0.049_wall": {"ID_m": (0.5 - 2*0.049)*0.0254, "eps_m": eps},   # ID ≈ 10.21 mm
    }
    if name not in table:
        raise ValueError(f"Unknown tube spec '{name}'. Options: {list(table.keys())}")
    return table[name]["ID_m"], table[name]["eps_m"]

# =============================
# Minor loss K library (typical values)
# =============================
def K_ball_valve_full_open(): return 0.05
def K_globe_valve_full_open(): return 10.0
def K_check_valve_swing(): return 2.0
def K_elbow_90_long(): return 0.3
def K_elbow_45(): return 0.2
def K_tee_run_through(): return 0.2
def K_tee_branch(): return 1.0
def K_sudden_enlargement(): return 0.5
def K_sudden_contraction(): return 0.5
def K_inlet_sharp(): return 0.5
def K_exit_free(): return 1.0

# =============================
# Friction factor (Churchill correlation)
# =============================
def friction_factor_churchill(Re, eps, D):
    if Re <= 0.0:
        return 0.0
    A = (-2.457 * math.log((7.0/Re)**0.9 + 0.27*(eps/D)))**16
    B = (37530.0/Re)**16
    f = 8.0 * ((8.0/Re)**12 + 1.0/(A + B)**1.5)**(1.0/12.0)
    return float(max(f, 1e-6))

# =============================
# Line pressure drop utilities
# =============================
def dp_major_darcy(mdot, rho, mu, D, L, eps):
    """Major loss ΔP = f (L/D) (ρ v^2 / 2)."""
    A = math.pi * (D**2) / 4.0
    v = mdot / max(rho*A, 1e-12)
    Re = rho * v * D / max(mu, 1e-12)
    if Re < 2300.0:
        f = 64.0 / max(Re, 1e-6)
    else:
        f = friction_factor_churchill(Re, eps, D)
    dp = f * (L / max(D, 1e-12)) * 0.5 * rho * v**2
    return float(max(dp, 0.0)), Re, v, f

def dp_minor_K(mdot, rho, D, K_total):
    """Minor loss ΔP = K (ρ v^2 / 2)."""
    A = math.pi * (D**2) / 4.0
    v = mdot / max(rho*A, 1e-12)
    dp = K_total * 0.5 * rho * v**2
    return float(max(dp, 0.0))

def K_sum(counts: dict):
    total = 0.0
    total += counts.get("inlet", 0) * K_inlet_sharp()
    total += counts.get("ball_valve", 0) * K_ball_valve_full_open()
    total += counts.get("globe_valve", 0) * K_globe_valve_full_open()
    total += counts.get("check_valve", 0) * K_check_valve_swing()
    total += counts.get("elbow_90", 0) * K_elbow_90_long()
    total += counts.get("elbow_45", 0) * K_elbow_45()
    total += counts.get("tee_run", 0) * K_tee_run_through()
    total += counts.get("tee_branch", 0) * K_tee_branch()
    total += counts.get("sudden_enlargement", 0) * K_sudden_enlargement()
    total += counts.get("sudden_contraction", 0) * K_sudden_contraction()
    total += counts.get("exit", 0) * K_exit_free()
    return total

# =============================
# Helium regulator + injector
# =============================
def sonic_mass_flux(gamma, R, P0, T0):
    """Choked mass flux (kg/m^2/s) for perfect gas at (P0,T0) upstream."""
    crit = (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
    return P0 * math.sqrt(gamma/(R*T0)) * crit

def compressible_orifice_mdot(Pu, Pd, T, A, Cd, gamma=GAM_HE, R=R_HE):
    """Mass flow through an orifice from Pu->Pd for perfect gas, upstream conditions (Pu,T)."""
    Pu = max(Pu, 1.0); Pd = max(Pd, 1.0)
    if Pu <= Pd:
        return 0.0
    pr = Pd / Pu
    pr_crit = (2.0/(gamma+1.0))**(gamma/(gamma-1.0))
    if pr <= pr_crit:
        # choked
        G = sonic_mass_flux(gamma, R, Pu, T)
        return Cd * A * G
    else:
        # subsonic (isentropic)
        term = (2.0*gamma/(gamma-1.0)) * ( (pr)**(2.0/gamma) - (pr)**((gamma+1.0)/gamma) )
        md = Cd * A * Pu * math.sqrt(term/(R*T))
        return max(md, 0.0)

# =============================
# Simulator
# =============================
def simulate(
    # Tank & initial state
    V_tank=0.034,           # [m^3] ~ 34 L
    ullage0=0.20,
    T0=293.15,              # [K]
    P_back=2.413e6,         # [Pa] baseline downstream (not used if chamber model on)

    # Valve & control
    Cd=0.80,                # discharge coefficient (metering orifice/valve)
    valve_tau=0.10,         # [s]

    # Fluid energy model
    cp_l=1550.0,            # [J/kg-K]
    dt=0.01, t_end=40.0,

    # Tank geometry
    aspect_ratio=3.0,       # H/D

    # Wall model
    wall_thickness=0.005,   # [m]
    rho_wall=2700.0, cp_wall=900.0,
    h_in_liq=800.0, h_in_vap=15.0, h_out=8.0,
    T_amb=293.15,

    # Feedline model (standard tube + components)
    tube_spec="3/8_OD_0.035_wall",
    L_pipe=1.5,             # [m]
    component_counts=None,  # dict of K components

    # Helium supercharge system
    he_enabled=True,
    he_set_MPa=0.0,         # base over-Psat setpoint (additional to no-flash demand)
    he_reg_tau=0.15,        # (kept for backwards compatibility; PI dominates)
    he_orifice_d=1.2e-3,    # m, He injector orifice diameter into tank
    he_Cd=0.85,             # -, He injector discharge coefficient
    he_bottle_P0=20e6,      # Pa
    he_bottle_T=300.0,      # K
    he_bottle_V=6.0e-3,     # m^3 (6 L)
    he_init_fill=False,     # Start with NO He in ullage

    # No-flash controller
    no_flash_enforce=True,
    no_flash_margin_MPa=0.25,

    # ===== Chamber model =====
    CSTAR=1382.22,          # [m/s]
    At=0.000447436,         # [m^2] throat area
    expansion_ratio=3.6631, # [-] info
    Lstar=0.74836,          # [m] info
    use_chamber_cstar_model=True,  # Pc = mdot_total * c* / At
    ox_mass_fraction=1.0,   # mdot_total = mdot_ox / ox_mass_fraction (set <1 when you add fuel)

    # ===== Injector drop model: fraction of Pc =====
    inj_dp_frac_of_Pc=0.30,     # ΔP_inj = 0.30 * Pc (typical 0.2–0.3 of Pc)
    inj_dp_floor_Pa=0.0         # optional floor on ΔP_inj (Pa)
):
    if component_counts is None:
        component_counts = {"inlet":1, "elbow_45":2, "ball_valve":1}

    props = N2OProps()

    # Geometry (tank)
    D, H = cylinder_from_volume(V_tank, aspect_ratio)
    A_side, A_cap, A_int_total = internal_areas(D, H)

    # Initial saturated state
    T = float(T0)
    rhoL = props.rho_l(T)
    rhoV = props.rho_v(T)
    Psat = props.Psat(T)

    V_v = V_tank * ullage0
    V_l = V_tank - V_v
    m_l = V_l * rhoL
    m_v = V_v * rhoV

    # Wall
    A_liq, A_vap = wetted_areas_from_fill(V_tank, D, H, A_side, A_cap, m_l, rhoL)
    T_w_liq = T
    T_w_vap = T
    Cw_areal = rho_wall * cp_wall * wall_thickness

    # Feedline (standard tube)
    D_pipe, eps_pipe = get_tube_spec(tube_spec)
    Ktot_global = K_sum(component_counts)

    # Helium states
    A_he = math.pi*(he_orifice_d**2)/4.0
    m_he_bottle = he_bottle_P0 * he_bottle_V / (R_HE * he_bottle_T)
    if he_enabled and he_init_fill:
        P_he0 = he_overpressure_setpoint(0.0, he_set_MPa) * 1e6
        m_he = P_he0 * V_v / (R_HE * T)
        m_he_bottle = max(m_he_bottle - m_he, 0.0)
        P_reg = Psat + P_he0
    else:
        m_he = 0.0
        P_reg = Psat

    # Valve initial sizing headroom (ignoring line drop at t=0)
    dP0 = max((Psat + (m_he*R_HE*T)/max(V_v,1e-12)) - P_back, 0.0)
    A_for_0p8 = 0.80 / (0.80 * math.sqrt(max(2.0 * rhoL * dP0, 1e-12)))  # conservative placeholder
    # Use provided Cd below; this initial headroom estimate is only for A_max scaling
    A_max = 1.5 * A_for_0p8
    A_min = 0.0
    A = 0.5 * A_for_0p8

    # ---- Initial summary ----
    muL0 = props.mu_l(T)
    mdot_ref = 0.52914
    P_he0 = (m_he*R_HE*T)/max(V_v,1e-12) if he_enabled else 0.0
    P_tank0 = Psat + P_he0
    dp_maj0, Re0, vpipe0, f0 = dp_major_darcy(mdot_ref, rhoL, muL0, D_pipe, L_pipe, eps_pipe)
    dp_min0 = dp_minor_K(mdot_ref, rhoL, D_pipe, Ktot_global)
    dP_line0 = dp_maj0 + dp_min0
    dP_valve0 = max(P_tank0 - dP_line0 - P_back, 0.0)
    A_for_0p8_line = (mdot_ref / (Cd * math.sqrt(max(2.0 * rhoL * dP_valve0, 1e-16)))) if dP_valve0 > 0 else float('nan')
    d_orifice_ideal_mm = 1e3 * math.sqrt(4.0 * A_for_0p8 / math.pi)
    d_orifice_line_mm  = 1e3 * math.sqrt(4.0 * A_for_0p8_line / math.pi) if dP_valve0 > 0 else float('nan')
    fill_frac = ((V_tank - V_v) / V_tank) * 100.0

    print("\n=== INITIAL GEOMETRY & HYDRAULICS ===")
    print(f"Tank volume: {V_tank:.6f} m^3  |  Ullage: {ullage0*100:.1f}%  |  Fill: {fill_frac:.1f}%")
    print(f"Liquid mass: {m_l:.2f} kg      |  Vapor mass: {m_v:.2f} kg")
    print(f"Sat Temp: {T-273.15:.2f} °C    |  Psat: {Psat/1e5:.2f} bar")
    print(f"He enabled: {he_enabled} | P_he0: {P_he0/1e5:.2f} bar | P_tank0: {P_tank0/1e5:.2f} bar")
    if he_enabled:
        print(f"He regulator (closed-loop): PI -> P_reg, tau legacy={he_reg_tau:.2f} s")
        print(f"No-flash control: {no_flash_enforce} | margin = {no_flash_margin_MPa:.2f} MPa")
        print(f"He injector orifice: d = {he_orifice_d*1e3:.2f} mm, Cd = {he_Cd:.2f}")
        print(f"He bottle: P0 = {he_bottle_P0/1e6:.1f} MPa, V = {he_bottle_V*1e3:.1f} L, T = {he_bottle_T:.1f} K")
        print(f"Initial bottle mass: {m_he_bottle:.3f} kg | Initial ullage He mass: {m_he:.3f} kg")
    print(f"Tank geom: D = {D*1000:.2f} mm, H = {H*1000:.2f} mm  |  A_side = {A_side:.4f} m^2, A_cap = {A_cap:.4f} m^2")
    print(f"Wetted areas: A_liq = {A_liq:.4f} m^2, A_vap = {A_vap:.4f} m^2")
    print(f"Wall: t = {wall_thickness*1e3:.1f} mm, rho = {rho_wall:.0f} kg/m^3, cp = {cp_wall:.0f} J/kg-K")
    print(f"Piping: {tube_spec}, ID = {D_pipe*1000:.2f} mm, L = {L_pipe:.2f} m, ε = {eps_pipe*1e6:.2f} μm")
    print(f"Minor K total (upstream): {Ktot_global:.2f}  (components: {component_counts})")
    print(f"Re @ 0.80 kg/s = {Re0:.0f}, f = {f0:.4f}, v = {vpipe0:.2f} m/s")
    print(f"ΔP_line @ 0.80 kg/s = {dP_line0/1e6:.3f} MPa  [major {dp_maj0/1e6:.3f}, minor {dp_min0/1e6:.3f}]")
    print(f"Valve ΔP available ≈ {dP_valve0/1e6:.3f} MPa  (P_tank - ΔP_line - P_back)")
    print(f"Valve sizing for 0.80 kg/s:")
    print(f"  ignoring line losses:  A = {A_for_0p8:.3e} m^2  (d ≈ {d_orifice_ideal_mm:.2f} mm)")
    print(f"  incl. line losses:     A ≈ {A_for_0p8_line:.3e} m^2  (d ≈ {d_orifice_line_mm:.2f} mm)")
    print(f"Nozzle/Chamber: c* = {CSTAR:.2f} m/s | At = {At:.9f} m^2 | eps = {expansion_ratio:.4f} | L* = {Lstar:.5f} m")
    print(f"Injector ΔP model: ΔP_inj = {inj_dp_frac_of_Pc:.2f} * Pc (floor {inj_dp_floor_Pa/1e6:.3f} MPa)")
    print(f"Ox mass fraction assumption for Pc: ox_mass_fraction = {ox_mass_fraction:.3f}  (1.0 = ox-only)")
    print("======================================\n")

    # --- Controllers (initial gains; tune for your plant) ---
    pi_he = PI(kp=0.6, ki=0.8, umin=Psat, umax=1e9)   # He tank-pressure PI → P_reg (abs pressure)
    pi_A  = PI(kp=0.4, ki=2.0, umin=-0.7, umax=+0.7)  # trims A around feedforward as a fraction

    # Logs
    N = int(t_end / dt) + 1
    t_arr    = np.zeros(N)
    T_arr    = np.zeros(N)
    Psat_arr = np.zeros(N)
    P_he_arr = np.zeros(N)
    P_tot_arr= np.zeros(N)
    P_tank_sp_arr = np.zeros(N)
    supply_lim_arr= np.zeros(N)
    ml_arr   = np.zeros(N)
    mv_arr   = np.zeros(N)
    mHe_arr  = np.zeros(N)
    mdot_arr = np.zeros(N)
    mdsp_arr = np.zeros(N)
    A_arr    = np.zeros(N)
    A_ff_arr = np.zeros(N)
    Twl_arr  = np.zeros(N)
    Twv_arr  = np.zeros(N)
    Aliq_arr = np.zeros(N)
    Avap_arr = np.zeros(N)
    dP_line_arr = np.zeros(N)
    dP_maj_arr  = np.zeros(N)
    dP_min_arr  = np.zeros(N)
    Re_arr      = np.zeros(N)
    vpipe_arr   = np.zeros(N)
    f_arr       = np.zeros(N)
    P_up_arr    = np.zeros(N)   # upstream-of-valve pressure (after line)
    mdot_he_arr = np.zeros(N)
    P_reg_arr   = np.zeros(N)
    Pc_arr      = np.zeros(N)   # chamber pressure
    Pc_sp_arr   = np.zeros(N)
    dP_inj_sp_arr  = np.zeros(N)
    dP_inj_act_arr = np.zeros(N)

    # Initialize chamber pressure state
    Pc_prev = float(P_back)

    # Helper: injector ΔP from Pc (with floor)
    def inj_dp_from_Pc(Pc):
        return max(inj_dp_frac_of_Pc * max(Pc, 0.0), inj_dp_floor_Pa)

    # ---- Inner thermo solver with wall heat ----
    def step_thermo_with_wall(T_old, m_l_old, m_v_old, mdot_out, A_liq_now, A_vap_now,
                              T_w_liq_now, T_w_vap_now, dt):
        if m_l_old <= 1e-9:
            return T_old, 0.0, max(m_v_old - mdot_out*dt, 0.0)

        # Allow both heating and cooling around previous state
        T_lo = max(props.T_min + 1e-6, T_old - 25.0)
        T_hi = min(props.T_max - 1e-6, T_old + 25.0)

        def energy_residual(Tg):
            rhoL_g = props.rho_l(Tg)
            rhoV_g = props.rho_v(Tg)
            denom = (1.0 - (rhoV_g / rhoL_g))
            if abs(denom) < 1e-12: denom = 1e-12
            rhs = rhoV_g * (V_tank - (m_l_old - mdot_out*dt)/rhoL_g - m_v_old/rhoL_g)
            m_v_new = max(rhs / denom, 0.0)
            dm_evap = m_v_new - m_v_old
            Q_w_in = (h_in_liq * A_liq_now * (T_w_liq_now - Tg) +
                      h_in_vap * A_vap_now * (T_w_vap_now - Tg)) * dt
            return (m_l_old * cp_l * (Tg - T_old) + dm_evap * props.h_fg(Tg) - Q_w_in)

        f_hi = energy_residual(T_hi)
        f_lo = energy_residual(T_lo)
        if f_hi * f_lo > 0.0:
            denomE = (m_l_old * cp_l + 1e-9)
            T_try = T_old - f_hi / denomE
            T_new = float(np.clip(T_try, T_lo, T_hi))
        else:
            a, b = T_lo, T_hi
            fa, fb = f_lo, f_hi
            T_new = 0.5*(a+b)
            for _ in range(60):
                c = 0.5*(a+b); fc = energy_residual(c); T_new = c
                if abs(fc) < 1e-6 or (b-a) < 1e-6: break
                if fa * fc <= 0.0: b, fb = c, fc
                else: a, fa = c, fc

        rhoL_n = props.rho_l(T_new)
        rhoV_n = props.rho_v(T_new)
        denom = (1.0 - (rhoV_n / rhoL_n))
        if abs(denom) < 1e-12: denom = 1e-12
        rhs = rhoV_n * (V_tank - (m_l_old - mdot_out*dt)/rhoL_n - m_v_old/rhoL_n)
        m_v_new = max(rhs / denom, 0.0)
        dm_evap = m_v_new - m_v_old
        m_l_new = max(m_l_old - mdot_out*dt - dm_evap, 0.0)
        return float(T_new), float(m_l_new), float(m_v_new)

    # =======================
    # Time loop
    # =======================
    mdot_prev = mdot_ref  # seed for iteration
    for k in range(N):
        t = k * dt

        # Current sat properties
        Psat = props.Psat(T)
        rhoL = props.rho_l(T)
        muL  = props.mu_l(T)

        # Update volumes from current liquid mass
        V_l = min(max(m_l / max(rhoL, 1e-9), 0.0), V_tank)
        V_v = max(V_tank - V_l, 1e-9)

        # ----- Commanded mdot & Pc setpoint (feedforward) -----
        md_sp = mdot_setpoint(t)
        if use_chamber_cstar_model:
            md_sp_total = md_sp / max(ox_mass_fraction, 1e-6)
            Pc_sp = (md_sp_total * CSTAR) / max(At, 1e-12)
        else:
            Pc_sp = P_back

        # --- Build He tank pressure setpoint (no-flash) using ACTUAL last-step values ---
        dp_maj_last, _, _, _ = dp_major_darcy(mdot_prev, rhoL, muL, D_pipe, L_pipe, eps_pipe)
        dp_min_last = dp_minor_K(mdot_prev, rhoL, D_pipe, Ktot_global)
        dP_line_last = dp_maj_last + dp_min_last
        dP_inj_need  = inj_dp_from_Pc(Pc_prev)
        P_tank_min_hyd = Pc_prev + dP_line_last + dP_inj_need + no_flash_margin_MPa*1e6
        P_tank_min_sat = Psat + he_overpressure_setpoint(t, he_set_MPa)*1e6
        P_tank_sp = max(P_tank_min_hyd, P_tank_min_sat)

        # --- Helium regulator: CLOSED LOOP on P_tank (PI -> P_reg) ---
        if he_enabled:
            # Current tank pressure from state masses (before adding He this step)
            P_he = (m_he * R_HE * T) / V_v
            P_tank = Psat + P_he

            # Bottle capability (isothermal ideal-gas approx)
            P_bottle = (m_he_bottle * R_HE * he_bottle_T) / max(he_bottle_V, 1e-12)
            P_reg_cmd_max = max(P_bottle - 0.15e6, 0.0)  # margin below bottle pressure
            pi_he.umax = P_reg_cmd_max
            pi_he.umin = Psat  # don't command below Psat (avoid dithering)

            e_tank = P_tank_sp - P_tank
            freeze_I = (P_bottle <= 0.2e6)  # freeze if nearly empty
            P_reg = pi_he.update(e_tank, dt, freeze_I=freeze_I)

            # He mass flow into tank
            mdot_he = 0.0
            if (P_reg > P_tank) and (m_he_bottle > 0.0):
                mdot_he = compressible_orifice_mdot(P_reg, P_tank, T, A_he, he_Cd, gamma=GAM_HE, R=R_HE)
                mdot_he = min(mdot_he, m_he_bottle/dt)

            # Update He masses & tank pressure
            m_he_bottle = max(m_he_bottle - mdot_he*dt, 0.0)
            m_he += mdot_he * dt
            P_he = (m_he * R_HE * T) / V_v
            P_tank = Psat + P_he
        else:
            P_he = 0.0; P_tank = Psat; mdot_he = 0.0; P_reg = Psat
            P_bottle = 0.0; P_reg_cmd_max = 0.0

        # --- Predict line ΔP for md_sp (sizing/FF) & injector ΔP request (based on Pc_prev) ---
        dp_maj_pred, _, _, _ = dp_major_darcy(md_sp, rhoL, muL, D_pipe, L_pipe, eps_pipe)
        dp_min_pred = dp_minor_K(md_sp, rhoL, D_pipe, Ktot_global)
        dP_line_pred = dp_maj_pred + dp_min_pred  # Pa

        dP_inj_req_sp = inj_dp_from_Pc(Pc_prev)  # use previous actual Pc

        # Upstream pressure predictor & available ΔP across valve (feedforward)
        P_back_pred = Pc_sp if use_chamber_cstar_model else P_back
        P_up_pred = max(P_tank - dP_line_pred, 1.0)
        dP_valve_ff = max(P_up_pred - (P_back_pred + dP_inj_req_sp), 0.0)

        # Feedforward area to meet mdot setpoint with estimated ΔP availability
        denom_ff = Cd * math.sqrt(max(2.0 * rhoL * dP_valve_ff, 1e-16))
        A_ff = md_sp / denom_ff if denom_ff > 0 else A  # keep old A if no ΔP
        A_ff = float(np.clip(A_ff, A_min, A_max))

        # --- Actual flow solve using CURRENT A (then we'll update A after we know error) ---
        md_guess = mdot_prev if k > 0 else md_sp
        P_up = None
        for _ in range(4):
            dp_maj, Re, vpipe, f = dp_major_darcy(md_guess, rhoL, muL, D_pipe, L_pipe, eps_pipe)
            dp_min = dp_minor_K(md_guess, rhoL, D_pipe, Ktot_global)
            dP_line = dp_maj + dp_min
            P_up = max(P_tank - dP_line, 1.0)

            # Pc from md_total guess
            if use_chamber_cstar_model:
                md_total_guess = md_guess / max(ox_mass_fraction, 1e-6)
                Pc_guess = (md_total_guess * CSTAR) / max(At, 1e-12)
            else:
                Pc_guess = P_back

            dP_inj_now = inj_dp_from_Pc(Pc_guess)
            dP_valve_now = max(P_up - (Pc_guess + dP_inj_now), 0.0)
            md_new = Cd * A * math.sqrt(max(2.0 * rhoL * dP_valve_now, 0.0))

            if abs(md_new - md_guess) < 1e-4:
                md_guess = md_new
                break
            md_guess = 0.5*(md_guess + md_new)

        mdot = md_guess
        mdot_prev = mdot

        # Final chamber pressure from actual mdot (total)
        if use_chamber_cstar_model:
            md_total = mdot / max(ox_mass_fraction, 1e-6)
            Pc_act = (md_total * CSTAR) / max(At, 1e-12)
        else:
            Pc_act = P_back

        # Recompute line metrics at actual mdot for logging
        dp_maj, Re, vpipe, f = dp_major_darcy(mdot, rhoL, muL, D_pipe, L_pipe, eps_pipe)
        dp_min = dp_minor_K(mdot, rhoL, D_pipe, Ktot_global)
        dP_line = dp_maj + dp_min
        P_up = max(P_tank - dP_line, 1.0)
        dP_inj_act = inj_dp_from_Pc(Pc_act)

        # ----- CLOSED-LOOP mdot trim on valve area around feedforward -----
        e_m = md_sp - mdot
        # Normalize error by setpoint to keep scale consistent (avoid windup at small flows)
        norm = max(abs(md_sp), 1e-6)
        frac_trim = pi_A.update(e_m / norm, dt)  # bounded in [-0.7, +0.7]
        A_cmd = float(np.clip(A_ff * (1.0 + frac_trim), A_min, A_max))
        # Apply actuator first-order lag
        A = A + (A_cmd - A) * (dt / max(valve_tau, 1e-6))

        # ----- Supply-aware soft clamp on mdot setpoint if regulator saturates -----
        supply_limited = (he_enabled and (P_reg >= P_reg_cmd_max - 1e3) and (Psat + (m_he*R_HE*T)/V_v < P_tank_sp - 0.02e6))
        if supply_limited:
            # soften demand by blending toward actual flow (prevents integrator fights upstream)
            md_sp = 0.8*md_sp + 0.2*mdot

        # ---- Fluid thermo step (with wall heat) ----
        T, m_l, m_v = step_thermo_with_wall(
            T_old=T, m_l_old=m_l, m_v_old=m_v, mdot_out=mdot,
            A_liq_now=A_liq, A_vap_now=A_vap,
            T_w_liq_now=T_w_liq, T_w_vap_now=T_w_vap, dt=dt
        )

        # Update wet areas
        rhoL_n = props.rho_l(T)
        A_liq, A_vap = wetted_areas_from_fill(V_tank, D, H, A_side, A_cap, m_l, rhoL_n)

        # Wall nodes (explicit Euler)
        C_w_liq = max(Cw_areal * A_liq, 1e-9)
        C_w_vap = max(Cw_areal * A_vap, 1e-9)
        dTw_liq_dt = (h_out*A_liq*(T_amb - T_w_liq) - h_in_liq*A_liq*(T_w_liq - T)) / C_w_liq if A_liq > 0 else 0.0
        dTw_vap_dt = (h_out*A_vap*(T_amb - T_w_vap) - h_in_vap*A_vap*(T_w_vap - T)) / C_w_vap if A_vap > 0 else 0.0
        T_w_liq += dTw_liq_dt * dt
        T_w_vap += dTw_vap_dt * dt

        # --- Logs ---
        t_arr[k]   = t
        T_arr[k]   = T
        Psat_arr[k]= Psat
        P_he_arr[k]= P_he
        P_tot_arr[k]= P_tank
        P_tank_sp_arr[k] = P_tank_sp
        supply_lim_arr[k]= 1.0 if supply_limited else 0.0
        ml_arr[k]  = m_l
        mv_arr[k]  = m_v
        mHe_arr[k] = m_he
        mdot_arr[k]= mdot
        mdsp_arr[k]= md_sp
        A_arr[k]   = A
        A_ff_arr[k]= A_ff
        Twl_arr[k] = T_w_liq
        Twv_arr[k] = T_w_vap
        Aliq_arr[k]= A_liq
        Avap_arr[k]= A_vap
        dP_line_arr[k] = dP_line
        dP_maj_arr[k]  = dp_maj
        dP_min_arr[k]  = dp_min
        Re_arr[k]      = Re
        vpipe_arr[k]   = vpipe
        f_arr[k]       = f
        P_up_arr[k]    = P_up
        mdot_he_arr[k] = mdot_he
        P_reg_arr[k]   = P_reg
        Pc_arr[k]      = Pc_act
        Pc_sp_arr[k]   = Pc_sp
        dP_inj_sp_arr[k]  = dP_inj_req_sp
        dP_inj_act_arr[k] = dP_inj_act

        # carry Pc forward
        Pc_prev = Pc_act

        # Termination guards
        if m_l <= 1e-6 or T <= props.T_min + 1e-6:
            k += 1
            # trim arrays
            def _trim(x): return x[:k]
            t_arr    = _trim(t_arr);   T_arr    = _trim(T_arr);  Psat_arr = _trim(Psat_arr); P_he_arr = _trim(P_he_arr); P_tot_arr=_trim(P_tot_arr)
            P_tank_sp_arr = _trim(P_tank_sp_arr); supply_lim_arr = _trim(supply_lim_arr)
            ml_arr   = _trim(ml_arr);  mv_arr   = _trim(mv_arr); mHe_arr  = _trim(mHe_arr)
            mdot_arr = _trim(mdot_arr);mdsp_arr = _trim(mdsp_arr);A_arr   = _trim(A_arr); A_ff_arr = _trim(A_ff_arr)
            Twl_arr  = _trim(Twl_arr); Twv_arr  = _trim(Twv_arr);Aliq_arr = _trim(Aliq_arr); Avap_arr = _trim(Avap_arr)
            dP_line_arr=_trim(dP_line_arr); dP_maj_arr=_trim(dP_maj_arr); dP_min_arr=_trim(dP_min_arr)
            Re_arr   = _trim(Re_arr); vpipe_arr=_trim(vpipe_arr); f_arr   = _trim(f_arr)
            P_up_arr = _trim(P_up_arr); mdot_he_arr=_trim(mdot_he_arr); P_reg_arr=_trim(P_reg_arr)
            Pc_arr   = _trim(Pc_arr); Pc_sp_arr=_trim(Pc_sp_arr); dP_inj_sp_arr=_trim(dP_inj_sp_arr); dP_inj_act_arr=_trim(dP_inj_act_arr)
            break

    return {
        "t": t_arr, "T": T_arr,
        "Psat": Psat_arr, "P_he": P_he_arr, "P_tot": P_tot_arr, "P_tank_sp": P_tank_sp_arr,
        "supply_limited": supply_lim_arr,
        "m_l": ml_arr, "m_v": mv_arr, "m_he": mHe_arr,
        "mdot": mdot_arr, "mdot_sp": mdsp_arr, "A": A_arr, "A_ff": A_ff_arr,
        "T_w_liq": Twl_arr, "T_w_vap": Twv_arr,
        "A_liq": Aliq_arr, "A_vap": Avap_arr,
        "geom": {"D": D, "H": H, "A_side": A_side, "A_cap": A_cap, "A_int_total": A_int_total},
        "line": {
            "tube_spec": tube_spec, "D_pipe": D_pipe, "eps": eps_pipe, "L_pipe": L_pipe,
            "dP_line": dP_line_arr, "dP_major": dP_maj_arr, "dP_minor": dP_min_arr,
            "Re": Re_arr, "v": vpipe_arr, "f": f_arr,
            "P_up": P_up_arr
        },
        "helium": {
            "enabled": he_enabled,
            "mdot_he": mdot_he_arr,
            "P_reg": P_reg_arr
        },
        "Pc": Pc_arr, "Pc_sp": Pc_sp_arr,
        "inj_dp": {
            "frac_of_Pc": inj_dp_frac_of_Pc,
            "dP_req_sp": dP_inj_sp_arr,
            "dP_req_act": dP_inj_act_arr
        },
        "nozzle": {
            "CSTAR": CSTAR,
            "At": At,
            "expansion_ratio": expansion_ratio,
            "Lstar": Lstar
        }
    }

# =============================
# Run + plots
# =============================
if __name__ == "__main__":
    out = simulate(
        V_tank=0.034, ullage0=0.20, T0=293.15,
        P_back=2.413e6,
        aspect_ratio=3.0,
        wall_thickness=0.005,
        rho_wall=2700.0, cp_wall=900.0,
        h_in_liq=800.0, h_in_vap=15.0, h_out=8.0,
        T_amb=293.15,
        tube_spec="1/2_OD_0.035_wall", # 1/2" stays above flashing and reduces He usage
        L_pipe=1.5,
        component_counts={"inlet":1, "elbow_45":2, "ball_valve":1},
        he_enabled=True,
        he_set_MPa=0.0,        # rely primarily on no-flash controller
        he_reg_tau=0.15,
        he_orifice_d=1.2e-3,
        he_Cd=0.85,
        he_bottle_P0=20e6,     # 200 bar
        he_bottle_T=300.0,
        he_bottle_V=6.0e-3,    # 6 L
        he_init_fill=False,
        no_flash_enforce=True,
        no_flash_margin_MPa=0.25,
        # Chamber & injector-ΔP model
        CSTAR=1382.22,
        At=0.000447436,
        expansion_ratio=3.6631,
        Lstar=0.74836,
        use_chamber_cstar_model=True,
        ox_mass_fraction=1.0,        # set <1.0 when you add the fuel stream
        inj_dp_frac_of_Pc=0.30,      # <-- key setting: ~30% of Pc
        inj_dp_floor_Pa=0.0
    )

    t = out["t"]
    import numpy as _np

    # mdot tracking
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, out["mdot"], label="mdot (actual) [kg/s]")
    plt.plot(t, out["mdot_sp"], "--", label="mdot (setpoint) [kg/s]")
    plt.xlabel("Time [s]"); plt.ylabel("Mass flow [kg/s]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Tank pressures
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e-6*_np.array(out["Psat"]), label="Psat(N2O) [MPa]")
    plt.plot(t, 1e-6*_np.array(out["P_he"]), label="P_He partial [MPa]")
    plt.plot(t, 1e-6*_np.array(out["P_tot"]), "--", label="P_tank total [MPa]")
    plt.plot(t, 1e-6*_np.array(out["P_tank_sp"]), ":", label="P_tank setpoint [MPa]")
    plt.xlabel("Time [s]"); plt.ylabel("Pressure [MPa]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Upstream-of-valve pressure and flashing margin
    P_up = _np.array(out["line"]["P_up"])
    Psat = _np.array(out["Psat"])
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e-6*P_up, label="P_upstream valve [MPa]")
    plt.plot(t, 1e-6*Psat, "--", label="Psat [MPa]")
    plt.plot(t, 1e-6*(P_up-Psat), ":", label="Flashing margin (P_up - Psat) [MPa]")
    plt.xlabel("Time [s]"); plt.ylabel("Pressure [MPa]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Chamber pressure (from c*)
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e-6*_np.array(out["Pc"]), label="Pc (actual) [MPa]")
    plt.plot(t, 1e-6*_np.array(out["Pc_sp"]), "--", label="Pc (feedforward) [MPa]")
    plt.xlabel("Time [s]"); plt.ylabel("Chamber Pressure [MPa]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Injector ΔP requirements (fraction-of-Pc model)
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e-6*_np.array(out["inj_dp"]["dP_req_act"]), label="ΔP_inj (actual req from Pc) [MPa]")
    plt.plot(t, 1e-6*_np.array(out["inj_dp"]["dP_req_sp"]), "--", label="ΔP_inj (step setpoint base) [MPa]")
    plt.xlabel("Time [s]"); plt.ylabel("Injector ΔP [MPa]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Tank temperature
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, out["T"]-273.15, label="Tank T [°C]")
    plt.xlabel("Time [s]"); plt.ylabel("Temperature [°C]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Liquid/Vapor/He masses
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, out["m_l"], label="m_liquid [kg]")
    plt.plot(t, out["m_v"], label="m_vapor [kg]")
    plt.plot(t, out["m_he"], label="m_He in ullage [kg]")
    plt.xlabel("Time [s]"); plt.ylabel("Mass [kg]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # He regulator and injector flow
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e-6*_np.array(out["helium"]["P_reg"]), label="P_reg (He) [MPa]")
    plt.plot(t, 1e3*_np.array(out["helium"]["mdot_he"]), label="He mdot [g/s]")
    plt.xlabel("Time [s]"); plt.ylabel("He Reg / Flow"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Valve area
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e6*out["A"], label="Valve area [mm²]")
    plt.plot(t, 1e6*out["A_ff"], "--", label="Valve area feedforward [mm²]")
    plt.xlabel("Time [s]"); plt.ylabel("Area [mm²]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # Line hydraulics
    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, 1e-6*_np.array(out["line"]["dP_line"]), label="ΔP_line total [MPa]")
    plt.plot(t, 1e-6*_np.array(out["line"]["dP_major"]), "--", label="ΔP_major [MPa]")
    plt.plot(t, 1e-6*_np.array(out["line"]["dP_minor"]), ":", label="ΔP_minor [MPa]")
    plt.xlabel("Time [s]"); plt.ylabel("Pressure drop [MPa]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, out["line"]["Re"], label="Re in pipe [-]")
    plt.xlabel("Time [s]"); plt.ylabel("Re"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8.6, 5.2))
    plt.plot(t, out["line"]["v"], label="Pipe velocity [m/s]")
    plt.xlabel("Time [s]"); plt.ylabel("Velocity [m/s]"); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # He usage summary
    he_used_kg   = float(out["m_he"][-1])                         # kg
    he_used_g    = 1e3 * he_used_kg                               # g
    he_used_int  = np.trapz(out["helium"]["mdot_he"], out["t"])   # kg (integration check)

    # Convert to standard volumes (STP ~ 0 °C, 1 atm; ρ_He ≈ 0.1785 kg/m^3)
    rho_he_stp = 0.1785  # kg/m^3
    he_std_m3  = he_used_kg / rho_he_stp
    he_std_L   = 1e3 * he_std_m3
    he_scf     = he_std_m3 * 35.3147  # standard cubic feet

    print(f"He used (from m_he): {he_used_g:.2f} g")
    print(f"He used (integrated mdot): {1e3*he_used_int:.2f} g")
    print(f"≈ {he_std_L:.1f} standard liters (STP)  |  {he_scf:.2f} scf")
