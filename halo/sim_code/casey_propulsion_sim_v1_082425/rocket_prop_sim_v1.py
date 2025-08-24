
# rocket_prop_sim_v1.py
# Casey — v1 end-to-end simulation of a pressure-fed N2O/IPA rocket propulsion system with He pressurization.
# Units: SI throughout.
# Requires: rocketcea (pip install rocketcea)
#
# Nodes (as requested; labels appear in comments alongside objects/variables):
# Pressurant: H1 He Tank, H2 Regulator Output, H3 Manifold
# Pressurant branches: H4 Ox restrictor -> N2O ullage (He+N2O vap), H5 Fuel restrictor -> IPA ullage (He only)
# Tanks: H6 Ox Ullage, H7 Fuel Ullage; O1 N2O liquid, O2 N2O outlet; F1 IPA liquid, F2 IPA outlet
# Valves: O3 Ox valve (throttling), F3 Fuel valve (throttling)
# Injectors: O4 Ox injector orifice (pintle area variable), F4 Fuel injector orifice (fixed area)
# Chamber/Nozzle: C1 Combustion chamber, N1 Nozzle Throat, N2 Nozzle Exit/Ambient
#
# Fidelity targets (v1):
# - Helium bottle: ideal gas, regulator ideal P-source
# - IPA tank: thermal mass (wall+liquid), ullage He energy-mixing, density variation allowed
# - N2O tank: two-phase with N2O vapor at saturation + He in ullage, simple energy balance with latent heat,
#             track m_liq, m_vap, n_He; wall+liquid isothermal lump
# - Lines: Darcy–Weisbach + minor K, variable density (liquids), no heat soak to environment
# - Injector/valves: orifice models. Pintle area A_ox is time-varying; fuel orifices fixed
# - Chamber: Pc tied to mdot via c* from RocketCEA; no heat losses
# - Nozzle: fixed expansion ratio eps, fixed ambient Pa; Isp from RocketCEA (no nozzle eta)
#
# -------------------------------------------------------------
# Legal & Safety Notice:
# This code is provided for educational simulation purposes only.
# It is NOT a construction or testing guide. Real propulsion systems are dangerous.
# Do not use this for hardware decisions without expert review, appropriate standards, and testing.
# -------------------------------------------------------------

from dataclasses import dataclass
import math
import numpy as np
from typing import Callable, Dict, Tuple

# ---- Thermochemistry via RocketCEA -----------------------------------------
# You must have rocketcea installed. Names for propellants may need adjustment to match library identifiers.
# Common options: 'N2O' for nitrous oxide and something like 'IPA'/'IsopropylAlcohol' (verify in your install).
try:
    from rocketcea.cea_obj import CEA_Obj  # RocketCEA
except Exception as e:
    raise ImportError(
        "This script requires the 'rocketcea' package. Install with 'pip install rocketcea'. "
        "Then verify oxName/fuelName strings match your local RocketCEA database."
    )

# ---- Physical constants -----------------------------------------------------
g0 = 9.80665  # m/s^2
R_univ = 8.314462618  # J/(mol*K)

# Helium properties (gas)
M_He = 4.002602e-3  # kg/mol
R_He = R_univ / M_He  # J/(kg*K)
gamma_He = 1.66
cp_He = gamma_He * R_He / (gamma_He - 1.0)  # J/kg-K

# IPA liquid (approx. at ~20 C)
rho_IPA_20C = 786.0  # kg/m^3
cp_IPA = 2550.0      # J/kg-K (approx.)
mu_IPA = 2.4e-3      # Pa*s (2.4 mPa*s at ~20 C)

# N2O properties (approximate where noted)
M_N2O = 44.0128e-3   # kg/mol
R_N2O_gas = R_univ / M_N2O  # J/(kg*K) gas
cp_N2O_liq = 2400.0  # J/kg-K (approximate)
cp_N2O_vap = 850.0   # J/kg-K (approximate)
L_N2O = 3.77e5       # J/kg latent heat of vaporization ~ 377 kJ/kg near room temp (approx.)
rho_N2O_liq_20C = 745.0  # kg/m^3 at ~20 C (approx. constant in v1)
mu_N2O_liq = 6.4e-5      # Pa*s (~0.064 mPa*s) liquid near 20 C

# Stainless steel/aluminum wall lump for tank (effective heat capacity)
cp_wall = 500.0  # J/kg-K (choose per material). 900 J/kg-K for Al, 500 J/kg-K for Stainless
# -----------------------------------------------------------------------------

# ---- Utility correlation functions ----------------------------

def density_IPA_liq(T: float) -> float:
    """
    Simple linear density model for IPA liquid [kg/m^3].
    Linear temp coefficient ~ -0.7 kg/m^3-K around 20 C; keep bounded by max and min functions.
    Min value 700 kg/m^3, max value 810 kg/m^3. Equation is: rho(T) = rho(20C) - alpha * (T - 293.15), T in Kelvin
    """
    return max(700.0, min(810.0, rho_IPA_20C - 0.7 * (T - 293.15))) 

def density_N2O_liq(T: float) -> float:
    """
    Saturation vapor pressure of N2O [bar] as a function of T [K].
    Ambrose-Walton correlation with constants b1..b4.
    Valid roughly for -90 °C to +36 °C (below Tc).
    """
    # Critical properties of N2O
    Tc = 309.57  # K
    Pc = 72.45   # bar

    # Coefficients
    b1 = -6.71893
    b2 = 1.35966
    b3 = -1.3779
    b4 = -4.051

    # Reduced temperature
    Tr = T / Tc

    # (Optional) guard: correlation is for Tr < 1
    if Tr >= 1.0:
        # At/above Tc, saturation curve terminates; return Pc at the limit
        return Pc

    x = (1.0 / Tr) * (
        b1 * (1.0 - Tr)
        + b2 * (1.0 - Tr) ** 1.5
        + b3 * (1.0 - Tr) ** 2.5
        + b4 * (1.0 - Tr) ** 5
    )
    return Pc * math.exp(x)

def Psat_N2O_Pa(T: float) -> float:
    return T * 1e5  # 1 bar = 1e5 Pa

def darcy_friction_factor(Re: float, rel_eps: float) -> float:
    """
    Haaland explicit equation for Darcy f in turbulent; Blasius for laminar fallback.
    Used for Darcy-Weisbach equation for pressure drop in a pipe
    rel_eps = eps / D, pipe roughness height divided by diameter
    returns Darcy friction factor
    """
    if Re < 1e-6: # No division by zero
        return 0.0
    if Re < 2300.0: # Laminar flow
        return 64.0 / max(Re, 1e-6)
    # Haaland equation (1978), Haaland explicit equation
    return 1.0 / ( -1.8*math.log10( (rel_eps/3.7)**1.11 + 6.9/max(Re,1.0) ) )**2

def incompressible_orifice_mdot(deltaP: float, rho: float, Cd: float, A: float) -> float:
    """
    Liquid orifice mass flow [kg/s] due to pressure drop across orifice.
    """
    if deltaP <= 0 or A <= 0:
        return 0.0
    return Cd * A * math.sqrt(2.0 * rho * deltaP)

def gas_orifice_mdot(P_up: float, T_up: float, P_dn: float, gamma: float, R_spec: float, Cd: float, A: float) -> float:
    """
    Compressible orifice flow for ideal gas with choked/subcritical check.
    Returns mass flow [kg/s].
    """
    if A <= 0 or P_up <= P_dn:
        return 0.0
    crit = (2.0/(gamma+1.0))**(gamma/(gamma-1.0))
    pr = P_dn / P_up

    if pr <= crit: # Choked
        term = (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
        return Cd * A * P_up * math.sqrt(gamma / (R_spec * T_up)) * term
    
    # Subsonic: comes from continuity + isentropic relations + energy for a perfect gas, taking the upstream state as stagnation (total) conditions
    term = (2.0*gamma/(R_spec*T_up*(gamma-1.0))) * (pr**(2.0/gamma) - pr**((gamma+1.0)/gamma))       
    return Cd * A * P_up * math.sqrt(max(term, 0.0))

def line_deltaP_liquid(mdot: float, rho: float, mu: float, L: float, D: float, K_minor: float, rough: float) -> float:
    """
    Pressure drop across a liquid line with Darcy-Weisbach + minor losses [Pa].
    """
    if mdot <= 0 or D <= 0 or L < 0:
        return 0.0
    A = math.pi * (D**2) / 4.0
    v = mdot / (rho * A)
    Re = rho * v * D / max(mu, 1e-9)
    f = darcy_friction_factor(Re, rough/D if D>0 else 1e-6)
    q = 0.5 * rho * v*v
    return (f * (L/D) + K_minor) * q

# ---- Components -------------------------------------------------------------

@dataclass
class HeliumBottle:  # H1 Node
    V: float        # m^3
    T: float        # K (assume constant for v1)
    P: float        # Pa (initial)
    n: float        # mol (derived from P V / R_u T)
    def __init__(self, V: float, T: float, P: float):
        self.V = V; self.T = T; self.P = P
        self.n = P*V / (R_univ * T)
    def withdraw(self, m_dot: float, dt: float) -> Tuple[float,float]:
        """Withdraw mass [kg] over dt; update bottle P [Pa]; return (m_withdrawn, T_out)."""
        if m_dot <= 0:
            return 0.0, self.T
        m = m_dot * dt # mass over timestep
        dn = m / M_He # num moles
        # Prevent overdraw
        dn = min(dn, self.n * 0.999)
        self.n -= dn
        self.P = self.n * R_univ * self.T / self.V
        return dn * M_He, self.T # outflow of mass [kg], outflow temp [K]

@dataclass
class IdealRegulator:  # H2 Node
    P_set: float  # Pa
    def outlet_pressure(self, P_in: float) -> float:
        return min(self.P_set, P_in)  # simple limit: can't exceed supply

@dataclass
class Restrictor:  # H4 Node
    Cd: float
    A: float
    def mdot(self, P_up: float, T_up: float, P_dn: float) -> float:
        return gas_orifice_mdot(P_up, T_up, P_dn, gamma_He, R_He, self.Cd, self.A)

@dataclass
class LiquidBranch:  # generic branch: line -> valve (orifice) -> line -> injector (orifice)
    # Geometry
    L1: float; D1: float; K1: float; rough1: float
    L2: float; D2: float; K2: float; rough2: float
    # Valve/orifices
    Cd_valve: float; A_valve_max: float  # throttling valve
    Cd_inj: float;   A_inj: float        # injector orifice
    # Fluid props functions (rho(T), mu(T))
    rho_fn: Callable[[float], float]
    mu_fn: Callable[[float], float]

    def mdot_from_upstream(self, P_up: float, T_liq: float, P_chamber: float, throttle: float) -> float:
        """
        Solve for mdot that satisfies ΔP across: line1 + valve + line2 + injector,
        where valve and injector are orifice losses for liquids.
        """
        rho = self.rho_fn(T_liq)
        mu = self.mu_fn(T_liq)
        A_valve = max(0.0, min(1.0, throttle)) * self.A_valve_max
        Cd_valve = self.Cd_valve
        Cd_inj = self.Cd_inj
        A_inj = self.A_inj

        if P_up <= P_chamber or (A_valve <= 0 and A_inj <= 0):
            return 0.0

        # Iteratively solve mdot: ΔP_total(mdot) = P_up - P_chamber
        target_dp = P_up - P_chamber
        mdot = 1e-6
        for _ in range(40):
            # Line 1 drop
            dp1 = line_deltaP_liquid(mdot, rho, mu, self.L1, self.D1, self.K1, self.rough1)
            # Valve drop
            dp_valve = 0.0 if A_valve <= 0 else (mdot / (Cd_valve * A_valve))**2 / (2.0 * rho)
            # Line 2 drop
            dp2 = line_deltaP_liquid(mdot, rho, mu, self.L2, self.D2, self.K2, self.rough2)
            # Injector drop
            dp_inj = 0.0 if A_inj <= 0 else (mdot / (Cd_inj * A_inj))**2 / (2.0 * rho)

            dp_total = dp1 + dp_valve + dp2 + dp_inj

            # Newton-like update on mdot (relaxation)
            err = dp_total - target_dp
            if abs(err) < 1.0:  # Pa tolerance
                break
            # numerical derivative via small perturbation
            mdot_eps = mdot * 1.01 + 1e-9
            dp1e = line_deltaP_liquid(mdot_eps, rho, mu, self.L1, self.D1, self.K1, self.rough1)
            dp2e = line_deltaP_liquid(mdot_eps, rho, mu, self.L2, self.D2, self.K2, self.rough2)
            dp_valvee = 0.0 if A_valve <= 0 else (mdot_eps / (Cd_valve * A_valve))**2 / (2.0 * rho)
            dp_inje = 0.0 if A_inj <= 0 else (mdot_eps / (Cd_inj * A_inj))**2 / (2.0 * rho)
            d_dp_d_m = (dp1e + dp_valvee + dp2e + dp_inje - dp_total) / (mdot_eps - mdot + 1e-12)
            if d_dp_d_m <= 0:
                mdot *= 1.2
            else:
                mdot = max(1e-9, mdot - err / d_dp_d_m)
        return max(0.0, float(mdot))

@dataclass
class IPATank:  # Fuel tank (H7 ullage He + F1 liquid IPA, F2 outlet)
    V_tank: float
    m_wall: float
    T: float        # lump temperature of wall+liquid+ullage [K]
    m_IPA: float    # kg liquid
    n_He: float     # mol in ullage
    def ullage_volume(self) -> float:
        V_liq = self.m_IPA / density_IPA_liq(self.T)
        return max(1e-6, self.V_tank - V_liq)
    def pressure(self) -> float:
        Vg = self.ullage_volume()
        return (self.n_He * R_univ * self.T) / Vg
    def add_He(self, m_He: float, T_in: float):
        if m_He <= 0: return
        # Energy mixing (very simple lumped model)
        C_eff = self.m_IPA*cp_IPA + self.m_wall*cp_wall + (self.n_He*M_He)*cp_He
        Q = m_He * cp_He * (T_in - self.T)
        dT = Q / max(C_eff, 1.0)
        self.T += dT
        self.n_He += m_He / M_He
    def withdraw_liquid(self, mdot: float, dt: float):
        dm = min(self.m_IPA, max(0.0, mdot*dt))
        self.m_IPA -= dm
        # Ullage expands -> P will drop automatically via ideal gas when we call pressure()

@dataclass
class N2OTank:  # Ox tank (H6 ullage: N2O vapor at Psat + He; O1 liquid; O2 outlet)
    V_tank: float
    m_wall: float
    T: float        # K (lumped wall+liquid+vapor)
    m_liq: float    # kg
    m_vap: float    # kg (N2O vapor)
    n_He: float     # mol in ullage

    def gas_volume(self) -> float:
        V_liq = self.m_liq / density_N2O_liq(self.T)
        return max(1e-6, self.V_tank - V_liq)

    def partial_pressures(self) -> Tuple[float,float,float]:
        Vg = self.gas_volume()
        # N2O vapor at saturation partial pressure
        P_sat = Psat_N2O_Pa(self.T)
        # He partial pressure
        P_he = (self.n_He * R_univ * self.T) / Vg
        P_tot = P_sat + P_he
        return P_sat, P_he, P_tot

    def add_He(self, m_He: float, T_in: float):
        if m_He <= 0: return
        # Simple energy mixing into the lumped tank temperature
        C_eff = self.m_liq*cp_N2O_liq + self.m_vap*cp_N2O_vap + self.m_wall*cp_wall + (self.n_He*M_He)*cp_He
        Q = m_He * cp_He * (T_in - self.T)
        dT = Q / max(C_eff, 1.0)
        self.T += dT
        self.n_He += m_He / M_He

    def withdraw_liquid_and_update_phase(self, mdot_liq: float, dt: float):
        """Remove liquid, then adjust vapor mass to maintain Psat via latent heat/condensation/evaporation energy balance (simple)."""
        dm = min(self.m_liq, max(0.0, mdot_liq * dt))
        self.m_liq -= dm

        # Recompute required vapor mass at saturation for current T and gas volume
        Vg = self.gas_volume()
        # Required moles of N2O vapor to have partial pressure = Psat(T)
        P_sat = Psat_N2O_Pa(self.T)
        n_vap_req = P_sat * Vg / (R_univ * self.T)
        m_vap_req = n_vap_req * M_N2O

        # Phase change to drive toward m_vap_req
        dm_phase = m_vap_req - self.m_vap  # + => evaporation needed (cooling), - => condensation (heating)
        # Energy from phase change:
        Q_lat = - L_N2O * dm_phase  # negative if evaporation (heat absorbed), positive if condensing (heat released)

        # Lumped heat capacity
        C_eff = self.m_liq*cp_N2O_liq + max(self.m_vap,1e-6)*cp_N2O_vap + self.m_wall*cp_wall + (self.n_He*M_He)*cp_He
        dT = Q_lat / max(C_eff, 1.0)
        self.T += dT

        # Apply the phase change
        self.m_vap += dm_phase
        self.m_liq -= max(0.0, dm_phase)  # if evaporation occurred, pull from liquid

        # Non-negativity guards
        self.m_vap = max(0.0, self.m_vap)
        self.m_liq = max(0.0, self.m_liq)

# ---- Chamber & Nozzle -------------------------------------------------------

@dataclass
class ChamberNozzle:
    At: float             # throat area [m^2]  (N1)
    eps: float            # expansion ratio Ae/At
    Pa: float             # ambient pressure [Pa] (N2)
    cea: CEA_Obj          # RocketCEA object
    ox_name: str
    fuel_name: str

    def cstar_gamma_Tc(self, MR: float, Pc_bar: float) -> Tuple[float,float,float]:
        """
        Return (c*, gamma, Tc) from CEA for given mixture ratio MR (=O/F) and chamber pressure Pc [bar].
        RocketCEA c* uses consistent SI units when using get_Cstar, get_Throat_MachNumber may not be needed here.
        """
        # CEA methods: get_Cstar(Pc, MR, eps=None) returns m/s
        cstar = self.cea.get_Cstar(Pc=Pc_bar, MR=MR)  # m/s
        gamma = self.cea.get_Chamber_MolWt_gamma(Pc=Pc_bar, MR=MR)  # ratio of specific heats in chamber
        Tc = self.cea.get_Tcomb(Pc=Pc_bar, MR=MR)     # K
        return float(cstar), float(gamma), float(Tc)

    def isp_amb(self, MR: float, Pc_bar: float, eps: float) -> float:
        """
        Ambient Isp [s] from RocketCEA (includes exit pressure effects at Pa).
        """
        Isp = self.cea.estimate_Ambient_Isp(Pc=Pc_bar, MR=MR, eps=eps, Pamb=self.Pa)
        return float(Isp)

    def Pc_from_mdot(self, mdot_total: float, MR: float, Pc_guess: float = 20e5) -> float:
        """
        Fixed-point solve Pc using m_dot = Pc*At/c*  -> Pc = m_dot*c*/At,
        with c* from CEA at that Pc and MR.
        Pc in Pa internally; CEA uses bar -> convert.
        """
        Pc = Pc_guess
        for _ in range(20):
            Pc_bar = Pc / 1e5
            cstar, _, _ = self.cstar_gamma_Tc(MR, Pc_bar)
            Pc_new = mdot_total * cstar / max(self.At, 1e-9)
            if abs(Pc_new - Pc) < 100.0:  # Pa tolerance
                Pc = Pc_new
                break
            Pc = 0.5*Pc + 0.5*Pc_new  # relax
        return max(1e3, Pc)

# ---- Simulation Orchestrator ------------------------------------------------

@dataclass
class SimulationConfig:
    t_end: float
    dt: float
    # Pintle oxidizer injector area schedule A_ox(t) [m^2]
    A_ox_pintle_fn: Callable[[float], float]
    # Valve throttle schedules [0..1]
    throttle_ox_fn: Callable[[float], float]
    throttle_fuel_fn: Callable[[float], float]

@dataclass
class SystemState:
    t: float
    # Tanks
    ipa: IPATank
    n2o: N2OTank
    # Helium supply & plumbing
    he: HeliumBottle
    reg: IdealRegulator
    rest_ox: Restrictor
    rest_fuel: Restrictor
    # Branches to chamber
    branch_fuel: LiquidBranch
    branch_ox: LiquidBranch
    # Chamber/nozzle
    ch_noz: ChamberNozzle

def run_sim(cfg: SimulationConfig, st: SystemState) -> Dict[str, np.ndarray]:
    # Storage
    N = int(cfg.t_end / cfg.dt) + 1
    t_log = np.zeros(N)
    Pc_log = np.zeros(N)
    MR_log = np.zeros(N)
    mdot_fuel_log = np.zeros(N)
    mdot_ox_log = np.zeros(N)
    thrust_log = np.zeros(N)
    P_ipa_log = np.zeros(N)
    P_n2o_log = np.zeros(N)

    for i in range(N):
        t = i * cfg.dt
        t_log[i] = t

        # --- Manifold pressure (H3) via regulator (H2)
        P_reg_out = st.reg.outlet_pressure(st.he.P)

        # --- Helium branch to IPA ullage (H5 -> H7)
        P_ipa = st.ipa.pressure()
        m_dot_he_fuel = st.rest_fuel.mdot(P_up=P_reg_out, T_up=st.he.T, P_dn=P_ipa)
        m_withdrawn, T_he_out = st.he.withdraw(m_dot_he_fuel, cfg.dt)
        st.ipa.add_He(m_withdrawn, T_he_out)

        # --- Helium branch to N2O ullage (H4 -> H6)
        P_sat, P_he, P_n2o_tot = st.n2o.partial_pressures()
        m_dot_he_ox = st.rest_ox.mdot(P_up=P_reg_out, T_up=st.he.T, P_dn=P_n2o_tot)
        m_withdrawn2, T_he_out2 = st.he.withdraw(m_dot_he_ox, cfg.dt)
        st.n2o.add_He(m_withdrawn2, T_he_out2)

        # Update tank pressures for logging
        P_ipa = st.ipa.pressure()
        _, _, P_n2o_tot = st.n2o.partial_pressures()
        P_ipa_log[i] = P_ipa
        P_n2o_log[i] = P_n2o_tot

        # --- Chamber coupling iteration -------------------------------------
        # Pintle area (oxidizer), valve throttles (both)
        A_ox_inj = max(0.0, cfg.A_ox_pintle_fn(t))
        st.branch_ox.A_inj = A_ox_inj  # O4
        th_ox = max(0.0, min(1.0, cfg.throttle_ox_fn(t)))  # O3
        th_f = max(0.0, min(1.0, cfg.throttle_fuel_fn(t)))  # F3

        # Iterate Pc until consistent with flows
        Pc = 2.0e6  # Pa initial guess ~20 bar
        for _ in range(25):
            # Fuel branch mdot (F1/F2 -> F3 -> F4 -> chamber)
            mdot_fuel = st.branch_fuel.mdot_from_upstream(P_ipa, st.ipa.T, Pc, th_f)
            # Ox branch mdot (O1/O2 -> O3 -> O4 -> chamber)
            mdot_ox = st.branch_ox.mdot_from_upstream(P_n2o_tot, st.n2o.T, Pc, th_ox)

            mdot_total = mdot_fuel + mdot_ox
            MR = (mdot_ox / max(mdot_fuel, 1e-9))

            Pc_new = st.ch_noz.Pc_from_mdot(mdot_total=mdot_total, MR=MR, Pc_guess=Pc)
            if abs(Pc_new - Pc) < 500.0:
                Pc = Pc_new
                break
            Pc = 0.5*Pc + 0.5*Pc_new

        Pc_log[i] = Pc
        MR_log[i] = MR
        mdot_fuel_log[i] = mdot_fuel
        mdot_ox_log[i] = mdot_ox

        # --- Consume propellants and update tank states ---------------------
        st.ipa.withdraw_liquid(mdot_fuel, cfg.dt)
        st.n2o.withdraw_liquid_and_update_phase(mdot_ox, cfg.dt)

        # --- Nozzle performance (thrust via Isp_amb) ------------------------
        Pc_bar = Pc / 1e5
        Isp = st.ch_noz.isp_amb(MR=MR, Pc_bar=Pc_bar, eps=st.ch_noz.eps)
        thrust = (mdot_total * Isp * g0)
        thrust_log[i] = thrust

    return dict(
        t=t_log, Pc=Pc_log, MR=MR_log, mdot_fuel=mdot_fuel_log, mdot_ox=mdot_ox_log,
        thrust=thrust_log, P_ipa=P_ipa_log, P_n2o=P_n2o_log
    )

# ---- Example wiring (values are placeholders; tune for your system) --------
def default_system() -> Tuple[SimulationConfig, SystemState]:
    # RocketCEA setup
    OX = "N2O"                 # verify string in RocketCEA
    FUEL = "Isopropanol"               # verify string in RocketCEA
    cea = CEA_Obj(oxName=OX, fuelName=FUEL)

    # He supply (H1)
    he = HeliumBottle(V=2.0e-3, T=300.0, P=20e6)  # 2 liters at 200 bar

    # Regulator (H2) and manifold (H3) — ideal P source
    reg = IdealRegulator(P_set=35e5)  # 35 bar setpoint

    # Restrictors to tanks (H4/H5)
    rest_ox = Restrictor(Cd=0.8, A=0.35e-6)    # ~0.67 mm dia equivalent
    rest_fuel = Restrictor(Cd=0.8, A=0.25e-6)  # tweak as needed

    # Tanks
    ipa_tank = IPATank(V_tank=12e-3, m_wall=3.0, T=293.15, m_IPA=2.5, n_He=0.0)  # 12 L tank, 2.5 kg IPA
    # N2O tank with both liquid and vapor initially
    n2o_tank = N2OTank(V_tank=12e-3, m_wall=5.0, T=293.15, m_liq=5.0, m_vap=0.05, n_He=0.0)

    # Branch geometries: (L, D, K_minor, roughness)
    rough = 1.5e-6  # m (stainless)
    # Fuel branch: H7 -> line -> F3 -> line -> F4
    branch_fuel = LiquidBranch(
        L1=1.0, D1=6.0e-3, K1=2.0, rough1=rough,
        L2=0.5, D2=4.0e-3, K2=1.5, rough2=rough,
        Cd_valve=0.8, A_valve_max=math.pi*(3.0e-3)**2/4.0,
        Cd_inj=0.75, A_inj=math.pi*(1.8e-3)**2/4.0,  # fixed fuel orifice
        rho_fn=lambda T: density_IPA_liq(T),
        mu_fn=lambda T: mu_IPA
    )
    # Ox branch: H6 -> line -> O3 -> line -> O4 (pintle: A_inj will be set via schedule)
    branch_ox = LiquidBranch(
        L1=1.2, D1=6.0e-3, K1=2.5, rough1=rough,
        L2=0.6, D2=4.0e-3, K2=1.5, rough2=rough,
        Cd_valve=0.8, A_valve_max=math.pi*(3.0e-3)**2/4.0,
        Cd_inj=0.80, A_inj=1e-8,   # will be overwritten each step by pintle schedule
        rho_fn=lambda T: density_N2O_liq(T),
        mu_fn=lambda T: mu_N2O_liq
    )

    # Chamber/Nozzle
    ch_noz = ChamberNozzle(
        At=math.pi * (9e-3/2)**2,  # throat dia ~9 mm
        eps=2.8,                   # expansion area ratio
        Pa=101325.0,              # ambient
        cea=cea, ox_name=OX, fuel_name=FUEL
    )

    # Schedules
    def A_ox_pintle_fn(t: float) -> float:
        # Ramp open from 0 to 3.2 mm eq. dia over 0.25 s, then hold
        d = (3.2e-3) * min(1.0, t/0.25)
        return math.pi*(d**2)/4.0

    throttle_flat = lambda t: 1.0  # valves fully open in v1

    cfg = SimulationConfig(
        t_end=2.0, dt=0.005,
        A_ox_pintle_fn=A_ox_pintle_fn,
        throttle_ox_fn=throttle_flat,
        throttle_fuel_fn=throttle_flat
    )

    st = SystemState(
        t=0.0,
        ipa=ipa_tank,
        n2o=n2o_tank,
        he=he, reg=reg,
        rest_ox=rest_ox, rest_fuel=rest_fuel,
        branch_fuel=branch_fuel, branch_ox=branch_ox,
        ch_noz=ch_noz
    )
    return cfg, st

# ---- Entry point for standalone use ----------------------------------------
if __name__ == "__main__":
    cfg, st = default_system()
    out = run_sim(cfg, st)

    # Minimal text output; in practice, plot with matplotlib or save CSV
    print("Final t = %.3f s" % out["t"][-1])
    print("Final Pc = %.1f bar" % (out["Pc"][-1]/1e5))
    print("Final MR (O/F) = %.3f" % out["MR"][-1])
    print("Final thrust = %.1f N" % out["thrust"][-1])
    print("Final P_ipa = %.1f bar, P_n2o = %.1f bar" % (out["P_ipa"][-1]/1e5, out["P_n2o"][-1]/1e5))
