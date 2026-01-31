"""
Continuous-Time Smartphone Battery Discharge Model
MCM 2026 Problem A Solution

This module implements a physics-based continuous-time model for lithium-ion
battery discharge in smartphones, considering multiple factors:
- Screen power consumption
- Processor load
- Network activity (WiFi, cellular, GPS)
- Background processes
- Temperature effects
- Battery aging

The model is based on electrochemical principles and energy balance equations.
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize, curve_fit
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Dict
import warnings


@dataclass
class BatteryParameters:
    """
    Physical and operational parameters for the battery model.
    
    Based on typical smartphone lithium-ion battery specifications.
    References:
    - Lithium-ion battery datasheet specifications (Samsung SDI, LG Chem)
    - "Modeling of Lithium-Ion Battery Degradation" (Xu et al., 2016)
    """
    # Nominal battery capacity (mAh) - typical smartphone range: 3000-5000 mAh
    Q_nominal: float = 4000.0  # mAh
    
    # Nominal voltage (V) - Li-ion cell nominal voltage
    V_nominal: float = 3.7  # V
    
    # Open circuit voltage parameters (modified Shepherd model)
    # V_oc = E0 - K*Q/(Q-q) + A*exp(-B*q)
    E0: float = 4.2  # Maximum voltage (V)
    K: float = 0.009  # Polarization constant (V/mAh)
    A: float = 0.4  # Exponential zone amplitude (V)
    B: float = 0.0002  # Exponential zone time constant inverse (1/mAh)
    
    # Internal resistance (Ohms) - typical range 50-150 mOhm
    R_internal: float = 0.1  # Ohms
    
    # Temperature reference (Celsius)
    T_ref: float = 25.0  # °C
    
    # Arrhenius activation energy for capacity (J/mol)
    E_a_capacity: float = 20000.0  # J/mol
    
    # Arrhenius activation energy for resistance (J/mol)
    E_a_resistance: float = 15000.0  # J/mol
    
    # Self-discharge rate (fraction per day at reference temperature)
    self_discharge_rate: float = 0.002  # 0.2% per day
    
    # Minimum operational SOC
    SOC_min: float = 0.03  # 3% - phone typically shuts down
    
    # Battery health (State of Health) - 1.0 = new battery
    SOH: float = 1.0
    
    # Cycle count for aging estimation
    cycle_count: float = 0.0


@dataclass
class DeviceParameters:
    """
    Smartphone device parameters affecting power consumption.
    
    Based on typical flagship smartphone specifications.
    References:
    - Various smartphone teardown analyses (iFixit)
    - "Power Consumption Analysis of Smartphone" (Carroll & Heiser, 2010)
    """
    # Screen parameters
    screen_size: float = 6.5  # diagonal inches
    screen_resolution: Tuple[int, int] = (1080, 2400)  # pixels
    max_brightness_power: float = 2.5  # W at maximum brightness
    min_brightness_power: float = 0.3  # W at minimum brightness
    
    # Processor parameters (typical Snapdragon/A-series)
    cpu_idle_power: float = 0.1  # W
    cpu_max_power: float = 5.0  # W at 100% load
    
    # GPU parameters
    gpu_idle_power: float = 0.05  # W
    gpu_max_power: float = 4.0  # W
    
    # Network power consumption
    wifi_idle_power: float = 0.03  # W (connected but idle)
    wifi_active_power: float = 0.8  # W (active transfer)
    cellular_idle_power: float = 0.1  # W (4G/5G standby)
    cellular_active_power: float = 2.0  # W (active transfer)
    gps_power: float = 0.5  # W
    bluetooth_power: float = 0.05  # W
    
    # Baseline/always-on power
    baseline_power: float = 0.15  # W (sensors, memory refresh, etc.)
    
    # Modem standby power
    modem_standby_power: float = 0.08  # W


@dataclass
class UsageProfile:
    """
    Time-varying usage profile for simulation.
    
    Defines functions for each usage component over time.
    """
    # Screen brightness (0-1)
    screen_brightness: Callable[[float], float] = field(default=lambda t: 0.5)
    
    # Screen on/off (0 or 1)
    screen_on: Callable[[float], float] = field(default=lambda t: 1.0)
    
    # CPU load (0-1)
    cpu_load: Callable[[float], float] = field(default=lambda t: 0.2)
    
    # GPU load (0-1)
    gpu_load: Callable[[float], float] = field(default=lambda t: 0.0)
    
    # WiFi state (0=off, 0-0.3=idle, 0.3-1=active transfer proportional)
    wifi_activity: Callable[[float], float] = field(default=lambda t: 0.1)
    
    # Cellular state (0=off, 0-0.3=idle, 0.3-1=active transfer proportional)
    cellular_activity: Callable[[float], float] = field(default=lambda t: 0.1)
    
    # GPS active (0 or 1)
    gps_active: Callable[[float], float] = field(default=lambda t: 0.0)
    
    # Bluetooth active (0 or 1)
    bluetooth_active: Callable[[float], float] = field(default=lambda t: 0.0)
    
    # Ambient temperature (Celsius)
    temperature: Callable[[float], float] = field(default=lambda t: 25.0)


# Physical constants
R_GAS = 8.314  # Universal gas constant (J/(mol·K))
KELVIN_OFFSET = 273.15  # Celsius to Kelvin


class SmartphoneBatteryModel:
    """
    Continuous-time smartphone battery discharge model.
    
    The core equation governing State of Charge (SOC) is:
    
    dSOC/dt = -I(t) / Q_eff(T, SOH)
    
    Where:
    - I(t) = P(t) / V(SOC, T) is the discharge current
    - P(t) is the total power consumption
    - V(SOC, T) is the terminal voltage
    - Q_eff is the effective capacity considering temperature and aging
    
    The model incorporates:
    1. Electrochemical voltage-SOC relationship (modified Shepherd model)
    2. Arrhenius temperature dependence for capacity and resistance
    3. Component-wise power consumption model
    4. Battery aging effects on capacity and resistance
    """
    
    def __init__(self, 
                 battery_params: Optional[BatteryParameters] = None,
                 device_params: Optional[DeviceParameters] = None):
        """
        Initialize the battery model.
        
        Args:
            battery_params: Battery physical parameters
            device_params: Smartphone device parameters
        """
        self.battery = battery_params or BatteryParameters()
        self.device = device_params or DeviceParameters()
        
    def temperature_factor_capacity(self, T: float) -> float:
        """
        Calculate temperature correction factor for battery capacity.
        
        Uses Arrhenius equation to model capacity reduction at low temperatures.
        
        Args:
            T: Temperature in Celsius
            
        Returns:
            Capacity factor (0-1), 1.0 at reference temperature
            
        Reference:
        - "Temperature dependent battery models" (Tremblay & Dessaint, 2009)
        """
        T_K = T + KELVIN_OFFSET
        T_ref_K = self.battery.T_ref + KELVIN_OFFSET
        
        # Arrhenius-type temperature dependence
        factor = np.exp(self.battery.E_a_capacity / R_GAS * (1/T_ref_K - 1/T_K))
        
        # Additional capacity reduction at very low temperatures
        if T < 0:
            factor *= (1 + 0.01 * T)  # Additional 1% loss per degree below 0
            factor = max(factor, 0.5)  # Minimum 50% capacity
        elif T > 45:
            # Slight reduction at high temperatures due to safety limits
            factor *= (1 - 0.005 * (T - 45))
            factor = max(factor, 0.8)
            
        return np.clip(factor, 0.3, 1.2)
    
    def temperature_factor_resistance(self, T: float) -> float:
        """
        Calculate temperature correction factor for internal resistance.
        
        Internal resistance increases at low temperatures.
        
        Args:
            T: Temperature in Celsius
            
        Returns:
            Resistance factor (>= 1 at low temps, ~1 at ref temp)
        """
        T_K = T + KELVIN_OFFSET
        T_ref_K = self.battery.T_ref + KELVIN_OFFSET
        
        # Arrhenius-type temperature dependence (inverse for resistance)
        factor = np.exp(self.battery.E_a_resistance / R_GAS * (1/T_K - 1/T_ref_K))
        
        return np.clip(factor, 0.8, 5.0)
    
    def effective_capacity(self, T: float) -> float:
        """
        Calculate effective battery capacity considering temperature and aging.
        
        Q_eff = Q_nominal × SOH × f_T(T)
        
        Args:
            T: Temperature in Celsius
            
        Returns:
            Effective capacity in mAh
        """
        temp_factor = self.temperature_factor_capacity(T)
        return self.battery.Q_nominal * self.battery.SOH * temp_factor
    
    def effective_resistance(self, T: float) -> float:
        """
        Calculate effective internal resistance considering temperature and aging.
        
        R_eff = R_internal × (1 + 0.5 × (1-SOH)) × f_R(T)
        
        Args:
            T: Temperature in Celsius
            
        Returns:
            Effective internal resistance in Ohms
        """
        # Resistance increases with aging
        aging_factor = 1 + 0.5 * (1 - self.battery.SOH)
        temp_factor = self.temperature_factor_resistance(T)
        return self.battery.R_internal * aging_factor * temp_factor
    
    def open_circuit_voltage(self, SOC: float) -> float:
        """
        Calculate open circuit voltage as function of SOC.
        
        Uses modified Shepherd model:
        V_oc = E0 - K×(1-SOC)/SOC + A×exp(-B×Q_nom×(1-SOC))
        
        Args:
            SOC: State of charge (0-1)
            
        Returns:
            Open circuit voltage in Volts
        """
        SOC = np.clip(SOC, 0.01, 1.0)  # Avoid division by zero
        
        Q_discharged = self.battery.Q_nominal * (1 - SOC)
        
        # Modified Shepherd equation
        V_oc = (self.battery.E0 
                - self.battery.K * (1 - SOC) / SOC
                + self.battery.A * np.exp(-self.battery.B * Q_discharged))
        
        return np.clip(V_oc, 2.8, 4.35)  # Typical Li-ion voltage limits
    
    def terminal_voltage(self, SOC: float, I: float, T: float) -> float:
        """
        Calculate terminal voltage under load.
        
        V = V_oc(SOC) - I × R_eff(T)
        
        Args:
            SOC: State of charge (0-1)
            I: Discharge current in Amperes (positive for discharge)
            T: Temperature in Celsius
            
        Returns:
            Terminal voltage in Volts
        """
        V_oc = self.open_circuit_voltage(SOC)
        R_eff = self.effective_resistance(T)
        V = V_oc - I * R_eff
        return np.clip(V, 2.5, 4.35)
    
    def power_consumption(self, t: float, profile: UsageProfile) -> Dict[str, float]:
        """
        Calculate component-wise power consumption.
        
        Args:
            t: Time in hours
            profile: Usage profile with time-varying parameters
            
        Returns:
            Dictionary with power consumption breakdown in Watts
        """
        # Screen power
        screen_on = profile.screen_on(t)
        brightness = profile.screen_brightness(t)
        P_screen = screen_on * (
            self.device.min_brightness_power + 
            brightness * (self.device.max_brightness_power - self.device.min_brightness_power)
        )
        
        # CPU power (quadratic relationship with load for modern CPUs)
        cpu_load = profile.cpu_load(t)
        P_cpu = (self.device.cpu_idle_power + 
                 cpu_load**1.5 * (self.device.cpu_max_power - self.device.cpu_idle_power))
        
        # GPU power
        gpu_load = profile.gpu_load(t)
        P_gpu = (self.device.gpu_idle_power + 
                 gpu_load**1.5 * (self.device.gpu_max_power - self.device.gpu_idle_power))
        
        # WiFi power
        wifi_act = profile.wifi_activity(t)
        if wifi_act <= 0:
            P_wifi = 0
        elif wifi_act <= 0.3:
            P_wifi = self.device.wifi_idle_power
        else:
            P_wifi = (self.device.wifi_idle_power + 
                     (wifi_act - 0.3) / 0.7 * 
                     (self.device.wifi_active_power - self.device.wifi_idle_power))
        
        # Cellular power
        cell_act = profile.cellular_activity(t)
        if cell_act <= 0:
            P_cellular = 0
        elif cell_act <= 0.3:
            P_cellular = self.device.cellular_idle_power
        else:
            P_cellular = (self.device.cellular_idle_power + 
                         (cell_act - 0.3) / 0.7 * 
                         (self.device.cellular_active_power - self.device.cellular_idle_power))
        
        # GPS power
        P_gps = profile.gps_active(t) * self.device.gps_power
        
        # Bluetooth power
        P_bluetooth = profile.bluetooth_active(t) * self.device.bluetooth_power
        
        # Baseline power (always-on components)
        P_baseline = self.device.baseline_power + self.device.modem_standby_power
        
        return {
            'screen': P_screen,
            'cpu': P_cpu,
            'gpu': P_gpu,
            'wifi': P_wifi,
            'cellular': P_cellular,
            'gps': P_gps,
            'bluetooth': P_bluetooth,
            'baseline': P_baseline,
            'total': (P_screen + P_cpu + P_gpu + P_wifi + 
                     P_cellular + P_gps + P_bluetooth + P_baseline)
        }
    
    def discharge_rate(self, SOC: float, t: float, profile: UsageProfile) -> float:
        """
        Calculate the instantaneous discharge rate dSOC/dt.
        
        Core differential equation:
        dSOC/dt = -I(t) / Q_eff = -P(t) / (V(SOC,T) × Q_eff)
        
        Including self-discharge:
        dSOC/dt = -P(t) / (V(SOC,T) × Q_eff) - k_sd × SOC
        
        Args:
            SOC: Current state of charge (0-1)
            t: Time in hours
            profile: Usage profile
            
        Returns:
            Rate of change of SOC (per hour, negative for discharge)
        """
        if SOC <= self.battery.SOC_min:
            return 0.0  # Battery depleted
            
        # Get temperature and power consumption
        T = profile.temperature(t)
        power_dict = self.power_consumption(t, profile)
        P_total = power_dict['total']
        
        # Calculate effective parameters
        Q_eff = self.effective_capacity(T)  # mAh
        
        # Estimate current from power (iterative approach for accuracy)
        # P = V × I, and V depends on I
        # Use simple approximation: I ≈ P / V_oc
        V_approx = self.open_circuit_voltage(SOC)
        I_approx = P_total / V_approx  # Amperes
        
        # Refined voltage calculation
        V_terminal = self.terminal_voltage(SOC, I_approx, T)
        I_refined = P_total / V_terminal  # Amperes
        
        # Convert to mA for matching with Q_eff in mAh
        I_mA = I_refined * 1000  # mA
        
        # Discharge rate (per hour)
        dSOC_dt = -I_mA / Q_eff
        
        # Add self-discharge (converted to per-hour rate)
        k_sd = self.battery.self_discharge_rate / 24  # per hour
        dSOC_dt -= k_sd * SOC
        
        return dSOC_dt
    
    def simulate(self, 
                 SOC_initial: float,
                 profile: UsageProfile,
                 t_max: float = 24.0,
                 dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Simulate battery discharge over time.
        
        Solves the ODE: dSOC/dt = f(SOC, t)
        
        Args:
            SOC_initial: Initial state of charge (0-1)
            profile: Usage profile
            t_max: Maximum simulation time in hours
            dt: Time step for output in hours
            
        Returns:
            t: Time array (hours)
            SOC: State of charge array
            details: Dictionary with additional simulation details
        """
        def ode_func(t, y):
            SOC = y[0]
            if SOC <= self.battery.SOC_min:
                return [0.0]
            return [self.discharge_rate(SOC, t, profile)]
        
        # Event function to detect battery depletion
        def battery_empty(t, y):
            return y[0] - self.battery.SOC_min
        battery_empty.terminal = True
        battery_empty.direction = -1
        
        # Solve ODE
        t_span = (0, t_max)
        t_eval = np.arange(0, t_max + dt, dt)
        
        sol = solve_ivp(ode_func, t_span, [SOC_initial], 
                       t_eval=t_eval, events=battery_empty,
                       method='RK45', max_step=0.1)
        
        # Calculate additional details
        t = sol.t
        SOC = sol.y[0]
        
        # Power consumption over time
        power_breakdown = []
        for ti in t:
            idx = min(int(ti / dt), len(SOC) - 1)
            power_breakdown.append(self.power_consumption(ti, profile))
        
        # Time to empty
        if len(sol.t_events[0]) > 0:
            t_empty = sol.t_events[0][0]
        else:
            t_empty = t_max if SOC[-1] > self.battery.SOC_min else t[-1]
        
        details = {
            'power_breakdown': power_breakdown,
            't_empty': t_empty,
            'final_SOC': SOC[-1],
            'solver_success': sol.success
        }
        
        return t, SOC, details
    
    def time_to_empty(self, SOC_initial: float, profile: UsageProfile) -> float:
        """
        Calculate time until battery depletes to minimum SOC.
        
        Args:
            SOC_initial: Initial state of charge (0-1)
            profile: Usage profile
            
        Returns:
            Time to empty in hours
        """
        _, _, details = self.simulate(SOC_initial, profile, t_max=48.0)
        return details['t_empty']
    
    def analytical_time_estimate(self, 
                                 SOC_initial: float, 
                                 P_avg: float, 
                                 T: float = 25.0) -> float:
        """
        Analytical approximation for time to empty (constant conditions).
        
        For constant power P and approximately constant voltage V:
        t_empty ≈ (SOC_initial - SOC_min) × Q_eff × V_avg / P
        
        Args:
            SOC_initial: Initial SOC (0-1)
            P_avg: Average power consumption in Watts
            T: Temperature in Celsius
            
        Returns:
            Estimated time to empty in hours
        """
        Q_eff = self.effective_capacity(T)  # mAh
        SOC_mid = (SOC_initial + self.battery.SOC_min) / 2
        V_avg = self.open_circuit_voltage(SOC_mid)
        
        # Energy available (Wh)
        E_available = (SOC_initial - self.battery.SOC_min) * Q_eff * V_avg / 1000
        
        # Time to empty
        if P_avg > 0:
            t_empty = E_available / P_avg
        else:
            t_empty = float('inf')
            
        return t_empty


class BatteryAgingModel:
    """
    Model for battery capacity degradation over time and cycles.
    
    Uses semi-empirical degradation model based on:
    - Calendar aging (time-dependent)
    - Cycle aging (usage-dependent)
    
    Reference:
    - "Calendar and Cycle Life Study of Li-Ion Batteries" (Xu et al., 2016)
    """
    
    def __init__(self, battery_params: BatteryParameters):
        """
        Initialize aging model.
        
        Args:
            battery_params: Battery parameters
        """
        self.battery = battery_params
        
        # Aging parameters
        self.calendar_aging_rate = 0.02  # 2% capacity loss per year at room temp
        self.cycle_aging_rate = 0.0002  # 0.02% capacity loss per cycle
        self.depth_of_discharge_factor = 1.0  # Aging accelerated by deep discharge
        
    def capacity_fade(self, 
                      years: float, 
                      cycles: float, 
                      avg_DOD: float = 0.7,
                      avg_temp: float = 25.0) -> float:
        """
        Calculate capacity fade (State of Health).
        
        SOH = 1 - L_cal - L_cyc
        
        Where:
        - L_cal = calendar aging loss
        - L_cyc = cycle aging loss
        
        Args:
            years: Time since manufacture in years
            cycles: Number of charge cycles
            avg_DOD: Average depth of discharge (0-1)
            avg_temp: Average temperature in Celsius
            
        Returns:
            State of Health (0-1)
        """
        # Calendar aging (Arrhenius temperature dependence)
        temp_factor = np.exp(0.05 * (avg_temp - 25))  # ~5% increase per 10°C
        L_cal = self.calendar_aging_rate * years * temp_factor
        
        # Cycle aging (depends on DOD)
        DOD_factor = 1 + 2 * (avg_DOD - 0.5)**2  # Higher DOD accelerates aging
        L_cyc = self.cycle_aging_rate * cycles * DOD_factor
        
        # Total capacity fade
        SOH = 1 - L_cal - L_cyc
        
        return np.clip(SOH, 0.5, 1.0)
    
    def resistance_increase(self, years: float, cycles: float) -> float:
        """
        Calculate internal resistance increase factor due to aging.
        
        Args:
            years: Time since manufacture in years
            cycles: Number of charge cycles
            
        Returns:
            Resistance multiplication factor (>= 1)
        """
        # Resistance increases faster than capacity decreases
        R_factor = 1 + 0.03 * years + 0.0003 * cycles
        return np.clip(R_factor, 1.0, 2.0)


def create_usage_scenarios() -> Dict[str, UsageProfile]:
    """
    Create predefined usage scenarios for simulation.
    
    Returns:
        Dictionary of named usage profiles
    """
    scenarios = {}
    
    # Scenario 1: Idle/Standby
    scenarios['idle'] = UsageProfile(
        screen_brightness=lambda t: 0.0,
        screen_on=lambda t: 0.0,
        cpu_load=lambda t: 0.02,
        gpu_load=lambda t: 0.0,
        wifi_activity=lambda t: 0.1,
        cellular_activity=lambda t: 0.1,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 25.0
    )
    
    # Scenario 2: Light usage (web browsing, messaging)
    scenarios['light'] = UsageProfile(
        screen_brightness=lambda t: 0.5,
        screen_on=lambda t: 1.0 if (t % 0.5) < 0.3 else 0.0,  # Screen on 60% of time
        cpu_load=lambda t: 0.15 + 0.1 * np.sin(2 * np.pi * t),
        gpu_load=lambda t: 0.05,
        wifi_activity=lambda t: 0.4,
        cellular_activity=lambda t: 0.0,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 25.0
    )
    
    # Scenario 3: Moderate usage (social media, video streaming)
    scenarios['moderate'] = UsageProfile(
        screen_brightness=lambda t: 0.7,
        screen_on=lambda t: 1.0,
        cpu_load=lambda t: 0.35,
        gpu_load=lambda t: 0.3,
        wifi_activity=lambda t: 0.7,
        cellular_activity=lambda t: 0.0,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 28.0
    )
    
    # Scenario 4: Heavy usage (gaming)
    scenarios['heavy_gaming'] = UsageProfile(
        screen_brightness=lambda t: 0.9,
        screen_on=lambda t: 1.0,
        cpu_load=lambda t: 0.85,
        gpu_load=lambda t: 0.9,
        wifi_activity=lambda t: 0.5,
        cellular_activity=lambda t: 0.0,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 35.0 + 5.0 * (1 - np.exp(-t))  # Temperature rises
    )
    
    # Scenario 5: Navigation (GPS intensive)
    scenarios['navigation'] = UsageProfile(
        screen_brightness=lambda t: 0.8,
        screen_on=lambda t: 1.0,
        cpu_load=lambda t: 0.4,
        gpu_load=lambda t: 0.3,
        wifi_activity=lambda t: 0.0,
        cellular_activity=lambda t: 0.6,
        gps_active=lambda t: 1.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 30.0
    )
    
    # Scenario 6: Video call
    scenarios['video_call'] = UsageProfile(
        screen_brightness=lambda t: 0.7,
        screen_on=lambda t: 1.0,
        cpu_load=lambda t: 0.5,
        gpu_load=lambda t: 0.4,
        wifi_activity=lambda t: 0.8,
        cellular_activity=lambda t: 0.0,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 32.0
    )
    
    # Scenario 7: Cold weather
    scenarios['cold_weather'] = UsageProfile(
        screen_brightness=lambda t: 0.5,
        screen_on=lambda t: 1.0 if (t % 0.5) < 0.3 else 0.0,
        cpu_load=lambda t: 0.2,
        gpu_load=lambda t: 0.05,
        wifi_activity=lambda t: 0.0,
        cellular_activity=lambda t: 0.4,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: -5.0
    )
    
    # Scenario 8: Mixed daily usage
    def daily_screen(t):
        # Simulate realistic daily pattern
        hour = t % 24
        if 7 <= hour < 9:  # Morning
            return 1.0
        elif 9 <= hour < 12:  # Morning work
            return 1.0 if np.random.random() < 0.3 else 0.0
        elif 12 <= hour < 13:  # Lunch
            return 1.0
        elif 13 <= hour < 18:  # Afternoon
            return 1.0 if np.random.random() < 0.4 else 0.0
        elif 18 <= hour < 23:  # Evening
            return 1.0
        else:  # Night
            return 0.0
    
    def daily_cpu(t):
        hour = t % 24
        if 7 <= hour < 23:
            return 0.15 + 0.25 * np.sin(np.pi * (hour - 7) / 16)
        return 0.02
    
    scenarios['daily_mixed'] = UsageProfile(
        screen_brightness=lambda t: 0.6,
        screen_on=daily_screen,
        cpu_load=daily_cpu,
        gpu_load=lambda t: 0.1 if 18 <= (t % 24) < 22 else 0.02,
        wifi_activity=lambda t: 0.5,
        cellular_activity=lambda t: 0.2,
        gps_active=lambda t: 1.0 if 8 <= (t % 24) < 9 else 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 25.0
    )
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    model = SmartphoneBatteryModel()
    scenarios = create_usage_scenarios()
    
    # Test with moderate usage
    profile = scenarios['moderate']
    t, SOC, details = model.simulate(1.0, profile, t_max=12.0)
    
    print(f"Simulation Results (Moderate Usage):")
    print(f"  Initial SOC: 100%")
    print(f"  Final SOC: {details['final_SOC']*100:.1f}%")
    print(f"  Time to Empty: {details['t_empty']:.2f} hours")
