"""
Continuous-Time Smartphone Battery Discharge Model
Based on electrochemical principles and energy consumption dynamics
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import matplotlib.pyplot as plt


@dataclass
class BatteryParameters:
    """Physical and operational parameters for lithium-ion battery"""
    
    # Battery specifications
    Q_nominal: float = 3000.0  # Nominal capacity in mAh
    V_nominal: float = 3.8  # Nominal voltage in V
    
    # Self-discharge and internal resistance
    k_self: float = 0.0001  # Self-discharge rate (1/hour)
    R_internal: float = 0.15  # Internal resistance (Ohm)
    
    # Temperature effects (Arrhenius-based)
    T_ref: float = 298.15  # Reference temperature (K) = 25°C
    E_a: float = 0.3  # Activation energy (eV)
    k_B: float = 8.617e-5  # Boltzmann constant (eV/K)
    
    # Component power consumption (mW)
    P_idle: float = 50.0  # Idle power
    P_screen_base: float = 200.0  # Screen base power
    P_screen_per_brightness: float = 500.0  # Additional per brightness unit
    P_cpu_idle: float = 100.0  # CPU idle power
    P_cpu_per_load: float = 1500.0  # Additional per CPU load unit
    P_network_2g: float = 100.0
    P_network_3g: float = 300.0
    P_network_4g: float = 400.0
    P_network_5g: float = 600.0
    P_wifi: float = 150.0
    P_gps: float = 250.0
    P_bluetooth: float = 50.0
    
    # Aging parameters
    capacity_fade_rate: float = 0.0002  # Per cycle equivalent
    resistance_growth_rate: float = 0.0001  # Per cycle equivalent
    
    # Thermal parameters
    thermal_mass: float = 50.0  # J/K
    thermal_resistance: float = 5.0  # K/W
    T_ambient: float = 298.15  # Ambient temperature (K)


class UsageProfile:
    """Defines smartphone usage patterns over time"""
    
    def __init__(self):
        self.brightness = lambda t: 0.5  # 0-1 scale
        self.cpu_load = lambda t: 0.2  # 0-1 scale
        self.network_type = lambda t: '4g'  # '2g', '3g', '4g', '5g', 'wifi', 'off'
        self.gps_active = lambda t: False
        self.bluetooth_active = lambda t: False
        self.screen_on = lambda t: False
        
    def set_screen_brightness(self, func: Callable[[float], float]):
        """Set screen brightness as function of time (0-1)"""
        self.brightness = func
        
    def set_cpu_load(self, func: Callable[[float], float]):
        """Set CPU load as function of time (0-1)"""
        self.cpu_load = func
        
    def set_network_type(self, func: Callable[[float], str]):
        """Set network type as function of time"""
        self.network_type = func
        
    def set_gps(self, func: Callable[[float], bool]):
        """Set GPS active as function of time"""
        self.gps_active = func
        
    def set_bluetooth(self, func: Callable[[float], bool]):
        """Set Bluetooth active as function of time"""
        self.bluetooth_active = func
        
    def set_screen_state(self, func: Callable[[float], bool]):
        """Set screen on/off as function of time"""
        self.screen_on = func


class BatteryModel:
    """
    Continuous-time battery discharge model for smartphones
    
    State variables:
    - SOC: State of Charge (0-1)
    - T: Battery temperature (K)
    - cycles: Equivalent charge cycles (for aging)
    """
    
    def __init__(self, params: BatteryParameters):
        self.params = params
        self.usage_profile = UsageProfile()
        
    def temperature_factor(self, T: float) -> float:
        """
        Temperature effect on reaction rates (Arrhenius equation)
        Returns multiplicative factor for discharge rate
        """
        params = self.params
        exp_term = (params.E_a / params.k_B) * (1/T - 1/params.T_ref)
        return np.exp(-exp_term)
    
    def internal_resistance(self, SOC: float, cycles: float, T: float) -> float:
        """
        Internal resistance as function of SOC, aging, and temperature
        """
        params = self.params
        
        # Base resistance increases with aging
        R_base = params.R_internal * (1 + params.resistance_growth_rate * cycles)
        
        # SOC dependency (increases at low SOC)
        R_soc = R_base * (1 + 0.5 * np.exp(-10 * SOC))
        
        # Temperature dependency
        temp_factor = 1 + 0.5 * np.exp(-(T - 273.15) / 10)
        
        return R_soc * temp_factor
    
    def effective_capacity(self, cycles: float) -> float:
        """
        Effective capacity accounting for aging
        Capacity fade follows empirical lithium-ion degradation
        """
        params = self.params
        # Typical capacity retention: 80% after 500 cycles
        fade = np.exp(-params.capacity_fade_rate * cycles)
        return params.Q_nominal * fade
    
    def power_consumption(self, t: float, SOC: float, T: float) -> float:
        """
        Total instantaneous power consumption (mW)
        Based on usage profile and operating conditions
        """
        params = self.params
        profile = self.usage_profile
        
        # Base idle power
        P_total = params.P_idle
        
        # Screen power (if on)
        if profile.screen_on(t):
            brightness = profile.brightness(t)
            P_total += params.P_screen_base + params.P_screen_per_brightness * brightness
        
        # CPU power
        cpu_load = profile.cpu_load(t)
        P_total += params.P_cpu_idle + params.P_cpu_per_load * cpu_load
        
        # Network power
        network = profile.network_type(t)
        network_power_map = {
            '2g': params.P_network_2g,
            '3g': params.P_network_3g,
            '4g': params.P_network_4g,
            '5g': params.P_network_5g,
            'wifi': params.P_wifi,
            'off': 0
        }
        P_total += network_power_map.get(network, 0)
        
        # GPS power
        if profile.gps_active(t):
            P_total += params.P_gps
        
        # Bluetooth power
        if profile.bluetooth_active(t):
            P_total += params.P_bluetooth
        
        # Temperature effect on all processes
        temp_factor = self.temperature_factor(T)
        P_total *= temp_factor
        
        return P_total
    
    def thermal_dynamics(self, T: float, P_dissipated: float) -> float:
        """
        Battery temperature rate of change (dT/dt)
        Based on heat generation and dissipation
        """
        params = self.params
        
        # Heat dissipation to ambient
        Q_out = (T - params.T_ambient) / params.thermal_resistance
        
        # Net heat flow
        dT_dt = (P_dissipated - Q_out) / params.thermal_mass
        
        return dT_dt
    
    def state_equations(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Continuous-time state equations
        
        d(SOC)/dt = -I_total / Q_eff
        d(T)/dt = (P_heat - Q_dissipation) / C_thermal
        d(cycles)/dt = |I_total| / (2 * Q_nominal)
        
        where:
        - I_total includes load current and self-discharge
        - P_heat comes from I²R losses and irreversible reactions
        - cycles accumulates for aging calculations
        """
        SOC, T, cycles = state
        params = self.params
        
        # Prevent negative SOC
        if SOC <= 0:
            return np.array([0.0, 0.0, 0.0])
        
        # Effective capacity with aging
        Q_eff = self.effective_capacity(cycles)
        
        # Power consumption
        P_load = self.power_consumption(t, SOC, T)
        
        # Current draw (mA) from power and voltage
        # P = V * I, so I = P / V
        # Include voltage drop due to internal resistance
        R_int = self.internal_resistance(SOC, cycles, T)
        
        # Solve for current: V_oc - I*R = V_terminal
        # P = V_terminal * I = (V_oc - I*R) * I
        # Approximate for small R: I ≈ P / V_oc
        V_oc = params.V_nominal * (0.95 + 0.1 * SOC)  # OCV varies with SOC
        I_load = P_load / V_oc
        
        # Self-discharge current
        I_self = params.k_self * Q_eff
        
        # Total current
        I_total = I_load + I_self
        
        # Heat generation (I²R losses + entropy)
        P_heat = (I_total / 1000) ** 2 * R_int * 1000  # Convert to mW
        P_heat += 0.1 * P_load  # Irreversible entropy (≈10% of power)
        
        # State derivatives
        dSOC_dt = -I_total / Q_eff  # Per hour
        dT_dt = self.thermal_dynamics(T, P_heat / 1000)  # Convert mW to W
        dcycles_dt = I_total / (2 * params.Q_nominal)  # Cycle accumulation
        
        return np.array([dSOC_dt, dT_dt, dcycles_dt])
    
    def simulate(self, 
                 t_span: tuple, 
                 initial_SOC: float = 1.0,
                 initial_T: float = 298.15,
                 initial_cycles: float = 0.0,
                 t_eval: Optional[np.ndarray] = None) -> dict:
        """
        Simulate battery discharge over time
        
        Parameters:
        -----------
        t_span: tuple
            (t_start, t_end) in hours
        initial_SOC: float
            Initial state of charge (0-1)
        initial_T: float
            Initial temperature (K)
        initial_cycles: float
            Initial equivalent cycles (for aged battery)
        t_eval: np.ndarray
            Time points for solution output
            
        Returns:
        --------
        dict with keys:
            - t: time array
            - SOC: state of charge array
            - T: temperature array
            - cycles: cycles array
            - voltage: terminal voltage array
            - current: current draw array
            - power: power consumption array
        """
        # Initial state
        state0 = np.array([initial_SOC, initial_T, initial_cycles])
        
        # Solve ODE
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        solution = solve_ivp(
            lambda t, y: self.state_equations(t, y),
            t_span,
            state0,
            t_eval=t_eval,
            method='LSODA',
            max_step=0.1
        )
        
        # Extract results
        t = solution.t
        SOC = solution.y[0]
        T = solution.y[1]
        cycles = solution.y[2]
        
        # Calculate derived quantities
        voltage = np.zeros_like(t)
        current = np.zeros_like(t)
        power = np.zeros_like(t)
        
        for i, ti in enumerate(t):
            if SOC[i] > 0:
                # Open circuit voltage
                V_oc = self.params.V_nominal * (0.95 + 0.1 * SOC[i])
                
                # Power and current
                power[i] = self.power_consumption(ti, SOC[i], T[i])
                current[i] = power[i] / V_oc
                
                # Terminal voltage with IR drop
                R_int = self.internal_resistance(SOC[i], cycles[i], T[i])
                voltage[i] = V_oc - (current[i] / 1000) * R_int
            else:
                voltage[i] = 0
                current[i] = 0
                power[i] = 0
        
        return {
            't': t,
            'SOC': SOC,
            'T': T,
            'cycles': cycles,
            'voltage': voltage,
            'current': current,
            'power': power
        }
    
    def time_to_empty(self, 
                      initial_SOC: float = 1.0,
                      initial_T: float = 298.15,
                      initial_cycles: float = 0.0,
                      SOC_cutoff: float = 0.05) -> float:
        """
        Calculate time until battery is empty (SOC < cutoff)
        
        Returns time in hours
        """
        # Simulate for a long time
        max_time = 50  # hours
        t_eval = np.linspace(0, max_time, 5000)
        
        result = self.simulate(
            (0, max_time),
            initial_SOC,
            initial_T,
            initial_cycles,
            t_eval
        )
        
        # Find when SOC drops below cutoff
        idx = np.where(result['SOC'] <= SOC_cutoff)[0]
        
        if len(idx) > 0:
            return result['t'][idx[0]]
        else:
            return max_time  # Battery lasted longer than simulation


def create_usage_scenarios():
    """Create different usage scenarios for testing"""
    
    scenarios = {}
    
    # Scenario 1: Light usage (mostly idle)
    profile1 = UsageProfile()
    profile1.set_screen_state(lambda t: t % 1 < 0.1)  # Screen on 10% of time
    profile1.set_screen_brightness(lambda t: 0.3)
    profile1.set_cpu_load(lambda t: 0.1)
    profile1.set_network_type(lambda t: 'wifi')
    scenarios['light'] = profile1
    
    # Scenario 2: Normal usage
    profile2 = UsageProfile()
    profile2.set_screen_state(lambda t: (t % 1 < 0.3))  # Screen on 30% of time
    profile2.set_screen_brightness(lambda t: 0.5)
    profile2.set_cpu_load(lambda t: 0.3 + 0.2 * np.sin(2 * np.pi * t))
    profile2.set_network_type(lambda t: '4g')
    scenarios['normal'] = profile2
    
    # Scenario 3: Heavy usage (gaming, video)
    profile3 = UsageProfile()
    profile3.set_screen_state(lambda t: True)  # Screen always on
    profile3.set_screen_brightness(lambda t: 0.8)
    profile3.set_cpu_load(lambda t: 0.7 + 0.2 * np.sin(2 * np.pi * t))
    profile3.set_network_type(lambda t: '4g')
    scenarios['heavy'] = profile3
    
    # Scenario 4: Navigation (GPS heavy)
    profile4 = UsageProfile()
    profile4.set_screen_state(lambda t: True)
    profile4.set_screen_brightness(lambda t: 0.9)
    profile4.set_cpu_load(lambda t: 0.4)
    profile4.set_network_type(lambda t: '4g')
    profile4.set_gps(lambda t: True)
    scenarios['navigation'] = profile4
    
    # Scenario 5: Video streaming
    profile5 = UsageProfile()
    profile5.set_screen_state(lambda t: True)
    profile5.set_screen_brightness(lambda t: 0.6)
    profile5.set_cpu_load(lambda t: 0.5)
    profile5.set_network_type(lambda t: 'wifi')
    scenarios['streaming'] = profile5
    
    return scenarios


if __name__ == "__main__":
    # Example usage
    params = BatteryParameters()
    model = BatteryModel(params)
    
    # Set a simple usage profile
    model.usage_profile.set_screen_state(lambda t: t % 2 < 1)  # 50% screen time
    model.usage_profile.set_screen_brightness(lambda t: 0.5)
    model.usage_profile.set_cpu_load(lambda t: 0.3)
    model.usage_profile.set_network_type(lambda t: '4g')
    
    # Simulate
    result = model.simulate((0, 10), initial_SOC=1.0)
    
    print(f"Time to empty: {model.time_to_empty()} hours")
    print(f"Final SOC: {result['SOC'][-1]:.2%}")
    print(f"Final temperature: {result['T'][-1] - 273.15:.1f}°C")
