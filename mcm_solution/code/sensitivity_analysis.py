"""
Sensitivity Analysis for Battery Discharge Model
Analyze how variations in parameters and usage patterns affect predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from battery_model import BatteryModel, BatteryParameters, UsageProfile
from typing import Dict, List, Callable
import seaborn as sns


class SensitivityAnalyzer:
    """Perform comprehensive sensitivity analysis"""
    
    def __init__(self, base_params: BatteryParameters):
        self.base_params = base_params
        
    def parameter_sensitivity(self, 
                             param_name: str,
                             param_range: np.ndarray,
                             usage_profile: UsageProfile) -> Dict:
        """
        Analyze sensitivity to a single parameter
        
        Parameters:
        -----------
        param_name: str
            Name of parameter to vary
        param_range: np.ndarray
            Range of values to test
        usage_profile: UsageProfile
            Usage pattern for simulation
            
        Returns:
        --------
        Dict with parameter values and corresponding time-to-empty
        """
        results = {
            'param_values': param_range,
            'time_to_empty': [],
            'final_temp': [],
            'avg_power': []
        }
        
        for param_value in param_range:
            # Create modified parameters
            params = BatteryParameters()
            setattr(params, param_name, param_value)
            
            # Create and simulate model
            model = BatteryModel(params)
            model.usage_profile = usage_profile
            
            tte = model.time_to_empty()
            
            # Get additional metrics
            sim_result = model.simulate((0, min(tte, 24)))
            
            results['time_to_empty'].append(tte)
            results['final_temp'].append(sim_result['T'][-1] - 273.15)
            results['avg_power'].append(np.mean(sim_result['power']))
        
        results['time_to_empty'] = np.array(results['time_to_empty'])
        results['final_temp'] = np.array(results['final_temp'])
        results['avg_power'] = np.array(results['avg_power'])
        
        return results
    
    def usage_pattern_sensitivity(self) -> Dict:
        """Analyze sensitivity to different usage patterns"""
        
        # Define parameter variations
        brightness_levels = np.linspace(0, 1, 6)
        cpu_loads = np.linspace(0, 1, 6)
        screen_times = np.linspace(0, 1, 6)  # Fraction of time screen is on
        
        results = {
            'brightness': {'values': brightness_levels, 'tte': []},
            'cpu_load': {'values': cpu_loads, 'tte': []},
            'screen_time': {'values': screen_times, 'tte': []}
        }
        
        model = BatteryModel(self.base_params)
        
        # Brightness sensitivity
        for brightness in brightness_levels:
            profile = UsageProfile()
            profile.set_screen_state(lambda t: True)
            profile.set_screen_brightness(lambda t: brightness)
            profile.set_cpu_load(lambda t: 0.3)
            profile.set_network_type(lambda t: '4g')
            
            model.usage_profile = profile
            tte = model.time_to_empty()
            results['brightness']['tte'].append(tte)
        
        # CPU load sensitivity
        for cpu_load in cpu_loads:
            profile = UsageProfile()
            profile.set_screen_state(lambda t: True)
            profile.set_screen_brightness(lambda t: 0.5)
            profile.set_cpu_load(lambda t: cpu_load)
            profile.set_network_type(lambda t: '4g')
            
            model.usage_profile = profile
            tte = model.time_to_empty()
            results['cpu_load']['tte'].append(tte)
        
        # Screen time sensitivity
        for screen_frac in screen_times:
            profile = UsageProfile()
            profile.set_screen_state(lambda t: (t % 1) < screen_frac)
            profile.set_screen_brightness(lambda t: 0.5)
            profile.set_cpu_load(lambda t: 0.3)
            profile.set_network_type(lambda t: '4g')
            
            model.usage_profile = profile
            tte = model.time_to_empty()
            results['screen_time']['tte'].append(tte)
        
        # Convert to arrays
        for key in results:
            results[key]['tte'] = np.array(results[key]['tte'])
        
        return results
    
    def network_type_comparison(self) -> Dict:
        """Compare impact of different network types"""
        
        network_types = ['wifi', '2g', '3g', '4g', '5g']
        results = {
            'network_types': network_types,
            'time_to_empty': [],
            'power_consumption': []
        }
        
        model = BatteryModel(self.base_params)
        
        for net_type in network_types:
            profile = UsageProfile()
            profile.set_screen_state(lambda t: (t % 1) < 0.3)
            profile.set_screen_brightness(lambda t: 0.5)
            profile.set_cpu_load(lambda t: 0.3)
            profile.set_network_type(lambda t: net_type)
            
            model.usage_profile = profile
            tte = model.time_to_empty()
            
            # Calculate average power
            sim = model.simulate((0, min(tte, 10)))
            avg_power = np.mean(sim['power'])
            
            results['time_to_empty'].append(tte)
            results['power_consumption'].append(avg_power)
        
        results['time_to_empty'] = np.array(results['time_to_empty'])
        results['power_consumption'] = np.array(results['power_consumption'])
        
        return results
    
    def temperature_sensitivity(self) -> Dict:
        """Analyze temperature impact on battery life"""
        
        temperatures_C = np.linspace(-10, 45, 12)
        temperatures_K = temperatures_C + 273.15
        
        results = {
            'temperature_C': temperatures_C,
            'time_to_empty': [],
            'capacity_factor': []
        }
        
        model = BatteryModel(self.base_params)
        
        # Normal usage profile
        profile = UsageProfile()
        profile.set_screen_state(lambda t: (t % 1) < 0.3)
        profile.set_screen_brightness(lambda t: 0.5)
        profile.set_cpu_load(lambda t: 0.3)
        profile.set_network_type(lambda t: '4g')
        
        model.usage_profile = profile
        
        # Reference at 25°C
        tte_ref = model.time_to_empty(initial_T=298.15)
        
        for temp_K in temperatures_K:
            tte = model.time_to_empty(initial_T=temp_K)
            capacity_factor = tte / tte_ref
            
            results['time_to_empty'].append(tte)
            results['capacity_factor'].append(capacity_factor)
        
        results['time_to_empty'] = np.array(results['time_to_empty'])
        results['capacity_factor'] = np.array(results['capacity_factor'])
        
        return results
    
    def aging_sensitivity(self) -> Dict:
        """Analyze battery aging impact"""
        
        cycle_counts = np.linspace(0, 1000, 11)
        
        results = {
            'cycles': cycle_counts,
            'time_to_empty': [],
            'capacity_retention': [],
            'resistance_increase': []
        }
        
        model = BatteryModel(self.base_params)
        
        # Normal usage profile
        profile = UsageProfile()
        profile.set_screen_state(lambda t: (t % 1) < 0.3)
        profile.set_screen_brightness(lambda t: 0.5)
        profile.set_cpu_load(lambda t: 0.3)
        profile.set_network_type(lambda t: '4g')
        
        model.usage_profile = profile
        
        for cycles in cycle_counts:
            tte = model.time_to_empty(initial_cycles=cycles)
            capacity_ret = model.effective_capacity(cycles) / model.params.Q_nominal
            resistance = model.internal_resistance(0.5, cycles, 298.15)
            
            results['time_to_empty'].append(tte)
            results['capacity_retention'].append(capacity_ret)
            results['resistance_increase'].append(resistance / self.base_params.R_internal)
        
        results['time_to_empty'] = np.array(results['time_to_empty'])
        results['capacity_retention'] = np.array(results['capacity_retention'])
        results['resistance_increase'] = np.array(results['resistance_increase'])
        
        return results
    
    def multi_parameter_sensitivity(self) -> Dict:
        """
        Perform global sensitivity analysis
        Calculate sensitivity indices for multiple parameters
        """
        
        # Parameters to analyze
        param_variations = {
            'Q_nominal': (0.8, 1.2),  # ±20%
            'P_screen_per_brightness': (0.8, 1.2),
            'P_cpu_per_load': (0.8, 1.2),
            'R_internal': (0.5, 1.5),
            'k_self': (0.5, 2.0),
        }
        
        n_samples = 100
        results = {
            'parameters': list(param_variations.keys()),
            'sensitivity_indices': [],
            'mean_tte': [],
            'std_tte': []
        }
        
        # Base case
        base_model = BatteryModel(self.base_params)
        profile = UsageProfile()
        profile.set_screen_state(lambda t: (t % 1) < 0.3)
        profile.set_screen_brightness(lambda t: 0.5)
        profile.set_cpu_load(lambda t: 0.3)
        profile.set_network_type(lambda t: '4g')
        base_model.usage_profile = profile
        tte_base = base_model.time_to_empty()
        
        # One-at-a-time sensitivity
        for param_name, (low_mult, high_mult) in param_variations.items():
            base_value = getattr(self.base_params, param_name)
            param_values = np.linspace(base_value * low_mult, base_value * high_mult, n_samples)
            
            tte_values = []
            for param_val in param_values:
                params = BatteryParameters()
                setattr(params, param_name, param_val)
                
                model = BatteryModel(params)
                model.usage_profile = profile
                tte = model.time_to_empty()
                tte_values.append(tte)
            
            tte_values = np.array(tte_values)
            
            # Sensitivity index: normalized standard deviation
            sensitivity = np.std(tte_values) / tte_base
            
            results['sensitivity_indices'].append(sensitivity)
            results['mean_tte'].append(np.mean(tte_values))
            results['std_tte'].append(np.std(tte_values))
        
        return results


def create_sensitivity_plots(analyzer: SensitivityAnalyzer):
    """Generate comprehensive sensitivity analysis plots"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Usage pattern sensitivity
    ax1 = fig.add_subplot(gs[0, 0])
    usage_results = analyzer.usage_pattern_sensitivity()
    
    ax1.plot(usage_results['brightness']['values'], usage_results['brightness']['tte'], 
             'o-', label='Brightness', linewidth=2, markersize=8)
    ax1.set_xlabel('Brightness Level (0-1)')
    ax1.set_ylabel('Time to Empty (hours)')
    ax1.set_title('Screen Brightness Sensitivity')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(usage_results['cpu_load']['values'], usage_results['cpu_load']['tte'], 
             's-', label='CPU Load', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('CPU Load (0-1)')
    ax2.set_ylabel('Time to Empty (hours)')
    ax2.set_title('CPU Load Sensitivity')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(usage_results['screen_time']['values'], usage_results['screen_time']['tte'], 
             '^-', label='Screen Time', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Screen On Fraction (0-1)')
    ax3.set_ylabel('Time to Empty (hours)')
    ax3.set_title('Screen Time Sensitivity')
    ax3.grid(True, alpha=0.3)
    
    # 2. Network type comparison
    ax4 = fig.add_subplot(gs[1, 0])
    network_results = analyzer.network_type_comparison()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(network_results['network_types'])))
    bars = ax4.barh(network_results['network_types'], network_results['time_to_empty'], color=colors)
    ax4.set_xlabel('Time to Empty (hours)')
    ax4.set_title('Network Type Impact')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 3. Temperature sensitivity
    ax5 = fig.add_subplot(gs[1, 1])
    temp_results = analyzer.temperature_sensitivity()
    
    ax5.plot(temp_results['temperature_C'], temp_results['time_to_empty'], 
             'o-', linewidth=2, markersize=6, color='orangered')
    ax5.axvline(x=25, color='gray', linestyle='--', alpha=0.5, label='25°C reference')
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('Time to Empty (hours)')
    ax5.set_title('Temperature Sensitivity')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 4. Temperature capacity factor
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(temp_results['temperature_C'], temp_results['capacity_factor'], 
             's-', linewidth=2, markersize=6, color='purple')
    ax6.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Temperature (°C)')
    ax6.set_ylabel('Relative Capacity')
    ax6.set_title('Temperature Effect on Capacity')
    ax6.grid(True, alpha=0.3)
    
    # 5. Aging effects
    ax7 = fig.add_subplot(gs[2, 0])
    aging_results = analyzer.aging_sensitivity()
    
    ax7.plot(aging_results['cycles'], aging_results['time_to_empty'], 
             'o-', linewidth=2, markersize=6, color='brown')
    ax7.set_xlabel('Charge Cycles')
    ax7.set_ylabel('Time to Empty (hours)')
    ax7.set_title('Battery Aging: Time to Empty')
    ax7.grid(True, alpha=0.3)
    
    # 6. Capacity retention vs cycles
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(aging_results['cycles'], aging_results['capacity_retention'] * 100, 
             's-', linewidth=2, markersize=6, color='teal')
    ax8.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    ax8.set_xlabel('Charge Cycles')
    ax8.set_ylabel('Capacity Retention (%)')
    ax8.set_title('Battery Aging: Capacity Fade')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 7. Multi-parameter sensitivity
    ax9 = fig.add_subplot(gs[2, 2])
    multi_results = analyzer.multi_parameter_sensitivity()
    
    params_short = ['Capacity', 'Screen\nPower', 'CPU\nPower', 'Resistance', 'Self-\nDischarge']
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(multi_results['sensitivity_indices'])))
    bars = ax9.bar(params_short, multi_results['sensitivity_indices'], color=colors)
    ax9.set_ylabel('Sensitivity Index')
    ax9.set_title('Parameter Sensitivity Ranking')
    ax9.tick_params(axis='x', rotation=0)
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comprehensive Sensitivity Analysis', fontsize=16, fontweight='bold')
    
    return fig


def generate_sensitivity_report(analyzer: SensitivityAnalyzer):
    """Generate text report of sensitivity analysis"""
    
    print("=" * 80)
    print("SENSITIVITY ANALYSIS REPORT")
    print("=" * 80)
    
    # Usage pattern sensitivity
    print("\n1. USAGE PATTERN SENSITIVITY")
    print("-" * 80)
    usage_results = analyzer.usage_pattern_sensitivity()
    
    for param, data in usage_results.items():
        tte_range = data['tte'].max() - data['tte'].min()
        tte_pct_change = (tte_range / data['tte'].mean()) * 100
        print(f"\n{param.upper()}:")
        print(f"  Range: {data['tte'].min():.2f} - {data['tte'].max():.2f} hours")
        print(f"  Variation: {tte_range:.2f} hours ({tte_pct_change:.1f}%)")
        print(f"  Sensitivity: {'HIGH' if tte_pct_change > 50 else 'MEDIUM' if tte_pct_change > 20 else 'LOW'}")
    
    # Network type comparison
    print("\n\n2. NETWORK TYPE COMPARISON")
    print("-" * 80)
    network_results = analyzer.network_type_comparison()
    
    for i, net_type in enumerate(network_results['network_types']):
        print(f"{net_type.upper():>10}: {network_results['time_to_empty'][i]:>6.2f} hours  "
              f"(Power: {network_results['power_consumption'][i]:>6.1f} mW)")
    
    best_idx = np.argmax(network_results['time_to_empty'])
    worst_idx = np.argmin(network_results['time_to_empty'])
    print(f"\nBest: {network_results['network_types'][best_idx]}")
    print(f"Worst: {network_results['network_types'][worst_idx]}")
    print(f"Difference: {network_results['time_to_empty'][best_idx] - network_results['time_to_empty'][worst_idx]:.2f} hours")
    
    # Temperature sensitivity
    print("\n\n3. TEMPERATURE SENSITIVITY")
    print("-" * 80)
    temp_results = analyzer.temperature_sensitivity()
    
    idx_cold = np.argmin(np.abs(temp_results['temperature_C'] - 0))
    idx_normal = np.argmin(np.abs(temp_results['temperature_C'] - 25))
    idx_hot = np.argmin(np.abs(temp_results['temperature_C'] - 40))
    
    print(f"At 0°C:  {temp_results['time_to_empty'][idx_cold]:.2f} hours "
          f"({temp_results['capacity_factor'][idx_cold]:.1%} capacity)")
    print(f"At 25°C: {temp_results['time_to_empty'][idx_normal]:.2f} hours "
          f"({temp_results['capacity_factor'][idx_normal]:.1%} capacity)")
    print(f"At 40°C: {temp_results['time_to_empty'][idx_hot]:.2f} hours "
          f"({temp_results['capacity_factor'][idx_hot]:.1%} capacity)")
    
    # Aging sensitivity
    print("\n\n4. BATTERY AGING IMPACT")
    print("-" * 80)
    aging_results = analyzer.aging_sensitivity()
    
    print(f"{'Cycles':<10} {'Time to Empty':<18} {'Capacity':<15} {'Resistance'}")
    print("-" * 70)
    for i in [0, 3, 6, 9, 10]:
        if i < len(aging_results['cycles']):
            cycles = aging_results['cycles'][i]
            tte = aging_results['time_to_empty'][i]
            cap = aging_results['capacity_retention'][i]
            res = aging_results['resistance_increase'][i]
            print(f"{cycles:<10.0f} {tte:<18.2f} {cap:<15.1%} {res:.2f}x")
    
    # Multi-parameter sensitivity
    print("\n\n5. PARAMETER SENSITIVITY RANKING")
    print("-" * 80)
    multi_results = analyzer.multi_parameter_sensitivity()
    
    # Sort by sensitivity
    sorted_indices = np.argsort(multi_results['sensitivity_indices'])[::-1]
    
    print(f"{'Parameter':<30} {'Sensitivity Index':<20} {'Impact'}")
    print("-" * 70)
    for idx in sorted_indices:
        param = multi_results['parameters'][idx]
        sens = multi_results['sensitivity_indices'][idx]
        impact = 'HIGH' if sens > 0.15 else 'MEDIUM' if sens > 0.05 else 'LOW'
        print(f"{param:<30} {sens:<20.4f} {impact}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run sensitivity analysis
    params = BatteryParameters()
    analyzer = SensitivityAnalyzer(params)
    
    # Generate report
    generate_sensitivity_report(analyzer)
    
    # Generate plots
    fig = create_sensitivity_plots(analyzer)
    fig.savefig('/workspace/mcm_solution/figures/sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    print("\nSensitivity plots saved to figures/sensitivity_analysis.png")
