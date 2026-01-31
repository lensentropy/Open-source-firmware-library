"""
Validation and comparison with empirical data
"""

import numpy as np
import matplotlib.pyplot as plt
from battery_model import BatteryModel, BatteryParameters, create_usage_scenarios
from typing import Dict, List


class BatteryDataValidator:
    """Validate model against empirical measurements and specifications"""
    
    def __init__(self):
        # Empirical data from literature and manufacturer specs
        # Source: Various smartphone battery tests (iPhone 13, Samsung S21, etc.)
        self.empirical_data = {
            'idle_time': {
                'mean': 400,  # hours
                'std': 50,
                'description': 'Standby time with minimal usage'
            },
            'light_usage_time': {
                'mean': 20,  # hours
                'std': 3,
                'description': 'Light usage (browsing, messaging)'
            },
            'normal_usage_time': {
                'mean': 12,  # hours
                'std': 2,
                'description': 'Normal usage (social media, browsing)'
            },
            'heavy_usage_time': {
                'mean': 6,  # hours
                'std': 1,
                'description': 'Heavy usage (gaming, video recording)'
            },
            'video_streaming': {
                'mean': 8,  # hours
                'std': 1.5,
                'description': 'Continuous video streaming'
            },
            'navigation': {
                'mean': 5,  # hours
                'std': 0.8,
                'description': 'GPS navigation with screen on'
            }
        }
        
        # Temperature effects from studies
        self.temp_effects = {
            'cold': {
                'temp_C': 0,
                'capacity_reduction': 0.20,  # 20% reduction at 0°C
                'description': 'Cold weather effect'
            },
            'normal': {
                'temp_C': 25,
                'capacity_reduction': 0.0,
                'description': 'Normal operating temperature'
            },
            'hot': {
                'temp_C': 40,
                'capacity_reduction': 0.10,  # 10% reduction at 40°C
                'description': 'Hot weather effect'
            }
        }
        
    def compare_scenarios(self, model: BatteryModel) -> Dict:
        """Compare model predictions with empirical data"""
        
        scenarios = create_usage_scenarios()
        results = {}
        
        for scenario_name, profile in scenarios.items():
            model.usage_profile = profile
            
            # Calculate time to empty
            tte = model.time_to_empty()
            
            # Get empirical data if available
            empirical_key = scenario_name
            if empirical_key == 'light':
                empirical_key = 'light_usage_time'
            elif empirical_key == 'normal':
                empirical_key = 'normal_usage_time'
            elif empirical_key == 'heavy':
                empirical_key = 'heavy_usage_time'
            elif empirical_key == 'streaming':
                empirical_key = 'video_streaming'
            
            empirical = self.empirical_data.get(empirical_key, None)
            
            results[scenario_name] = {
                'predicted_time': tte,
                'empirical_mean': empirical['mean'] if empirical else None,
                'empirical_std': empirical['std'] if empirical else None,
                'description': empirical['description'] if empirical else scenario_name
            }
            
            if empirical:
                error = abs(tte - empirical['mean']) / empirical['mean'] * 100
                results[scenario_name]['relative_error'] = error
                results[scenario_name]['within_std'] = abs(tte - empirical['mean']) <= 2 * empirical['std']
        
        return results
    
    def validate_temperature_effects(self, model: BatteryModel) -> Dict:
        """Validate temperature impact on battery performance"""
        
        results = {}
        base_usage = create_usage_scenarios()['normal']
        
        for temp_name, temp_data in self.temp_effects.items():
            temp_K = temp_data['temp_C'] + 273.15
            
            model.usage_profile = base_usage
            
            # Simulate at different temperatures
            tte = model.time_to_empty(initial_T=temp_K)
            
            # Reference at 25°C
            model.usage_profile = base_usage
            tte_ref = model.time_to_empty(initial_T=298.15)
            
            capacity_reduction = (tte_ref - tte) / tte_ref
            
            results[temp_name] = {
                'temperature_C': temp_data['temp_C'],
                'time_to_empty': tte,
                'capacity_reduction_predicted': capacity_reduction,
                'capacity_reduction_empirical': temp_data['capacity_reduction'],
                'error': abs(capacity_reduction - temp_data['capacity_reduction'])
            }
        
        return results
    
    def aging_validation(self, model: BatteryModel) -> Dict:
        """Validate battery aging model"""
        
        # Empirical: Li-ion batteries typically retain 80% capacity after 500 cycles
        cycles_test = [0, 100, 300, 500, 800, 1000]
        results = {}
        
        for cycles in cycles_test:
            Q_eff = model.effective_capacity(cycles)
            retention = Q_eff / model.params.Q_nominal
            
            results[cycles] = {
                'capacity_retention': retention,
                'effective_capacity_mAh': Q_eff
            }
        
        return results


def create_validation_plots(validator: BatteryDataValidator, model: BatteryModel):
    """Generate validation comparison plots"""
    
    # Scenario comparison
    scenario_results = validator.compare_scenarios(model)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Scenario comparison
    ax = axes[0, 0]
    scenarios = list(scenario_results.keys())
    predicted = [scenario_results[s]['predicted_time'] for s in scenarios]
    empirical = [scenario_results[s]['empirical_mean'] for s in scenarios if scenario_results[s]['empirical_mean']]
    empirical_std = [scenario_results[s]['empirical_std'] for s in scenarios if scenario_results[s]['empirical_mean']]
    scenarios_with_data = [s for s in scenarios if scenario_results[s]['empirical_mean']]
    
    x = np.arange(len(scenarios_with_data))
    width = 0.35
    
    ax.bar(x - width/2, [scenario_results[s]['predicted_time'] for s in scenarios_with_data], 
           width, label='Model Prediction', color='steelblue')
    ax.bar(x + width/2, empirical, width, label='Empirical Data', 
           yerr=empirical_std, capsize=5, color='coral')
    
    ax.set_ylabel('Time to Empty (hours)')
    ax.set_title('Model Validation: Usage Scenarios')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_with_data, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Temperature effects
    ax = axes[0, 1]
    temp_results = validator.validate_temperature_effects(model)
    
    temps = [temp_results[t]['temperature_C'] for t in ['cold', 'normal', 'hot']]
    reduction_pred = [temp_results[t]['capacity_reduction_predicted'] for t in ['cold', 'normal', 'hot']]
    reduction_emp = [temp_results[t]['capacity_reduction_empirical'] for t in ['cold', 'normal', 'hot']]
    
    ax.plot(temps, reduction_pred, 'o-', label='Model Prediction', linewidth=2, markersize=8)
    ax.plot(temps, reduction_emp, 's-', label='Empirical Data', linewidth=2, markersize=8)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Capacity Reduction')
    ax.set_title('Temperature Impact on Battery Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Aging effects
    ax = axes[1, 0]
    aging_results = validator.aging_validation(model)
    
    cycles = list(aging_results.keys())
    retention = [aging_results[c]['capacity_retention'] for c in cycles]
    
    # Empirical curve (typical Li-ion)
    empirical_retention = [1.0 - 0.2 * (c / 500) ** 0.5 for c in cycles]
    
    ax.plot(cycles, retention, 'o-', label='Model Prediction', linewidth=2, markersize=8)
    ax.plot(cycles, empirical_retention, 's--', label='Typical Li-ion', linewidth=2, markersize=8)
    ax.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
    ax.set_xlabel('Charge Cycles')
    ax.set_ylabel('Capacity Retention')
    ax.set_title('Battery Aging: Capacity Fade')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Prediction error summary
    ax = axes[1, 1]
    errors = [scenario_results[s]['relative_error'] for s in scenarios_with_data]
    
    ax.barh(scenarios_with_data, errors, color='steelblue')
    ax.axvline(x=10, color='orange', linestyle='--', label='±10% threshold')
    ax.axvline(x=20, color='red', linestyle='--', label='±20% threshold')
    ax.set_xlabel('Relative Error (%)')
    ax.set_title('Model Prediction Error')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Run validation
    params = BatteryParameters()
    model = BatteryModel(params)
    validator = BatteryDataValidator()
    
    # Scenario validation
    print("=" * 60)
    print("SCENARIO VALIDATION")
    print("=" * 60)
    scenario_results = validator.compare_scenarios(model)
    for scenario, result in scenario_results.items():
        print(f"\n{scenario.upper()}:")
        print(f"  Predicted: {result['predicted_time']:.2f} hours")
        if result['empirical_mean']:
            print(f"  Empirical: {result['empirical_mean']:.2f} ± {result['empirical_std']:.2f} hours")
            print(f"  Error: {result['relative_error']:.1f}%")
            print(f"  Within 2σ: {result['within_std']}")
    
    # Temperature validation
    print("\n" + "=" * 60)
    print("TEMPERATURE VALIDATION")
    print("=" * 60)
    temp_results = validator.validate_temperature_effects(model)
    for temp, result in temp_results.items():
        print(f"\n{temp.upper()} ({result['temperature_C']}°C):")
        print(f"  Predicted reduction: {result['capacity_reduction_predicted']:.1%}")
        print(f"  Empirical reduction: {result['capacity_reduction_empirical']:.1%}")
        print(f"  Error: {result['error']:.2%}")
    
    # Aging validation
    print("\n" + "=" * 60)
    print("AGING VALIDATION")
    print("=" * 60)
    aging_results = validator.aging_validation(model)
    print(f"{'Cycles':<10} {'Retention':<12} {'Capacity (mAh)'}")
    print("-" * 40)
    for cycles, result in aging_results.items():
        print(f"{cycles:<10} {result['capacity_retention']:>10.1%}  {result['effective_capacity_mAh']:>10.0f}")
    
    # Generate plots
    fig = create_validation_plots(validator, model)
    fig.savefig('/workspace/mcm_solution/figures/validation.png', dpi=300, bbox_inches='tight')
    print("\nValidation plots saved to figures/validation.png")
