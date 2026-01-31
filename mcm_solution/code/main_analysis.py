"""
Main analysis script for MCM Problem A
Generates all figures and results for the report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from battery_model import BatteryModel, BatteryParameters, create_usage_scenarios
from validation import BatteryDataValidator, create_validation_plots
from sensitivity_analysis import SensitivityAnalyzer, create_sensitivity_plots, generate_sensitivity_report
import os

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def create_output_dirs():
    """Create output directories if they don't exist"""
    dirs = ['/workspace/mcm_solution/figures', '/workspace/mcm_solution/data']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def plot_scenario_simulations():
    """Generate time series plots for different usage scenarios"""
    
    params = BatteryParameters()
    scenarios = create_usage_scenarios()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    scenario_names = ['light', 'normal', 'heavy', 'navigation', 'streaming']
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenario_names)))
    
    for idx, (scenario_name, profile) in enumerate(scenarios.items()):
        if idx >= len(scenario_names):
            break
            
        model = BatteryModel(params)
        model.usage_profile = profile
        
        # Simulate
        tte = model.time_to_empty()
        t_end = min(tte * 1.1, 24)
        result = model.simulate((0, t_end), initial_SOC=1.0)
        
        # Plot SOC
        ax = axes[idx]
        ax.plot(result['t'], result['SOC'] * 100, linewidth=2, color=colors[idx])
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
        ax.axhline(y=5, color='darkred', linestyle='--', alpha=0.5, label='5% cutoff')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('State of Charge (%)')
        ax.set_title(f'{scenario_name.capitalize()} Usage\n(TTE: {tte:.2f} hours)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, t_end])
        ax.set_ylim([0, 105])
        if idx == 0:
            ax.legend()
    
    # Last subplot: comparison
    ax = axes[-1]
    for idx, (scenario_name, profile) in enumerate(scenarios.items()):
        if idx >= len(scenario_names):
            break
            
        model = BatteryModel(params)
        model.usage_profile = profile
        
        tte = model.time_to_empty()
        t_end = min(tte * 1.1, 24)
        result = model.simulate((0, t_end), initial_SOC=1.0)
        
        ax.plot(result['t'], result['SOC'] * 100, linewidth=2, 
                label=scenario_name, color=colors[idx])
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('All Scenarios Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 24])
    
    plt.tight_layout()
    return fig


def plot_power_breakdown():
    """Show power consumption breakdown for different components"""
    
    params = BatteryParameters()
    
    # Define components
    components = {
        'Idle': params.P_idle,
        'Screen (50%)': params.P_screen_base + 0.5 * params.P_screen_per_brightness,
        'CPU (30%)': params.P_cpu_idle + 0.3 * params.P_cpu_per_load,
        '4G Network': params.P_network_4g,
        'WiFi': params.P_wifi,
        'GPS': params.P_gps,
        'Bluetooth': params.P_bluetooth
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar chart
    names = list(components.keys())
    powers = list(components.values())
    colors = plt.cm.Spectral(np.linspace(0, 1, len(names)))
    
    bars = ax1.barh(names, powers, color=colors)
    ax1.set_xlabel('Power Consumption (mW)')
    ax1.set_title('Component Power Consumption')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Cumulative scenario analysis
    scenarios_power = {
        'Idle': params.P_idle + params.P_cpu_idle + params.P_wifi,
        'Light\nUsage': params.P_idle + params.P_screen_base + 0.3 * params.P_screen_per_brightness + 
                        params.P_cpu_idle + 0.2 * params.P_cpu_per_load + params.P_wifi,
        'Normal\nUsage': params.P_idle + params.P_screen_base + 0.5 * params.P_screen_per_brightness + 
                         params.P_cpu_idle + 0.3 * params.P_cpu_per_load + params.P_network_4g,
        'Heavy\nUsage': params.P_idle + params.P_screen_base + 0.8 * params.P_screen_per_brightness + 
                        params.P_cpu_idle + 0.7 * params.P_cpu_per_load + params.P_network_4g,
        'Navigation': params.P_idle + params.P_screen_base + 0.9 * params.P_screen_per_brightness + 
                      params.P_cpu_idle + 0.4 * params.P_cpu_per_load + params.P_network_4g + params.P_gps
    }
    
    scenario_names = list(scenarios_power.keys())
    scenario_powers = list(scenarios_power.values())
    
    # Calculate battery life for each
    battery_energy = params.Q_nominal * params.V_nominal  # mWh
    battery_life = [battery_energy / p for p in scenario_powers]
    
    ax2.bar(scenario_names, battery_life, color=plt.cm.viridis(np.linspace(0, 1, len(scenario_names))))
    ax2.set_ylabel('Estimated Battery Life (hours)')
    ax2.set_title('Battery Life by Usage Scenario')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(battery_life):
        ax2.text(i, v + 0.5, f'{v:.1f}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_temperature_dynamics():
    """Show temperature evolution during heavy usage"""
    
    params = BatteryParameters()
    model = BatteryModel(params)
    
    # Heavy usage scenario
    profile = create_usage_scenarios()['heavy']
    model.usage_profile = profile
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Simulate at different ambient temperatures
    ambient_temps = [273.15, 288.15, 298.15, 308.15]  # 0, 15, 25, 35°C
    temp_labels = ['0°C', '15°C', '25°C', '35°C']
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(ambient_temps)))
    
    for i, (T_amb, label) in enumerate(zip(ambient_temps, temp_labels)):
        model.params.T_ambient = T_amb
        result = model.simulate((0, 8), initial_SOC=1.0, initial_T=T_amb)
        
        # SOC
        axes[0, 0].plot(result['t'], result['SOC'] * 100, linewidth=2, 
                       label=f'Ambient {label}', color=colors[i])
        
        # Temperature
        axes[0, 1].plot(result['t'], result['T'] - 273.15, linewidth=2, 
                       label=f'Ambient {label}', color=colors[i])
        
        # Voltage
        axes[1, 0].plot(result['t'], result['voltage'], linewidth=2, 
                       label=f'Ambient {label}', color=colors[i])
        
        # Power
        axes[1, 1].plot(result['t'], result['power'], linewidth=2, 
                       label=f'Ambient {label}', color=colors[i])
    
    # Reset to default
    model.params.T_ambient = 298.15
    
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('State of Charge (%)')
    axes[0, 0].set_title('SOC vs Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Battery Temperature (°C)')
    axes[0, 1].set_title('Temperature Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Terminal Voltage (V)')
    axes[1, 0].set_title('Voltage vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Power Consumption (mW)')
    axes[1, 1].set_title('Power Consumption')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Temperature Effects on Battery Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_aging_comparison():
    """Compare battery performance at different aging levels"""
    
    params = BatteryParameters()
    model = BatteryModel(params)
    
    # Normal usage
    profile = create_usage_scenarios()['normal']
    model.usage_profile = profile
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    cycle_counts = [0, 200, 500, 800]
    labels = ['New', '200 cycles', '500 cycles', '800 cycles']
    colors = plt.cm.autumn(np.linspace(0, 1, len(cycle_counts)))
    
    for i, (cycles, label) in enumerate(zip(cycle_counts, labels)):
        result = model.simulate((0, 15), initial_SOC=1.0, initial_cycles=cycles)
        
        # SOC
        axes[0, 0].plot(result['t'], result['SOC'] * 100, linewidth=2, 
                       label=label, color=colors[i])
        
        # Voltage
        axes[0, 1].plot(result['t'], result['voltage'], linewidth=2, 
                       label=label, color=colors[i])
        
        # Current
        axes[1, 0].plot(result['t'], result['current'], linewidth=2, 
                       label=label, color=colors[i])
        
        # Temperature
        axes[1, 1].plot(result['t'], result['T'] - 273.15, linewidth=2, 
                       label=label, color=colors[i])
    
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('State of Charge (%)')
    axes[0, 0].set_title('SOC Degradation with Age')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Terminal Voltage (V)')
    axes[0, 1].set_title('Voltage Drop with Age')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Current Draw (mA)')
    axes[1, 0].set_title('Current Requirements')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].set_title('Temperature Rise')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Battery Aging Effects', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_recommendations_impact():
    """Visualize impact of different user recommendations"""
    
    params = BatteryParameters()
    
    recommendations = {
        'Baseline\n(Normal)': lambda: create_usage_scenarios()['normal'],
        'Reduce\nBrightness\n50%→30%': lambda: modify_scenario('normal', brightness=0.3),
        'Use WiFi\ninstead of 4G': lambda: modify_scenario('normal', network='wifi'),
        'Reduce\nScreen Time\n30%→20%': lambda: modify_scenario('normal', screen_time=0.2),
        'All\nOptimizations': lambda: modify_scenario('normal', brightness=0.3, 
                                                      network='wifi', screen_time=0.2)
    }
    
    results = {}
    for name, scenario_func in recommendations.items():
        model = BatteryModel(params)
        model.usage_profile = scenario_func()
        tte = model.time_to_empty()
        results[name] = tte
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = list(results.keys())
    times = list(results.values())
    baseline = times[0]
    improvements = [(t - baseline) for t in times]
    improvement_pct = [(t - baseline) / baseline * 100 for t in times]
    
    # Absolute times
    colors = ['gray'] + ['green'] * (len(names) - 1)
    bars1 = ax1.bar(names, times, color=colors, alpha=0.7)
    ax1.set_ylabel('Time to Empty (hours)')
    ax1.set_title('Battery Life with Optimizations')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars1, times)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}h',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Improvements
    colors2 = ['gray'] + ['green' if x > 0 else 'red' for x in improvements[1:]]
    bars2 = ax2.bar(names[1:], improvement_pct[1:], color=colors2[1:], alpha=0.7)
    ax2.set_ylabel('Battery Life Improvement (%)')
    ax2.set_title('Relative Improvement over Baseline')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars2, improvement_pct[1:]):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'+{val:.1f}%' if val > 0 else f'{val:.1f}%',
                ha='center', va='bottom' if val > 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig


def modify_scenario(base_scenario: str, brightness=None, network=None, screen_time=None):
    """Helper to modify usage scenarios"""
    scenarios = create_usage_scenarios()
    profile = scenarios[base_scenario]
    
    if brightness is not None:
        profile.set_screen_brightness(lambda t: brightness)
    if network is not None:
        profile.set_network_type(lambda t: network)
    if screen_time is not None:
        profile.set_screen_state(lambda t: (t % 1) < screen_time)
    
    return profile


def main():
    """Run all analyses and generate figures"""
    
    print("Starting comprehensive battery model analysis...")
    print("=" * 80)
    
    # Create output directories
    create_output_dirs()
    
    # Initialize
    params = BatteryParameters()
    model = BatteryModel(params)
    validator = BatteryDataValidator()
    analyzer = SensitivityAnalyzer(params)
    
    # 1. Scenario simulations
    print("\n1. Generating scenario simulation plots...")
    fig1 = plot_scenario_simulations()
    fig1.savefig('/workspace/mcm_solution/figures/scenario_simulations.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: scenario_simulations.png")
    
    # 2. Power breakdown
    print("\n2. Generating power breakdown analysis...")
    fig2 = plot_power_breakdown()
    fig2.savefig('/workspace/mcm_solution/figures/power_breakdown.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: power_breakdown.png")
    
    # 3. Temperature dynamics
    print("\n3. Analyzing temperature effects...")
    fig3 = plot_temperature_dynamics()
    fig3.savefig('/workspace/mcm_solution/figures/temperature_dynamics.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: temperature_dynamics.png")
    
    # 4. Aging comparison
    print("\n4. Analyzing battery aging...")
    fig4 = plot_aging_comparison()
    fig4.savefig('/workspace/mcm_solution/figures/aging_comparison.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: aging_comparison.png")
    
    # 5. Validation
    print("\n5. Running model validation...")
    fig5 = create_validation_plots(validator, model)
    fig5.savefig('/workspace/mcm_solution/figures/validation.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: validation.png")
    
    # 6. Sensitivity analysis
    print("\n6. Performing sensitivity analysis...")
    fig6 = create_sensitivity_plots(analyzer)
    fig6.savefig('/workspace/mcm_solution/figures/sensitivity_analysis.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: sensitivity_analysis.png")
    
    # 7. Recommendations
    print("\n7. Evaluating optimization recommendations...")
    fig7 = plot_recommendations_impact()
    fig7.savefig('/workspace/mcm_solution/figures/recommendations.png', 
                 dpi=300, bbox_inches='tight')
    print("   Saved: recommendations.png")
    
    # 8. Generate text reports
    print("\n8. Generating analysis reports...")
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    generate_sensitivity_report(analyzer)
    
    print("\n" + "=" * 80)
    print("All analyses complete! Figures saved to mcm_solution/figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()
