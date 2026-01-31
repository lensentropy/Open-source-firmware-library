"""
Visualization Module for Smartphone Battery Discharge Model
MCM 2026 Problem A Solution

Generates publication-quality figures for the model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_soc_discharge_curves(results: Dict[str, Dict], 
                               save_path: str = None,
                               title: str = "Battery Discharge Under Different Usage Scenarios"):
    """
    Plot SOC vs time for multiple scenarios.
    
    Args:
        results: Dictionary from ScenarioComparison.compare_scenarios()
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    
    for (name, result), color in zip(results.items(), colors):
        if 'error' in result:
            continue
        
        t = result['t']
        SOC = result['SOC'] * 100  # Convert to percentage
        t_empty = result['t_empty']
        
        ax.plot(t, SOC, label=f"{name.replace('_', ' ').title()} ({t_empty:.1f}h)", 
               color=color, linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title(title)
    ax.set_xlim(0, None)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='Shutdown threshold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_power_breakdown_bars(rankings: Dict[str, Dict],
                              scenarios_to_plot: List[str] = None,
                              save_path: str = None):
    """
    Plot power consumption breakdown as stacked bars.
    
    Args:
        rankings: Results from ScenarioComparison.rank_power_components()
        scenarios_to_plot: List of scenarios to include
        save_path: Path to save figure
    """
    if scenarios_to_plot is None:
        scenarios_to_plot = list(rankings.keys())
    
    components = ['screen', 'cpu', 'gpu', 'wifi', 'cellular', 'gps', 'bluetooth', 'baseline']
    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios_to_plot))
    width = 0.6
    
    bottom = np.zeros(len(scenarios_to_plot))
    
    for component, color in zip(components, colors):
        values = []
        for scenario in scenarios_to_plot:
            if scenario in rankings:
                comp_data = rankings[scenario]['components'].get(component, {})
                values.append(comp_data.get('power_W', 0))
            else:
                values.append(0)
        
        ax.bar(x, values, width, bottom=bottom, label=component.capitalize(), color=color)
        bottom += values
    
    ax.set_xlabel('Usage Scenario')
    ax.set_ylabel('Power Consumption (W)')
    ax.set_title('Power Consumption Breakdown by Component')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios_to_plot], rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_sensitivity_tornado(oat_results: Dict[str, Dict],
                             metric: str = 'elasticity',
                             save_path: str = None,
                             title: str = "Parameter Sensitivity (Tornado Diagram)"):
    """
    Create tornado diagram showing parameter sensitivities.
    
    Args:
        oat_results: Results from SensitivityAnalyzer.oat_sensitivity()
        metric: 'elasticity' or 'local_sensitivity'
        save_path: Path to save figure
        title: Plot title
    """
    # Extract sensitivities
    params = []
    sensitivities = []
    
    for param_name, data in oat_results.items():
        params.append(param_name)
        sensitivities.append(data[metric])
    
    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(sensitivities))[::-1]
    params = [params[i] for i in sorted_idx]
    sensitivities = [sensitivities[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(params))
    colors = ['green' if s > 0 else 'red' for s in sensitivities]
    
    ax.barh(y_pos, sensitivities, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('_', ' ').title() for p in params])
    ax.set_xlabel('Sensitivity (Elasticity)' if metric == 'elasticity' else 'Local Sensitivity')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Add annotations
    for i, (p, s) in enumerate(zip(params, sensitivities)):
        ax.annotate(f'{s:.3f}', xy=(s, i), 
                   xytext=(5 if s > 0 else -5, 0),
                   textcoords='offset points',
                   ha='left' if s > 0 else 'right',
                   va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_monte_carlo_histogram(mc_results: Dict,
                               save_path: str = None,
                               title: str = "Time-to-Empty Distribution (Monte Carlo)"):
    """
    Plot histogram of time-to-empty from Monte Carlo analysis.
    
    Args:
        mc_results: Results from SensitivityAnalyzer.monte_carlo_uncertainty()
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_empty = mc_results['t_empty_samples']
    valid = t_empty[~np.isnan(t_empty)]
    
    # Histogram
    n, bins, patches = ax.hist(valid, bins=40, density=True, alpha=0.7, 
                               color='steelblue', edgecolor='white')
    
    # Statistics
    mean = mc_results['mean']
    std = mc_results['std']
    p5 = mc_results['p5']
    p95 = mc_results['p95']
    
    # Add vertical lines
    ax.axvline(mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean:.2f}h')
    ax.axvline(p5, color='orange', linestyle='--', linewidth=1.5, label=f'5th percentile: {p5:.2f}h')
    ax.axvline(p95, color='orange', linestyle='--', linewidth=1.5, label=f'95th percentile: {p95:.2f}h')
    
    # Fill confidence interval
    ax.axvspan(p5, p95, alpha=0.2, color='orange', label='90% CI')
    
    ax.set_xlabel('Time-to-Empty (hours)')
    ax.set_ylabel('Probability Density')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    # Add statistics box
    stats_text = (f'Mean: {mean:.2f}h\n'
                  f'Std Dev: {std:.2f}h\n'
                  f'Median: {mc_results["median"]:.2f}h\n'
                  f'90% CI: [{p5:.2f}, {p95:.2f}]h')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_temperature_effects(model, save_path: str = None):
    """
    Plot temperature effects on battery performance.
    
    Args:
        model: SmartphoneBatteryModel instance
        save_path: Path to save figure
    """
    from battery_model import UsageProfile
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    temperatures = np.linspace(-20, 50, 100)
    
    # Capacity factor
    cap_factors = [model.temperature_factor_capacity(T) for T in temperatures]
    axes[0].plot(temperatures, cap_factors, 'b-', linewidth=2)
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Capacity Factor')
    axes[0].set_title('Effective Capacity vs Temperature')
    axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(25, color='green', linestyle=':', alpha=0.5, label='Reference (25°C)')
    axes[0].legend()
    
    # Resistance factor
    res_factors = [model.temperature_factor_resistance(T) for T in temperatures]
    axes[1].plot(temperatures, res_factors, 'r-', linewidth=2)
    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_ylabel('Resistance Factor')
    axes[1].set_title('Internal Resistance vs Temperature')
    axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(25, color='green', linestyle=':', alpha=0.5, label='Reference (25°C)')
    axes[1].legend()
    
    # Time-to-empty at different temperatures
    temps_to_test = [-10, 0, 10, 20, 25, 30, 40]
    t_empty_values = []
    
    for T in temps_to_test:
        profile = UsageProfile(
            screen_brightness=lambda t: 0.5,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.3,
            gpu_load=lambda t: 0.1,
            wifi_activity=lambda t: 0.4,
            cellular_activity=lambda t: 0.0,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t, temp=T: temp
        )
        try:
            _, _, details = model.simulate(1.0, profile, t_max=48.0)
            t_empty_values.append(details['t_empty'])
        except:
            t_empty_values.append(np.nan)
    
    axes[2].bar(range(len(temps_to_test)), t_empty_values, color='steelblue', alpha=0.7)
    axes[2].set_xticks(range(len(temps_to_test)))
    axes[2].set_xticklabels([f'{T}°C' for T in temps_to_test])
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Time-to-Empty (hours)')
    axes[2].set_title('Battery Life vs Temperature')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_voltage_soc_curve(model, save_path: str = None):
    """
    Plot voltage-SOC relationship.
    
    Args:
        model: SmartphoneBatteryModel instance
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    SOC_values = np.linspace(0.05, 1.0, 100)
    
    # Open circuit voltage
    V_oc = [model.open_circuit_voltage(s) for s in SOC_values]
    axes[0].plot(SOC_values * 100, V_oc, 'b-', linewidth=2, label='Open Circuit')
    
    # Terminal voltage at different currents
    currents = [0.5, 1.0, 2.0]
    colors = ['orange', 'red', 'darkred']
    for I, color in zip(currents, colors):
        V_terminal = [model.terminal_voltage(s, I, 25.0) for s in SOC_values]
        axes[0].plot(SOC_values * 100, V_terminal, '--', color=color, 
                    linewidth=1.5, label=f'I = {I}A')
    
    axes[0].set_xlabel('State of Charge (%)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('Voltage-SOC Characteristics')
    axes[0].legend()
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(2.5, 4.5)
    
    # Discharge curves at different C-rates
    from battery_model import UsageProfile
    
    power_levels = [1.0, 2.5, 5.0, 8.0]  # Watts
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(power_levels)))
    
    for P, color in zip(power_levels, colors):
        # Create constant power profile
        profile = UsageProfile(
            screen_brightness=lambda t: 0.0,
            screen_on=lambda t: 0.0,
            cpu_load=lambda t, p=P: min(p/5.0, 1.0),  # Map to CPU load
            gpu_load=lambda t: 0.0,
            wifi_activity=lambda t: 0.0,
            cellular_activity=lambda t: 0.0,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )
        
        try:
            t, SOC, details = model.simulate(1.0, profile, t_max=24.0)
            axes[1].plot(t, SOC * 100, color=color, linewidth=2, 
                        label=f'P ≈ {P}W')
        except:
            pass
    
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('State of Charge (%)')
    axes[1].set_title('Discharge at Different Power Levels')
    axes[1].legend()
    axes[1].set_xlim(0, 24)
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_aging_effects(save_path: str = None):
    """
    Plot battery aging effects.
    
    Args:
        save_path: Path to save figure
    """
    from battery_model import BatteryAgingModel, BatteryParameters
    
    aging_model = BatteryAgingModel(BatteryParameters())
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Capacity fade over time
    years = np.linspace(0, 4, 50)
    cycles_per_year = 365  # Assume daily charging
    
    soh_time = [aging_model.capacity_fade(y, y * cycles_per_year) for y in years]
    axes[0].plot(years, np.array(soh_time) * 100, 'b-', linewidth=2)
    axes[0].set_xlabel('Years of Use')
    axes[0].set_ylabel('State of Health (%)')
    axes[0].set_title('Capacity Fade Over Time')
    axes[0].set_xlim(0, 4)
    axes[0].set_ylim(50, 105)
    axes[0].axhline(80, color='red', linestyle='--', alpha=0.5, 
                   label='80% threshold (typical warranty)')
    axes[0].legend()
    
    # SOH vs cycles
    cycles = np.linspace(0, 1500, 50)
    soh_cycles = [aging_model.capacity_fade(2, c) for c in cycles]  # Fixed 2 years
    axes[1].plot(cycles, np.array(soh_cycles) * 100, 'g-', linewidth=2)
    axes[1].set_xlabel('Number of Cycles')
    axes[1].set_ylabel('State of Health (%)')
    axes[1].set_title('Capacity Fade vs Cycles (at 2 years)')
    axes[1].set_xlim(0, 1500)
    axes[1].set_ylim(50, 105)
    axes[1].axhline(80, color='red', linestyle='--', alpha=0.5)
    
    # Temperature effect on aging
    temps = [15, 25, 35, 45]
    for T in temps:
        soh_temp = [aging_model.capacity_fade(y, y * 365, avg_temp=T) for y in years]
        axes[2].plot(years, np.array(soh_temp) * 100, linewidth=2, 
                    label=f'{T}°C')
    
    axes[2].set_xlabel('Years of Use')
    axes[2].set_ylabel('State of Health (%)')
    axes[2].set_title('Temperature Effect on Aging')
    axes[2].legend()
    axes[2].set_xlim(0, 4)
    axes[2].set_ylim(50, 105)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def plot_model_validation(observed_data: Dict = None, save_path: str = None):
    """
    Plot model validation against observed/literature data.
    
    Args:
        observed_data: Dictionary with observed discharge data
        save_path: Path to save figure
    """
    from battery_model import SmartphoneBatteryModel, UsageProfile, create_usage_scenarios
    
    # Use typical literature values if no data provided
    if observed_data is None:
        # Based on typical smartphone reviews and battery tests
        # Sources: GSMArena, Anandtech, Tom's Hardware battery tests
        observed_data = {
            'idle': {'mean': 72.0, 'std': 12.0},  # ~3 days standby
            'light': {'mean': 12.0, 'std': 2.0},  # Light use
            'moderate': {'mean': 6.0, 'std': 1.0},  # Video playback
            'heavy_gaming': {'mean': 3.0, 'std': 0.5},  # Gaming
            'navigation': {'mean': 4.0, 'std': 0.8},  # GPS navigation
        }
    
    model = SmartphoneBatteryModel()
    scenarios = create_usage_scenarios()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenario_names = list(observed_data.keys())
    x = np.arange(len(scenario_names))
    width = 0.35
    
    # Model predictions
    model_values = []
    for name in scenario_names:
        if name in scenarios:
            _, _, details = model.simulate(1.0, scenarios[name], t_max=96.0)
            model_values.append(details['t_empty'])
        else:
            model_values.append(0)
    
    # Observed values
    obs_means = [observed_data[name]['mean'] for name in scenario_names]
    obs_stds = [observed_data[name]['std'] for name in scenario_names]
    
    # Plot bars
    bars1 = ax.bar(x - width/2, model_values, width, label='Model Prediction', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, obs_means, width, yerr=obs_stds, 
                   label='Literature Values', color='coral', alpha=0.8,
                   capsize=5)
    
    ax.set_xlabel('Usage Scenario')
    ax.set_ylabel('Time-to-Empty (hours)')
    ax.set_title('Model Validation: Predicted vs Literature Values')
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenario_names])
    ax.legend()
    
    # Calculate RMSE and R²
    model_arr = np.array(model_values)
    obs_arr = np.array(obs_means)
    rmse = np.sqrt(np.mean((model_arr - obs_arr)**2))
    ss_res = np.sum((obs_arr - model_arr)**2)
    ss_tot = np.sum((obs_arr - np.mean(obs_arr))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Add metrics text
    metrics_text = f'RMSE: {rmse:.2f}h\nR²: {r2:.3f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_recommendations_impact(save_path: str = None):
    """
    Visualize impact of power-saving recommendations.
    
    Args:
        save_path: Path to save figure
    """
    from battery_model import SmartphoneBatteryModel, UsageProfile
    
    model = SmartphoneBatteryModel()
    
    # Baseline: typical moderate usage
    baseline = UsageProfile(
        screen_brightness=lambda t: 0.7,
        screen_on=lambda t: 1.0,
        cpu_load=lambda t: 0.35,
        gpu_load=lambda t: 0.2,
        wifi_activity=lambda t: 0.5,
        cellular_activity=lambda t: 0.3,
        gps_active=lambda t: 0.0,
        bluetooth_active=lambda t: 0.0,
        temperature=lambda t: 25.0
    )
    
    # Recommendations
    recommendations = {
        'Baseline': baseline,
        'Lower Brightness\n(70%→30%)': UsageProfile(
            screen_brightness=lambda t: 0.3,  # Reduced
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.35,
            gpu_load=lambda t: 0.2,
            wifi_activity=lambda t: 0.5,
            cellular_activity=lambda t: 0.3,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        ),
        'Dark Mode\n(OLED savings)': UsageProfile(
            screen_brightness=lambda t: 0.5,  # Effectively reduced due to dark pixels
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.35,
            gpu_load=lambda t: 0.2,
            wifi_activity=lambda t: 0.5,
            cellular_activity=lambda t: 0.3,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        ),
        'Disable Background\nApps': UsageProfile(
            screen_brightness=lambda t: 0.7,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.2,  # Reduced
            gpu_load=lambda t: 0.1,
            wifi_activity=lambda t: 0.3,  # Reduced
            cellular_activity=lambda t: 0.1,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        ),
        'WiFi Instead of\nCellular': UsageProfile(
            screen_brightness=lambda t: 0.7,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.35,
            gpu_load=lambda t: 0.2,
            wifi_activity=lambda t: 0.6,  # Use WiFi
            cellular_activity=lambda t: 0.0,  # Disable cellular data
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        ),
        'All Optimizations\nCombined': UsageProfile(
            screen_brightness=lambda t: 0.3,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.2,
            gpu_load=lambda t: 0.1,
            wifi_activity=lambda t: 0.4,
            cellular_activity=lambda t: 0.0,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        ),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time-to-empty comparison
    t_empty_values = {}
    for name, profile in recommendations.items():
        _, _, details = model.simulate(1.0, profile, t_max=24.0)
        t_empty_values[name] = details['t_empty']
    
    names = list(t_empty_values.keys())
    values = list(t_empty_values.values())
    baseline_val = values[0]
    
    colors = ['gray'] + ['steelblue'] * (len(names) - 2) + ['green']
    bars = axes[0].barh(names, values, color=colors, alpha=0.8)
    axes[0].set_xlabel('Time-to-Empty (hours)')
    axes[0].set_title('Impact of Power-Saving Recommendations')
    axes[0].axvline(baseline_val, color='red', linestyle='--', alpha=0.5)
    
    # Add percentage improvement labels
    for i, (name, val) in enumerate(zip(names, values)):
        if name != 'Baseline':
            improvement = (val - baseline_val) / baseline_val * 100
            axes[0].annotate(f'+{improvement:.0f}%', xy=(val, i), 
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=9, va='center')
    
    # Power breakdown comparison
    components = ['screen', 'cpu', 'gpu', 'wifi', 'cellular', 'baseline']
    component_colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    
    x = np.arange(len(names))
    width = 0.8
    bottom = np.zeros(len(names))
    
    for comp, color in zip(components, component_colors):
        comp_values = []
        for name, profile in recommendations.items():
            power = model.power_consumption(0, profile)
            comp_values.append(power.get(comp, 0))
        
        axes[1].bar(x, comp_values, width, bottom=bottom, label=comp.capitalize(), color=color)
        bottom += comp_values
    
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Power Consumption (W)')
    axes[1].set_title('Power Breakdown by Component')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace('\n', ' ') for n in names], rotation=45, ha='right')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig, axes


def generate_all_figures(output_dir: str = '../figures/'):
    """
    Generate all figures for the report.
    
    Args:
        output_dir: Directory to save figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    from battery_model import SmartphoneBatteryModel, create_usage_scenarios
    from sensitivity_analysis import SensitivityAnalyzer, ScenarioComparison
    
    print("Generating figures...")
    
    model = SmartphoneBatteryModel()
    scenarios = create_usage_scenarios()
    
    # 1. Discharge curves
    print("  1. Discharge curves...")
    comparison = ScenarioComparison(model)
    scenario_results = comparison.compare_scenarios(scenarios=scenarios)
    plot_soc_discharge_curves(scenario_results, 
                              save_path=os.path.join(output_dir, 'discharge_curves.png'))
    
    # 2. Power breakdown
    print("  2. Power breakdown...")
    rankings = comparison.rank_power_components(scenario_results)
    plot_power_breakdown_bars(rankings, 
                              scenarios_to_plot=['idle', 'light', 'moderate', 'heavy_gaming', 'navigation', 'video_call'],
                              save_path=os.path.join(output_dir, 'power_breakdown.png'))
    
    # 3. Temperature effects
    print("  3. Temperature effects...")
    plot_temperature_effects(model, 
                            save_path=os.path.join(output_dir, 'temperature_effects.png'))
    
    # 4. Voltage-SOC curves
    print("  4. Voltage-SOC curves...")
    plot_voltage_soc_curve(model, 
                          save_path=os.path.join(output_dir, 'voltage_soc.png'))
    
    # 5. Aging effects
    print("  5. Aging effects...")
    plot_aging_effects(save_path=os.path.join(output_dir, 'aging_effects.png'))
    
    # 6. Model validation
    print("  6. Model validation...")
    plot_model_validation(save_path=os.path.join(output_dir, 'validation.png'))
    
    # 7. Sensitivity analysis
    print("  7. Sensitivity analysis...")
    analyzer = SensitivityAnalyzer(model)
    oat_results = analyzer.oat_sensitivity(scenarios['moderate'], n_points=15)
    plot_sensitivity_tornado(oat_results, 
                            save_path=os.path.join(output_dir, 'sensitivity_tornado.png'))
    
    # 8. Monte Carlo uncertainty
    print("  8. Monte Carlo uncertainty...")
    mc_results = analyzer.monte_carlo_uncertainty(scenarios['moderate'], n_samples=300)
    plot_monte_carlo_histogram(mc_results, 
                              save_path=os.path.join(output_dir, 'monte_carlo.png'))
    
    # 9. Recommendations impact
    print("  9. Recommendations impact...")
    plot_recommendations_impact(save_path=os.path.join(output_dir, 'recommendations.png'))
    
    print(f"\nAll figures saved to {output_dir}")
    plt.close('all')


if __name__ == "__main__":
    generate_all_figures()
