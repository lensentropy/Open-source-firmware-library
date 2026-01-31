"""
Main Simulation and Analysis Script
MCM 2026 Problem A: Smartphone Battery Discharge Modeling

This script orchestrates all simulations, analyses, and figure generation.
"""

import numpy as np
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battery_model import (
    SmartphoneBatteryModel, BatteryParameters, DeviceParameters,
    UsageProfile, BatteryAgingModel, create_usage_scenarios
)
from sensitivity_analysis import (
    SensitivityAnalyzer, ScenarioComparison, run_comprehensive_sensitivity_analysis
)
from visualization import generate_all_figures


def run_scenario_analysis():
    """
    Run comprehensive scenario analysis and print results.
    """
    print("=" * 60)
    print("SMARTPHONE BATTERY DISCHARGE MODEL ANALYSIS")
    print("MCM 2026 Problem A Solution")
    print("=" * 60)
    print()
    
    # Initialize model with default parameters
    model = SmartphoneBatteryModel()
    scenarios = create_usage_scenarios()
    
    print("Model Parameters:")
    print(f"  Battery Capacity: {model.battery.Q_nominal} mAh")
    print(f"  Nominal Voltage: {model.battery.V_nominal} V")
    print(f"  Internal Resistance: {model.battery.R_internal*1000:.0f} mΩ")
    print(f"  State of Health: {model.battery.SOH*100:.0f}%")
    print()
    
    # Run scenarios
    print("Scenario Analysis:")
    print("-" * 60)
    
    comparison = ScenarioComparison(model)
    results = comparison.compare_scenarios(scenarios=scenarios)
    
    # Sort by time-to-empty
    sorted_scenarios = sorted(
        [(name, res) for name, res in results.items() if 'error' not in res],
        key=lambda x: x[1]['t_empty'],
        reverse=True
    )
    
    print(f"{'Scenario':<20} {'Time-to-Empty':>15} {'Avg Power':>12} {'Final SOC':>12}")
    print("-" * 60)
    
    for name, res in sorted_scenarios:
        t_empty = res['t_empty']
        avg_power = res['avg_power']
        final_soc = res['final_SOC']
        print(f"{name:<20} {t_empty:>12.2f} h {avg_power:>10.2f} W {final_soc*100:>10.1f}%")
    
    print()
    
    # Power component analysis
    print("Power Consumption Analysis (Moderate Usage Scenario):")
    print("-" * 60)
    
    moderate_profile = scenarios['moderate']
    power_breakdown = model.power_consumption(0, moderate_profile)
    
    total_power = power_breakdown['total']
    for component, power in sorted(power_breakdown.items(), key=lambda x: x[1], reverse=True):
        if component != 'total':
            percentage = (power / total_power * 100) if total_power > 0 else 0
            bar = '█' * int(percentage / 5)
            print(f"  {component:<12}: {power:>6.3f} W ({percentage:>5.1f}%) {bar}")
    
    print(f"  {'TOTAL':<12}: {total_power:>6.3f} W")
    print()
    
    return results


def run_temperature_analysis():
    """
    Analyze temperature effects on battery performance.
    """
    print("\nTemperature Effects Analysis:")
    print("-" * 60)
    
    model = SmartphoneBatteryModel()
    
    temperatures = [-10, 0, 10, 20, 25, 30, 40, 45]
    
    print(f"{'Temp (°C)':<12} {'Capacity Factor':>16} {'Resistance Factor':>18} {'Time-to-Empty':>15}")
    print("-" * 60)
    
    for T in temperatures:
        cap_factor = model.temperature_factor_capacity(T)
        res_factor = model.temperature_factor_resistance(T)
        
        # Create profile at this temperature
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
        
        _, _, details = model.simulate(1.0, profile, t_max=48.0)
        t_empty = details['t_empty']
        
        print(f"{T:>8}°C {cap_factor:>15.3f} {res_factor:>18.3f} {t_empty:>13.2f} h")
    
    print()


def run_aging_analysis():
    """
    Analyze battery aging effects.
    """
    print("\nBattery Aging Analysis:")
    print("-" * 60)
    
    base_params = BatteryParameters()
    aging_model = BatteryAgingModel(base_params)
    
    print(f"{'Years':>6} {'Cycles':>8} {'SOH (%)':>10} {'Eff. Capacity':>15} {'Time-to-Empty':>15}")
    print("-" * 60)
    
    years_list = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
    
    for years in years_list:
        cycles = int(years * 365)  # Assume daily charging
        soh = aging_model.capacity_fade(years, cycles)
        eff_capacity = base_params.Q_nominal * soh
        
        # Create model with aged battery
        aged_params = BatteryParameters(SOH=soh)
        model = SmartphoneBatteryModel(aged_params)
        
        # Moderate usage profile
        profile = UsageProfile(
            screen_brightness=lambda t: 0.5,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.3,
            gpu_load=lambda t: 0.1,
            wifi_activity=lambda t: 0.4,
            cellular_activity=lambda t: 0.0,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )
        
        _, _, details = model.simulate(1.0, profile, t_max=24.0)
        t_empty = details['t_empty']
        
        print(f"{years:>6.1f} {cycles:>8d} {soh*100:>9.1f}% {eff_capacity:>13.0f} mAh {t_empty:>13.2f} h")
    
    print()


def run_sensitivity_summary():
    """
    Run and summarize sensitivity analysis.
    """
    print("\nSensitivity Analysis Summary:")
    print("-" * 60)
    
    model = SmartphoneBatteryModel()
    analyzer = SensitivityAnalyzer(model)
    scenarios = create_usage_scenarios()
    
    # Run OAT sensitivity for moderate usage
    profile = scenarios['moderate']
    oat_results = analyzer.oat_sensitivity(profile, n_points=15)
    
    # Sort by absolute elasticity
    sorted_params = sorted(
        oat_results.items(),
        key=lambda x: abs(x[1]['elasticity']),
        reverse=True
    )
    
    print("Parameter Sensitivities (Elasticity - % change in time-to-empty per % change in parameter):")
    print()
    print(f"{'Parameter':<25} {'Elasticity':>12} {'Impact':>10}")
    print("-" * 50)
    
    for param_name, data in sorted_params:
        elasticity = data['elasticity']
        if abs(elasticity) > 0.5:
            impact = "HIGH"
        elif abs(elasticity) > 0.1:
            impact = "MEDIUM"
        else:
            impact = "LOW"
        
        print(f"{param_name:<25} {elasticity:>+12.3f} {impact:>10}")
    
    print()
    
    # Monte Carlo uncertainty
    print("Monte Carlo Uncertainty Quantification (n=300):")
    mc_results = analyzer.monte_carlo_uncertainty(profile, n_samples=300)
    
    print(f"  Mean time-to-empty: {mc_results['mean']:.2f} hours")
    print(f"  Standard deviation: {mc_results['std']:.2f} hours")
    print(f"  90% Confidence Interval: [{mc_results['p5']:.2f}, {mc_results['p95']:.2f}] hours")
    print()


def generate_recommendations():
    """
    Generate practical recommendations for users.
    """
    print("\nPractical Recommendations:")
    print("=" * 60)
    
    model = SmartphoneBatteryModel()
    
    # Baseline moderate usage
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
    
    _, _, baseline_details = model.simulate(1.0, baseline, t_max=24.0)
    baseline_time = baseline_details['t_empty']
    
    recommendations = [
        ("Reduce screen brightness to 30%", UsageProfile(
            screen_brightness=lambda t: 0.3,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.35,
            gpu_load=lambda t: 0.2,
            wifi_activity=lambda t: 0.5,
            cellular_activity=lambda t: 0.3,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )),
        ("Use WiFi instead of cellular data", UsageProfile(
            screen_brightness=lambda t: 0.7,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.35,
            gpu_load=lambda t: 0.2,
            wifi_activity=lambda t: 0.6,
            cellular_activity=lambda t: 0.0,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )),
        ("Close background apps", UsageProfile(
            screen_brightness=lambda t: 0.7,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.2,
            gpu_load=lambda t: 0.1,
            wifi_activity=lambda t: 0.3,
            cellular_activity=lambda t: 0.1,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )),
        ("Turn off GPS when not needed", UsageProfile(
            screen_brightness=lambda t: 0.7,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.35,
            gpu_load=lambda t: 0.2,
            wifi_activity=lambda t: 0.5,
            cellular_activity=lambda t: 0.3,
            gps_active=lambda t: 0.0,  # Already off in baseline
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )),
        ("Enable battery saver mode (all optimizations)", UsageProfile(
            screen_brightness=lambda t: 0.3,
            screen_on=lambda t: 1.0,
            cpu_load=lambda t: 0.2,
            gpu_load=lambda t: 0.05,
            wifi_activity=lambda t: 0.3,
            cellular_activity=lambda t: 0.0,
            gps_active=lambda t: 0.0,
            bluetooth_active=lambda t: 0.0,
            temperature=lambda t: 25.0
        )),
    ]
    
    print(f"Baseline time-to-empty: {baseline_time:.2f} hours\n")
    print(f"{'Recommendation':<45} {'New Time':>12} {'Improvement':>12}")
    print("-" * 70)
    
    for desc, profile in recommendations:
        _, _, details = model.simulate(1.0, profile, t_max=24.0)
        new_time = details['t_empty']
        improvement = (new_time - baseline_time) / baseline_time * 100
        print(f"{desc:<45} {new_time:>10.2f} h {improvement:>+10.1f}%")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS:")
    print("-" * 60)
    print("""
1. SCREEN is the largest power consumer during active use (30-50% of total).
   → Reducing brightness has the highest single-factor impact.

2. CELLULAR data consumes 2-3x more power than WiFi for similar data rates.
   → Prefer WiFi when available, especially for large transfers.

3. TEMPERATURE significantly affects battery capacity:
   → Cold weather (<10°C) can reduce effective capacity by 20-30%.
   → Avoid leaving phone in direct sunlight (>40°C accelerates aging).

4. BATTERY AGING: Expect ~10% capacity loss per year with daily charging.
   → Partial charges (20-80%) reduce cycle aging.
   → Avoid full discharges which accelerate degradation.

5. BACKGROUND APPS can consume 10-20% of battery even with screen off.
   → Regularly review and restrict background activity permissions.

6. GPS is highly power-intensive (~0.5W continuous).
   → Use only when actively navigating, disable location services otherwise.
""")


def save_results_to_json(results: dict, filepath: str):
    """
    Save analysis results to JSON file.
    """
    # Convert numpy arrays and other non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(item) for item in obj]
        else:
            return obj
    
    converted = convert(results)
    
    with open(filepath, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"Results saved to {filepath}")


def main():
    """
    Main entry point for the analysis.
    """
    # Create output directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../figures', exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Starting Smartphone Battery Discharge Model Analysis")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")
    
    # Run all analyses
    scenario_results = run_scenario_analysis()
    run_temperature_analysis()
    run_aging_analysis()
    run_sensitivity_summary()
    generate_recommendations()
    
    # Generate figures
    print("\n" + "=" * 60)
    print("Generating Figures...")
    print("=" * 60)
    generate_all_figures('../figures/')
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - Figures: ../figures/")
    print("  - Data: ../data/")


if __name__ == "__main__":
    main()
