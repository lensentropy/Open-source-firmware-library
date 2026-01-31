"""
Sensitivity Analysis Module for Smartphone Battery Discharge Model
MCM 2026 Problem A Solution

This module provides comprehensive sensitivity analysis including:
- One-at-a-time (OAT) parameter sensitivity
- Global sensitivity analysis (Sobol indices)
- Monte Carlo uncertainty quantification
"""

import numpy as np
from scipy.stats import uniform, norm, lognorm
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import warnings
from battery_model import (
    SmartphoneBatteryModel, BatteryParameters, DeviceParameters,
    UsageProfile, create_usage_scenarios
)


@dataclass
class ParameterRange:
    """Define a parameter with its uncertainty range."""
    name: str
    base_value: float
    lower_bound: float
    upper_bound: float
    distribution: str = 'uniform'  # 'uniform', 'normal', 'lognormal'
    std_dev: Optional[float] = None


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on the battery discharge model.
    """
    
    def __init__(self, model: SmartphoneBatteryModel):
        """
        Initialize sensitivity analyzer.
        
        Args:
            model: SmartphoneBatteryModel instance
        """
        self.model = model
        self.base_battery_params = BatteryParameters()
        self.base_device_params = DeviceParameters()
        
    def define_parameter_ranges(self) -> List[ParameterRange]:
        """
        Define parameter ranges for sensitivity analysis.
        
        Returns:
            List of ParameterRange objects
        """
        return [
            # Battery parameters
            ParameterRange('Q_nominal', 4000, 3000, 5000),
            ParameterRange('R_internal', 0.1, 0.05, 0.2),
            ParameterRange('SOH', 1.0, 0.7, 1.0),
            
            # Device parameters
            ParameterRange('max_brightness_power', 2.5, 1.5, 4.0),
            ParameterRange('cpu_max_power', 5.0, 3.0, 8.0),
            ParameterRange('cellular_active_power', 2.0, 1.5, 3.5),
            
            # Usage parameters
            ParameterRange('screen_brightness', 0.5, 0.0, 1.0),
            ParameterRange('cpu_load', 0.3, 0.0, 1.0),
            
            # Environmental
            ParameterRange('temperature', 25.0, -10.0, 45.0),
        ]
    
    def oat_sensitivity(self, 
                        profile: UsageProfile,
                        SOC_initial: float = 1.0,
                        n_points: int = 20) -> Dict[str, np.ndarray]:
        """
        One-at-a-time sensitivity analysis.
        
        Varies each parameter individually while holding others constant.
        
        Args:
            profile: Base usage profile
            SOC_initial: Initial SOC
            n_points: Number of points to sample for each parameter
            
        Returns:
            Dictionary with sensitivity results for each parameter
        """
        param_ranges = self.define_parameter_ranges()
        results = {}
        
        # Base case
        t_base, SOC_base, details_base = self.model.simulate(
            SOC_initial, profile, t_max=24.0
        )
        t_empty_base = details_base['t_empty']
        
        for param in param_ranges:
            values = np.linspace(param.lower_bound, param.upper_bound, n_points)
            t_empty_values = []
            
            for val in values:
                # Create modified model
                modified_model = self._create_modified_model(param.name, val)
                
                # Create modified profile if parameter is usage-related
                modified_profile = self._create_modified_profile(
                    profile, param.name, val
                )
                
                # Simulate
                try:
                    _, _, details = modified_model.simulate(
                        SOC_initial, modified_profile, t_max=48.0
                    )
                    t_empty_values.append(details['t_empty'])
                except Exception as e:
                    warnings.warn(f"Simulation failed for {param.name}={val}: {e}")
                    t_empty_values.append(np.nan)
            
            # Calculate sensitivity metrics
            t_empty_array = np.array(t_empty_values)
            
            # Local sensitivity (derivative at base value)
            idx_base = np.argmin(np.abs(values - param.base_value))
            if idx_base > 0 and idx_base < len(values) - 1:
                local_sens = (t_empty_array[idx_base+1] - t_empty_array[idx_base-1]) / \
                            (values[idx_base+1] - values[idx_base-1])
            else:
                local_sens = 0.0
            
            # Normalized sensitivity (elasticity)
            if t_empty_base > 0 and param.base_value != 0:
                elasticity = local_sens * param.base_value / t_empty_base
            else:
                elasticity = 0.0
            
            results[param.name] = {
                'values': values,
                't_empty': t_empty_array,
                'local_sensitivity': local_sens,
                'elasticity': elasticity,
                't_empty_base': t_empty_base,
                'range': (param.lower_bound, param.upper_bound)
            }
        
        return results
    
    def _create_modified_model(self, param_name: str, value: float) -> SmartphoneBatteryModel:
        """Create a model with a modified parameter."""
        battery_params = BatteryParameters()
        device_params = DeviceParameters()
        
        # Battery parameters
        if param_name == 'Q_nominal':
            battery_params.Q_nominal = value
        elif param_name == 'R_internal':
            battery_params.R_internal = value
        elif param_name == 'SOH':
            battery_params.SOH = value
        
        # Device parameters
        elif param_name == 'max_brightness_power':
            device_params.max_brightness_power = value
        elif param_name == 'cpu_max_power':
            device_params.cpu_max_power = value
        elif param_name == 'cellular_active_power':
            device_params.cellular_active_power = value
        
        return SmartphoneBatteryModel(battery_params, device_params)
    
    def _create_modified_profile(self, 
                                  base_profile: UsageProfile,
                                  param_name: str, 
                                  value: float) -> UsageProfile:
        """Create a profile with a modified parameter."""
        # Create a copy by using the same functions initially
        profile = UsageProfile(
            screen_brightness=base_profile.screen_brightness,
            screen_on=base_profile.screen_on,
            cpu_load=base_profile.cpu_load,
            gpu_load=base_profile.gpu_load,
            wifi_activity=base_profile.wifi_activity,
            cellular_activity=base_profile.cellular_activity,
            gps_active=base_profile.gps_active,
            bluetooth_active=base_profile.bluetooth_active,
            temperature=base_profile.temperature
        )
        
        # Modify specific parameters
        if param_name == 'screen_brightness':
            profile.screen_brightness = lambda t, v=value: v
        elif param_name == 'cpu_load':
            profile.cpu_load = lambda t, v=value: v
        elif param_name == 'temperature':
            profile.temperature = lambda t, v=value: v
        
        return profile
    
    def monte_carlo_uncertainty(self,
                                profile: UsageProfile,
                                SOC_initial: float = 1.0,
                                n_samples: int = 500,
                                param_ranges: Optional[List[ParameterRange]] = None
                                ) -> Dict[str, np.ndarray]:
        """
        Monte Carlo uncertainty quantification.
        
        Randomly samples parameters from their distributions and
        propagates uncertainty through the model.
        
        Args:
            profile: Base usage profile
            SOC_initial: Initial SOC
            n_samples: Number of Monte Carlo samples
            param_ranges: Parameter ranges (uses defaults if None)
            
        Returns:
            Dictionary with uncertainty quantification results
        """
        if param_ranges is None:
            param_ranges = self.define_parameter_ranges()
        
        t_empty_samples = []
        param_samples = {p.name: [] for p in param_ranges}
        
        for i in range(n_samples):
            # Sample parameters
            sampled_values = {}
            for param in param_ranges:
                if param.distribution == 'uniform':
                    val = np.random.uniform(param.lower_bound, param.upper_bound)
                elif param.distribution == 'normal':
                    std = param.std_dev or (param.upper_bound - param.lower_bound) / 4
                    val = np.clip(
                        np.random.normal(param.base_value, std),
                        param.lower_bound,
                        param.upper_bound
                    )
                else:
                    val = param.base_value
                
                sampled_values[param.name] = val
                param_samples[param.name].append(val)
            
            # Create modified model and profile
            battery_params = BatteryParameters(
                Q_nominal=sampled_values.get('Q_nominal', 4000),
                R_internal=sampled_values.get('R_internal', 0.1),
                SOH=sampled_values.get('SOH', 1.0)
            )
            device_params = DeviceParameters(
                max_brightness_power=sampled_values.get('max_brightness_power', 2.5),
                cpu_max_power=sampled_values.get('cpu_max_power', 5.0),
                cellular_active_power=sampled_values.get('cellular_active_power', 2.0)
            )
            
            model = SmartphoneBatteryModel(battery_params, device_params)
            
            modified_profile = UsageProfile(
                screen_brightness=lambda t, v=sampled_values.get('screen_brightness', 0.5): v,
                screen_on=profile.screen_on,
                cpu_load=lambda t, v=sampled_values.get('cpu_load', 0.3): v,
                gpu_load=profile.gpu_load,
                wifi_activity=profile.wifi_activity,
                cellular_activity=profile.cellular_activity,
                gps_active=profile.gps_active,
                bluetooth_active=profile.bluetooth_active,
                temperature=lambda t, v=sampled_values.get('temperature', 25.0): v
            )
            
            # Simulate
            try:
                _, _, details = model.simulate(SOC_initial, modified_profile, t_max=48.0)
                t_empty_samples.append(details['t_empty'])
            except Exception as e:
                t_empty_samples.append(np.nan)
        
        t_empty_array = np.array(t_empty_samples)
        valid_samples = t_empty_array[~np.isnan(t_empty_array)]
        
        results = {
            't_empty_samples': t_empty_array,
            'mean': np.nanmean(t_empty_array),
            'std': np.nanstd(t_empty_array),
            'median': np.nanmedian(t_empty_array),
            'p5': np.nanpercentile(t_empty_array, 5),
            'p25': np.nanpercentile(t_empty_array, 25),
            'p75': np.nanpercentile(t_empty_array, 75),
            'p95': np.nanpercentile(t_empty_array, 95),
            'param_samples': param_samples
        }
        
        # Calculate correlation coefficients
        correlations = {}
        for param_name, samples in param_samples.items():
            valid_mask = ~np.isnan(t_empty_array)
            if np.sum(valid_mask) > 10:
                corr = np.corrcoef(
                    np.array(samples)[valid_mask], 
                    t_empty_array[valid_mask]
                )[0, 1]
                correlations[param_name] = corr
        results['correlations'] = correlations
        
        return results
    
    def sobol_indices(self,
                      profile: UsageProfile,
                      SOC_initial: float = 1.0,
                      n_samples: int = 256) -> Dict[str, float]:
        """
        Calculate first-order Sobol sensitivity indices.
        
        Uses Saltelli sampling scheme for variance-based sensitivity.
        
        Args:
            profile: Base usage profile
            SOC_initial: Initial SOC
            n_samples: Base sample size (total evaluations = n_samples * (2*k + 2))
            
        Returns:
            Dictionary with Sobol indices for each parameter
        """
        param_ranges = self.define_parameter_ranges()
        k = len(param_ranges)
        
        # Generate Saltelli sample matrices
        # A and B are independent samples, AB_i has A except column i from B
        np.random.seed(42)  # Reproducibility
        
        A = np.random.random((n_samples, k))
        B = np.random.random((n_samples, k))
        
        # Scale to parameter ranges
        for i, param in enumerate(param_ranges):
            A[:, i] = param.lower_bound + A[:, i] * (param.upper_bound - param.lower_bound)
            B[:, i] = param.lower_bound + B[:, i] * (param.upper_bound - param.lower_bound)
        
        # Evaluate model
        def evaluate(params_matrix):
            results = []
            for params in params_matrix:
                try:
                    battery_params = BatteryParameters(
                        Q_nominal=params[0],
                        R_internal=params[1],
                        SOH=params[2]
                    )
                    device_params = DeviceParameters(
                        max_brightness_power=params[3],
                        cpu_max_power=params[4],
                        cellular_active_power=params[5]
                    )
                    
                    modified_profile = UsageProfile(
                        screen_brightness=lambda t, v=params[6]: v,
                        screen_on=profile.screen_on,
                        cpu_load=lambda t, v=params[7]: v,
                        gpu_load=profile.gpu_load,
                        wifi_activity=profile.wifi_activity,
                        cellular_activity=profile.cellular_activity,
                        gps_active=profile.gps_active,
                        bluetooth_active=profile.bluetooth_active,
                        temperature=lambda t, v=params[8]: v
                    )
                    
                    model = SmartphoneBatteryModel(battery_params, device_params)
                    _, _, details = model.simulate(SOC_initial, modified_profile, t_max=48.0)
                    results.append(details['t_empty'])
                except:
                    results.append(np.nan)
            return np.array(results)
        
        # Evaluate A and B
        Y_A = evaluate(A)
        Y_B = evaluate(B)
        
        # Calculate first-order indices
        var_total = np.nanvar(np.concatenate([Y_A, Y_B]))
        sobol_indices = {}
        
        for i, param in enumerate(param_ranges):
            # Create AB_i matrix (A with column i from B)
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            
            Y_AB_i = evaluate(AB_i)
            
            # First-order Sobol index
            valid = ~(np.isnan(Y_A) | np.isnan(Y_B) | np.isnan(Y_AB_i))
            if np.sum(valid) > n_samples // 2 and var_total > 0:
                V_i = np.mean(Y_B[valid] * (Y_AB_i[valid] - Y_A[valid]))
                S_i = V_i / var_total
                sobol_indices[param.name] = np.clip(S_i, 0, 1)
            else:
                sobol_indices[param.name] = 0.0
        
        return sobol_indices


class ScenarioComparison:
    """
    Compare battery performance across different usage scenarios.
    """
    
    def __init__(self, model: SmartphoneBatteryModel):
        """
        Initialize scenario comparison.
        
        Args:
            model: SmartphoneBatteryModel instance
        """
        self.model = model
        
    def compare_scenarios(self,
                          SOC_initial: float = 1.0,
                          scenarios: Optional[Dict[str, UsageProfile]] = None
                          ) -> Dict[str, Dict]:
        """
        Compare battery performance across multiple scenarios.
        
        Args:
            SOC_initial: Initial SOC
            scenarios: Dictionary of usage scenarios (uses defaults if None)
            
        Returns:
            Dictionary with comparison results for each scenario
        """
        if scenarios is None:
            scenarios = create_usage_scenarios()
        
        results = {}
        
        for name, profile in scenarios.items():
            try:
                t, SOC, details = self.model.simulate(SOC_initial, profile, t_max=48.0)
                
                # Calculate average power consumption
                power_samples = [
                    self.model.power_consumption(ti, profile)['total'] 
                    for ti in np.linspace(0, min(details['t_empty'], 24), 50)
                ]
                avg_power = np.mean(power_samples)
                
                # Power breakdown at midpoint
                t_mid = details['t_empty'] / 2 if details['t_empty'] < 48 else 12
                power_breakdown = self.model.power_consumption(t_mid, profile)
                
                results[name] = {
                    't_empty': details['t_empty'],
                    'final_SOC': details['final_SOC'],
                    'avg_power': avg_power,
                    'power_breakdown': power_breakdown,
                    't': t,
                    'SOC': SOC
                }
            except Exception as e:
                warnings.warn(f"Scenario {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def rank_power_components(self, scenario_results: Dict) -> Dict[str, Dict]:
        """
        Rank power-consuming components by their impact.
        
        Args:
            scenario_results: Results from compare_scenarios()
            
        Returns:
            Dictionary with ranked power components for each scenario
        """
        rankings = {}
        
        for name, result in scenario_results.items():
            if 'error' in result:
                continue
            
            breakdown = result['power_breakdown']
            total = breakdown['total']
            
            # Calculate percentage contribution
            components = {}
            for key, value in breakdown.items():
                if key != 'total':
                    components[key] = {
                        'power_W': value,
                        'percentage': (value / total * 100) if total > 0 else 0
                    }
            
            # Sort by power consumption
            sorted_components = sorted(
                components.items(), 
                key=lambda x: x[1]['power_W'], 
                reverse=True
            )
            
            rankings[name] = {
                'total_power_W': total,
                'components': dict(sorted_components)
            }
        
        return rankings


def run_comprehensive_sensitivity_analysis(save_dir: str = '../data/') -> Dict:
    """
    Run complete sensitivity analysis suite.
    
    Args:
        save_dir: Directory to save results
        
    Returns:
        Dictionary with all analysis results
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model = SmartphoneBatteryModel()
    analyzer = SensitivityAnalyzer(model)
    scenarios = create_usage_scenarios()
    
    results = {}
    
    # Run for multiple scenarios
    for scenario_name in ['light', 'moderate', 'heavy_gaming']:
        profile = scenarios[scenario_name]
        
        print(f"\nAnalyzing scenario: {scenario_name}")
        
        # OAT sensitivity
        print("  Running OAT sensitivity analysis...")
        oat_results = analyzer.oat_sensitivity(profile, n_points=15)
        
        # Monte Carlo
        print("  Running Monte Carlo uncertainty analysis...")
        mc_results = analyzer.monte_carlo_uncertainty(profile, n_samples=300)
        
        results[scenario_name] = {
            'oat': oat_results,
            'monte_carlo': mc_results
        }
        
        # Print summary
        print(f"  Time-to-empty: {mc_results['mean']:.2f} ± {mc_results['std']:.2f} hours")
        print(f"  95% CI: [{mc_results['p5']:.2f}, {mc_results['p95']:.2f}] hours")
        
        # Top correlations
        print("  Top parameter sensitivities:")
        corr_sorted = sorted(
            mc_results['correlations'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for param, corr in corr_sorted[:5]:
            print(f"    {param}: {corr:.3f}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_sensitivity_analysis()
