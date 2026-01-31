"""
MCM 2026 Problem A: Smartphone Battery Discharge Model
======================================================

A physics-based continuous-time model for lithium-ion battery discharge
in smartphones under realistic usage conditions.

Modules:
- battery_model: Core battery discharge model
- sensitivity_analysis: Sensitivity and uncertainty analysis
- visualization: Figure generation for reports

Example usage:
    from src.battery_model import SmartphoneBatteryModel, UsageProfile
    
    model = SmartphoneBatteryModel()
    profile = UsageProfile()
    t, SOC, details = model.simulate(1.0, profile)
    print(f"Time to empty: {details['t_empty']:.2f} hours")
"""

from .battery_model import (
    SmartphoneBatteryModel,
    BatteryParameters,
    DeviceParameters,
    UsageProfile,
    BatteryAgingModel,
    create_usage_scenarios
)

from .sensitivity_analysis import (
    SensitivityAnalyzer,
    ScenarioComparison,
    ParameterRange
)

__version__ = "1.0.0"
__author__ = "MCM Team"
