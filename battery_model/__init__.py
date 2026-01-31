"""
智能手机电池连续时间模型

基于电化学原理和电子学理论的锂离子电池SOC建模

模块结构:
- core_battery_model: 核心电池模型 (库仑计数, Peukert定律, 温度影响)
- submodules/
  - network_model: 网络连接功耗模型 (WiFi, LTE, 5G)
  - gps_model: GPS/GNSS功耗模型
  - background_tasks_model: 后台任务功耗模型
- visualization/
  - power_analysis: 功耗分析与可视化

使用方法:
    from battery_model import ContinuousTimeBatteryModel
    from battery_model.submodules import NetworkPowerModel, GPSPowerModel
    
    model = ContinuousTimeBatteryModel()
    t, soc = model.simulate(power_func, initial_soc=1.0, duration=3600)

作者: Battery Modeling Team
版本: 1.0.0
日期: 2026-01-31
"""

from .core_battery_model import (
    ContinuousTimeBatteryModel,
    BatteryParameters,
    calculate_energy_consumption
)

__version__ = "1.0.0"
__author__ = "Battery Modeling Team"

__all__ = [
    'ContinuousTimeBatteryModel',
    'BatteryParameters',
    'calculate_energy_consumption'
]
