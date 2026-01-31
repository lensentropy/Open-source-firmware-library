"""
电池功耗子模块包

包含:
- network_model: 网络连接功耗模型
- gps_model: GPS/GNSS功耗模型
- background_tasks_model: 后台任务功耗模型
"""

from .network_model import (
    NetworkPowerModel, 
    NetworkParameters, 
    NetworkType, 
    NetworkState,
    NetworkActivityProfile
)

from .gps_model import (
    GPSPowerModel, 
    GPSParameters, 
    GNSSMode, 
    GNSSConstellation,
    GPSUsageScenarios
)

from .background_tasks_model import (
    BackgroundPowerModel, 
    CPUParameters, 
    CPUState,
    BackgroundTaskProfile,
    TaskType,
    CommonBackgroundTasks,
    BackgroundScenarios
)

__all__ = [
    # Network
    'NetworkPowerModel', 'NetworkParameters', 'NetworkType', 'NetworkState',
    'NetworkActivityProfile',
    # GPS
    'GPSPowerModel', 'GPSParameters', 'GNSSMode', 'GNSSConstellation',
    'GPSUsageScenarios',
    # Background
    'BackgroundPowerModel', 'CPUParameters', 'CPUState', 
    'BackgroundTaskProfile', 'TaskType',
    'CommonBackgroundTasks', 'BackgroundScenarios'
]
