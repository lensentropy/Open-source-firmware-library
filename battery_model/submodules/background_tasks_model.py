"""
后台任务功耗子模块 - 连续时间模型

基于计算机体系结构和操作系统原理的后台进程功耗建模

理论基础:
1. CMOS动态功耗模型: P = C × V² × f
2. CPU频率-电压缩放 (DVFS)
3. 进程调度和唤醒开销
4. 内存访问功耗模型

文献参考:
- Pathak, A., et al. (2012). "Fine-grained power modeling for smartphones using system 
  call tracing." EuroSys'12. ACM. [系统调用级功耗建模]

- Zhang, L., et al. (2010). "Accurate online power estimation and automatic battery 
  behavior based power model generation for smartphones." CODES+ISSS'10. ACM.

- Carroll, A., & Heiser, G. (2010). "An Analysis of Power Consumption in a Smartphone."
  USENIX ATC'10. [CPU功耗: 活跃74-377mW, 空闲7-35mW]

- Mittal, R., et al. (2012). "Empowering developers to estimate app energy consumption."
  MobiCom'12. ACM. [应用能耗估计方法]

- Shye, A., et al. (2009). "Into the wild: Studying real user activity patterns to guide 
  power optimizations for mobile architectures." MICRO'09. [真实用户行为研究]

硬件参考:
- ARM Cortex-A系列技术参考手册 (公开)
- Qualcomm Snapdragon功耗数据 (公开)
- Android电池统计API文档 (公开)

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class CPUState(Enum):
    """
    CPU状态枚举
    
    基于ARM big.LITTLE架构状态:
    - DEEP_SLEEP: 深度睡眠 (仅保持RAM自刷新)
    - LIGHT_SLEEP: 浅睡眠 (可快速唤醒)
    - IDLE: 空闲 (时钟门控)
    - ACTIVE_LOW: 低频活跃 (省电核心)
    - ACTIVE_HIGH: 高频活跃 (性能核心)
    """
    DEEP_SLEEP = "deep_sleep"
    LIGHT_SLEEP = "light_sleep"
    IDLE = "idle"
    ACTIVE_LOW = "active_low"
    ACTIVE_HIGH = "active_high"


class TaskType(Enum):
    """后台任务类型"""
    SYSTEM_SERVICE = "system"        # 系统服务
    SYNC_SERVICE = "sync"            # 同步服务 (邮件/云同步)
    PUSH_SERVICE = "push"            # 推送通知服务
    LOCATION_SERVICE = "location"    # 位置服务
    MEDIA_SERVICE = "media"          # 媒体服务 (后台音乐)
    SENSOR_SERVICE = "sensor"        # 传感器服务
    ALARM_SERVICE = "alarm"          # 闹钟/定时服务
    APP_BACKGROUND = "app_bg"        # 应用后台任务


@dataclass
class CPUParameters:
    """
    CPU功耗参数
    
    数据来源:
    1. Carroll & Heiser (2010) - Openmoko Neo Freerunner测量
       - CPU活跃: 74-377mW (取决于负载)
       - CPU空闲: 7-35mW
    
    2. ARM Cortex-A技术手册
       - 典型big.LITTLE配置功耗
       - DVFS功耗缩放关系
    
    3. Qualcomm Snapdragon 865数据表 (2020)
       - 大核 (Cortex-A77): ~300mW/核心 @峰值
       - 小核 (Cortex-A55): ~30mW/核心 @峰值
    """
    
    # ==================== 睡眠状态功耗 ====================
    deep_sleep_power: float = 0.003         # 深度睡眠 (W), ~3mW
    light_sleep_power: float = 0.010        # 浅睡眠 (W), ~10mW
    idle_power: float = 0.025               # 空闲状态 (W), ~25mW
    
    # ==================== 活跃状态功耗 ====================
    # 小核心 (效率核心)
    little_core_power_min: float = 0.010    # 最低频率功耗 (W)
    little_core_power_max: float = 0.050    # 最高频率功耗 (W)
    num_little_cores: int = 4
    
    # 大核心 (性能核心)
    big_core_power_min: float = 0.050       # 最低频率功耗 (W)
    big_core_power_max: float = 0.350       # 最高频率功耗 (W)
    num_big_cores: int = 4
    
    # ==================== DVFS参数 ====================
    # 频率-电压关系 (基于CMOS理论)
    freq_min: float = 0.3e9                 # 最小频率 (Hz), 300MHz
    freq_max: float = 2.8e9                 # 最大频率 (Hz), 2.8GHz
    voltage_min: float = 0.6                # 最小电压 (V)
    voltage_max: float = 1.1                # 最大电压 (V)
    
    # ==================== 内存功耗参数 ====================
    memory_idle_power: float = 0.050        # 内存空闲功耗 (W)
    memory_active_power: float = 0.200      # 内存活跃功耗 (W)
    
    # ==================== 唤醒开销 ====================
    wakeup_energy_deep: float = 0.005       # 从深度睡眠唤醒能量 (J)
    wakeup_energy_light: float = 0.001      # 从浅睡眠唤醒能量 (J)
    wakeup_latency_deep: float = 0.100      # 深度睡眠唤醒延迟 (s)
    wakeup_latency_light: float = 0.010     # 浅睡眠唤醒延迟 (s)


@dataclass
class BackgroundTaskProfile:
    """
    后台任务配置
    
    描述单个后台任务的行为模式
    """
    task_type: TaskType
    name: str
    
    # 周期性参数
    period: float = 900.0           # 执行周期 (s), 默认15分钟
    duration: float = 2.0           # 执行时长 (s)
    
    # 资源使用
    cpu_load: float = 0.1           # CPU负载 (0-1)
    memory_usage: float = 0.05      # 内存使用 (0-1)
    uses_network: bool = True       # 是否使用网络
    network_data: float = 10e3      # 网络数据量 (bytes)
    
    # 唤醒类型
    can_defer: bool = True          # 是否可延迟


class BackgroundPowerModel:
    """
    后台任务功耗连续时间模型
    
    核心理论:
    
    1. CMOS动态功耗模型:
       P_dynamic = α × C × V² × f
       
       其中:
       - α: 活动因子 (0-1)
       - C: 负载电容
       - V: 工作电压
       - f: 工作频率
    
    2. DVFS功耗缩放:
       频率和电压近似线性关系: V ∝ f
       因此: P ∝ f³ (立方关系)
       
       实际测量显示: P ∝ f^(2-2.5) (由于漏电流等因素)
    
    3. 静态功耗 (漏电流):
       P_static = I_leak × V
       I_leak ∝ exp(-V_th / (n×V_t))
       
       静态功耗随温度指数增加
    
    4. 周期性任务功耗:
       对于周期T、持续时间τ、功率P的任务:
       P_avg = P × τ/T + P_idle × (1 - τ/T)
       
    5. 进程调度开销:
       唤醒和上下文切换带来额外功耗:
       E_overhead = E_wakeup + E_context_switch
    """
    
    def __init__(self, params: Optional[CPUParameters] = None):
        """初始化后台任务功耗模型"""
        self.params = params or CPUParameters()
        self.tasks: List[BackgroundTaskProfile] = []
        self.current_state = CPUState.IDLE
    
    def add_task(self, task: BackgroundTaskProfile):
        """添加后台任务"""
        self.tasks.append(task)
    
    def dvfs_power(self, frequency: float, load: float) -> float:
        """
        计算DVFS状态下的CPU功耗
        
        基于CMOS功耗方程:
        P = P_static + P_dynamic
        P_dynamic = C_eff × V² × f
        
        使用频率-电压线性关系和测量数据拟合
        
        参数:
            frequency: 工作频率 (Hz)
            load: CPU负载 (0-1)
            
        返回:
            功率 (W)
        """
        p = self.params
        
        # 归一化频率
        freq_norm = (frequency - p.freq_min) / (p.freq_max - p.freq_min)
        freq_norm = np.clip(freq_norm, 0, 1)
        
        # 电压跟随频率 (线性近似)
        voltage = p.voltage_min + freq_norm * (p.voltage_max - p.voltage_min)
        
        # 动态功耗 (P ∝ V² × f)
        # 使用实测数据校准的系数
        p_ref = p.big_core_power_max
        v_ref = p.voltage_max
        f_ref = p.freq_max
        
        p_dynamic = p_ref * (voltage/v_ref)**2 * (frequency/f_ref)
        
        # 静态功耗 (与电压相关,与频率无关)
        p_static = 0.01 * (voltage/v_ref)**2
        
        # 总功耗受负载调制
        return (p_dynamic * load + p_static)
    
    def cpu_power_for_state(self, state: CPUState, load: float = 0) -> float:
        """
        计算指定CPU状态的功耗
        
        参数:
            state: CPU状态
            load: CPU负载
            
        返回:
            功率 (W)
        """
        p = self.params
        
        if state == CPUState.DEEP_SLEEP:
            return p.deep_sleep_power
        
        elif state == CPUState.LIGHT_SLEEP:
            return p.light_sleep_power
        
        elif state == CPUState.IDLE:
            return p.idle_power
        
        elif state == CPUState.ACTIVE_LOW:
            # 使用效率核心,低频率
            freq = p.freq_min + 0.3 * (p.freq_max - p.freq_min)
            return self.dvfs_power(freq, load)
        
        else:  # ACTIVE_HIGH
            # 使用性能核心,高频率
            freq = p.freq_min + 0.8 * (p.freq_max - p.freq_min)
            return self.dvfs_power(freq, load)
    
    def task_power(self, task: BackgroundTaskProfile, t: float) -> float:
        """
        计算单个任务的瞬时功耗
        
        周期性任务模型:
        - 在每个周期的开始执行
        - 执行期间产生CPU和内存功耗
        - 非执行期间功耗为零 (不计入)
        
        参数:
            task: 任务配置
            t: 当前时间 (s)
            
        返回:
            功率 (W)
        """
        p = self.params
        
        # 判断任务是否在执行
        t_in_period = t % task.period
        
        if t_in_period < task.duration:
            # 任务正在执行
            
            # CPU功耗
            if task.cpu_load > 0.3:
                state = CPUState.ACTIVE_HIGH
            else:
                state = CPUState.ACTIVE_LOW
            
            cpu_power = self.cpu_power_for_state(state, task.cpu_load)
            
            # 内存功耗
            mem_power = (p.memory_idle_power + 
                        (p.memory_active_power - p.memory_idle_power) * task.memory_usage)
            
            return cpu_power + mem_power
        else:
            return 0  # 任务休眠
    
    def total_background_power(self, t: float) -> float:
        """
        计算所有后台任务的总功耗
        
        连续时间模型:
        P_total(t) = P_base + Σ P_task_i(t)
        
        其中:
        - P_base: 基础系统功耗
        - P_task_i: 第i个任务的功耗
        
        参数:
            t: 时间 (s)
            
        返回:
            总功率 (W)
        """
        p = self.params
        
        # 基础系统功耗 (系统服务、内核等)
        base_power = p.idle_power + p.memory_idle_power
        
        # 累加所有任务功耗
        task_power = sum(self.task_power(task, t) for task in self.tasks)
        
        return base_power + task_power
    
    def create_power_function(self) -> Callable[[float], float]:
        """
        创建后台功耗的连续时间函数
        
        返回可用于ODE求解器的函数 P(t)
        """
        return lambda t: self.total_background_power(t)


class CommonBackgroundTasks:
    """
    常见后台任务预设
    
    数据来源:
    - Android Battery Historian工具分析
    - Pathak et al. (2012) - 系统调用功耗分析
    - 实际应用行为观察
    """
    
    @staticmethod
    def email_sync() -> BackgroundTaskProfile:
        """
        邮件同步服务
        
        典型行为:
        - 每15分钟检查一次新邮件
        - 需要网络连接
        - 中等CPU使用
        """
        return BackgroundTaskProfile(
            task_type=TaskType.SYNC_SERVICE,
            name="email_sync",
            period=900.0,           # 15分钟
            duration=3.0,           # 3秒执行时间
            cpu_load=0.2,
            memory_usage=0.05,
            uses_network=True,
            network_data=50e3,      # 50KB
            can_defer=True
        )
    
    @staticmethod
    def cloud_backup() -> BackgroundTaskProfile:
        """
        云备份服务 (如Google Photos, iCloud)
        
        典型行为:
        - 每30分钟检查一次
        - 高CPU使用 (图片处理)
        - 大量数据传输
        """
        return BackgroundTaskProfile(
            task_type=TaskType.SYNC_SERVICE,
            name="cloud_backup",
            period=1800.0,          # 30分钟
            duration=30.0,          # 30秒
            cpu_load=0.6,
            memory_usage=0.15,
            uses_network=True,
            network_data=1e6,       # 1MB
            can_defer=True
        )
    
    @staticmethod
    def push_notification() -> BackgroundTaskProfile:
        """
        推送通知服务 (FCM/APNs)
        
        典型行为:
        - 维持长连接
        - 周期性心跳
        - 低CPU使用
        """
        return BackgroundTaskProfile(
            task_type=TaskType.PUSH_SERVICE,
            name="push_notification",
            period=300.0,           # 5分钟心跳
            duration=0.5,           # 0.5秒
            cpu_load=0.05,
            memory_usage=0.02,
            uses_network=True,
            network_data=1e3,       # 1KB
            can_defer=False         # 不可延迟
        )
    
    @staticmethod
    def location_update() -> BackgroundTaskProfile:
        """
        位置更新服务
        
        典型行为:
        - 周期性位置更新
        - 触发GPS或网络定位
        - 中等CPU使用
        """
        return BackgroundTaskProfile(
            task_type=TaskType.LOCATION_SERVICE,
            name="location_update",
            period=600.0,           # 10分钟
            duration=5.0,           # 5秒
            cpu_load=0.15,
            memory_usage=0.03,
            uses_network=True,
            network_data=5e3,
            can_defer=True
        )
    
    @staticmethod
    def social_media_refresh() -> BackgroundTaskProfile:
        """
        社交媒体后台刷新
        
        典型行为:
        - 频繁刷新 (用户偏好)
        - 图片预加载
        - 较高数据使用
        """
        return BackgroundTaskProfile(
            task_type=TaskType.APP_BACKGROUND,
            name="social_media",
            period=600.0,           # 10分钟
            duration=5.0,
            cpu_load=0.25,
            memory_usage=0.08,
            uses_network=True,
            network_data=200e3,     # 200KB
            can_defer=True
        )
    
    @staticmethod
    def sensor_data_collection() -> BackgroundTaskProfile:
        """
        传感器数据收集 (如健康应用)
        
        典型行为:
        - 高频采样
        - 本地处理
        - 低网络使用
        """
        return BackgroundTaskProfile(
            task_type=TaskType.SENSOR_SERVICE,
            name="sensor_collection",
            period=60.0,            # 1分钟
            duration=2.0,
            cpu_load=0.1,
            memory_usage=0.03,
            uses_network=False,
            network_data=0,
            can_defer=True
        )
    
    @staticmethod
    def system_maintenance() -> BackgroundTaskProfile:
        """
        系统维护任务 (垃圾回收、优化等)
        
        典型行为:
        - 低频执行
        - 高CPU使用
        - 无网络
        """
        return BackgroundTaskProfile(
            task_type=TaskType.SYSTEM_SERVICE,
            name="system_maintenance",
            period=3600.0,          # 1小时
            duration=10.0,
            cpu_load=0.7,
            memory_usage=0.2,
            uses_network=False,
            network_data=0,
            can_defer=True
        )


class BackgroundScenarios:
    """
    后台使用场景配置
    """
    
    @staticmethod
    def minimal_background() -> List[BackgroundTaskProfile]:
        """
        最小后台活动 (省电模式)
        
        仅保留必要的系统服务
        """
        return [
            CommonBackgroundTasks.push_notification(),
        ]
    
    @staticmethod
    def typical_background() -> List[BackgroundTaskProfile]:
        """
        典型后台活动
        
        一般用户的标准配置
        """
        return [
            CommonBackgroundTasks.push_notification(),
            CommonBackgroundTasks.email_sync(),
            CommonBackgroundTasks.location_update(),
            CommonBackgroundTasks.social_media_refresh(),
        ]
    
    @staticmethod
    def heavy_background() -> List[BackgroundTaskProfile]:
        """
        重度后台活动
        
        多应用同步、频繁更新
        """
        return [
            CommonBackgroundTasks.push_notification(),
            CommonBackgroundTasks.email_sync(),
            CommonBackgroundTasks.cloud_backup(),
            CommonBackgroundTasks.location_update(),
            CommonBackgroundTasks.social_media_refresh(),
            CommonBackgroundTasks.sensor_data_collection(),
            CommonBackgroundTasks.system_maintenance(),
        ]


# 使用示例
if __name__ == "__main__":
    model = BackgroundPowerModel()
    
    print("=== 后台任务功耗模型测试 ===\n")
    
    # CPU状态功耗
    print("CPU状态功耗:")
    print(f"  深度睡眠: {model.cpu_power_for_state(CPUState.DEEP_SLEEP)*1000:.1f} mW")
    print(f"  浅睡眠: {model.cpu_power_for_state(CPUState.LIGHT_SLEEP)*1000:.1f} mW")
    print(f"  空闲: {model.cpu_power_for_state(CPUState.IDLE)*1000:.1f} mW")
    print(f"  低频活跃(50%负载): {model.cpu_power_for_state(CPUState.ACTIVE_LOW, 0.5)*1000:.1f} mW")
    print(f"  高频活跃(100%负载): {model.cpu_power_for_state(CPUState.ACTIVE_HIGH, 1.0)*1000:.1f} mW")
    
    # DVFS功耗曲线
    print("\nDVFS功耗 (50%负载):")
    for freq_ghz in [0.5, 1.0, 1.5, 2.0, 2.5]:
        freq = freq_ghz * 1e9
        power = model.dvfs_power(freq, 0.5)
        print(f"  {freq_ghz} GHz: {power*1000:.1f} mW")
    
    # 添加典型后台任务
    print("\n典型后台配置功耗:")
    for task in BackgroundScenarios.typical_background():
        model.add_task(task)
    
    # 计算不同时间点的功耗
    for t in [0, 1, 100, 500, 1000]:
        power = model.total_background_power(t)
        print(f"  t={t}s: {power*1000:.1f} mW")
