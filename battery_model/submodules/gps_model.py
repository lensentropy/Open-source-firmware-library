"""
全球定位系统(GPS/GNSS)功耗子模块 - 连续时间模型

基于射频电子学和卫星导航原理的GPS功耗建模

理论基础:
1. GPS接收机架构和信号处理
2. 相关器和跟踪环路功耗
3. 冷启动/热启动功耗差异
4. 辅助GPS (A-GPS) 功耗优化

文献参考:
- Zandbergen, P.A., & Barbeau, S.J. (2011). "Positional accuracy of assisted GPS data 
  from high-sensitivity GPS-enabled mobile phones." Journal of Navigation, 64(3), 381-399.

- Kjærgaard, M.B., et al. (2011). "EnTracked: energy-efficient robust position tracking 
  for mobile devices." MobiSys'11. ACM. [GPS功耗优化研究]

- Carroll, A., & Heiser, G. (2010). "An Analysis of Power Consumption in a Smartphone."
  USENIX ATC'10. [测量GPS功耗: 活跃状态~143mW, 含基带处理]

- Lin, K., et al. (2010). "Energy-accuracy trade-off for continuous mobile device 
  location." MobiSys'10. [GPS能量-精度权衡分析]

- Karunanayake, A., et al. (2007). "GPS Receiver Power Consumption Analysis."
  ION GPS/GNSS 2007. [GPS接收机功耗详细分析]

硬件参考:
- u-blox NEO系列GPS模块数据表 (公开)
- Qualcomm gpsOne规格 (公开)
- Broadcom BCM4774 GNSS芯片规格 (公开)

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class GNSSMode(Enum):
    """
    GNSS工作模式枚举
    
    基于接收机状态机:
    - OFF: 完全关闭
    - COLD_START: 冷启动 (无星历数据,需要完整搜索)
    - WARM_START: 热启动 (有部分星历数据)
    - HOT_START: 高速启动 (有完整星历和时间)
    - TRACKING: 持续跟踪模式
    - LOW_POWER: 低功耗定位模式
    """
    OFF = "off"
    COLD_START = "cold_start"
    WARM_START = "warm_start"
    HOT_START = "hot_start"
    TRACKING = "tracking"
    LOW_POWER = "low_power"


class GNSSConstellation(Enum):
    """GNSS星座类型"""
    GPS_ONLY = "gps"
    GPS_GLONASS = "gps_glonass"
    MULTI_GNSS = "multi_gnss"  # GPS + GLONASS + Galileo + BeiDou


@dataclass
class GPSParameters:
    """
    GPS/GNSS功耗参数
    
    数据来源:
    1. Carroll & Heiser (2010) - 智能手机GPS测量
       - GPS活跃: 143mW (含基带)
       - GPS接收机: ~30mW
    
    2. u-blox NEO-M8系列数据表 (公开)
       - 持续跟踪: 25-30mW
       - 捕获模式: 40-50mW
       - 待机: ~0.5mW
    
    3. Qualcomm Snapdragon Location技术白皮书
       - 多GNSS: 50-100mW
       - A-GPS: 降低TTFF功耗30-50%
    """
    
    # ==================== 基础功耗参数 ====================
    # 完全关闭功耗
    off_power: float = 0.0                  # 关闭状态 (W)
    
    # 待机功耗 (保持RTC运行)
    standby_power: float = 0.0005           # 待机功耗 (W), ~0.5mW
    
    # ==================== 启动功耗参数 ====================
    # 冷启动功耗 (需要完整星历下载和信号捕获)
    cold_start_power: float = 0.120         # 冷启动功率 (W)
    cold_start_duration: float = 45.0       # 冷启动时间 (s), 典型30-60s
    
    # 热启动功耗
    warm_start_power: float = 0.080         # 热启动功率 (W)
    warm_start_duration: float = 30.0       # 热启动时间 (s)
    
    # 高速启动功耗 (有A-GPS辅助)
    hot_start_power: float = 0.060          # 热启动功率 (W)
    hot_start_duration: float = 3.0         # 高速启动时间 (s)
    
    # ==================== 持续跟踪功耗 ====================
    # 单GPS星座
    tracking_power_gps: float = 0.030       # GPS单星座跟踪 (W)
    
    # 双星座 (GPS + GLONASS)
    tracking_power_dual: float = 0.045      # 双星座跟踪 (W)
    
    # 多星座 (GPS + GLONASS + Galileo + BeiDou)
    tracking_power_multi: float = 0.065     # 多星座跟踪 (W)
    
    # ==================== 基带处理功耗 ====================
    # 位置计算和滤波 (卡尔曼滤波等)
    baseband_power: float = 0.100           # 基带处理功率 (W)
    
    # ==================== 低功耗模式参数 ====================
    # 周期性定位模式
    low_power_period: float = 1.0           # 定位周期 (s)
    low_power_active: float = 0.100         # 活跃时功率 (W)
    low_power_sleep: float = 0.001          # 休眠时功率 (W)
    
    # ==================== 环境影响参数 ====================
    # 信号衰减影响功耗
    weak_signal_power_factor: float = 1.3   # 弱信号时功耗增加因子
    indoor_power_factor: float = 1.5        # 室内定位功耗增加因子


class GPSPowerModel:
    """
    GPS/GNSS功耗连续时间模型
    
    核心理论:
    
    1. GPS接收机功耗组成:
       P_total = P_RF + P_correlator + P_baseband + P_memory
       
       其中:
       - P_RF: 射频前端功耗 (LNA, 混频器, ADC)
       - P_correlator: 相关器功耗 (信号捕获和跟踪)
       - P_baseband: 基带处理功耗 (位置计算)
       - P_memory: 存储器访问功耗 (星历存储)
    
    2. 相关器功耗模型:
       GPS信号捕获需要对多普勒频移和码相位进行二维搜索
       
       P_correlator = N_ch × N_correlators × P_per_correlator
       
       其中:
       - N_ch: 跟踪通道数 (典型12-32个)
       - N_correlators: 每通道相关器数
       - P_per_correlator: 单相关器功耗
    
    3. 信号质量影响:
       弱信号条件下,接收机需要更长的积分时间和更多计算
       
       P_weak = P_normal × (1 + k × (C/N0_ref - C/N0) / 10)
       
       其中:
       - C/N0: 载噪比 (dB-Hz)
       - k: 功耗调整系数
    
    4. 状态转换功耗:
       启动过程有显著功耗峰值
       
       E_startup = ∫_{0}^{T_startup} P_startup(t) dt
    """
    
    def __init__(self, params: Optional[GPSParameters] = None):
        """初始化GPS功耗模型"""
        self.params = params or GPSParameters()
        self.current_mode = GNSSMode.OFF
        self.constellation = GNSSConstellation.GPS_ONLY
        self.startup_time = 0.0
        self.is_starting = False
    
    def get_tracking_power(self, constellation: GNSSConstellation) -> float:
        """
        获取持续跟踪功耗
        
        多星座支持增加功耗,但提高定位可靠性:
        - 更多通道需要更多相关器
        - 不同频率需要更多RF前端
        
        参数:
            constellation: GNSS星座配置
            
        返回:
            跟踪功率 (W)
        """
        p = self.params
        
        if constellation == GNSSConstellation.GPS_ONLY:
            return p.tracking_power_gps
        elif constellation == GNSSConstellation.GPS_GLONASS:
            return p.tracking_power_dual
        else:
            return p.tracking_power_multi
    
    def signal_quality_factor(self, cn0: float = 40.0) -> float:
        """
        计算信号质量对功耗的影响
        
        基于GPS接收机信号处理原理:
        - 弱信号需要更长积分时间 (相干/非相干积分)
        - 更长积分时间意味着更高功耗
        
        载噪比C/N0参考值:
        - 开阔天空: 45-50 dB-Hz
        - 城市环境: 35-45 dB-Hz
        - 室内: 20-35 dB-Hz (A-GPS辅助)
        
        参数:
            cn0: 载噪比 (dB-Hz)
            
        返回:
            功耗调整因子
        """
        p = self.params
        
        # 参考载噪比
        cn0_ref = 45.0  # dB-Hz
        
        # 功耗随载噪比下降而增加
        if cn0 >= cn0_ref:
            return 1.0
        else:
            # 每10dB载噪比下降,功耗增加约30%
            delta = (cn0_ref - cn0) / 10.0
            return 1.0 + 0.3 * delta
    
    def startup_power(self, t: float, mode: GNSSMode) -> float:
        """
        计算启动过程功耗
        
        启动功耗模型:
        P_startup(t) = P_peak × f(t/T_startup)
        
        f(x) 是归一化功耗曲线:
        - 初期: 信号搜索阶段,功耗较高
        - 中期: 星历下载阶段
        - 后期: 位置计算,功耗降低
        
        参数:
            t: 启动后的时间 (s)
            mode: 启动模式
            
        返回:
            功率 (W)
        """
        p = self.params
        
        if mode == GNSSMode.COLD_START:
            peak_power = p.cold_start_power
            duration = p.cold_start_duration
        elif mode == GNSSMode.WARM_START:
            peak_power = p.warm_start_power
            duration = p.warm_start_duration
        else:  # HOT_START
            peak_power = p.hot_start_power
            duration = p.hot_start_duration
        
        if t >= duration:
            return 0  # 启动完成
        
        # 归一化时间
        x = t / duration
        
        # 启动功耗曲线 (初期高,后期低)
        # 使用sigmoid函数模拟
        power_profile = 1.0 - 0.3 * x  # 简化的线性衰减
        
        return peak_power * power_profile
    
    def continuous_tracking_power(self, 
                                   t: float,
                                   constellation: GNSSConstellation,
                                   cn0: float = 40.0,
                                   update_rate: float = 1.0) -> float:
        """
        计算持续跟踪模式下的功耗
        
        连续时间方程:
        P(t) = P_rf + P_tracking(constellation) + P_baseband(rate)
        
        参数:
            t: 时间 (s)
            constellation: 星座配置
            cn0: 载噪比 (dB-Hz)
            update_rate: 位置更新率 (Hz)
            
        返回:
            功率 (W)
        """
        p = self.params
        
        # 基础跟踪功耗
        tracking = self.get_tracking_power(constellation)
        
        # 信号质量影响
        sig_factor = self.signal_quality_factor(cn0)
        
        # 基带处理功耗 (与更新率相关)
        baseband = p.baseband_power * update_rate
        
        # 总功耗
        total = (tracking * sig_factor + baseband)
        
        return total
    
    def low_power_mode_power(self, t: float, period: float = 1.0) -> float:
        """
        低功耗模式功耗计算
        
        周期性定位模式:
        - 短暂激活获取位置
        - 大部分时间休眠
        
        平均功耗:
        P_avg = P_active × T_active/T_period + P_sleep × (1 - T_active/T_period)
        
        参数:
            t: 时间 (s)
            period: 定位周期 (s)
            
        返回:
            瞬时功率 (W)
        """
        p = self.params
        
        # 活跃时间 (假设100ms足够获取位置更新)
        active_duration = 0.1
        
        # 周期内位置
        t_in_period = t % period
        
        if t_in_period < active_duration:
            return p.low_power_active
        else:
            return p.low_power_sleep
    
    def get_power(self, 
                  mode: GNSSMode,
                  constellation: GNSSConstellation = GNSSConstellation.GPS_ONLY,
                  cn0: float = 40.0,
                  t_since_start: float = 0.0,
                  update_rate: float = 1.0) -> float:
        """
        获取指定模式下的GPS功耗
        
        统一接口函数
        
        参数:
            mode: 工作模式
            constellation: 星座配置
            cn0: 载噪比
            t_since_start: 启动后时间
            update_rate: 位置更新率
            
        返回:
            功率 (W)
        """
        p = self.params
        
        if mode == GNSSMode.OFF:
            return p.off_power
        
        elif mode in [GNSSMode.COLD_START, GNSSMode.WARM_START, GNSSMode.HOT_START]:
            return self.startup_power(t_since_start, mode)
        
        elif mode == GNSSMode.TRACKING:
            return self.continuous_tracking_power(0, constellation, cn0, update_rate)
        
        elif mode == GNSSMode.LOW_POWER:
            return self.low_power_mode_power(t_since_start)
        
        return 0
    
    def navigation_session_power(self,
                                  duration: float,
                                  start_mode: GNSSMode = GNSSMode.HOT_START,
                                  constellation: GNSSConstellation = GNSSConstellation.MULTI_GNSS,
                                  cn0: float = 42.0) -> Callable[[float], float]:
        """
        生成导航会话的功率函数
        
        模拟完整的GPS使用会话:
        1. 启动阶段 (冷/热/高速启动)
        2. 持续跟踪阶段
        3. 关闭
        
        参数:
            duration: 会话总时长 (s)
            start_mode: 启动模式
            constellation: 星座配置
            cn0: 信号质量
            
        返回:
            功率函数 P(t)
        """
        p = self.params
        
        # 确定启动时间
        if start_mode == GNSSMode.COLD_START:
            startup_duration = p.cold_start_duration
        elif start_mode == GNSSMode.WARM_START:
            startup_duration = p.warm_start_duration
        else:
            startup_duration = p.hot_start_duration
        
        def power_func(t: float) -> float:
            if t >= duration:
                return p.off_power
            
            if t < startup_duration:
                # 启动阶段
                return self.startup_power(t, start_mode)
            else:
                # 跟踪阶段
                return self.continuous_tracking_power(t, constellation, cn0)
        
        return power_func
    
    def calculate_ttff_energy(self, mode: GNSSMode) -> float:
        """
        计算首次定位时间(TTFF)的能量消耗
        
        TTFF能量 = ∫_{0}^{T_ttff} P(t) dt
        
        这是衡量GPS效率的重要指标
        
        参数:
            mode: 启动模式
            
        返回:
            能量消耗 (J)
        """
        p = self.params
        
        if mode == GNSSMode.COLD_START:
            # 冷启动: 高功率 × 长时间
            energy = p.cold_start_power * p.cold_start_duration * 0.9
        elif mode == GNSSMode.WARM_START:
            energy = p.warm_start_power * p.warm_start_duration * 0.85
        else:  # HOT_START
            energy = p.hot_start_power * p.hot_start_duration * 0.8
        
        return energy


class GPSUsageScenarios:
    """
    GPS使用场景配置
    
    定义典型使用模式的GPS活动
    
    数据来源:
    - Kjærgaard et al. (2011) - 移动设备位置追踪研究
    - Lin et al. (2010) - GPS能量-精度权衡
    """
    
    @staticmethod
    def navigation_app(t: float, session_duration: float = 1800) -> Tuple[GNSSMode, float, float]:
        """
        导航应用场景 (如Google Maps导航)
        
        特征:
        - 持续高精度定位
        - 高更新率 (1-5Hz)
        - 多星座支持
        
        返回:
            (模式, 载噪比, 更新率)
        """
        if t < 0 or t > session_duration:
            return (GNSSMode.OFF, 0, 0)
        
        if t < 3:  # 前3秒启动
            return (GNSSMode.HOT_START, 42.0, 1.0)
        else:
            return (GNSSMode.TRACKING, 42.0, 1.0)
    
    @staticmethod
    def fitness_tracker(t: float, session_duration: float = 3600) -> Tuple[GNSSMode, float, float]:
        """
        运动追踪场景 (跑步/骑行)
        
        特征:
        - 中等精度需求
        - 周期性更新 (1Hz)
        - 可能在复杂环境 (树林、城市)
        """
        if t < 0 or t > session_duration:
            return (GNSSMode.OFF, 0, 0)
        
        if t < 5:
            return (GNSSMode.WARM_START, 38.0, 1.0)
        else:
            return (GNSSMode.TRACKING, 38.0, 1.0)
    
    @staticmethod
    def location_tagging(t: float) -> Tuple[GNSSMode, float, float]:
        """
        位置标记场景 (拍照地理标记)
        
        特征:
        - 短暂单次定位
        - 中等精度即可
        - 快速启动优先
        """
        # 模拟30秒一次的位置标记
        cycle = 30
        active_duration = 5
        
        t_in_cycle = t % cycle
        
        if t_in_cycle < 3:  # 启动
            return (GNSSMode.HOT_START, 40.0, 1.0)
        elif t_in_cycle < active_duration:  # 获取位置
            return (GNSSMode.TRACKING, 40.0, 0.2)
        else:
            return (GNSSMode.OFF, 0, 0)
    
    @staticmethod
    def background_location(t: float) -> Tuple[GNSSMode, float, float]:
        """
        后台位置更新场景
        
        特征:
        - 低频更新 (每几分钟一次)
        - 使用低功耗模式
        - 与网络定位混合使用
        """
        # 每5分钟一次定位
        cycle = 300
        active_duration = 10
        
        t_in_cycle = t % cycle
        
        if t_in_cycle < active_duration:
            return (GNSSMode.LOW_POWER, 35.0, 0.1)
        else:
            return (GNSSMode.OFF, 0, 0)


# 使用示例和测试
if __name__ == "__main__":
    model = GPSPowerModel()
    
    print("=== GPS/GNSS功耗模型测试 ===\n")
    
    # 启动功耗测试
    print("启动功耗:")
    print(f"  冷启动 (t=0): {model.startup_power(0, GNSSMode.COLD_START)*1000:.1f} mW")
    print(f"  冷启动 (t=20s): {model.startup_power(20, GNSSMode.COLD_START)*1000:.1f} mW")
    print(f"  热启动 (t=0): {model.startup_power(0, GNSSMode.HOT_START)*1000:.1f} mW")
    
    # 跟踪功耗测试
    print("\n持续跟踪功耗:")
    print(f"  GPS单星座: {model.get_tracking_power(GNSSConstellation.GPS_ONLY)*1000:.1f} mW")
    print(f"  双星座 (GPS+GLONASS): {model.get_tracking_power(GNSSConstellation.GPS_GLONASS)*1000:.1f} mW")
    print(f"  多星座: {model.get_tracking_power(GNSSConstellation.MULTI_GNSS)*1000:.1f} mW")
    
    # 信号质量影响
    print("\n信号质量影响:")
    print(f"  良好信号 (C/N0=45): 因子 = {model.signal_quality_factor(45):.2f}")
    print(f"  一般信号 (C/N0=35): 因子 = {model.signal_quality_factor(35):.2f}")
    print(f"  弱信号 (C/N0=25): 因子 = {model.signal_quality_factor(25):.2f}")
    
    # TTFF能量
    print("\nTTFF能量消耗:")
    print(f"  冷启动: {model.calculate_ttff_energy(GNSSMode.COLD_START):.2f} J")
    print(f"  热启动: {model.calculate_ttff_energy(GNSSMode.WARM_START):.2f} J")
    print(f"  高速启动: {model.calculate_ttff_energy(GNSSMode.HOT_START):.2f} J")
