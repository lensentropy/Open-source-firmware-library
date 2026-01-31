"""
网络连接功耗子模块 - 连续时间模型

基于电子学原理和实测数据的WiFi/蜂窝网络功耗建模

理论基础:
1. 无线电功率放大器效率模型
2. 信号衰减与功率补偿 (Friis传输方程)
3. 调制解调器状态机模型 (DRX/DTX周期)

文献参考:
- Huang, J., et al. (2012). "A Close Examination of Performance and Power Characteristics 
  of 4G LTE Networks." MobiSys'12. ACM. [测量了LTE功耗特性]
  
- Balasubramanian, N., et al. (2009). "Energy Consumption in Mobile Phones: A Measurement 
  Study and Implications for Network Applications." IMC'09. ACM. 
  [WiFi vs 3G功耗对比研究]
  
- Perrucci, G.P., et al. (2011). "Survey on Energy Consumption Entities on the Smartphone 
  Platform." VTC 2011-Spring. IEEE. [智能手机各组件功耗综述]
  
- Carroll, A., & Heiser, G. (2010). "An Analysis of Power Consumption in a Smartphone."
  USENIX ATC'10. [详细的智能手机功耗分析]

数据来源:
- Qualcomm Snapdragon功耗规格 (公开数据表)
- 3GPP TS 36.321 (LTE MAC层规范, DRX参数)
- IEEE 802.11规范 (WiFi省电模式)

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class NetworkType(Enum):
    """网络类型枚举"""
    WIFI = "wifi"
    LTE_4G = "lte"
    NR_5G = "5g"
    IDLE = "idle"  # 飞行模式或无连接


class NetworkState(Enum):
    """
    网络状态枚举
    
    基于3GPP DRX (Discontinuous Reception) 状态机:
    - IDLE: 空闲状态 (最低功耗)
    - DRX: 间歇接收状态 (中等功耗)  
    - ACTIVE: 活跃传输状态 (最高功耗)
    
    参考: 3GPP TS 36.321 Section 5.7 (DRX operation)
    """
    IDLE = "idle"
    DRX = "drx"
    ACTIVE = "active"


@dataclass
class NetworkParameters:
    """
    网络功耗参数
    
    数据来源:
    1. Huang et al. (2012) - LTE实测数据
       - LTE活跃状态: 1.2-2.5W
       - LTE DRX状态: 0.1-0.6W
       
    2. Carroll & Heiser (2010) - 智能手机功耗分析
       - WiFi接收: 0.3-0.7W
       - WiFi发送: 0.7-1.2W
       
    3. Qualcomm SDM865规格表 (2020, 公开)
       - 5G NR: 2.5-5.0W (取决于配置)
       - 调制解调器待机: ~50mW
    """
    
    # ==================== WiFi参数 ====================
    # IEEE 802.11n/ac典型功耗 (来自Carroll & Heiser, 2010)
    wifi_idle_power: float = 0.010          # 空闲功率 (W), PSM模式
    wifi_rx_power: float = 0.450            # 接收功率 (W)
    wifi_tx_power_base: float = 0.800       # 发送基础功率 (W)
    wifi_tx_power_max: float = 1.200        # 最大发送功率 (W)
    
    # WiFi功率放大器效率 (典型值10-20%)
    wifi_pa_efficiency: float = 0.15
    
    # ==================== LTE参数 ====================
    # 来自Huang et al. (2012) MobiSys论文
    lte_idle_power: float = 0.050           # RRC_IDLE状态功率 (W)
    lte_drx_power: float = 0.300            # DRX状态平均功率 (W)
    lte_rx_power: float = 1.000             # 活跃接收功率 (W)
    lte_tx_power_base: float = 1.500        # 发送基础功率 (W)
    lte_tx_power_max: float = 2.500         # 最大发送功率 (W)
    
    # LTE DRX周期参数 (3GPP典型配置)
    lte_drx_cycle: float = 0.320            # DRX周期 (s), 320ms典型值
    lte_drx_on_duration: float = 0.002      # 每周期唤醒时间 (s)
    
    # ==================== 5G NR参数 ====================
    # 来自Qualcomm技术规格和早期测量
    nr_idle_power: float = 0.100            # 空闲功率 (W)
    nr_drx_power: float = 0.500             # DRX状态功率 (W)
    nr_rx_power: float = 1.500              # 接收功率 (W)
    nr_tx_power_base: float = 2.500         # 发送基础功率 (W)
    nr_tx_power_max: float = 5.000          # 最大发送功率 (W)
    
    # ==================== 信号强度影响 ====================
    # 功率随信号强度调整 (基于Friis方程推导)
    rssi_reference: float = -50.0           # 参考信号强度 (dBm)
    power_adjustment_factor: float = 0.02   # 每dB信号变化的功率调整因子


class NetworkPowerModel:
    """
    网络功耗连续时间模型
    
    核心理论:
    
    1. 无线发射功率模型:
       发射功率需要补偿路径损耗以维持链路预算
       
       P_tx = P_tx_base × 10^((RSSI_ref - RSSI) × k / 10)
       
       其中:
       - RSSI: 接收信号强度指示 (dBm)
       - k: 功率调整因子
    
    2. 功率放大器功耗:
       射频功率放大器是主要耗电组件
       
       P_PA = P_rf_out / η_PA
       
       其中:
       - P_rf_out: 射频输出功率
       - η_PA: 功率放大器效率 (典型10-30%)
    
    3. 基带处理功耗:
       数字信号处理功耗与数据速率相关
       
       P_baseband = P_0 + α × R
       
       其中:
       - P_0: 静态功耗
       - R: 数据速率 (bps)
       - α: 每比特功耗系数
    
    4. 状态转换功耗:
       网络状态机转换带来额外功耗
       
       E_transition = ∫_{t_0}^{t_0+τ} P_trans(t) dt
    """
    
    def __init__(self, params: Optional[NetworkParameters] = None):
        """初始化网络功耗模型"""
        self.params = params or NetworkParameters()
        self.current_state = NetworkState.IDLE
        self.current_type = NetworkType.WIFI
    
    def signal_strength_adjustment(self, rssi: float) -> float:
        """
        计算信号强度引起的功率调整
        
        基于Friis传输方程:
        P_r = P_t × G_t × G_r × (λ/(4πd))²
        
        反推发射功率调整:
        ΔP(dB) = (RSSI_ref - RSSI) × k
        
        物理意义: 信号弱时,需要更高发射功率来维持连接质量
        
        参数:
            rssi: 接收信号强度 (dBm)
            
        返回:
            功率调整因子 (线性)
        """
        p = self.params
        delta_rssi = p.rssi_reference - rssi
        
        # 功率调整 (dB转线性)
        # 使用软限制防止极端值
        delta_db = np.clip(delta_rssi * p.power_adjustment_factor * 10, -10, 20)
        
        return 10 ** (delta_db / 10)
    
    def wifi_power(self, 
                   data_rate: float,  # bps
                   rssi: float = -50.0,  # dBm
                   is_transmitting: bool = False) -> float:
        """
        WiFi功耗模型
        
        连续时间功率方程:
        P_wifi(t) = P_idle + P_baseband(R) + P_rf(RSSI, tx)
        
        组成部分:
        1. 空闲功耗 (始终存在)
        2. 基带处理功耗 (与数据速率成比例)
        3. 射频功耗 (发送/接收,受信号强度影响)
        
        参数:
            data_rate: 数据传输速率 (bps)
            rssi: 信号强度 (dBm), 典型范围 -30 到 -90
            is_transmitting: 是否正在发送数据
            
        返回:
            瞬时功率 (W)
        """
        p = self.params
        
        # 空闲功耗 (维持连接的基础功耗)
        power = p.wifi_idle_power
        
        if data_rate > 0:
            # 基带处理功耗
            # 参考: 802.11n最大速率约600Mbps
            max_rate = 600e6  # bps
            baseband_power = 0.1 * min(data_rate / max_rate, 1.0)
            power += baseband_power
            
            # 射频功耗
            if is_transmitting:
                # 发送功率 (主要功耗来源)
                sig_adj = self.signal_strength_adjustment(rssi)
                rf_power = p.wifi_tx_power_base * sig_adj
                rf_power = min(rf_power, p.wifi_tx_power_max)
            else:
                # 接收功率
                rf_power = p.wifi_rx_power
            
            power += rf_power
        
        return power
    
    def lte_power(self,
                  data_rate: float,  # bps
                  rsrp: float = -80.0,  # dBm, LTE参考信号功率
                  state: NetworkState = NetworkState.DRX) -> float:
        """
        LTE功耗模型
        
        基于Huang et al. (2012)的测量结果建立的连续时间模型:
        
        状态功耗方程:
        
        1. IDLE状态 (RRC_IDLE):
           P = P_idle (恒定,约50mW)
        
        2. DRX状态 (RRC_CONNECTED, DRX活跃):
           P = P_drx_base + P_drx_monitor × duty_cycle
           
           其中 duty_cycle = T_on / T_cycle
        
        3. ACTIVE状态 (活跃传输):
           P = P_baseband + P_rf(RSRP)
        
        状态转换延迟和功耗 (来自3GPP规范):
        - IDLE -> CONNECTED: ~100ms, 功耗峰值~2W
        - DRX -> ACTIVE: ~10ms
        
        参数:
            data_rate: 数据传输速率 (bps)
            rsrp: LTE参考信号接收功率 (dBm)
            state: 当前网络状态
            
        返回:
            瞬时功率 (W)
        """
        p = self.params
        
        if state == NetworkState.IDLE:
            return p.lte_idle_power
        
        elif state == NetworkState.DRX:
            # DRX状态: 周期性唤醒监听
            # 平均功耗 = 空闲功耗 + (活跃功耗 - 空闲功耗) × 占空比
            duty_cycle = p.lte_drx_on_duration / p.lte_drx_cycle
            return p.lte_idle_power + (p.lte_rx_power - p.lte_idle_power) * duty_cycle
        
        else:  # ACTIVE
            # 活跃状态功耗
            power = p.lte_rx_power
            
            if data_rate > 0:
                # 基带处理功耗
                max_rate = 150e6  # LTE Cat 4 峰值速率
                baseband_power = 0.3 * min(data_rate / max_rate, 1.0)
                power += baseband_power
                
                # 上行发送功率调整
                sig_adj = self.signal_strength_adjustment(rsrp)
                tx_power = p.lte_tx_power_base * sig_adj
                tx_power = min(tx_power, p.lte_tx_power_max)
                
                # 假设50%时间用于上行
                power += tx_power * 0.5
            
            return power
    
    def nr_5g_power(self,
                    data_rate: float,  # bps
                    ss_rsrp: float = -85.0,  # dBm
                    state: NetworkState = NetworkState.DRX) -> float:
        """
        5G NR功耗模型
        
        5G NR相比LTE有更高的功耗,主要原因:
        1. 更宽的信道带宽 (100MHz vs 20MHz)
        2. 更高的载波频率 (毫米波需要更多功率)
        3. 更复杂的信号处理 (Massive MIMO, beamforming)
        
        功耗模型:
        P_5G = P_LTE × k_bandwidth × k_frequency × k_mimo
        
        参数:
            data_rate: 数据传输速率 (bps)
            ss_rsrp: 5G同步信号参考功率 (dBm)
            state: 网络状态
            
        返回:
            瞬时功率 (W)
        """
        p = self.params
        
        if state == NetworkState.IDLE:
            return p.nr_idle_power
        
        elif state == NetworkState.DRX:
            # 5G DRX功耗高于LTE
            duty_cycle = 0.01  # 更短的DRX周期
            return p.nr_idle_power + (p.nr_rx_power - p.nr_idle_power) * duty_cycle
        
        else:  # ACTIVE
            power = p.nr_rx_power
            
            if data_rate > 0:
                # 基带处理功耗 (5G需要更强的处理能力)
                max_rate = 1e9  # 1Gbps峰值
                baseband_power = 0.8 * min(data_rate / max_rate, 1.0)
                power += baseband_power
                
                # 射频功耗
                sig_adj = self.signal_strength_adjustment(ss_rsrp)
                tx_power = p.nr_tx_power_base * sig_adj
                tx_power = min(tx_power, p.nr_tx_power_max)
                power += tx_power * 0.5
            
            return power
    
    def get_power(self,
                  network_type: NetworkType,
                  data_rate: float = 0,
                  signal_strength: float = -60.0,
                  state: NetworkState = NetworkState.DRX) -> float:
        """
        获取指定网络类型的功耗
        
        统一接口,根据网络类型调用相应模型
        
        参数:
            network_type: 网络类型
            data_rate: 数据速率 (bps)
            signal_strength: 信号强度 (dBm)
            state: 网络状态
            
        返回:
            功率 (W)
        """
        if network_type == NetworkType.IDLE:
            return 0.005  # 最小功耗
        
        elif network_type == NetworkType.WIFI:
            is_tx = data_rate > 1e6  # 假设高速率时有上传
            return self.wifi_power(data_rate, signal_strength, is_tx)
        
        elif network_type == NetworkType.LTE_4G:
            return self.lte_power(data_rate, signal_strength, state)
        
        elif network_type == NetworkType.NR_5G:
            return self.nr_5g_power(data_rate, signal_strength, state)
        
        return 0
    
    def continuous_power_function(self,
                                  network_type: NetworkType,
                                  data_rate_func: Callable[[float], float],
                                  signal_func: Callable[[float], float],
                                  state_func: Callable[[float], NetworkState]
                                  ) -> Callable[[float], float]:
        """
        生成连续时间功率函数 P_network(t)
        
        返回一个可以用于ODE求解器的功率函数
        
        参数:
            network_type: 网络类型
            data_rate_func: 数据速率函数 R(t)
            signal_func: 信号强度函数 S(t)
            state_func: 状态函数 state(t)
            
        返回:
            功率函数 P(t)
        """
        def power_func(t: float) -> float:
            data_rate = data_rate_func(t)
            signal = signal_func(t)
            state = state_func(t)
            return self.get_power(network_type, data_rate, signal, state)
        
        return power_func


class NetworkActivityProfile:
    """
    网络活动配置类
    
    定义典型使用场景的网络活动模式
    
    数据来源:
    - Falaki, H., et al. (2010). "Diversity in smartphone usage."
      MobiSys'10. [用户行为多样性研究]
    - Oliver, E. (2010). "The challenges in large-scale smartphone 
      user studies." HotPlanet'10. [大规模用户研究]
    """
    
    @staticmethod
    def idle_profile(t: float) -> Tuple[float, float, NetworkState]:
        """
        空闲模式 (锁屏,后台同步)
        
        特征:
        - 周期性短数据包 (推送通知检查)
        - 大部分时间处于DRX状态
        """
        # 每5分钟一次短暂活动
        cycle = 300  # 5分钟
        active_duration = 2  # 2秒
        
        time_in_cycle = t % cycle
        if time_in_cycle < active_duration:
            return (50e3, -70.0, NetworkState.ACTIVE)  # 50kbps
        else:
            return (0, -70.0, NetworkState.DRX)
    
    @staticmethod
    def browsing_profile(t: float) -> Tuple[float, float, NetworkState]:
        """
        网页浏览模式
        
        特征:
        - 间歇性高速下载 (页面加载)
        - 较长空闲期 (阅读)
        """
        cycle = 30  # 30秒一个周期
        active_duration = 5  # 5秒下载
        
        time_in_cycle = t % cycle
        if time_in_cycle < active_duration:
            # 页面加载期间高速率
            rate = 5e6  # 5Mbps
            return (rate, -65.0, NetworkState.ACTIVE)
        else:
            return (0, -65.0, NetworkState.IDLE)
    
    @staticmethod
    def streaming_profile(t: float) -> Tuple[float, float, NetworkState]:
        """
        流媒体播放模式 (视频/音乐)
        
        特征:
        - 持续中等带宽
        - 缓冲区策略导致的周期性流量
        """
        # 视频: 720p约3Mbps, 1080p约5-8Mbps
        buffer_cycle = 10  # 10秒缓冲周期
        buffer_duration = 3  # 3秒高速下载
        
        time_in_cycle = t % buffer_cycle
        if time_in_cycle < buffer_duration:
            return (8e6, -60.0, NetworkState.ACTIVE)  # 8Mbps
        else:
            return (0.1e6, -60.0, NetworkState.DRX)  # 维持连接
    
    @staticmethod
    def gaming_profile(t: float) -> Tuple[float, float, NetworkState]:
        """
        在线游戏模式
        
        特征:
        - 持续低延迟小数据包
        - 始终保持活跃连接
        """
        # 游戏需要低延迟,持续活跃
        base_rate = 0.5e6  # 500kbps基础
        # 添加一些波动
        rate = base_rate * (1 + 0.5 * np.sin(t * 0.1))
        return (rate, -55.0, NetworkState.ACTIVE)


# 使用示例
if __name__ == "__main__":
    model = NetworkPowerModel()
    
    # 测试不同网络类型的功耗
    print("=== 网络功耗模型测试 ===\n")
    
    # WiFi功耗测试
    print("WiFi功耗:")
    print(f"  空闲: {model.wifi_power(0, -50):.3f} W")
    print(f"  低速率接收 (1Mbps): {model.wifi_power(1e6, -60, False):.3f} W")
    print(f"  高速率发送 (50Mbps): {model.wifi_power(50e6, -70, True):.3f} W")
    
    # LTE功耗测试
    print("\nLTE功耗:")
    print(f"  IDLE状态: {model.lte_power(0, -80, NetworkState.IDLE):.3f} W")
    print(f"  DRX状态: {model.lte_power(0, -80, NetworkState.DRX):.3f} W")
    print(f"  ACTIVE (50Mbps): {model.lte_power(50e6, -90, NetworkState.ACTIVE):.3f} W")
    
    # 5G功耗测试
    print("\n5G NR功耗:")
    print(f"  IDLE状态: {model.nr_5g_power(0, -85, NetworkState.IDLE):.3f} W")
    print(f"  ACTIVE (500Mbps): {model.nr_5g_power(500e6, -95, NetworkState.ACTIVE):.3f} W")
