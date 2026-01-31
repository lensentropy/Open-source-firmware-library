"""
智能手机锂离子电池连续时间模型 - 核心模块

基于电化学原理的电池充电状态(SOC)连续时间数学模型

理论基础:
1. 库仑计数法 (Coulomb Counting)
2. 等效电路模型 (Equivalent Circuit Model)
3. Peukert定律修正
4. 温度影响的Arrhenius模型

文献参考:
- Plett, G.L. (2015). Battery Management Systems, Volume I: Battery Modeling. Artech House.
- Tremblay, O., et al. (2007). "Experimental validation of a battery dynamic model for EV applications."
  World Electric Vehicle Journal, Vol. 1.
- Doyle, M., et al. (1993). "Modeling of galvanostatic charge and discharge of the 
  lithium/polymer/insertion cell." Journal of the Electrochemical Society, 140(6), 1526-1533.

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
from scipy.integrate import odeint, solve_ivp
from typing import Callable, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BatteryParameters:
    """
    锂离子电池参数类
    
    参数来源:
    - 典型智能手机电池规格 (如Samsung SDI, LG Chem电芯数据)
    - 文献: Ecker, M., et al. (2015). "Parameterization of a Physico-Chemical Model 
      of a Lithium-Ion Battery." Journal of The Electrochemical Society, 162(9), A1836-A1848.
    """
    # 电池容量参数
    nominal_capacity: float = 4000.0  # 标称容量 (mAh), 典型智能手机电池
    nominal_voltage: float = 3.85     # 标称电压 (V), 锂离子电池典型值
    max_voltage: float = 4.35         # 最大充电电压 (V)
    min_voltage: float = 3.0          # 截止放电电压 (V)
    
    # 内阻参数 (来自文献测量数据)
    internal_resistance_25C: float = 0.08  # 25°C时内阻 (Ω)
    
    # 温度系数 (Arrhenius模型参数)
    activation_energy: float = 20000.0  # 活化能 (J/mol), 锂离子电池典型值
    reference_temperature: float = 298.15  # 参考温度 (K), 25°C
    
    # Peukert系数 (锂离子电池接近1.0)
    peukert_coefficient: float = 1.05  # Peukert指数, 锂离子电池典型值1.02-1.1
    
    # 自放电系数 (每月约1-2%)
    self_discharge_rate: float = 5.0e-7  # 自放电速率 (1/s), 约1.5%/月


class ContinuousTimeBatteryModel:
    """
    连续时间电池模型
    
    核心方程 (基于电化学原理):
    
    1. SOC动态方程 (库仑计数):
       dSOC/dt = -I(t) / Q_eff(T, I)
       
       其中:
       - SOC: 充电状态 (0-1)
       - I(t): 瞬时放电电流 (A)
       - Q_eff: 有效容量 (As), 受温度和电流影响
    
    2. 有效容量修正 (Peukert定律):
       Q_eff = Q_nom × (I_nom/I)^(k-1) × f(T)
       
       其中:
       - k: Peukert系数
       - f(T): 温度影响因子
    
    3. 温度影响 (Arrhenius模型):
       f(T) = exp[-Ea/R × (1/T - 1/T_ref)]
       
    4. 功率-电流关系:
       I = P / (V_oc(SOC) - I×R_int)
       
       简化为: I ≈ P / V_nom (对于低内阻情况)
    """
    
    R = 8.314  # 通用气体常数 (J/(mol·K))
    
    def __init__(self, params: Optional[BatteryParameters] = None):
        """初始化电池模型"""
        self.params = params or BatteryParameters()
        self._validate_parameters()
    
    def _validate_parameters(self):
        """验证参数合理性"""
        p = self.params
        assert 0 < p.nominal_capacity < 10000, "容量参数超出合理范围"
        assert 3.0 <= p.nominal_voltage <= 4.5, "标称电压超出锂离子电池范围"
        assert 1.0 <= p.peukert_coefficient <= 1.5, "Peukert系数超出合理范围"
    
    def temperature_factor(self, T: float) -> float:
        """
        计算温度影响因子
        
        基于Arrhenius方程:
        f(T) = exp[-Ea/R × (1/T - 1/T_ref)]
        
        参数:
            T: 当前温度 (K)
            
        返回:
            温度影响因子 (无量纲)
        """
        p = self.params
        return np.exp(-p.activation_energy / self.R * 
                     (1/T - 1/p.reference_temperature))
    
    def effective_capacity(self, current: float, temperature: float) -> float:
        """
        计算有效容量
        
        结合Peukert定律和温度影响:
        Q_eff = Q_nom × (I_nom/I)^(k-1) × f(T)
        
        参数:
            current: 放电电流 (A)
            temperature: 温度 (K)
            
        返回:
            有效容量 (mAh)
        """
        p = self.params
        # 标称电流 (0.5C放电)
        nominal_current = p.nominal_capacity / 2000  # mA -> A, 0.5C
        
        # 防止除零
        current = max(current, 1e-6)
        
        # Peukert修正
        peukert_factor = (nominal_current / current) ** (p.peukert_coefficient - 1)
        
        # 温度修正
        temp_factor = self.temperature_factor(temperature)
        
        return p.nominal_capacity * peukert_factor * temp_factor
    
    def internal_resistance(self, temperature: float, soc: float) -> float:
        """
        计算内阻
        
        内阻受温度和SOC影响:
        R_int = R_25C × exp[Ea_R/R × (1/T - 1/T_ref)] × f(SOC)
        
        文献来源:
        - Waag, W., et al. (2014). "Critical review of the methods for monitoring 
          of lithium-ion batteries in electric and hybrid vehicles."
          Journal of Power Sources, 258, 321-339.
        """
        p = self.params
        
        # 温度影响 (低温时内阻显著增加)
        temp_factor = np.exp(5000 / self.R * (1/temperature - 1/p.reference_temperature))
        
        # SOC影响 (低SOC时内阻增加)
        soc_factor = 1 + 0.5 * (1 - soc) ** 2
        
        return p.internal_resistance_25C * temp_factor * soc_factor
    
    def open_circuit_voltage(self, soc: float) -> float:
        """
        计算开路电压 OCV(SOC)
        
        采用多项式拟合模型:
        V_oc(SOC) = a0 + a1×SOC + a2×SOC² + ... + a_n×SOC^n
        
        系数来源:
        - 典型LiCoO2/石墨电池OCV-SOC曲线拟合
        - 参考: Chen, M., & Rincon-Mora, G.A. (2006). "Accurate electrical battery 
          model capable of predicting runtime and IV performance."
          IEEE Trans. Energy Conversion, 21(2), 504-511.
        """
        p = self.params
        
        # OCV多项式系数 (基于典型锂离子电池数据)
        coefficients = [
            3.0,     # a0
            0.85,    # a1
            0.35,    # a2
            -0.15,   # a3
            0.10     # a4
        ]
        
        soc_clipped = np.clip(soc, 0, 1)
        ocv = sum(c * soc_clipped**i for i, c in enumerate(coefficients))
        
        return np.clip(ocv, p.min_voltage, p.max_voltage)
    
    def power_to_current(self, power: float, soc: float, temperature: float) -> float:
        """
        将功率转换为电流
        
        基于电路方程:
        P = V × I = (V_oc - I×R_int) × I
        
        解二次方程得:
        I = (V_oc - sqrt(V_oc² - 4×R×P)) / (2×R)
        
        参数:
            power: 功率消耗 (W)
            soc: 当前SOC
            temperature: 温度 (K)
            
        返回:
            放电电流 (A)
        """
        v_oc = self.open_circuit_voltage(soc)
        r_int = self.internal_resistance(temperature, soc)
        
        # 解二次方程
        discriminant = v_oc**2 - 4 * r_int * power
        
        if discriminant < 0:
            # 功率需求超过电池能力,返回最大可能电流
            return v_oc / (2 * r_int)
        
        return (v_oc - np.sqrt(discriminant)) / (2 * r_int)
    
    def soc_dynamics(self, t: float, soc: float, 
                     power_func: Callable[[float], float],
                     temperature: float = 298.15) -> float:
        """
        SOC动态方程 (核心微分方程)
        
        dSOC/dt = -I(t) / (Q_eff × 3600/1000) - k_sd × SOC
        
        其中:
        - I(t): 瞬时电流
        - Q_eff: 有效容量 (mAh)
        - 3600/1000: 单位转换因子 (mAh -> As)
        - k_sd: 自放电系数
        
        参数:
            t: 时间 (s)
            soc: 当前SOC
            power_func: 功率随时间变化的函数 P(t)
            temperature: 温度 (K)
            
        返回:
            dSOC/dt
        """
        p = self.params
        
        # 获取当前功率消耗
        power = power_func(t)
        
        # 将功率转换为电流
        current = self.power_to_current(power, soc, temperature)
        
        # 计算有效容量
        q_eff = self.effective_capacity(current, temperature)
        
        # 容量单位转换: mAh -> As
        q_eff_As = q_eff * 3.6  # 3600s/1000
        
        # SOC变化率
        dsoc_dt = -current / q_eff_As - p.self_discharge_rate * soc
        
        return dsoc_dt
    
    def simulate(self, 
                 power_func: Callable[[float], float],
                 initial_soc: float = 1.0,
                 duration: float = 3600,
                 temperature: float = 298.15,
                 num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟电池SOC随时间的变化
        
        参数:
            power_func: 功率函数 P(t)
            initial_soc: 初始SOC
            duration: 模拟时长 (s)
            temperature: 温度 (K)
            num_points: 输出点数
            
        返回:
            (时间数组, SOC数组)
        """
        t_span = (0, duration)
        t_eval = np.linspace(0, duration, num_points)
        
        def ode_func(t, y):
            return self.soc_dynamics(t, y[0], power_func, temperature)
        
        solution = solve_ivp(
            ode_func,
            t_span,
            [initial_soc],
            t_eval=t_eval,
            method='RK45',
            events=lambda t, y: y[0] - 0.01  # SOC达到1%时停止
        )
        
        return solution.t, solution.y[0]
    
    def estimate_remaining_time(self, 
                                current_soc: float,
                                power_func: Callable[[float], float],
                                temperature: float = 298.15,
                                soc_threshold: float = 0.05) -> float:
        """
        估算剩余使用时间
        
        通过数值积分估算SOC从当前值下降到阈值所需时间
        
        参数:
            current_soc: 当前SOC
            power_func: 功率函数
            temperature: 温度
            soc_threshold: SOC阈值 (关机点)
            
        返回:
            剩余时间 (s)
        """
        max_time = 24 * 3600  # 最长24小时
        
        def event_func(t, y):
            return y[0] - soc_threshold
        event_func.terminal = True
        event_func.direction = -1
        
        def ode_func(t, y):
            return self.soc_dynamics(t, y[0], power_func, temperature)
        
        solution = solve_ivp(
            ode_func,
            (0, max_time),
            [current_soc],
            method='RK45',
            events=event_func
        )
        
        if solution.t_events[0].size > 0:
            return solution.t_events[0][0]
        else:
            return max_time


def calculate_energy_consumption(voltage: float, current: float) -> float:
    """
    计算能量消耗
    
    P = V × I
    
    参数:
        voltage: 工作电压 (V)
        current: 工作电流 (A)
        
    返回:
        功率 (W)
    """
    return voltage * current


if __name__ == "__main__":
    # 示例: 创建模型并进行简单模拟
    model = ContinuousTimeBatteryModel()
    
    # 定义恒定功率消耗 (2W, 典型待机功率)
    power_func = lambda t: 2.0
    
    # 模拟
    t, soc = model.simulate(power_func, initial_soc=1.0, duration=3600)
    
    print(f"初始SOC: {soc[0]:.4f}")
    print(f"1小时后SOC: {soc[-1]:.4f}")
    print(f"SOC下降: {(soc[0] - soc[-1])*100:.2f}%")
