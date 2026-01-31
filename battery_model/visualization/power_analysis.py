"""
功耗分析与可视化模块

整合所有子模块,生成综合功耗分析和可视化图表

功能:
1. 各子模块独立功耗分析
2. 综合功耗时间序列
3. SOC仿真和剩余时间预测
4. 参数敏感性分析

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from scipy.integrate import solve_ivp
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_battery_model import ContinuousTimeBatteryModel, BatteryParameters
from submodules.network_model import (
    NetworkPowerModel, NetworkType, NetworkState, NetworkActivityProfile
)
from submodules.gps_model import (
    GPSPowerModel, GNSSMode, GNSSConstellation, GPSUsageScenarios
)
from submodules.background_tasks_model import (
    BackgroundPowerModel, CPUState, BackgroundScenarios, CommonBackgroundTasks
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class PowerAnalyzer:
    """
    综合功耗分析器
    
    整合网络、GPS和后台任务三个子模块
    """
    
    def __init__(self):
        self.network_model = NetworkPowerModel()
        self.gps_model = GPSPowerModel()
        self.background_model = BackgroundPowerModel()
        self.battery_model = ContinuousTimeBatteryModel()
        
        # 添加典型后台任务
        for task in BackgroundScenarios.typical_background():
            self.background_model.add_task(task)
    
    def analyze_network_power(self, duration: float = 3600) -> dict:
        """
        分析网络功耗随时间变化
        
        参数:
            duration: 分析时长 (s)
            
        返回:
            分析结果字典
        """
        t = np.linspace(0, duration, 1000)
        
        # 不同场景的功耗
        results = {
            'time': t,
            'wifi_idle': [],
            'wifi_browsing': [],
            'wifi_streaming': [],
            'lte_idle': [],
            'lte_browsing': [],
            'lte_streaming': [],
            '5g_streaming': []
        }
        
        for ti in t:
            # WiFi场景
            idle_params = NetworkActivityProfile.idle_profile(ti)
            browse_params = NetworkActivityProfile.browsing_profile(ti)
            stream_params = NetworkActivityProfile.streaming_profile(ti)
            
            results['wifi_idle'].append(
                self.network_model.wifi_power(idle_params[0], idle_params[1], False))
            results['wifi_browsing'].append(
                self.network_model.wifi_power(browse_params[0], browse_params[1], browse_params[0] > 1e6))
            results['wifi_streaming'].append(
                self.network_model.wifi_power(stream_params[0], stream_params[1], False))
            
            # LTE场景
            results['lte_idle'].append(
                self.network_model.lte_power(idle_params[0], -80, idle_params[2]))
            results['lte_browsing'].append(
                self.network_model.lte_power(browse_params[0], -75, browse_params[2]))
            results['lte_streaming'].append(
                self.network_model.lte_power(stream_params[0], -70, stream_params[2]))
            
            # 5G场景
            results['5g_streaming'].append(
                self.network_model.nr_5g_power(stream_params[0], -85, stream_params[2]))
        
        return results
    
    def analyze_gps_power(self, duration: float = 1800) -> dict:
        """
        分析GPS功耗随时间变化
        
        参数:
            duration: 分析时长 (s)
            
        返回:
            分析结果字典
        """
        t = np.linspace(0, duration, 500)
        
        results = {
            'time': t,
            'navigation': [],
            'fitness': [],
            'tagging': [],
            'background': [],
            'constellation_comparison': {}
        }
        
        for ti in t:
            # 导航场景
            nav_mode, nav_cn0, nav_rate = GPSUsageScenarios.navigation_app(ti, duration)
            results['navigation'].append(
                self.gps_model.get_power(nav_mode, GNSSConstellation.MULTI_GNSS, nav_cn0, ti, nav_rate))
            
            # 健身场景
            fit_mode, fit_cn0, fit_rate = GPSUsageScenarios.fitness_tracker(ti, duration)
            results['fitness'].append(
                self.gps_model.get_power(fit_mode, GNSSConstellation.GPS_GLONASS, fit_cn0, ti, fit_rate))
            
            # 位置标记
            tag_mode, tag_cn0, tag_rate = GPSUsageScenarios.location_tagging(ti)
            results['tagging'].append(
                self.gps_model.get_power(tag_mode, GNSSConstellation.GPS_ONLY, tag_cn0, ti, tag_rate))
            
            # 后台定位
            bg_mode, bg_cn0, bg_rate = GPSUsageScenarios.background_location(ti)
            results['background'].append(
                self.gps_model.get_power(bg_mode, GNSSConstellation.GPS_ONLY, bg_cn0, ti, bg_rate))
        
        # 星座对比
        for const in GNSSConstellation:
            results['constellation_comparison'][const.value] = \
                self.gps_model.get_tracking_power(const)
        
        return results
    
    def analyze_background_power(self, duration: float = 7200) -> dict:
        """
        分析后台任务功耗
        
        参数:
            duration: 分析时长 (s)
            
        返回:
            分析结果字典
        """
        t = np.linspace(0, duration, 1000)
        
        # 不同后台配置
        minimal_model = BackgroundPowerModel()
        for task in BackgroundScenarios.minimal_background():
            minimal_model.add_task(task)
        
        typical_model = BackgroundPowerModel()
        for task in BackgroundScenarios.typical_background():
            typical_model.add_task(task)
        
        heavy_model = BackgroundPowerModel()
        for task in BackgroundScenarios.heavy_background():
            heavy_model.add_task(task)
        
        results = {
            'time': t,
            'minimal': [minimal_model.total_background_power(ti) for ti in t],
            'typical': [typical_model.total_background_power(ti) for ti in t],
            'heavy': [heavy_model.total_background_power(ti) for ti in t],
            'cpu_states': {},
            'dvfs_curve': {}
        }
        
        # CPU状态功耗
        for state in CPUState:
            load = 0.5 if 'ACTIVE' in state.name else 0
            results['cpu_states'][state.name] = \
                self.background_model.cpu_power_for_state(state, load)
        
        # DVFS曲线
        freqs = np.linspace(0.3e9, 2.8e9, 50)
        results['dvfs_curve']['frequencies'] = freqs
        results['dvfs_curve']['power_50'] = [self.background_model.dvfs_power(f, 0.5) for f in freqs]
        results['dvfs_curve']['power_100'] = [self.background_model.dvfs_power(f, 1.0) for f in freqs]
        
        return results
    
    def simulate_combined_soc(self, 
                              duration: float = 14400,  # 4小时
                              scenario: str = 'typical') -> dict:
        """
        综合SOC仿真
        
        将所有子模块组合进行电池SOC仿真
        
        参数:
            duration: 仿真时长 (s)
            scenario: 使用场景
            
        返回:
            仿真结果
        """
        # 定义综合功率函数
        def combined_power(t):
            # 基础功耗 (屏幕关闭时的最小系统功耗)
            base_power = 0.1  # 100mW
            
            # 网络功耗 (周期性活动)
            net_params = NetworkActivityProfile.idle_profile(t)
            network_power = self.network_model.wifi_power(
                net_params[0], net_params[1], False)
            
            # GPS功耗 (偶尔使用)
            gps_params = GPSUsageScenarios.background_location(t)
            gps_power = self.gps_model.get_power(
                gps_params[0], GNSSConstellation.GPS_ONLY, 
                gps_params[1], t, gps_params[2])
            
            # 后台功耗
            bg_power = self.background_model.total_background_power(t)
            
            return base_power + network_power + gps_power + bg_power
        
        # 运行SOC仿真
        t, soc = self.battery_model.simulate(
            combined_power,
            initial_soc=1.0,
            duration=duration,
            temperature=298.15,
            num_points=500
        )
        
        # 分解各组件功耗
        network_powers = []
        gps_powers = []
        bg_powers = []
        
        for ti in t:
            net_params = NetworkActivityProfile.idle_profile(ti)
            network_powers.append(self.network_model.wifi_power(
                net_params[0], net_params[1], False))
            
            gps_params = GPSUsageScenarios.background_location(ti)
            gps_powers.append(self.gps_model.get_power(
                gps_params[0], GNSSConstellation.GPS_ONLY,
                gps_params[1], ti, gps_params[2]))
            
            bg_powers.append(self.background_model.total_background_power(ti))
        
        return {
            'time': t,
            'soc': soc,
            'network_power': network_powers,
            'gps_power': gps_powers,
            'background_power': bg_powers,
            'total_power': [combined_power(ti) for ti in t]
        }


def plot_network_analysis(results: dict, save_path: str = None):
    """绘制网络功耗分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_min = results['time'] / 60  # 转换为分钟
    
    # WiFi功耗对比
    ax = axes[0, 0]
    ax.plot(t_min[:200], np.array(results['wifi_idle'][:200])*1000, 
            label='Idle Mode', alpha=0.8)
    ax.plot(t_min[:200], np.array(results['wifi_browsing'][:200])*1000, 
            label='Web Browsing', alpha=0.8)
    ax.plot(t_min[:200], np.array(results['wifi_streaming'][:200])*1000, 
            label='Video Streaming', alpha=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('WiFi Power Consumption - Different Usage Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_min[199])
    
    # LTE功耗对比
    ax = axes[0, 1]
    ax.plot(t_min[:200], np.array(results['lte_idle'][:200])*1000, 
            label='Idle/DRX', alpha=0.8)
    ax.plot(t_min[:200], np.array(results['lte_browsing'][:200])*1000, 
            label='Web Browsing', alpha=0.8)
    ax.plot(t_min[:200], np.array(results['lte_streaming'][:200])*1000, 
            label='Video Streaming', alpha=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('LTE/4G Power Consumption - Different Usage Scenarios')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_min[199])
    
    # 网络类型对比 (平均功耗)
    ax = axes[1, 0]
    scenarios = ['Idle', 'Browsing', 'Streaming']
    wifi_avg = [np.mean(results['wifi_idle'])*1000, 
                np.mean(results['wifi_browsing'])*1000,
                np.mean(results['wifi_streaming'])*1000]
    lte_avg = [np.mean(results['lte_idle'])*1000,
               np.mean(results['lte_browsing'])*1000,
               np.mean(results['lte_streaming'])*1000]
    
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, wifi_avg, width, label='WiFi', color='#2196F3')
    ax.bar(x + width/2, lte_avg, width, label='LTE/4G', color='#FF9800')
    ax.set_xlabel('Usage Scenario')
    ax.set_ylabel('Average Power (mW)')
    ax.set_title('Network Power Comparison: WiFi vs LTE')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5G vs LTE对比
    ax = axes[1, 1]
    t_short = t_min[:100]
    ax.plot(t_short, np.array(results['lte_streaming'][:100])*1000, 
            label='LTE/4G', linewidth=2)
    ax.plot(t_short, np.array(results['5g_streaming'][:100])*1000, 
            label='5G NR', linewidth=2)
    ax.fill_between(t_short, 
                    np.array(results['lte_streaming'][:100])*1000,
                    np.array(results['5g_streaming'][:100])*1000,
                    alpha=0.3, color='red', label='5G Overhead')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('5G vs LTE Power Consumption (Video Streaming)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_gps_analysis(results: dict, save_path: str = None):
    """绘制GPS功耗分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_min = results['time'] / 60
    
    # 不同场景功耗
    ax = axes[0, 0]
    ax.plot(t_min, np.array(results['navigation'])*1000, 
            label='Navigation App', linewidth=1.5)
    ax.plot(t_min, np.array(results['fitness'])*1000, 
            label='Fitness Tracking', linewidth=1.5)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('GPS Power: Navigation vs Fitness Tracking')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 间歇性使用功耗
    ax = axes[0, 1]
    ax.plot(t_min[:150], np.array(results['tagging'][:150])*1000, 
            label='Location Tagging', linewidth=1.5, color='#4CAF50')
    ax.plot(t_min[:150], np.array(results['background'][:150])*1000, 
            label='Background Updates', linewidth=1.5, color='#9C27B0')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('GPS Power: Intermittent Usage Patterns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 星座对比
    ax = axes[1, 0]
    constellations = list(results['constellation_comparison'].keys())
    powers = [results['constellation_comparison'][c]*1000 for c in constellations]
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    bars = ax.bar(constellations, powers, color=colors)
    ax.set_xlabel('GNSS Constellation')
    ax.set_ylabel('Tracking Power (mW)')
    ax.set_title('Power Consumption by GNSS Constellation')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, power in zip(bars, powers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{power:.1f}', ha='center', va='bottom')
    
    # 启动能量对比
    ax = axes[1, 1]
    gps_model = GPSPowerModel()
    start_modes = ['Cold Start', 'Warm Start', 'Hot Start']
    energies = [
        gps_model.calculate_ttff_energy(GNSSMode.COLD_START),
        gps_model.calculate_ttff_energy(GNSSMode.WARM_START),
        gps_model.calculate_ttff_energy(GNSSMode.HOT_START)
    ]
    colors = ['#f44336', '#FF9800', '#4CAF50']
    bars = ax.bar(start_modes, energies, color=colors)
    ax.set_xlabel('Start Mode')
    ax.set_ylabel('TTFF Energy (J)')
    ax.set_title('GPS Time-To-First-Fix Energy Consumption')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, energy in zip(bars, energies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{energy:.2f}J', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_background_analysis(results: dict, save_path: str = None):
    """绘制后台任务功耗分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_min = results['time'] / 60
    
    # 后台配置对比
    ax = axes[0, 0]
    ax.plot(t_min, np.array(results['minimal'])*1000, 
            label='Minimal (Power Saver)', linewidth=1.5, alpha=0.8)
    ax.plot(t_min, np.array(results['typical'])*1000, 
            label='Typical Usage', linewidth=1.5, alpha=0.8)
    ax.plot(t_min, np.array(results['heavy'])*1000, 
            label='Heavy Background', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Background Power: Different Activity Levels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)  # 显示前30分钟
    
    # 平均功耗对比
    ax = axes[0, 1]
    configs = ['Minimal', 'Typical', 'Heavy']
    avg_powers = [np.mean(results['minimal'])*1000,
                  np.mean(results['typical'])*1000,
                  np.mean(results['heavy'])*1000]
    colors = ['#4CAF50', '#2196F3', '#f44336']
    bars = ax.bar(configs, avg_powers, color=colors)
    ax.set_xlabel('Background Configuration')
    ax.set_ylabel('Average Power (mW)')
    ax.set_title('Average Background Power Consumption')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, power in zip(bars, avg_powers):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{power:.1f}', ha='center', va='bottom')
    
    # CPU状态功耗
    ax = axes[1, 0]
    states = list(results['cpu_states'].keys())
    state_powers = [results['cpu_states'][s]*1000 for s in states]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(states)))
    bars = ax.barh(states, state_powers, color=colors)
    ax.set_xlabel('Power (mW)')
    ax.set_ylabel('CPU State')
    ax.set_title('CPU Power by Operating State')
    ax.grid(True, alpha=0.3, axis='x')
    
    # DVFS功耗曲线
    ax = axes[1, 1]
    freqs_ghz = results['dvfs_curve']['frequencies'] / 1e9
    ax.plot(freqs_ghz, np.array(results['dvfs_curve']['power_50'])*1000,
            label='50% Load', linewidth=2)
    ax.plot(freqs_ghz, np.array(results['dvfs_curve']['power_100'])*1000,
            label='100% Load', linewidth=2)
    ax.fill_between(freqs_ghz,
                    np.array(results['dvfs_curve']['power_50'])*1000,
                    np.array(results['dvfs_curve']['power_100'])*1000,
                    alpha=0.3)
    ax.set_xlabel('CPU Frequency (GHz)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('DVFS Power Scaling Curve (P ~ V²f)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加理论曲线说明
    ax.annotate('Dynamic Power: P = C×V²×f\nVoltage scales with frequency',
                xy=(1.5, 150), fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_soc_simulation(results: dict, save_path: str = None):
    """绘制SOC仿真结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    t_hours = results['time'] / 3600
    
    # SOC曲线
    ax = axes[0, 0]
    ax.plot(t_hours, results['soc'] * 100, linewidth=2, color='#2196F3')
    ax.fill_between(t_hours, 0, results['soc'] * 100, alpha=0.3, color='#2196F3')
    ax.axhline(y=20, color='orange', linestyle='--', label='Low Battery Warning (20%)')
    ax.axhline(y=5, color='red', linestyle='--', label='Critical (5%)')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('State of Charge (%)')
    ax.set_title('Battery SOC Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_hours[-1])
    ax.set_ylim(0, 105)
    
    # 功耗分解
    ax = axes[0, 1]
    ax.stackplot(t_hours,
                 np.array(results['network_power'])*1000,
                 np.array(results['gps_power'])*1000,
                 np.array(results['background_power'])*1000,
                 labels=['Network', 'GPS', 'Background Tasks'],
                 colors=['#2196F3', '#4CAF50', '#FF9800'],
                 alpha=0.8)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Power Breakdown by Component')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_hours[-1])
    
    # 总功耗
    ax = axes[1, 0]
    ax.plot(t_hours, np.array(results['total_power'])*1000, 
            linewidth=1, alpha=0.8, color='#673AB7')
    # 添加移动平均
    window = 20
    if len(results['total_power']) > window:
        moving_avg = np.convolve(np.array(results['total_power'])*1000, 
                                 np.ones(window)/window, mode='valid')
        ax.plot(t_hours[window-1:], moving_avg, 
                linewidth=2, color='red', label='Moving Average')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (mW)')
    ax.set_title('Total Power Consumption Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_hours[-1])
    
    # 能量消耗饼图
    ax = axes[1, 1]
    # 计算各组件总能量
    dt = results['time'][1] - results['time'][0] if len(results['time']) > 1 else 1
    network_energy = np.sum(results['network_power']) * dt
    gps_energy = np.sum(results['gps_power']) * dt
    bg_energy = np.sum(results['background_power']) * dt
    base_energy = 0.1 * results['time'][-1]  # 基础功耗
    
    energies = [network_energy, gps_energy, bg_energy, base_energy]
    labels = ['Network', 'GPS', 'Background', 'Base System']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9E9E9E']
    
    wedges, texts, autotexts = ax.pie(energies, labels=labels, colors=colors,
                                       autopct='%1.1f%%', startangle=90)
    ax.set_title('Energy Distribution by Component')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_comprehensive_report():
    """生成综合分析报告"""
    print("="*60)
    print("智能手机电池功耗模型 - 综合分析报告")
    print("="*60)
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    analyzer = PowerAnalyzer()
    
    # 1. 网络分析
    print("\n[1/4] 分析网络功耗...")
    network_results = analyzer.analyze_network_power()
    fig1 = plot_network_analysis(network_results, 
                                  os.path.join(script_dir, 'network_power_analysis.png'))
    
    # 2. GPS分析
    print("[2/4] 分析GPS功耗...")
    gps_results = analyzer.analyze_gps_power()
    fig2 = plot_gps_analysis(gps_results,
                             os.path.join(script_dir, 'gps_power_analysis.png'))
    
    # 3. 后台任务分析
    print("[3/4] 分析后台任务功耗...")
    bg_results = analyzer.analyze_background_power()
    fig3 = plot_background_analysis(bg_results,
                                     os.path.join(script_dir, 'background_power_analysis.png'))
    
    # 4. 综合SOC仿真
    print("[4/4] 运行综合SOC仿真...")
    soc_results = analyzer.simulate_combined_soc(duration=14400)  # 4小时
    fig4 = plot_soc_simulation(soc_results,
                                os.path.join(script_dir, 'soc_simulation.png'))
    
    # 打印摘要
    print("\n" + "="*60)
    print("分析摘要")
    print("="*60)
    
    print("\n网络功耗 (平均值):")
    print(f"  WiFi空闲: {np.mean(network_results['wifi_idle'])*1000:.1f} mW")
    print(f"  WiFi浏览: {np.mean(network_results['wifi_browsing'])*1000:.1f} mW")
    print(f"  LTE空闲: {np.mean(network_results['lte_idle'])*1000:.1f} mW")
    print(f"  LTE流媒体: {np.mean(network_results['lte_streaming'])*1000:.1f} mW")
    print(f"  5G流媒体: {np.mean(network_results['5g_streaming'])*1000:.1f} mW")
    
    print("\nGPS功耗 (平均值):")
    print(f"  导航应用: {np.mean(gps_results['navigation'])*1000:.1f} mW")
    print(f"  运动追踪: {np.mean(gps_results['fitness'])*1000:.1f} mW")
    print(f"  后台定位: {np.mean(gps_results['background'])*1000:.1f} mW")
    
    print("\n后台任务功耗 (平均值):")
    print(f"  最小配置: {np.mean(bg_results['minimal'])*1000:.1f} mW")
    print(f"  典型配置: {np.mean(bg_results['typical'])*1000:.1f} mW")
    print(f"  重度配置: {np.mean(bg_results['heavy'])*1000:.1f} mW")
    
    print("\nSOC仿真结果:")
    print(f"  初始SOC: {soc_results['soc'][0]*100:.1f}%")
    print(f"  4小时后SOC: {soc_results['soc'][-1]*100:.1f}%")
    print(f"  平均功耗: {np.mean(soc_results['total_power'])*1000:.1f} mW")
    
    # 估算续航时间
    avg_power = np.mean(soc_results['total_power'])
    battery_capacity = 4000 * 3.85 / 1000  # Wh
    estimated_runtime = battery_capacity / avg_power if avg_power > 0 else float('inf')
    print(f"  估计续航时间: {estimated_runtime:.1f} 小时")
    
    print("\n" + "="*60)
    print(f"分析完成! 图表已保存到 {script_dir}")
    print("="*60)
    
    return {
        'network': network_results,
        'gps': gps_results,
        'background': bg_results,
        'soc': soc_results
    }


if __name__ == "__main__":
    results = generate_comprehensive_report()
    plt.show()
