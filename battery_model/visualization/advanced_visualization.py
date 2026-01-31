"""
高级可视化模块 - 现代化图表设计

采用更新颖的可视化技术:
1. 极坐标雷达图 - 多维度对比
2. 桑基图 - 能量流向
3. 热力图 - 参数敏感性
4. 瀑布图 - 功耗分解
5. 动态风格时间序列
6. 信息图表风格布局

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_battery_model import ContinuousTimeBatteryModel
from submodules.network_model import NetworkPowerModel, NetworkType, NetworkState, NetworkActivityProfile
from submodules.gps_model import GPSPowerModel, GNSSMode, GNSSConstellation, GPSUsageScenarios
from submodules.background_tasks_model import BackgroundPowerModel, CPUState, BackgroundScenarios

# 现代配色方案
COLORS = {
    'primary': '#6366F1',      # Indigo
    'secondary': '#8B5CF6',    # Purple
    'accent': '#EC4899',       # Pink
    'success': '#10B981',      # Emerald
    'warning': '#F59E0B',      # Amber
    'danger': '#EF4444',       # Red
    'info': '#3B82F6',         # Blue
    'dark': '#1F2937',         # Gray-800
    'light': '#F3F4F6',        # Gray-100
    'network': '#06B6D4',      # Cyan
    'gps': '#22C55E',          # Green
    'background': '#A855F7',   # Purple
    'gradient_start': '#667eea',
    'gradient_end': '#764ba2'
}

# 创建渐变色映射
def create_gradient_cmap(color1, color2, name='custom'):
    """创建双色渐变色映射"""
    from matplotlib.colors import hex2color
    c1 = hex2color(color1)
    c2 = hex2color(color2)
    return LinearSegmentedColormap.from_list(name, [c1, c2])


def set_modern_style():
    """设置现代化图表样式"""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': '#FAFAFA',
        'axes.facecolor': '#FFFFFF',
        'axes.edgecolor': '#E5E7EB',
        'axes.labelcolor': '#374151',
        'axes.titlecolor': '#111827',
        'axes.grid': True,
        'grid.color': '#F3F4F6',
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'text.color': '#374151',
        'xtick.color': '#6B7280',
        'ytick.color': '#6B7280',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'legend.frameon': True,
        'legend.facecolor': 'white',
        'legend.edgecolor': '#E5E7EB',
        'legend.fontsize': 9,
    })


def plot_radar_comparison(save_path=None):
    """
    雷达图 - 多维度功耗对比
    
    展示不同使用场景下各组件的功耗贡献
    """
    set_modern_style()
    
    fig = plt.figure(figsize=(14, 6))
    
    # 数据准备
    categories = ['Network\nActivity', 'GPS\nUsage', 'Background\nTasks', 
                  'Data\nTransfer', 'Signal\nQuality', 'Update\nFrequency']
    
    # 不同使用场景的归一化分数 (0-10)
    scenarios = {
        'Power Saver': [2, 1, 2, 1, 8, 2],
        'Normal Use': [5, 3, 5, 5, 6, 5],
        'Navigation': [6, 10, 4, 4, 7, 8],
        'Streaming': [9, 1, 3, 10, 5, 3],
        'Gaming': [8, 2, 6, 7, 6, 10]
    }
    
    colors = [COLORS['success'], COLORS['info'], COLORS['warning'], 
              COLORS['danger'], COLORS['accent']]
    
    # 创建雷达图
    ax1 = fig.add_subplot(121, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for (scenario, values), color in zip(scenarios.items(), colors):
        values_closed = values + values[:1]
        ax1.plot(angles, values_closed, 'o-', linewidth=2, label=scenario, color=color)
        ax1.fill(angles, values_closed, alpha=0.15, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=9)
    ax1.set_ylim(0, 10)
    ax1.set_yticks([2, 4, 6, 8, 10])
    ax1.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.set_title('Power Consumption Profile by Scenario', pad=20, fontsize=13, fontweight='bold')
    
    # 右侧: 堆叠条形图 - 功耗组成
    ax2 = fig.add_subplot(122)
    
    scenarios_list = list(scenarios.keys())
    network_power = [120, 250, 350, 800, 600]
    gps_power = [5, 30, 180, 10, 20]
    bg_power = [50, 80, 70, 60, 120]
    
    x = np.arange(len(scenarios_list))
    width = 0.6
    
    bars1 = ax2.bar(x, network_power, width, label='Network', color=COLORS['network'], alpha=0.9)
    bars2 = ax2.bar(x, gps_power, width, bottom=network_power, label='GPS', color=COLORS['gps'], alpha=0.9)
    bars3 = ax2.bar(x, bg_power, width, bottom=np.array(network_power)+np.array(gps_power), 
                    label='Background', color=COLORS['background'], alpha=0.9)
    
    ax2.set_ylabel('Power Consumption (mW)', fontweight='bold')
    ax2.set_xlabel('Usage Scenario', fontweight='bold')
    ax2.set_title('Power Breakdown by Component', pad=15, fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios_list, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    
    # 添加总功耗标签
    totals = np.array(network_power) + np.array(gps_power) + np.array(bg_power)
    for i, total in enumerate(totals):
        ax2.annotate(f'{total}mW', xy=(i, total + 20), ha='center', fontsize=9, fontweight='bold')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
        print(f"Saved: {save_path}")
    
    return fig


def plot_energy_flow_sankey(save_path=None):
    """
    桑基图风格 - 能量流向可视化
    
    展示电池能量如何分配到各个子系统
    """
    set_modern_style()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(5, 9.5, 'Battery Energy Flow Distribution', 
            fontsize=16, fontweight='bold', ha='center', color=COLORS['dark'])
    ax.text(5, 9.0, 'Continuous-Time Power Model Analysis', 
            fontsize=11, ha='center', color='#6B7280', style='italic')
    
    # 电池源 (左侧)
    battery_box = FancyBboxPatch((0.5, 3), 1.5, 4, 
                                  boxstyle="round,pad=0.1,rounding_size=0.3",
                                  facecolor=COLORS['primary'], edgecolor='white', linewidth=2)
    ax.add_patch(battery_box)
    ax.text(1.25, 5, 'BATTERY\n4000mAh\n15.4Wh', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='white')
    
    # 中间分流节点
    flow_colors = [COLORS['network'], COLORS['gps'], COLORS['background'], '#9CA3AF']
    flow_labels = ['Network\n35%', 'GPS\n15%', 'Background\n25%', 'Base System\n25%']
    flow_values = [35, 15, 25, 25]
    
    # 绘制流向箭头和目标框
    y_positions = [7.5, 5.5, 3.5, 1.5]
    
    for i, (y, color, label, value) in enumerate(zip(y_positions, flow_colors, flow_labels, flow_values)):
        # 流向线 (使用贝塞尔曲线效果)
        arrow_width = value / 15
        
        # 创建流向区域
        from matplotlib.patches import Polygon
        flow_points = [
            (2.2, 5 + arrow_width/2),
            (4.5, y + 0.8),
            (4.5, y - 0.8),
            (2.2, 5 - arrow_width/2)
        ]
        flow_poly = Polygon(flow_points, closed=True, facecolor=color, alpha=0.6, edgecolor='none')
        ax.add_patch(flow_poly)
        
        # 目标框
        target_box = FancyBboxPatch((4.8, y - 0.9), 2.2, 1.8,
                                     boxstyle="round,pad=0.05,rounding_size=0.2",
                                     facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(target_box)
        ax.text(5.9, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # 右侧详细分解
    detail_x = 7.5
    details = [
        ('WiFi: 15%\nLTE: 18%\n5G: 2%', COLORS['network'], 7.5),
        ('Navigation: 8%\nBackground: 5%\nA-GPS: 2%', COLORS['gps'], 5.5),
        ('Sync: 10%\nPush: 8%\nSystem: 7%', COLORS['background'], 3.5),
        ('CPU Idle: 15%\nMemory: 10%', '#9CA3AF', 1.5)
    ]
    
    for detail, color, y in details:
        # 连接线
        ax.annotate('', xy=(7.3, y), xytext=(7.0, y),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        
        detail_box = FancyBboxPatch((7.4, y - 0.7), 2.2, 1.4,
                                     boxstyle="round,pad=0.05,rounding_size=0.15",
                                     facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(detail_box)
        ax.text(8.5, y, detail, ha='center', va='center', fontsize=8, color=COLORS['dark'])
    
    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['network'], label='Network (WiFi/LTE/5G)'),
        mpatches.Patch(facecolor=COLORS['gps'], label='GPS/GNSS'),
        mpatches.Patch(facecolor=COLORS['background'], label='Background Tasks'),
        mpatches.Patch(facecolor='#9CA3AF', label='Base System')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=9)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
        print(f"Saved: {save_path}")
    
    return fig


def plot_heatmap_sensitivity(save_path=None):
    """
    热力图 - 参数敏感性分析
    
    展示不同参数组合对功耗的影响
    """
    set_modern_style()
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 网络: 信号强度 vs 数据速率 热力图
    ax1 = fig.add_subplot(gs[0, 0])
    
    signal_strengths = np.linspace(-90, -40, 20)  # dBm
    data_rates = np.linspace(0, 100, 20)  # Mbps
    
    network_model = NetworkPowerModel()
    power_matrix = np.zeros((len(data_rates), len(signal_strengths)))
    
    for i, rate in enumerate(data_rates):
        for j, signal in enumerate(signal_strengths):
            power_matrix[i, j] = network_model.lte_power(rate * 1e6, signal, NetworkState.ACTIVE) * 1000
    
    im1 = ax1.imshow(power_matrix, aspect='auto', cmap='YlOrRd', origin='lower',
                     extent=[-90, -40, 0, 100])
    ax1.set_xlabel('Signal Strength (dBm)', fontweight='bold')
    ax1.set_ylabel('Data Rate (Mbps)', fontweight='bold')
    ax1.set_title('LTE Power Consumption\nHeatmap', fontweight='bold', pad=10)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Power (mW)')
    
    # 添加等高线
    ax1.contour(signal_strengths, data_rates, power_matrix, colors='white', 
                alpha=0.5, linewidths=0.5, levels=5)
    
    # 2. GPS: 载噪比 vs 更新率 热力图
    ax2 = fig.add_subplot(gs[0, 1])
    
    cn0_values = np.linspace(20, 50, 20)  # dB-Hz
    update_rates = np.linspace(0.1, 5, 20)  # Hz
    
    gps_model = GPSPowerModel()
    gps_power_matrix = np.zeros((len(update_rates), len(cn0_values)))
    
    for i, rate in enumerate(update_rates):
        for j, cn0 in enumerate(cn0_values):
            gps_power_matrix[i, j] = gps_model.continuous_tracking_power(
                0, GNSSConstellation.MULTI_GNSS, cn0, rate) * 1000
    
    im2 = ax2.imshow(gps_power_matrix, aspect='auto', cmap='YlGn', origin='lower',
                     extent=[20, 50, 0.1, 5])
    ax2.set_xlabel('C/N₀ (dB-Hz)', fontweight='bold')
    ax2.set_ylabel('Update Rate (Hz)', fontweight='bold')
    ax2.set_title('GPS Power Consumption\nHeatmap', fontweight='bold', pad=10)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Power (mW)')
    ax2.contour(cn0_values, update_rates, gps_power_matrix, colors='white', 
                alpha=0.5, linewidths=0.5, levels=5)
    
    # 3. CPU: 频率 vs 负载 热力图
    ax3 = fig.add_subplot(gs[0, 2])
    
    frequencies = np.linspace(0.5, 2.8, 20)  # GHz
    loads = np.linspace(0, 1, 20)
    
    bg_model = BackgroundPowerModel()
    cpu_power_matrix = np.zeros((len(loads), len(frequencies)))
    
    for i, load in enumerate(loads):
        for j, freq in enumerate(frequencies):
            cpu_power_matrix[i, j] = bg_model.dvfs_power(freq * 1e9, load) * 1000
    
    im3 = ax3.imshow(cpu_power_matrix, aspect='auto', cmap='PuRd', origin='lower',
                     extent=[0.5, 2.8, 0, 100])
    ax3.set_xlabel('CPU Frequency (GHz)', fontweight='bold')
    ax3.set_ylabel('CPU Load (%)', fontweight='bold')
    ax3.set_title('CPU Power (DVFS)\nHeatmap', fontweight='bold', pad=10)
    cbar3 = plt.colorbar(im3, ax=ax3, label='Power (mW)')
    ax3.contour(frequencies, loads * 100, cpu_power_matrix, colors='white', 
                alpha=0.5, linewidths=0.5, levels=5)
    
    # 4. 综合敏感性条形图
    ax4 = fig.add_subplot(gs[1, :])
    
    parameters = ['Signal\nStrength\n(-90→-40 dBm)', 'Data\nRate\n(0→100 Mbps)', 
                  'GPS C/N₀\n(20→50 dB-Hz)', 'Update\nRate\n(0.1→5 Hz)',
                  'CPU Freq\n(0.5→2.8 GHz)', 'CPU\nLoad\n(0→100%)']
    
    # 计算敏感性 (参数变化范围内的功耗变化)
    sensitivities = [
        (power_matrix[-1, -1] - power_matrix[-1, 0]),  # 信号强度影响
        (power_matrix[-1, 10] - power_matrix[0, 10]),  # 数据率影响
        (gps_power_matrix[10, 0] - gps_power_matrix[10, -1]),  # CN0影响
        (gps_power_matrix[-1, 10] - gps_power_matrix[0, 10]),  # 更新率影响
        (cpu_power_matrix[10, -1] - cpu_power_matrix[10, 0]),  # CPU频率影响
        (cpu_power_matrix[-1, 10] - cpu_power_matrix[0, 10])   # CPU负载影响
    ]
    
    colors = [COLORS['network'], COLORS['network'], COLORS['gps'], 
              COLORS['gps'], COLORS['background'], COLORS['background']]
    
    bars = ax4.barh(parameters, sensitivities, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax4.set_xlabel('Power Sensitivity (mW change over parameter range)', fontweight='bold')
    ax4.set_title('Parameter Sensitivity Analysis', fontweight='bold', pad=15, fontsize=13)
    ax4.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    
    # 添加数值标签
    for bar, sensitivity in zip(bars, sensitivities):
        width = bar.get_width()
        ax4.text(width + 5, bar.get_y() + bar.get_height()/2, 
                 f'{abs(sensitivity):.0f}mW', va='center', fontsize=9, fontweight='bold')
    
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=COLORS['network'], label='Network Parameters'),
        mpatches.Patch(color=COLORS['gps'], label='GPS Parameters'),
        mpatches.Patch(color=COLORS['background'], label='CPU Parameters')
    ]
    ax4.legend(handles=legend_patches, loc='lower right')
    
    plt.suptitle('Power Consumption Sensitivity Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
        print(f"Saved: {save_path}")
    
    return fig


def plot_dynamic_timeline(save_path=None):
    """
    动态时间线 - 展示功耗随时间的变化
    
    使用现代化的时间序列可视化
    """
    set_modern_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1.5, 1.5], hspace=0.25)
    
    # 模拟数据
    duration = 3600  # 1小时
    t = np.linspace(0, duration, 500)
    t_min = t / 60
    
    network_model = NetworkPowerModel()
    gps_model = GPSPowerModel()
    bg_model = BackgroundPowerModel()
    for task in BackgroundScenarios.typical_background():
        bg_model.add_task(task)
    
    # 计算各组件功耗
    network_power = []
    gps_power = []
    bg_power = []
    
    for ti in t:
        # 网络: 混合场景
        if (ti % 300) < 30:  # 每5分钟有30秒活跃
            net_p = network_model.wifi_power(10e6, -60, True)
        else:
            net_p = network_model.wifi_power(0, -60, False)
        network_power.append(net_p * 1000)
        
        # GPS: 偶尔使用
        gps_params = GPSUsageScenarios.background_location(ti)
        gps_p = gps_model.get_power(gps_params[0], GNSSConstellation.GPS_ONLY, 
                                     gps_params[1], ti, gps_params[2])
        gps_power.append(gps_p * 1000)
        
        # 后台
        bg_p = bg_model.total_background_power(ti)
        bg_power.append(bg_p * 1000)
    
    network_power = np.array(network_power)
    gps_power = np.array(gps_power)
    bg_power = np.array(bg_power)
    total_power = network_power + gps_power + bg_power + 100  # +100mW基础功耗
    
    # 1. 主图: 堆叠面积图
    ax1 = fig.add_subplot(gs[0])
    
    ax1.fill_between(t_min, 0, network_power, alpha=0.7, color=COLORS['network'], label='Network')
    ax1.fill_between(t_min, network_power, network_power + gps_power, alpha=0.7, 
                     color=COLORS['gps'], label='GPS')
    ax1.fill_between(t_min, network_power + gps_power, network_power + gps_power + bg_power, 
                     alpha=0.7, color=COLORS['background'], label='Background')
    ax1.fill_between(t_min, network_power + gps_power + bg_power, total_power, 
                     alpha=0.5, color='#9CA3AF', label='Base System')
    
    # 添加总功耗线
    ax1.plot(t_min, total_power, color=COLORS['dark'], linewidth=1.5, linestyle='--', 
             label='Total Power', alpha=0.8)
    
    ax1.set_ylabel('Power (mW)', fontweight='bold', fontsize=11)
    ax1.set_title('Real-Time Power Consumption Analysis', fontweight='bold', fontsize=14, pad=15)
    ax1.legend(loc='upper right', ncol=5, fontsize=9)
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, max(total_power) * 1.1)
    
    # 添加事件标记
    events = [
        (5, 'Network\nBurst', COLORS['network']),
        (10, 'GPS\nUpdate', COLORS['gps']),
        (15, 'Email\nSync', COLORS['background']),
        (25, 'Network\nBurst', COLORS['network']),
        (35, 'GPS\nUpdate', COLORS['gps']),
    ]
    
    for time, label, color in events:
        ax1.axvline(x=time, color=color, linestyle=':', alpha=0.5, linewidth=1.5)
        ax1.annotate(label, xy=(time, max(total_power) * 0.9), fontsize=7, 
                     ha='center', color=color, fontweight='bold')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. 网络活动详细视图
    ax2 = fig.add_subplot(gs[1])
    
    # 绘制脉冲式网络活动
    ax2.fill_between(t_min, 0, network_power, alpha=0.3, color=COLORS['network'])
    ax2.plot(t_min, network_power, color=COLORS['network'], linewidth=1.5)
    
    # 添加状态标注
    ax2.axhspan(0, 50, alpha=0.1, color='green', label='Idle')
    ax2.axhspan(50, 500, alpha=0.1, color='yellow', label='DRX')
    ax2.axhspan(500, 1500, alpha=0.1, color='red', label='Active')
    
    ax2.set_ylabel('Network Power (mW)', fontweight='bold')
    ax2.set_title('Network Activity Pattern', fontweight='bold', fontsize=12)
    ax2.set_xlim(0, 60)
    ax2.legend(loc='upper right', ncol=3, fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. SOC变化
    ax3 = fig.add_subplot(gs[2])
    
    # 简化的SOC计算
    battery_capacity = 4000 * 3.85  # mWh
    cumulative_energy = np.cumsum(total_power) * (t[1] - t[0]) / 3600  # mWh
    soc = 100 * (1 - cumulative_energy / battery_capacity)
    
    # 渐变填充效果
    gradient_cmap = create_gradient_cmap('#22C55E', '#EF4444')
    for i in range(len(t_min) - 1):
        ax3.fill_between(t_min[i:i+2], 0, soc[i:i+2], 
                         color=gradient_cmap(1 - soc[i]/100), alpha=0.8)
    
    ax3.plot(t_min, soc, color=COLORS['dark'], linewidth=2)
    
    # 添加SOC区域标注
    ax3.axhline(y=20, color=COLORS['warning'], linestyle='--', linewidth=1.5, label='Low Battery')
    ax3.axhline(y=5, color=COLORS['danger'], linestyle='--', linewidth=1.5, label='Critical')
    
    ax3.set_xlabel('Time (minutes)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Battery SOC (%)', fontweight='bold')
    ax3.set_title('Battery State of Charge', fontweight='bold', fontsize=12)
    ax3.set_xlim(0, 60)
    ax3.set_ylim(0, 105)
    ax3.legend(loc='lower left', fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 添加剩余时间估算
    avg_power = np.mean(total_power)
    remaining_hours = (soc[-1] / 100 * battery_capacity) / avg_power
    ax3.text(55, soc[-1] + 5, f'Est. Remaining:\n{remaining_hours:.1f}h', 
             fontsize=9, ha='center', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['info']))
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
        print(f"Saved: {save_path}")
    
    return fig


def plot_infographic_summary(save_path=None):
    """
    信息图表风格 - 综合摘要
    
    使用图标和简洁设计展示关键数据
    """
    set_modern_style()
    
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#F8FAFC')
    
    # 主标题
    fig.text(0.5, 0.96, 'Smartphone Battery Power Model', fontsize=20, 
             fontweight='bold', ha='center', color=COLORS['dark'])
    fig.text(0.5, 0.93, 'Continuous-Time Mathematical Model for Network, GPS & Background Tasks', 
             fontsize=12, ha='center', color='#6B7280', style='italic')
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3,
                  top=0.88, bottom=0.08, left=0.05, right=0.95)
    
    # ============ 第一行: 关键指标卡片 ============
    metrics = [
        ('NET', 'Network', '14.5 - 1309 mW', 'WiFi to 5G range', COLORS['network']),
        ('GPS', 'GPS', '30 - 180 mW', 'Single to Multi-GNSS', COLORS['gps']),
        ('BG', 'Background', '75 - 85 mW', 'Minimal to Heavy', COLORS['background']),
        ('BAT', 'Battery', '4000 mAh', '15.4 Wh capacity', COLORS['primary'])
    ]
    
    for i, (icon, title, value, subtitle, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 卡片背景
        card = FancyBboxPatch((0.5, 0.5), 9, 9,
                               boxstyle="round,pad=0.1,rounding_size=0.5",
                               facecolor='white', edgecolor=color, linewidth=3)
        ax.add_patch(card)
        
        # 顶部色条
        top_bar = FancyBboxPatch((0.5, 8), 9, 1.5,
                                  boxstyle="round,pad=0,rounding_size=0.5",
                                  facecolor=color, edgecolor='none')
        ax.add_patch(top_bar)
        
        ax.text(5, 8.7, icon, fontsize=20, ha='center', va='center')
        ax.text(5, 6.5, title, fontsize=12, fontweight='bold', ha='center', color=COLORS['dark'])
        ax.text(5, 4.5, value, fontsize=14, fontweight='bold', ha='center', color=color)
        ax.text(5, 2.5, subtitle, fontsize=9, ha='center', color='#6B7280')
    
    # ============ 第二行: 功耗对比图 ============
    
    # 左: 网络技术对比
    ax_net = fig.add_subplot(gs[1, :2])
    
    technologies = ['WiFi\nIdle', 'WiFi\nActive', 'LTE\nIdle', 'LTE\nActive', '5G\nActive']
    powers = [14.5, 450, 78, 1500, 2500]
    colors_net = [COLORS['network']] * 5
    alphas = [0.4, 0.7, 0.4, 0.7, 0.9]
    
    bars = ax_net.barh(technologies, powers, color=colors_net, alpha=0.8, 
                       edgecolor='white', linewidth=2)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)
    
    ax_net.set_xlabel('Power Consumption (mW)', fontweight='bold')
    ax_net.set_title('Network Technology Comparison', fontweight='bold', fontsize=12, pad=10)
    ax_net.spines['top'].set_visible(False)
    ax_net.spines['right'].set_visible(False)
    
    # 添加数值
    for i, (bar, power) in enumerate(zip(bars, powers)):
        ax_net.text(power + 50, i, f'{power}mW', va='center', fontsize=9, fontweight='bold')
    
    # 右: GPS模式对比
    ax_gps = fig.add_subplot(gs[1, 2:])
    
    gps_modes = ['Off', 'Standby', 'GPS Only\nTracking', 'Dual\nConstellation', 
                 'Multi-GNSS\nTracking', 'Navigation\nApp']
    gps_powers = [0, 0.5, 30, 45, 65, 180]
    
    # 使用气泡图风格
    x_pos = np.arange(len(gps_modes))
    sizes = np.array(gps_powers) * 3 + 50
    
    scatter = ax_gps.scatter(x_pos, gps_powers, s=sizes, c=COLORS['gps'], 
                              alpha=0.7, edgecolors='white', linewidths=2)
    ax_gps.plot(x_pos, gps_powers, color=COLORS['gps'], linewidth=2, alpha=0.5)
    
    ax_gps.set_xticks(x_pos)
    ax_gps.set_xticklabels(gps_modes, fontsize=9)
    ax_gps.set_ylabel('Power (mW)', fontweight='bold')
    ax_gps.set_title('GPS Power by Operating Mode', fontweight='bold', fontsize=12, pad=10)
    ax_gps.spines['top'].set_visible(False)
    ax_gps.spines['right'].set_visible(False)
    
    # 添加标签
    for x, y in zip(x_pos, gps_powers):
        if y > 0:
            ax_gps.annotate(f'{y}mW', (x, y + 10), ha='center', fontsize=8, fontweight='bold')
    
    # ============ 第三行: 数学模型核心方程 ============
    ax_eq = fig.add_subplot(gs[2, :])
    ax_eq.axis('off')
    
    # 方程卡片背景
    eq_card = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor='white', edgecolor=COLORS['primary'], 
                              linewidth=2, transform=ax_eq.transAxes)
    ax_eq.add_patch(eq_card)
    
    ax_eq.text(0.5, 0.85, 'Core Mathematical Equations', fontsize=14, 
               fontweight='bold', ha='center', transform=ax_eq.transAxes, color=COLORS['dark'])
    
    equations = [
        ('SOC Dynamics:', r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}(T)} - k_{sd} \cdot SOC$', COLORS['primary']),
        ('Network Power:', r'$P_{net} = P_{base} + P_{PA} \cdot 10^{\frac{\Delta RSSI \cdot k}{10}}$', COLORS['network']),
        ('GPS Power:', r'$P_{GPS} = P_{RF} + P_{corr} + P_{baseband}(rate)$', COLORS['gps']),
        ('CPU Power (DVFS):', r'$P_{CPU} = \alpha \cdot C \cdot V^2 \cdot f \propto f^{2.5}$', COLORS['background'])
    ]
    
    for i, (label, eq, color) in enumerate(equations):
        y_pos = 0.65 - i * 0.18
        ax_eq.text(0.08, y_pos, label, fontsize=10, fontweight='bold', 
                   transform=ax_eq.transAxes, color=color)
        ax_eq.text(0.35, y_pos, eq, fontsize=12, transform=ax_eq.transAxes, 
                   color=COLORS['dark'])
    
    # 底部参考文献提示
    fig.text(0.5, 0.02, 'Based on: Carroll & Heiser (2010), Huang et al. (2012), Pathak et al. (2012), 3GPP Specifications',
             fontsize=8, ha='center', color='#9CA3AF', style='italic')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#F8FAFC')
        print(f"Saved: {save_path}")
    
    return fig


def plot_3d_power_surface(save_path=None):
    """
    3D曲面图 - 功耗参数空间可视化
    """
    from mpl_toolkits.mplot3d import Axes3D
    set_modern_style()
    
    fig = plt.figure(figsize=(16, 6))
    
    # 1. 网络功耗3D曲面
    ax1 = fig.add_subplot(131, projection='3d')
    
    signal = np.linspace(-90, -40, 30)
    rate = np.linspace(0, 100, 30)
    SIGNAL, RATE = np.meshgrid(signal, rate)
    
    network_model = NetworkPowerModel()
    POWER = np.zeros_like(SIGNAL)
    for i in range(len(rate)):
        for j in range(len(signal)):
            POWER[i, j] = network_model.lte_power(rate[i] * 1e6, signal[j], NetworkState.ACTIVE) * 1000
    
    surf1 = ax1.plot_surface(SIGNAL, RATE, POWER, cmap='coolwarm', alpha=0.8,
                              linewidth=0, antialiased=True)
    ax1.set_xlabel('Signal (dBm)', fontsize=9)
    ax1.set_ylabel('Data Rate (Mbps)', fontsize=9)
    ax1.set_zlabel('Power (mW)', fontsize=9)
    ax1.set_title('LTE Power Surface', fontweight='bold', pad=10)
    ax1.view_init(elev=25, azim=45)
    
    # 2. GPS功耗3D曲面
    ax2 = fig.add_subplot(132, projection='3d')
    
    cn0 = np.linspace(20, 50, 30)
    update = np.linspace(0.1, 5, 30)
    CN0, UPDATE = np.meshgrid(cn0, update)
    
    gps_model = GPSPowerModel()
    GPS_POWER = np.zeros_like(CN0)
    for i in range(len(update)):
        for j in range(len(cn0)):
            GPS_POWER[i, j] = gps_model.continuous_tracking_power(
                0, GNSSConstellation.MULTI_GNSS, cn0[j], update[i]) * 1000
    
    surf2 = ax2.plot_surface(CN0, UPDATE, GPS_POWER, cmap='Greens', alpha=0.8,
                              linewidth=0, antialiased=True)
    ax2.set_xlabel('C/N₀ (dB-Hz)', fontsize=9)
    ax2.set_ylabel('Update Rate (Hz)', fontsize=9)
    ax2.set_zlabel('Power (mW)', fontsize=9)
    ax2.set_title('GPS Power Surface', fontweight='bold', pad=10)
    ax2.view_init(elev=25, azim=45)
    
    # 3. CPU功耗3D曲面
    ax3 = fig.add_subplot(133, projection='3d')
    
    freq = np.linspace(0.5, 2.8, 30)
    load = np.linspace(0, 1, 30)
    FREQ, LOAD = np.meshgrid(freq, load)
    
    bg_model = BackgroundPowerModel()
    CPU_POWER = np.zeros_like(FREQ)
    for i in range(len(load)):
        for j in range(len(freq)):
            CPU_POWER[i, j] = bg_model.dvfs_power(freq[j] * 1e9, load[i]) * 1000
    
    surf3 = ax3.plot_surface(FREQ, LOAD * 100, CPU_POWER, cmap='Purples', alpha=0.8,
                              linewidth=0, antialiased=True)
    ax3.set_xlabel('Frequency (GHz)', fontsize=9)
    ax3.set_ylabel('Load (%)', fontsize=9)
    ax3.set_zlabel('Power (mW)', fontsize=9)
    ax3.set_title('CPU Power Surface', fontweight='bold', pad=10)
    ax3.view_init(elev=25, azim=45)
    
    plt.suptitle('3D Power Consumption Surfaces', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#FAFAFA')
        print(f"Saved: {save_path}")
    
    return fig


def generate_advanced_visualizations():
    """生成所有高级可视化图表"""
    print("="*60)
    print("生成高级可视化图表...")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/5] 雷达对比图...")
    plot_radar_comparison(os.path.join(script_dir, 'advanced_radar_comparison.png'))
    
    print("[2/5] 能量流向图...")
    plot_energy_flow_sankey(os.path.join(script_dir, 'advanced_energy_flow.png'))
    
    print("[3/5] 参数敏感性热力图...")
    plot_heatmap_sensitivity(os.path.join(script_dir, 'advanced_heatmap_sensitivity.png'))
    
    print("[4/5] 动态时间线...")
    plot_dynamic_timeline(os.path.join(script_dir, 'advanced_dynamic_timeline.png'))
    
    print("[5/5] 信息图表摘要...")
    plot_infographic_summary(os.path.join(script_dir, 'advanced_infographic_summary.png'))
    
    print("\n[Bonus] 3D功耗曲面...")
    plot_3d_power_surface(os.path.join(script_dir, 'advanced_3d_surface.png'))
    
    print("\n" + "="*60)
    print("高级可视化图表生成完成!")
    print("="*60)


if __name__ == "__main__":
    generate_advanced_visualizations()
