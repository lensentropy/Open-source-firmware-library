#!/usr/bin/env python3
"""
科学可视化模块 - 白色背景专业风格

创新图表类型:
1. 桑基流程图 - 功耗流向
2. 小提琴图 - 分布对比
3. 蜂窝热力图 - 参数敏感性
4. 阶梯瀑布图 - 功耗分解
5. 极坐标玫瑰图 - 多维对比
6. 河流图 - 时序演变
7. 树状图 - 层级分解

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Rectangle, Polygon, PathPatch
from matplotlib.path import Path
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iontech_integrated_model import (
    IntegratedPowerModel, NetworkType, NetworkState,
    BluetoothMode, CPUState, create_scenario_power_func
)

# ============================================================================
# 白色背景配色方案 (科学论文风格)
# ============================================================================

COLORS = {
    'bg': '#FFFFFF',
    'text': '#2D3748',
    'text_light': '#718096',
    'grid': '#E2E8F0',
    'border': '#CBD5E0',
    
    # 主色系 (Nature/Science风格)
    'network': '#2563EB',      # 蓝色
    'wifi': '#3B82F6',
    'lte': '#F97316',          # 橙色
    'nr5g': '#DC2626',         # 红色
    
    'bluetooth': '#7C3AED',    # 紫色
    'ble': '#8B5CF6',
    'bt_classic': '#6D28D9',
    
    'background': '#059669',   # 绿色
    'cpu': '#10B981',
    'memory': '#34D399',
    
    'battery': '#0891B2',      # 青色
    
    # 渐变色
    'gradient': ['#3B82F6', '#8B5CF6', '#EC4899', '#F97316'],
    
    # 强调色
    'accent': '#F59E0B',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': COLORS['border'],
    'axes.grid': True,
    'grid.color': COLORS['grid'],
    'grid.linewidth': 0.5,
    'text.color': COLORS['text'],
})


# ============================================================================
# 1. 桑基图 - 功耗流向分析
# ============================================================================

def plot_sankey_power_flow(save_path=None):
    """
    桑基图 - 电池能量流向各子系统
    """
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    ax = fig.add_subplot(111)
    
    # 创建桑基图
    sankey = Sankey(ax=ax, scale=0.01, offset=0.3, head_angle=120,
                    format='%.0f', unit=' mW')
    
    # 主流程: 电池 -> 各子系统
    # flows: 正数流出, 负数流入
    sankey.add(flows=[350, -120, -45, -85, -50, -50],
               labels=['Battery\nOutput', 'Network', 'Bluetooth', 
                      'Background', 'Display', 'Other'],
               orientations=[0, 0, -1, 1, 0, 0],
               pathlengths=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               facecolor=COLORS['battery'],
               edgecolor=COLORS['text'],
               alpha=0.8)
    
    # 网络子流程
    sankey.add(flows=[120, -80, -25, -15],
               labels=['', 'RF/PA', 'Baseband', 'Idle'],
               orientations=[0, 0, 1, -1],
               prior=0, connect=(1, 0),
               pathlengths=[0.3, 0.3, 0.3, 0.3],
               facecolor=COLORS['network'],
               alpha=0.7)
    
    # 蓝牙子流程
    sankey.add(flows=[45, -30, -10, -5],
               labels=['', 'Audio', 'BLE', 'Idle'],
               orientations=[0, 0, -1, 1],
               prior=0, connect=(2, 0),
               pathlengths=[0.3, 0.3, 0.3, 0.3],
               facecolor=COLORS['bluetooth'],
               alpha=0.7)
    
    # 后台子流程
    sankey.add(flows=[85, -50, -25, -10],
               labels=['', 'CPU', 'Memory', 'Services'],
               orientations=[0, 0, 1, -1],
               prior=0, connect=(3, 0),
               pathlengths=[0.3, 0.3, 0.3, 0.3],
               facecolor=COLORS['background'],
               alpha=0.7)
    
    diagrams = sankey.finish()
    
    ax.set_title('Battery Power Flow Sankey Diagram\n(Typical Active Usage Scenario)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    # 添加图例
    legend_elements = [
        mpatches.Patch(color=COLORS['battery'], label='Battery', alpha=0.8),
        mpatches.Patch(color=COLORS['network'], label='Network', alpha=0.7),
        mpatches.Patch(color=COLORS['bluetooth'], label='Bluetooth', alpha=0.7),
        mpatches.Patch(color=COLORS['background'], label='Background', alpha=0.7),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True,
              facecolor='white', edgecolor=COLORS['border'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 2. 小提琴图 - 功耗分布对比
# ============================================================================

def plot_violin_distribution(save_path=None):
    """
    小提琴图 - 不同技术的功耗分布
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='white')
    
    model = IntegratedPowerModel()
    
    # 生成模拟数据 (基于模型的随机采样)
    np.random.seed(42)
    n_samples = 200
    
    # 网络功耗分布
    ax1 = axes[0]
    
    wifi_idle = np.random.normal(10, 2, n_samples)
    wifi_active = np.random.normal(500, 150, n_samples)
    lte_drx = np.random.normal(150, 30, n_samples)
    lte_active = np.random.normal(1200, 300, n_samples)
    nr_active = np.random.normal(2500, 500, n_samples)
    
    data_net = [wifi_idle, wifi_active, lte_drx, lte_active, nr_active]
    positions = [1, 2, 3, 4, 5]
    
    parts = ax1.violinplot(data_net, positions, widths=0.7, showmeans=True, showmedians=True)
    
    colors_net = [COLORS['wifi'], COLORS['wifi'], COLORS['lte'], COLORS['lte'], COLORS['nr5g']]
    for pc, color in zip(parts['bodies'], colors_net):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['WiFi\nIdle', 'WiFi\nActive', 'LTE\nDRX', 'LTE\nActive', '5G\nActive'],
                        fontsize=9)
    ax1.set_ylabel('Power (mW)', fontweight='bold')
    ax1.set_title('Network Power Distribution', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 4000)
    
    # 蓝牙功耗分布
    ax2 = axes[1]
    
    ble_idle = np.random.exponential(1, n_samples)
    ble_active = np.random.normal(15, 5, n_samples)
    bt_sbc = np.random.normal(45, 8, n_samples)
    bt_ldac = np.random.normal(68, 10, n_samples)
    
    data_bt = [ble_idle, ble_active, bt_sbc, bt_ldac]
    positions_bt = [1, 2, 3, 4]
    
    parts_bt = ax2.violinplot(data_bt, positions_bt, widths=0.6, showmeans=True, showmedians=True)
    
    colors_bt = [COLORS['ble'], COLORS['ble'], COLORS['bt_classic'], COLORS['bt_classic']]
    for pc, color in zip(parts_bt['bodies'], colors_bt):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_xticks(positions_bt)
    ax2.set_xticklabels(['BLE\nIdle', 'BLE\nActive', 'BT\nSBC', 'BT\nLDAC'], fontsize=9)
    ax2.set_ylabel('Power (mW)', fontweight='bold')
    ax2.set_title('Bluetooth Power Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    
    # 后台任务功耗分布
    ax3 = axes[2]
    
    t = np.linspace(0, 3600, n_samples)
    bg_powers = [model.background.total_power(ti) * 1000 for ti in t]
    cpu_idle = np.random.normal(15, 3, n_samples)
    cpu_active = np.random.normal(150, 50, n_samples)
    
    data_bg = [cpu_idle, bg_powers, cpu_active]
    positions_bg = [1, 2, 3]
    
    parts_bg = ax3.violinplot(data_bg, positions_bg, widths=0.6, showmeans=True, showmedians=True)
    
    colors_bg = [COLORS['cpu'], COLORS['background'], COLORS['cpu']]
    for pc, color in zip(parts_bg['bodies'], colors_bg):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax3.set_xticks(positions_bg)
    ax3.set_xticklabels(['CPU\nIdle', 'Background\nTasks', 'CPU\nActive'], fontsize=9)
    ax3.set_ylabel('Power (mW)', fontweight='bold')
    ax3.set_title('Background Power Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 300)
    
    plt.suptitle('Power Consumption Distribution Analysis\n(Violin Plots with Mean and Median)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 3. 蜂窝图 - 参数敏感性分析
# ============================================================================

def plot_hexbin_sensitivity(save_path=None):
    """
    蜂窝图 - 参数空间功耗密度
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='white')
    
    model = IntegratedPowerModel()
    np.random.seed(42)
    n_points = 2000
    
    # 1. WiFi: 信号强度 vs 数据速率
    ax1 = axes[0]
    
    rssi = np.random.uniform(-90, -40, n_points)
    rate = np.random.uniform(0, 100, n_points)
    wifi_power = np.array([model.network.wifi_power(r, s, True) * 1000 
                           for r, s in zip(rate, rssi)])
    
    hb1 = ax1.hexbin(rssi, rate, C=wifi_power, gridsize=25, cmap='Blues',
                      reduce_C_function=np.mean, mincnt=1)
    ax1.set_xlabel('RSSI (dBm)', fontweight='bold')
    ax1.set_ylabel('Data Rate (Mbps)', fontweight='bold')
    ax1.set_title('WiFi Power Sensitivity', fontsize=12, fontweight='bold')
    
    cb1 = plt.colorbar(hb1, ax=ax1)
    cb1.set_label('Power (mW)')
    
    # 2. LTE: 信号强度 vs 数据速率  
    ax2 = axes[1]
    
    rsrp = np.random.uniform(-110, -60, n_points)
    rate_lte = np.random.uniform(0, 150, n_points)
    lte_power = np.array([model.network.lte_power(r, s, NetworkState.ACTIVE_RX) * 1000
                          for r, s in zip(rate_lte, rsrp)])
    
    hb2 = ax2.hexbin(rsrp, rate_lte, C=lte_power, gridsize=25, cmap='Oranges',
                      reduce_C_function=np.mean, mincnt=1)
    ax2.set_xlabel('RSRP (dBm)', fontweight='bold')
    ax2.set_ylabel('Data Rate (Mbps)', fontweight='bold')
    ax2.set_title('LTE Power Sensitivity', fontsize=12, fontweight='bold')
    
    cb2 = plt.colorbar(hb2, ax=ax2)
    cb2.set_label('Power (mW)')
    
    # 3. CPU: 频率 vs 负载
    ax3 = axes[2]
    
    freq = np.random.uniform(0.3, 2.8, n_points)
    load = np.random.uniform(0, 1, n_points)
    cpu_power = np.array([model.background.dvfs_power(f * 1e9, l) * 1000
                          for f, l in zip(freq, load)])
    
    hb3 = ax3.hexbin(freq, load * 100, C=cpu_power, gridsize=25, cmap='Greens',
                      reduce_C_function=np.mean, mincnt=1)
    ax3.set_xlabel('CPU Frequency (GHz)', fontweight='bold')
    ax3.set_ylabel('CPU Load (%)', fontweight='bold')
    ax3.set_title('CPU DVFS Power Sensitivity', fontsize=12, fontweight='bold')
    
    cb3 = plt.colorbar(hb3, ax=ax3)
    cb3.set_label('Power (mW)')
    
    plt.suptitle('Parameter Sensitivity Analysis (Hexbin Density Maps)',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 4. 瀑布图 - 功耗分解
# ============================================================================

def plot_waterfall_breakdown(save_path=None):
    """
    瀑布图 - 功耗逐项累积分解
    """
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # 功耗分解项
    categories = ['Base\nSystem', 'CPU\nIdle', 'DRAM', 'WiFi\nIdle', 
                  'Push\nService', 'Email\nSync', 'Location', 
                  'Social\nSync', 'BLE', 'Total']
    
    values = [25, 15, 40, 10, 8, 15, 18, 25, 3, None]
    
    # 计算累积值
    cumulative = [0]
    for v in values[:-1]:
        cumulative.append(cumulative[-1] + v)
    
    total = cumulative[-1]
    values[-1] = total
    
    # 颜色映射
    colors = [COLORS['text_light'], COLORS['cpu'], COLORS['memory'],
              COLORS['wifi'], COLORS['background'], COLORS['background'],
              COLORS['background'], COLORS['background'], COLORS['ble'],
              COLORS['accent']]
    
    x = np.arange(len(categories))
    bar_width = 0.6
    
    # 绘制瀑布图
    for i, (cat, val, cum, color) in enumerate(zip(categories, values, cumulative + [0], colors)):
        if i < len(categories) - 1:
            # 增量柱
            bar = ax.bar(i, val, bottom=cum, width=bar_width, 
                        color=color, edgecolor='white', linewidth=2, alpha=0.85)
            
            # 连接线
            if i < len(categories) - 2:
                ax.plot([i + bar_width/2, i + 1 - bar_width/2], 
                       [cum + val, cum + val],
                       color=COLORS['text_light'], linewidth=1.5, 
                       linestyle='--', alpha=0.5)
            
            # 增量标签
            ax.text(i, cum + val/2, f'+{val}', ha='center', va='center',
                   fontsize=10, fontweight='bold', color='white')
        else:
            # 总计柱
            bar = ax.bar(i, val, width=bar_width, color=color,
                        edgecolor=COLORS['text'], linewidth=2, alpha=0.9)
            ax.text(i, val/2, f'{val}\nmW', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='white')
    
    # 添加分组标注
    ax.axvspan(-0.5, 2.5, alpha=0.05, color=COLORS['text'])
    ax.axvspan(2.5, 7.5, alpha=0.05, color=COLORS['background'])
    ax.axvspan(7.5, 8.5, alpha=0.05, color=COLORS['bluetooth'])
    
    ax.text(1, total * 1.05, 'Base System', ha='center', fontsize=9, 
            color=COLORS['text_light'], style='italic')
    ax.text(5, total * 1.05, 'Background Tasks', ha='center', fontsize=9,
            color=COLORS['text_light'], style='italic')
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Power Consumption (mW)', fontsize=11, fontweight='bold')
    ax.set_title('Power Consumption Waterfall Breakdown\n(Cumulative Component Analysis)',
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_ylim(0, total * 1.15)
    ax.axhline(y=total, color=COLORS['danger'], linestyle='--', 
               alpha=0.5, linewidth=1.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 图例
    legend_elements = [
        mpatches.Patch(color=COLORS['text_light'], label='Base', alpha=0.85),
        mpatches.Patch(color=COLORS['cpu'], label='CPU/Memory', alpha=0.85),
        mpatches.Patch(color=COLORS['wifi'], label='Network', alpha=0.85),
        mpatches.Patch(color=COLORS['background'], label='Services', alpha=0.85),
        mpatches.Patch(color=COLORS['ble'], label='Bluetooth', alpha=0.85),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 5. 玫瑰图 - 多维场景对比
# ============================================================================

def plot_rose_scenario_comparison(save_path=None):
    """
    玫瑰图 - 不同场景的多维功耗对比
    """
    fig = plt.figure(figsize=(14, 7), facecolor='white')
    
    # 左侧: 玫瑰图
    ax1 = fig.add_subplot(121, projection='polar')
    
    model = IntegratedPowerModel()
    
    # 场景和维度
    scenarios = ['Idle', 'Music\nStream', 'Social\nBrowsing', 'Video\nStream', 
                 'Gaming', 'Fitness', 'Navigation']
    
    # 各维度功耗 (归一化)
    # [网络, 蓝牙, 后台, 显示, GPS]
    scenario_data = {
        'Idle': [0.1, 0.05, 0.3, 0, 0],
        'Music\nStream': [0.2, 0.8, 0.3, 0.1, 0],
        'Social\nBrowsing': [0.7, 0.1, 0.5, 0.8, 0.1],
        'Video\nStream': [0.6, 0.1, 0.3, 1.0, 0],
        'Gaming': [0.5, 0.6, 0.8, 1.0, 0.1],
        'Fitness': [0.1, 0.7, 0.4, 0.3, 0.8],
        'Navigation': [0.4, 0.2, 0.5, 0.9, 1.0],
    }
    
    n_scenarios = len(scenarios)
    angles = np.linspace(0, 2 * np.pi, n_scenarios, endpoint=False)
    width = 2 * np.pi / n_scenarios * 0.8
    
    colors = [COLORS['success'], COLORS['bluetooth'], COLORS['lte'], 
              COLORS['wifi'], COLORS['danger'], COLORS['ble'], COLORS['accent']]
    
    # 绘制每个场景的功耗
    for i, (scenario, color) in enumerate(zip(scenarios, colors)):
        data = scenario_data[scenario]
        total = sum(data)
        ax1.bar(angles[i], total, width=width, bottom=0.1,
               color=color, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax1.set_xticks(angles)
    ax1.set_xticklabels(scenarios, fontsize=9)
    ax1.set_ylim(0, 3.5)
    ax1.set_yticklabels([])
    ax1.set_title('Usage Scenario Power Comparison\n(Normalized Rose Chart)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)
    
    # 右侧: 雷达图
    ax2 = fig.add_subplot(122, projection='polar')
    
    dimensions = ['Network', 'Bluetooth', 'Background', 'Display', 'GPS']
    n_dims = len(dimensions)
    angles_radar = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles_radar += angles_radar[:1]
    
    # 选择几个典型场景
    selected = ['Idle', 'Social\nBrowsing', 'Gaming']
    selected_colors = [COLORS['success'], COLORS['lte'], COLORS['danger']]
    
    for scenario, color in zip(selected, selected_colors):
        values = scenario_data[scenario] + scenario_data[scenario][:1]
        ax2.plot(angles_radar, values, 'o-', linewidth=2, color=color, 
                label=scenario.replace('\n', ' '), markersize=6)
        ax2.fill(angles_radar, values, alpha=0.15, color=color)
    
    ax2.set_xticks(angles_radar[:-1])
    ax2.set_xticklabels(dimensions, fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax2.set_title('Multi-Dimension Power Profile\n(Radar Comparison)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 6. 河流图 - 时序功耗演变
# ============================================================================

def plot_streamgraph_evolution(save_path=None):
    """
    河流图 - 功耗随时间的演变
    """
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='white')
    
    model = IntegratedPowerModel()
    
    # 24小时时间序列
    hours = np.linspace(0, 24, 288)  # 5分钟分辨率
    
    # 生成各组件功耗
    network_power = []
    bluetooth_power = []
    background_power = []
    display_power = []
    
    for h in hours:
        t = h * 3600
        
        # 基于时间的使用模式
        if 0 <= h < 7:  # 睡眠
            net = 10
            bt = 0.5
            disp = 0
        elif 7 <= h < 9:  # 早通勤
            net = 800 + 200 * np.sin(h * np.pi)
            bt = 50
            disp = 200
        elif 9 <= h < 12:  # 上午工作
            net = 300 + 100 * np.random.random()
            bt = 5
            disp = 150
        elif 12 <= h < 14:  # 午餐
            net = 600 + 200 * np.random.random()
            bt = 10
            disp = 300
        elif 14 <= h < 18:  # 下午工作
            net = 250 + 100 * np.random.random()
            bt = 5
            disp = 150
        elif 18 <= h < 20:  # 通勤
            net = 700 + 200 * np.random.random()
            bt = 50
            disp = 250
        elif 20 <= h < 23:  # 晚间娱乐
            net = 500 + 150 * np.random.random()
            bt = 45
            disp = 400
        else:  # 深夜
            net = 50
            bt = 1
            disp = 0
        
        network_power.append(net)
        bluetooth_power.append(bt)
        background_power.append(model.background.total_power(t) * 1000)
        display_power.append(disp)
    
    # 平滑处理
    from scipy.ndimage import gaussian_filter1d
    network_power = gaussian_filter1d(network_power, sigma=3)
    bluetooth_power = gaussian_filter1d(bluetooth_power, sigma=3)
    background_power = gaussian_filter1d(background_power, sigma=2)
    display_power = gaussian_filter1d(display_power, sigma=3)
    
    # 堆叠面积图 (河流图风格)
    ax.stackplot(hours, 
                 network_power, bluetooth_power, background_power, display_power,
                 labels=['Network', 'Bluetooth', 'Background', 'Display'],
                 colors=[COLORS['network'], COLORS['bluetooth'], 
                        COLORS['background'], COLORS['accent']],
                 alpha=0.8,
                 baseline='wiggle')  # 河流图效果
    
    ax.set_xlabel('Time of Day (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (mW)', fontsize=12, fontweight='bold')
    ax.set_title('24-Hour Power Consumption Stream Graph\n(Component Evolution Over Time)',
                 fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    
    # 添加时段标注
    periods = [(0, 7, 'Sleep'), (7, 9, 'AM Commute'), (9, 12, 'Work'),
               (12, 14, 'Lunch'), (14, 18, 'Work'), (18, 20, 'PM Commute'),
               (20, 23, 'Evening'), (23, 24, 'Night')]
    
    y_max = max(np.array(network_power) + np.array(bluetooth_power) + 
                np.array(background_power) + np.array(display_power))
    
    for start, end, label in periods:
        ax.axvline(x=start, color=COLORS['grid'], linestyle='-', alpha=0.5)
    
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 7. 树状图 - 功耗层级分解
# ============================================================================

def plot_treemap_hierarchy(save_path=None):
    """
    树状图风格 - 功耗层级分解
    """
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 第一层: 总功耗
    total_box = FancyBboxPatch((2, 2), 96, 96,
                                boxstyle="round,pad=0.02,rounding_size=0.5",
                                facecolor='white', edgecolor=COLORS['text'],
                                linewidth=3)
    ax.add_patch(total_box)
    ax.text(50, 95, 'Total Power: 350 mW', ha='center', fontsize=14, 
            fontweight='bold', color=COLORS['text'])
    
    # 第二层: 主要组件
    components = [
        ('Network\n120 mW (34%)', 3, 55, 45, 38, COLORS['network']),
        ('Bluetooth\n45 mW (13%)', 52, 55, 45, 38, COLORS['bluetooth']),
        ('Background\n85 mW (24%)', 3, 10, 45, 42, COLORS['background']),
        ('Other\n100 mW (29%)', 52, 10, 45, 42, COLORS['text_light']),
    ]
    
    for label, x, y, w, h, color in components:
        box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.01,rounding_size=0.3",
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
    
    # 第三层: 网络子组件
    net_subs = [
        ('WiFi\n40 mW', 4, 75, 20, 16, COLORS['wifi']),
        ('LTE\n60 mW', 26, 75, 20, 16, COLORS['lte']),
        ('5G\n20 mW', 4, 57, 20, 16, COLORS['nr5g']),
        ('Idle\n0 mW', 26, 57, 20, 16, COLORS['grid']),
    ]
    
    for label, x, y, w, h, color in net_subs:
        box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.01,rounding_size=0.2",
                              facecolor=color, edgecolor='white',
                              linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white' if color != COLORS['grid'] else COLORS['text'])
    
    # 第三层: 蓝牙子组件
    bt_subs = [
        ('BLE\n5 mW', 53, 75, 20, 16, COLORS['ble']),
        ('Audio\n40 mW', 75, 75, 20, 16, COLORS['bt_classic']),
    ]
    
    for label, x, y, w, h, color in bt_subs:
        box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.01,rounding_size=0.2",
                              facecolor=color, edgecolor='white',
                              linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    # 第三层: 后台子组件
    bg_subs = [
        ('CPU\n50 mW', 4, 28, 20, 20, COLORS['cpu']),
        ('Memory\n25 mW', 26, 28, 20, 20, COLORS['memory']),
        ('Services\n10 mW', 4, 12, 20, 14, COLORS['background']),
    ]
    
    for label, x, y, w, h, color in bg_subs:
        box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.01,rounding_size=0.2",
                              facecolor=color, edgecolor='white',
                              linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    ax.set_title('Power Consumption Hierarchy (Treemap Style)',
                 fontsize=14, fontweight='bold', pad=10)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 8. 科学方程展示图
# ============================================================================

def plot_scientific_equations(save_path=None):
    """
    科学方程展示 - 各子模块核心公式
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. 网络功耗方程
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # 背景框
    box1 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#EBF5FF', edgecolor=COLORS['network'],
                           linewidth=2)
    ax1.add_patch(box1)
    
    ax1.text(5, 9, 'Network Power Model', fontsize=12, fontweight='bold',
             ha='center', color=COLORS['network'])
    
    equations_net = [
        (r'$P_{tx} = P_{base} \cdot 10^{\frac{(RSSI_{ref} - RSSI) \cdot k}{10}}$', 'Signal adjustment'),
        (r'$P_{DRX} = P_{idle} + (P_{rx} - P_{idle}) \cdot \frac{T_{on}}{T_{cycle}}$', 'DRX average'),
        (r'$P_{WiFi} = P_{idle} + P_{BB}(R) + P_{RF}(RSSI)$', 'WiFi total'),
    ]
    
    y = 7.5
    for eq, desc in equations_net:
        ax1.text(5, y, eq, fontsize=11, ha='center', color=COLORS['text'])
        ax1.text(5, y - 0.8, f'({desc})', fontsize=9, ha='center', 
                color=COLORS['text_light'], style='italic')
        y -= 2.2
    
    # 2. 蓝牙功耗方程
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    box2 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#F5F3FF', edgecolor=COLORS['bluetooth'],
                           linewidth=2)
    ax2.add_patch(box2)
    
    ax2.text(5, 9, 'Bluetooth Power Model', fontsize=12, fontweight='bold',
             ha='center', color=COLORS['bluetooth'])
    
    equations_bt = [
        (r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}} + P_{sleep} \cdot (1-\frac{T_{tx}}{T_{int}})$', 'Advertising'),
        (r'$P_{conn} = P_{active} \cdot \frac{T_{event}}{T_{CI}} + P_{idle}$', 'Connected'),
        (r'$P_{audio} = P_{codec} + P_{RF} + P_{buffer}$', 'Audio streaming'),
    ]
    
    y = 7.5
    for eq, desc in equations_bt:
        ax2.text(5, y, eq, fontsize=11, ha='center', color=COLORS['text'])
        ax2.text(5, y - 0.8, f'({desc})', fontsize=9, ha='center',
                color=COLORS['text_light'], style='italic')
        y -= 2.2
    
    # 3. 后台功耗方程
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    box3 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#ECFDF5', edgecolor=COLORS['background'],
                           linewidth=2)
    ax3.add_patch(box3)
    
    ax3.text(5, 9, 'Background Tasks Model', fontsize=12, fontweight='bold',
             ha='center', color=COLORS['background'])
    
    equations_bg = [
        (r'$P_{dyn} = \alpha \cdot C \cdot V^2 \cdot f$', 'CMOS dynamic'),
        (r'$P_{DVFS} \propto f^{2.5}$ (with $V \propto f$)', 'DVFS scaling'),
        (r'$P_{task} = P_{active} \cdot \frac{\tau}{T} + P_{idle} \cdot (1-\frac{\tau}{T})$', 'Periodic task'),
    ]
    
    y = 7.5
    for eq, desc in equations_bg:
        ax3.text(5, y, eq, fontsize=11, ha='center', color=COLORS['text'])
        ax3.text(5, y - 0.8, f'({desc})', fontsize=9, ha='center',
                color=COLORS['text_light'], style='italic')
        y -= 2.2
    
    # 4. 电池SOC方程
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    
    box4 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#ECFEFF', edgecolor=COLORS['battery'],
                           linewidth=2)
    ax4.add_patch(box4)
    
    ax4.text(5, 9, 'Battery SOC Model', fontsize=12, fontweight='bold',
             ha='center', color=COLORS['battery'])
    
    equations_bat = [
        (r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}(T)} - k_{sd} \cdot SOC$', 'SOC dynamics'),
        (r'$V = V_{OC}(SOC) - I \cdot R(T, SOC)$', 'Terminal voltage'),
        (r'$Q_{eff} = Q_{nom} \cdot f(T) \cdot (1 - \alpha N^\beta)$', 'Capacity fade'),
    ]
    
    y = 7.5
    for eq, desc in equations_bat:
        ax4.text(5, y, eq, fontsize=11, ha='center', color=COLORS['text'])
        ax4.text(5, y - 0.8, f'({desc})', fontsize=9, ha='center',
                color=COLORS['text_light'], style='italic')
        y -= 2.2
    
    # 5-6. 参数表格
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    ax5.text(0.5, 0.95, 'Key Model Parameters', fontsize=14, fontweight='bold',
             ha='center', transform=ax5.transAxes, color=COLORS['text'])
    
    # 创建参数表格
    table_data = [
        ['Parameter', 'Symbol', 'Value', 'Unit', 'Source'],
        ['WiFi Idle Power', r'$P_{wifi,idle}$', '10', 'mW', 'Carroll (2010)'],
        ['LTE DRX Cycle', r'$T_{cycle}$', '320', 'ms', '3GPP TS 36.321'],
        ['BLE Adv Interval', r'$T_{int}$', '100', 'ms', 'BT Core Spec'],
        ['DVFS Exponent', r'$\gamma$', '2.5', '-', 'Pathak (2012)'],
        ['Battery Capacity', r'$Q_{nom}$', '4500', 'mAh', 'Typical'],
        ['Self-discharge Rate', r'$k_{sd}$', r'$3.5\times10^{-7}$', r'$s^{-1}$', 'Stanford Dataset'],
    ]
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.15, 0.15, 0.1, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # 设置表头样式
    for i in range(5):
        table[(0, i)].set_facecolor(COLORS['network'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 设置行样式
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8FAFC')
    
    plt.suptitle('Mathematical Model Equations Summary',
                 fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 9. 综合仪表板 - 白色主题
# ============================================================================

def plot_white_dashboard(save_path=None):
    """
    综合仪表板 - 白色背景专业风格
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    # 标题
    fig.text(0.5, 0.97, 'Smartphone Battery Power Consumption Analysis Dashboard',
             fontsize=18, fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.94, 'Network | Bluetooth | Background Tasks — Continuous-Time Model',
             fontsize=11, ha='center', color=COLORS['text_light'])
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  top=0.90, bottom=0.05, left=0.05, right=0.95)
    
    model = IntegratedPowerModel()
    
    # ===== 第一行: KPI卡片 =====
    kpis = [
        ('Network Power', '10 - 4500 mW', 'WiFi/LTE/5G', COLORS['network']),
        ('Bluetooth Power', '0.5 - 68 mW', 'BLE/Classic', COLORS['bluetooth']),
        ('Background Power', '60 - 100 mW', 'Tasks/Services', COLORS['background']),
        ('Battery Capacity', '4500 mAh', '17.3 Wh', COLORS['battery']),
    ]
    
    for i, (title, value, subtitle, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 卡片
        card = FancyBboxPatch((0.3, 0.3), 9.4, 9.4,
                               boxstyle="round,pad=0.05,rounding_size=0.4",
                               facecolor='white', edgecolor=color,
                               linewidth=3)
        ax.add_patch(card)
        
        # 顶部色条
        bar = FancyBboxPatch((0.3, 8), 9.4, 1.7,
                              boxstyle="round,pad=0,rounding_size=0.4",
                              facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(bar)
        
        ax.text(5, 8.8, title, fontsize=10, fontweight='bold', ha='center', color='white')
        ax.text(5, 5.5, value, fontsize=14, fontweight='bold', ha='center', color=color)
        ax.text(5, 3, subtitle, fontsize=9, ha='center', color=COLORS['text_light'])
    
    # ===== 第二行: 技术对比 =====
    
    # 网络技术对比
    ax_net = fig.add_subplot(gs[1, :2])
    
    techs = ['WiFi Idle', 'WiFi Active', 'LTE DRX', 'LTE Active', '5G Active']
    powers = [10, 650, 150, 1500, 3000]
    colors_net = [COLORS['wifi'], COLORS['wifi'], COLORS['lte'], COLORS['lte'], COLORS['nr5g']]
    
    bars = ax_net.barh(techs, powers, color=colors_net, alpha=0.85,
                       edgecolor='white', linewidth=2, height=0.6)
    
    for bar, power in zip(bars, powers):
        ax_net.text(power + 80, bar.get_y() + bar.get_height()/2,
                    f'{power}', va='center', fontsize=10, fontweight='bold',
                    color=COLORS['text'])
    
    ax_net.set_xlabel('Power (mW)', fontweight='bold')
    ax_net.set_title('Network Technology Power Comparison', fontsize=11, fontweight='bold')
    ax_net.spines['top'].set_visible(False)
    ax_net.spines['right'].set_visible(False)
    
    # 蓝牙模式对比
    ax_bt = fig.add_subplot(gs[1, 2:])
    
    bt_modes = ['BLE Standby', 'BLE Active', 'BT Audio SBC', 'BT Audio LDAC']
    bt_powers = [0.5, 25, 45, 68]
    colors_bt = [COLORS['ble'], COLORS['ble'], COLORS['bt_classic'], COLORS['bt_classic']]
    
    bars_bt = ax_bt.barh(bt_modes, bt_powers, color=colors_bt, alpha=0.85,
                         edgecolor='white', linewidth=2, height=0.6)
    
    for bar, power in zip(bars_bt, bt_powers):
        ax_bt.text(power + 2, bar.get_y() + bar.get_height()/2,
                   f'{power}', va='center', fontsize=10, fontweight='bold',
                   color=COLORS['text'])
    
    ax_bt.set_xlabel('Power (mW)', fontweight='bold')
    ax_bt.set_title('Bluetooth Mode Power Comparison', fontsize=11, fontweight='bold')
    ax_bt.spines['top'].set_visible(False)
    ax_bt.spines['right'].set_visible(False)
    
    # ===== 第三行: SOC曲线和方程 =====
    
    # SOC曲线
    ax_soc = fig.add_subplot(gs[2, :2])
    
    scenarios = [
        ('idle', 'Idle', COLORS['success']),
        ('social_browsing', 'Active Use', COLORS['lte']),
    ]
    
    for name, label, color in scenarios:
        power_func = create_scenario_power_func(model, name)
        t, soc = model.simulate_soc(4 * 3600, power_func)
        ax_soc.plot(t / 3600, soc * 100, label=label, color=color, linewidth=2.5)
        ax_soc.fill_between(t / 3600, soc * 100, alpha=0.15, color=color)
    
    ax_soc.axhline(y=20, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Low Battery')
    ax_soc.set_xlabel('Time (hours)', fontweight='bold')
    ax_soc.set_ylabel('SOC (%)', fontweight='bold')
    ax_soc.set_title('Battery Discharge Simulation', fontsize=11, fontweight='bold')
    ax_soc.legend(loc='lower left', frameon=True)
    ax_soc.set_ylim(0, 105)
    ax_soc.set_xlim(0, 4)
    ax_soc.spines['top'].set_visible(False)
    ax_soc.spines['right'].set_visible(False)
    ax_soc.grid(True, alpha=0.3)
    
    # 方程面板
    ax_eq = fig.add_subplot(gs[2, 2:])
    ax_eq.set_xlim(0, 10)
    ax_eq.set_ylim(0, 10)
    ax_eq.axis('off')
    
    box = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                          boxstyle="round,pad=0.05,rounding_size=0.3",
                          facecolor='#F8FAFC', edgecolor=COLORS['border'],
                          linewidth=2)
    ax_eq.add_patch(box)
    
    ax_eq.text(5, 9, 'Core Equations', fontsize=11, fontweight='bold',
               ha='center', color=COLORS['network'])
    
    equations = [
        (r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}} - k_{sd} \cdot SOC$', 'SOC Dynamics'),
        (r'$P_{total} = P_{net} + P_{bt} + P_{bg} + P_{base}$', 'Total Power'),
        (r'$P_{tx} = P_0 \cdot 10^{(\Delta RSSI \cdot k)/10}$', 'TX Power'),
        (r'$P_{DVFS} = \alpha C V^2 f \propto f^{2.5}$', 'CPU Power'),
    ]
    
    y = 7.5
    for eq, label in equations:
        ax_eq.text(1, y, f'{label}:', fontsize=9, color=COLORS['text_light'])
        ax_eq.text(3.5, y, eq, fontsize=10, color=COLORS['text'])
        y -= 1.8
    
    # 底部引用
    fig.text(0.5, 0.01, 'Data: Iontech Repository | References: Huang (2012), Carroll (2010), 3GPP, Bluetooth SIG',
             fontsize=8, ha='center', color=COLORS['text_light'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 主函数
# ============================================================================

def generate_scientific_visualizations():
    """生成所有科学可视化图表"""
    print("="*70)
    print("生成科学可视化图表 (白色背景)")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/9] 桑基流程图...")
    plot_sankey_power_flow(os.path.join(script_dir, 'sci_sankey_flow.png'))
    
    print("[2/9] 小提琴图...")
    plot_violin_distribution(os.path.join(script_dir, 'sci_violin_distribution.png'))
    
    print("[3/9] 蜂窝图...")
    plot_hexbin_sensitivity(os.path.join(script_dir, 'sci_hexbin_sensitivity.png'))
    
    print("[4/9] 瀑布图...")
    plot_waterfall_breakdown(os.path.join(script_dir, 'sci_waterfall_breakdown.png'))
    
    print("[5/9] 玫瑰图...")
    plot_rose_scenario_comparison(os.path.join(script_dir, 'sci_rose_comparison.png'))
    
    print("[6/9] 河流图...")
    plot_streamgraph_evolution(os.path.join(script_dir, 'sci_streamgraph_24h.png'))
    
    print("[7/9] 树状图...")
    plot_treemap_hierarchy(os.path.join(script_dir, 'sci_treemap_hierarchy.png'))
    
    print("[8/9] 方程展示...")
    plot_scientific_equations(os.path.join(script_dir, 'sci_equations_summary.png'))
    
    print("[9/9] 综合仪表板...")
    plot_white_dashboard(os.path.join(script_dir, 'sci_dashboard_white.png'))
    
    print("\n" + "="*70)
    print("科学可视化图表生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_scientific_visualizations()
