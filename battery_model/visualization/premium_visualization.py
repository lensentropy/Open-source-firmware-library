#!/usr/bin/env python3
"""
高级美学可视化模块 - 新颖美观有层次感

设计理念:
1. 现代渐变色彩系统
2. 立体阴影与深度效果
3. 流畅曲线与动态感
4. 信息层次与视觉引导
5. 专业数据可视化美学

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Circle, Wedge, Rectangle, 
                                 Polygon, PathPatch, Arc, Ellipse)
from matplotlib.path import Path
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iontech_integrated_model import (
    IntegratedPowerModel, NetworkType, NetworkState,
    BluetoothMode, CPUState, create_scenario_power_func
)

# ============================================================================
# 现代渐变配色系统
# ============================================================================

# 主色渐变
GRADIENT_BLUE = ['#667eea', '#764ba2']
GRADIENT_PURPLE = ['#a855f7', '#6366f1']
GRADIENT_CYAN = ['#06b6d4', '#3b82f6']
GRADIENT_GREEN = ['#10b981', '#059669']
GRADIENT_ORANGE = ['#f97316', '#ea580c']
GRADIENT_PINK = ['#ec4899', '#be185d']
GRADIENT_RED = ['#ef4444', '#dc2626']

# 创建渐变色图
def create_gradient_cmap(colors):
    return LinearSegmentedColormap.from_list('custom', colors, N=256)

CMAP_BLUE = create_gradient_cmap(GRADIENT_BLUE)
CMAP_PURPLE = create_gradient_cmap(GRADIENT_PURPLE)
CMAP_CYAN = create_gradient_cmap(GRADIENT_CYAN)

# 现代配色
COLORS = {
    'bg': '#0f172a',           # 深色背景
    'bg_card': '#1e293b',      # 卡片背景
    'bg_light': '#334155',     # 浅色背景
    'text': '#f8fafc',         # 主文字
    'text_secondary': '#94a3b8', # 次要文字
    'border': '#475569',       # 边框
    'grid': '#334155',         # 网格
    
    # 功能色
    'network': '#3b82f6',
    'bluetooth': '#8b5cf6', 
    'background': '#10b981',
    'battery': '#06b6d4',
    
    # 强调色
    'accent': '#f59e0b',
    'success': '#22c55e',
    'warning': '#eab308',
    'danger': '#ef4444',
    
    # 渐变起止色
    'grad_start': '#667eea',
    'grad_end': '#764ba2',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor': COLORS['bg_card'],
    'axes.edgecolor': COLORS['border'],
    'axes.labelcolor': COLORS['text'],
    'axes.titlecolor': COLORS['text'],
    'xtick.color': COLORS['text_secondary'],
    'ytick.color': COLORS['text_secondary'],
    'text.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.3,
})


# ============================================================================
# 辅助函数: 创建渐变填充和阴影效果
# ============================================================================

def add_glow_effect(ax, x, y, color, alpha=0.3, sigma=3):
    """添加发光效果"""
    for i in range(5):
        ax.plot(x, y, color=color, alpha=alpha * (1 - i/5), 
                linewidth=10 - i*2, solid_capstyle='round')

def create_gradient_bar(ax, x, height, width, gradient_colors, bottom=0):
    """创建渐变色柱状图"""
    n_segments = 50
    for i in range(n_segments):
        ratio = i / n_segments
        color = [c1 + (c2 - c1) * ratio for c1, c2 in 
                 zip(to_rgba(gradient_colors[0])[:3], to_rgba(gradient_colors[1])[:3])]
        segment_height = height / n_segments
        rect = Rectangle((x - width/2, bottom + i * segment_height), 
                         width, segment_height, 
                         facecolor=color + [0.9], edgecolor='none')
        ax.add_patch(rect)

def smooth_curve(x, y, num_points=300):
    """平滑曲线"""
    if len(x) < 4:
        return x, y
    x_new = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)
    return x_new, y_new


# ============================================================================
# 1. 3D层叠环形图 - 功耗分布
# ============================================================================

def plot_layered_donut(save_path=None):
    """
    3D层叠环形图 - 展示各子系统功耗分布
    """
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 数据
    categories = ['Network', 'Bluetooth', 'Background', 'Display', 'Other']
    values = [120, 45, 85, 150, 50]
    total = sum(values)
    
    gradients = [
        GRADIENT_BLUE, GRADIENT_PURPLE, GRADIENT_GREEN, 
        GRADIENT_ORANGE, ['#64748b', '#475569']
    ]
    
    # 绘制多层环形
    layers = [
        (6, 4, values, 0),      # 外层
        (3.5, 2.5, [v*0.7 for v in values], 0.3),  # 中层
        (2, 1.2, [v*0.4 for v in values], 0.5),    # 内层
    ]
    
    for outer_r, inner_r, layer_values, offset_y in layers:
        angles = np.cumsum([0] + [v/total * 360 for v in layer_values[:-1]])
        
        for i, (cat, val, start_angle, grad) in enumerate(zip(categories, layer_values, angles, gradients)):
            extent = val / sum(layer_values) * 360
            
            # 创建扇形
            theta1, theta2 = start_angle, start_angle + extent
            
            # 外弧
            wedge = Wedge((0, offset_y), outer_r, theta1, theta2, width=outer_r-inner_r,
                         facecolor=grad[0], edgecolor=COLORS['bg'], linewidth=2, alpha=0.9)
            ax.add_patch(wedge)
            
            # 添加高光效果
            highlight = Wedge((0, offset_y), outer_r - 0.1, theta1, theta1 + extent * 0.3, 
                             width=0.3, facecolor='white', alpha=0.2)
            ax.add_patch(highlight)
    
    # 中心文字
    center_circle = Circle((0, 0), 1, facecolor=COLORS['bg'], edgecolor=COLORS['border'],
                           linewidth=2)
    ax.add_patch(center_circle)
    
    ax.text(0, 0.3, f'{total}', fontsize=28, fontweight='bold', ha='center', va='center',
            color=COLORS['text'])
    ax.text(0, -0.5, 'mW', fontsize=14, ha='center', va='center', color=COLORS['text_secondary'])
    
    # 图例
    legend_y = 8
    for i, (cat, val, grad) in enumerate(zip(categories, values, gradients)):
        rect = FancyBboxPatch((-14, legend_y - i*2 - 0.4), 1.5, 0.8,
                               boxstyle="round,pad=0.02,rounding_size=0.2",
                               facecolor=grad[0], edgecolor='none')
        ax.add_patch(rect)
        ax.text(-12, legend_y - i*2, f'{cat}', fontsize=11, va='center', 
                color=COLORS['text'], fontweight='bold')
        ax.text(-12, legend_y - i*2 - 0.7, f'{val} mW ({val/total*100:.1f}%)', 
                fontsize=9, va='center', color=COLORS['text_secondary'])
    
    # 标题
    ax.text(0, 11, 'Power Consumption Distribution', fontsize=18, fontweight='bold',
            ha='center', color=COLORS['text'])
    ax.text(0, 9.8, 'Layered Analysis by Component', fontsize=11,
            ha='center', color=COLORS['text_secondary'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 2. 流体渐变曲线图 - 功耗动态
# ============================================================================

def plot_fluid_curves(save_path=None):
    """
    流体渐变曲线图 - 展示功耗随参数变化
    """
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    model = IntegratedPowerModel()
    
    # ===== 左上: 网络功耗曲线 =====
    ax1 = fig.add_subplot(gs[0, 0], facecolor=COLORS['bg_card'])
    
    # 数据
    rssi = np.linspace(-90, -40, 100)
    wifi_power = np.array([model.network.wifi_power(50, r, True) * 1000 for r in rssi])
    
    # 平滑曲线
    rssi_smooth, wifi_smooth = smooth_curve(rssi, wifi_power)
    
    # 发光效果
    for i in range(5):
        ax1.plot(rssi_smooth, wifi_smooth, color=GRADIENT_BLUE[0], 
                alpha=0.1 * (5-i), linewidth=15 - i*3)
    
    # 主曲线
    ax1.plot(rssi_smooth, wifi_smooth, color=GRADIENT_BLUE[0], linewidth=3)
    
    # 渐变填充
    ax1.fill_between(rssi_smooth, wifi_smooth, alpha=0.3,
                     color=GRADIENT_BLUE[0])
    ax1.fill_between(rssi_smooth, wifi_smooth * 0.5, alpha=0.15,
                     color=GRADIENT_BLUE[1])
    
    ax1.set_xlabel('Signal Strength (dBm)', fontweight='bold', color=COLORS['text'])
    ax1.set_ylabel('Power (mW)', fontweight='bold', color=COLORS['text'])
    ax1.set_title('WiFi Power vs Signal Strength', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ===== 右上: 蓝牙功耗曲线 =====
    ax2 = fig.add_subplot(gs[0, 1], facecolor=COLORS['bg_card'])
    
    intervals = np.linspace(20, 2000, 100)
    bt_power = np.array([model.bluetooth.ble_advertising_power(i) * 1000 for i in intervals])
    
    int_smooth, bt_smooth = smooth_curve(intervals, bt_power)
    
    for i in range(5):
        ax2.plot(int_smooth, bt_smooth, color=GRADIENT_PURPLE[0],
                alpha=0.1 * (5-i), linewidth=15 - i*3)
    
    ax2.plot(int_smooth, bt_smooth, color=GRADIENT_PURPLE[0], linewidth=3)
    ax2.fill_between(int_smooth, bt_smooth, alpha=0.3, color=GRADIENT_PURPLE[0])
    
    ax2.set_xlabel('Advertising Interval (ms)', fontweight='bold', color=COLORS['text'])
    ax2.set_ylabel('Power (mW)', fontweight='bold', color=COLORS['text'])
    ax2.set_title('BLE Advertising Power', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    ax2.grid(True, alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ===== 左下: CPU功耗曲线 =====
    ax3 = fig.add_subplot(gs[1, 0], facecolor=COLORS['bg_card'])
    
    freq = np.linspace(0.5, 2.8, 100)
    loads = [0.25, 0.5, 0.75, 1.0]
    colors_load = [GRADIENT_GREEN[0], GRADIENT_CYAN[0], GRADIENT_ORANGE[0], GRADIENT_RED[0]]
    
    for load, color in zip(loads, colors_load):
        cpu_power = np.array([model.background.dvfs_power(f * 1e9, load) * 1000 for f in freq])
        freq_smooth, cpu_smooth = smooth_curve(freq, cpu_power)
        
        for i in range(3):
            ax3.plot(freq_smooth, cpu_smooth, color=color, alpha=0.08 * (3-i), linewidth=10 - i*3)
        ax3.plot(freq_smooth, cpu_smooth, color=color, linewidth=2.5, label=f'{int(load*100)}% Load')
    
    ax3.set_xlabel('CPU Frequency (GHz)', fontweight='bold', color=COLORS['text'])
    ax3.set_ylabel('Power (mW)', fontweight='bold', color=COLORS['text'])
    ax3.set_title('CPU DVFS Power Scaling', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    ax3.legend(loc='upper left', frameon=True, facecolor=COLORS['bg_card'],
               edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax3.grid(True, alpha=0.2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ===== 右下: SOC曲线 =====
    ax4 = fig.add_subplot(gs[1, 1], facecolor=COLORS['bg_card'])
    
    scenarios = [
        ('Idle', 'idle', GRADIENT_GREEN),
        ('Light Use', 'light_use', GRADIENT_CYAN),
        ('Active', 'social_browsing', GRADIENT_ORANGE),
        ('Heavy', 'gaming', GRADIENT_RED),
    ]
    
    for name, scenario, grad in scenarios:
        power_func = create_scenario_power_func(model, scenario)
        t, soc = model.simulate_soc(4 * 3600, power_func)
        t_h = t / 3600
        soc_pct = soc * 100
        
        t_smooth, soc_smooth = smooth_curve(t_h, soc_pct)
        
        for i in range(3):
            ax4.plot(t_smooth, soc_smooth, color=grad[0], alpha=0.08 * (3-i), linewidth=10 - i*3)
        ax4.plot(t_smooth, soc_smooth, color=grad[0], linewidth=2.5, label=name)
    
    ax4.axhline(y=20, color=COLORS['warning'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.set_xlabel('Time (hours)', fontweight='bold', color=COLORS['text'])
    ax4.set_ylabel('SOC (%)', fontweight='bold', color=COLORS['text'])
    ax4.set_title('Battery Discharge Simulation', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    ax4.legend(loc='lower left', frameon=True, facecolor=COLORS['bg_card'],
               edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax4.set_ylim(0, 105)
    ax4.grid(True, alpha=0.2)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle('Power Consumption Analysis - Fluid Gradient Visualization',
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 3. 蜂窝密度热力图 - 参数空间
# ============================================================================

def plot_honeycomb_heatmap(save_path=None):
    """
    蜂窝密度热力图 - 参数空间功耗分布
    """
    fig = plt.figure(figsize=(16, 6), facecolor=COLORS['bg'])
    
    model = IntegratedPowerModel()
    np.random.seed(42)
    
    titles = ['WiFi Power Map', 'LTE Power Map', 'CPU Power Map']
    
    for idx, title in enumerate(titles):
        ax = fig.add_subplot(1, 3, idx + 1, facecolor=COLORS['bg_card'])
        
        if idx == 0:
            # WiFi
            x = np.random.uniform(-90, -40, 2000)
            y = np.random.uniform(0, 100, 2000)
            z = np.array([model.network.wifi_power(yi, xi, True) * 1000 for xi, yi in zip(x, y)])
            xlabel, ylabel = 'RSSI (dBm)', 'Data Rate (Mbps)'
            cmap = CMAP_BLUE
        elif idx == 1:
            # LTE
            x = np.random.uniform(-110, -60, 2000)
            y = np.random.uniform(0, 150, 2000)
            z = np.array([model.network.lte_power(yi, xi, NetworkState.ACTIVE_RX) * 1000 
                         for xi, yi in zip(x, y)])
            xlabel, ylabel = 'RSRP (dBm)', 'Data Rate (Mbps)'
            cmap = create_gradient_cmap(GRADIENT_ORANGE)
        else:
            # CPU
            x = np.random.uniform(0.3, 2.8, 2000)
            y = np.random.uniform(0, 100, 2000)
            z = np.array([model.background.dvfs_power(xi * 1e9, yi/100) * 1000 
                         for xi, yi in zip(x, y)])
            xlabel, ylabel = 'Frequency (GHz)', 'Load (%)'
            cmap = create_gradient_cmap(GRADIENT_GREEN)
        
        # 蜂窝图
        hb = ax.hexbin(x, y, C=z, gridsize=25, cmap=cmap, reduce_C_function=np.mean,
                       mincnt=1, edgecolors=COLORS['bg'], linewidths=0.5)
        
        # 颜色条
        cb = plt.colorbar(hb, ax=ax, shrink=0.8)
        cb.set_label('Power (mW)', color=COLORS['text'])
        cb.ax.yaxis.set_tick_params(color=COLORS['text_secondary'])
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=COLORS['text_secondary'])
        
        ax.set_xlabel(xlabel, fontweight='bold', color=COLORS['text'])
        ax.set_ylabel(ylabel, fontweight='bold', color=COLORS['text'])
        ax.set_title(title, fontsize=12, fontweight='bold', color=COLORS['text'], pad=10)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Parameter Space Power Distribution - Honeycomb Visualization',
                 fontsize=14, fontweight='bold', color=COLORS['text'], y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 4. 动态时间轴 - 24小时功耗
# ============================================================================

def plot_dynamic_timeline(save_path=None):
    """
    动态时间轴 - 24小时功耗演变
    """
    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111, facecolor=COLORS['bg_card'])
    
    model = IntegratedPowerModel()
    
    # 生成24小时数据
    hours = np.linspace(0, 24, 288)
    
    network_power = []
    bluetooth_power = []
    background_power = []
    display_power = []
    
    for h in hours:
        t = h * 3600
        
        if 0 <= h < 7:
            net, bt, disp = 10, 0.5, 0
        elif 7 <= h < 9:
            net, bt, disp = 600, 50, 200
        elif 9 <= h < 12:
            net, bt, disp = 300, 5, 150
        elif 12 <= h < 14:
            net, bt, disp = 800, 10, 350
        elif 14 <= h < 18:
            net, bt, disp = 250, 5, 150
        elif 18 <= h < 20:
            net, bt, disp = 700, 50, 300
        elif 20 <= h < 23:
            net, bt, disp = 600, 45, 400
        else:
            net, bt, disp = 50, 1, 0
        
        # 添加随机波动
        net += np.random.normal(0, net * 0.1)
        
        network_power.append(max(net, 0))
        bluetooth_power.append(bt)
        background_power.append(model.background.total_power(t) * 1000)
        display_power.append(disp)
    
    # 平滑
    network_power = gaussian_filter1d(network_power, sigma=3)
    bluetooth_power = gaussian_filter1d(bluetooth_power, sigma=2)
    display_power = gaussian_filter1d(display_power, sigma=3)
    
    # 堆叠面积图
    colors = [GRADIENT_BLUE[0], GRADIENT_PURPLE[0], GRADIENT_GREEN[0], GRADIENT_ORANGE[0]]
    
    ax.stackplot(hours, network_power, bluetooth_power, background_power, display_power,
                 colors=colors, alpha=0.8,
                 labels=['Network', 'Bluetooth', 'Background', 'Display'])
    
    # 添加时段标记
    periods = [
        (0, 7, 'Sleep', '🌙'),
        (7, 9, 'Commute', '🚗'),
        (9, 12, 'Work AM', '💼'),
        (12, 14, 'Lunch', '🍽'),
        (14, 18, 'Work PM', '💼'),
        (18, 20, 'Commute', '🚗'),
        (20, 23, 'Evening', '📱'),
        (23, 24, 'Night', '🌙'),
    ]
    
    y_max = max(np.array(network_power) + np.array(bluetooth_power) + 
                np.array(background_power) + np.array(display_power))
    
    for start, end, label, icon in periods:
        mid = (start + end) / 2
        ax.axvline(x=start, color=COLORS['grid'], linestyle='-', alpha=0.3, linewidth=1)
        ax.text(mid, y_max * 1.02, label, ha='center', fontsize=9, 
                color=COLORS['text_secondary'], fontweight='bold')
    
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max * 1.1)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Power (mW)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('24-Hour Power Consumption Timeline', fontsize=16, fontweight='bold',
                 color=COLORS['text'], pad=20)
    
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    
    ax.legend(loc='upper right', frameon=True, facecolor=COLORS['bg_card'],
              edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 5. 辐射雷达图 - 多维场景对比
# ============================================================================

def plot_radial_comparison(save_path=None):
    """
    辐射雷达图 - 多维场景功耗对比
    """
    fig = plt.figure(figsize=(14, 12), facecolor=COLORS['bg'])
    
    # 数据
    scenarios = ['Idle', 'Music', 'Social', 'Video', 'Gaming', 'Navigation']
    dimensions = ['Network', 'Bluetooth', 'CPU', 'Display', 'GPS', 'Memory']
    
    # 各场景在各维度的功耗 (归一化0-1)
    data = {
        'Idle': [0.05, 0.02, 0.1, 0, 0, 0.05],
        'Music': [0.15, 0.8, 0.2, 0.1, 0, 0.1],
        'Social': [0.7, 0.1, 0.4, 0.8, 0.05, 0.3],
        'Video': [0.6, 0.1, 0.3, 1.0, 0, 0.2],
        'Gaming': [0.5, 0.6, 0.9, 1.0, 0.1, 0.8],
        'Navigation': [0.4, 0.2, 0.5, 0.9, 1.0, 0.4],
    }
    
    colors_radar = [GRADIENT_GREEN[0], GRADIENT_PURPLE[0], GRADIENT_BLUE[0],
                   GRADIENT_ORANGE[0], GRADIENT_RED[0], GRADIENT_CYAN[0]]
    
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = fig.add_subplot(111, projection='polar', facecolor=COLORS['bg_card'])
    ax.set_facecolor(COLORS['bg_card'])
    
    # 绘制每个场景
    for (scenario, values), color in zip(data.items(), colors_radar):
        values_plot = values + values[:1]
        
        # 发光效果
        for i in range(3):
            ax.plot(angles, values_plot, color=color, alpha=0.1 * (3-i), linewidth=10 - i*3)
        
        # 主线
        ax.plot(angles, values_plot, 'o-', color=color, linewidth=2.5, 
                label=scenario, markersize=8)
        ax.fill(angles, values_plot, color=color, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11, fontweight='bold', color=COLORS['text'])
    ax.set_ylim(0, 1.1)
    
    # 设置网格样式
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color=COLORS['text_secondary'], fontsize=9)
    ax.grid(True, alpha=0.3, color=COLORS['grid'])
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True,
              facecolor=COLORS['bg_card'], edgecolor=COLORS['border'],
              labelcolor=COLORS['text'], fontsize=10)
    
    plt.title('Multi-Scenario Power Profile Comparison\n(Radial Visualization)',
              fontsize=16, fontweight='bold', color=COLORS['text'], pad=30, y=1.08)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 6. 层级仪表盘 - 综合展示
# ============================================================================

def plot_layered_dashboard(save_path=None):
    """
    层级仪表盘 - 综合数据展示
    """
    fig = plt.figure(figsize=(20, 14), facecolor=COLORS['bg'])
    
    # 标题
    fig.text(0.5, 0.97, 'Smartphone Power Consumption Analysis',
             fontsize=22, fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.94, 'Integrated Model Dashboard — Network | Bluetooth | Background',
             fontsize=12, ha='center', color=COLORS['text_secondary'])
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  top=0.90, bottom=0.05, left=0.05, right=0.95)
    
    model = IntegratedPowerModel()
    
    # ===== 第一行: KPI卡片 =====
    kpis = [
        ('Total Power', '350 mW', 'Active Usage', GRADIENT_BLUE),
        ('Battery Life', '8.2 hrs', 'Predicted', GRADIENT_GREEN),
        ('Efficiency', '87%', 'Power Saving', GRADIENT_CYAN),
        ('Temperature', '32°C', 'Normal', GRADIENT_ORANGE),
    ]
    
    for i, (title, value, subtitle, grad) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 卡片背景
        card = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                               boxstyle="round,pad=0.1,rounding_size=0.5",
                               facecolor=COLORS['bg_card'], 
                               edgecolor=grad[0], linewidth=2)
        ax.add_patch(card)
        
        # 顶部渐变条
        for j in range(50):
            ratio = j / 50
            color = [c1 + (c2 - c1) * ratio for c1, c2 in 
                     zip(to_rgba(grad[0])[:3], to_rgba(grad[1])[:3])]
            rect = Rectangle((0.2 + j * 9.6/50, 8.3), 9.6/50, 1.3,
                             facecolor=color + [0.9], edgecolor='none')
            ax.add_patch(rect)
        
        ax.text(5, 9, title, fontsize=10, fontweight='bold', ha='center', 
                va='center', color='white')
        ax.text(5, 5, value, fontsize=24, fontweight='bold', ha='center',
                va='center', color=grad[0])
        ax.text(5, 2.5, subtitle, fontsize=9, ha='center', va='center',
                color=COLORS['text_secondary'])
    
    # ===== 第二行: 技术对比条形图 =====
    
    # 网络技术
    ax_net = fig.add_subplot(gs[1, :2], facecolor=COLORS['bg_card'])
    
    techs = ['WiFi Idle', 'WiFi Active', 'LTE DRX', 'LTE Active', '5G Active']
    powers = [10, 650, 150, 1500, 3000]
    
    bars = ax_net.barh(techs, powers, color=GRADIENT_BLUE[0], alpha=0.85,
                       edgecolor='none', height=0.6)
    
    # 添加渐变效果
    for bar, power in zip(bars, powers):
        ax_net.text(power + 80, bar.get_y() + bar.get_height()/2,
                    f'{power} mW', va='center', fontsize=10, fontweight='bold',
                    color=COLORS['text'])
    
    ax_net.set_xlabel('Power (mW)', fontweight='bold', color=COLORS['text'])
    ax_net.set_title('Network Technology Power', fontsize=12, fontweight='bold',
                     color=COLORS['text'], pad=10)
    ax_net.spines['top'].set_visible(False)
    ax_net.spines['right'].set_visible(False)
    ax_net.tick_params(colors=COLORS['text_secondary'])
    
    # 蓝牙模式
    ax_bt = fig.add_subplot(gs[1, 2:], facecolor=COLORS['bg_card'])
    
    bt_modes = ['BLE Standby', 'BLE Active', 'BT Audio SBC', 'BT Audio LDAC']
    bt_powers = [0.5, 25, 45, 68]
    
    bars_bt = ax_bt.barh(bt_modes, bt_powers, color=GRADIENT_PURPLE[0], alpha=0.85,
                         edgecolor='none', height=0.6)
    
    for bar, power in zip(bars_bt, bt_powers):
        ax_bt.text(power + 2, bar.get_y() + bar.get_height()/2,
                   f'{power} mW', va='center', fontsize=10, fontweight='bold',
                   color=COLORS['text'])
    
    ax_bt.set_xlabel('Power (mW)', fontweight='bold', color=COLORS['text'])
    ax_bt.set_title('Bluetooth Mode Power', fontsize=12, fontweight='bold',
                    color=COLORS['text'], pad=10)
    ax_bt.spines['top'].set_visible(False)
    ax_bt.spines['right'].set_visible(False)
    ax_bt.tick_params(colors=COLORS['text_secondary'])
    
    # ===== 第三行: SOC曲线和方程 =====
    
    ax_soc = fig.add_subplot(gs[2, :2], facecolor=COLORS['bg_card'])
    
    scenarios = [
        ('Idle', 'idle', GRADIENT_GREEN),
        ('Active', 'social_browsing', GRADIENT_ORANGE),
        ('Heavy', 'gaming', GRADIENT_RED),
    ]
    
    for name, scenario, grad in scenarios:
        power_func = create_scenario_power_func(model, scenario)
        t, soc = model.simulate_soc(6 * 3600, power_func)
        
        t_h = t / 3600
        soc_pct = soc * 100
        
        t_smooth, soc_smooth = smooth_curve(t_h, soc_pct)
        
        for i in range(3):
            ax_soc.plot(t_smooth, soc_smooth, color=grad[0], alpha=0.08 * (3-i), linewidth=10 - i*3)
        ax_soc.plot(t_smooth, soc_smooth, color=grad[0], linewidth=2.5, label=name)
    
    ax_soc.axhline(y=20, color=COLORS['warning'], linestyle='--', alpha=0.7, linewidth=1.5)
    ax_soc.set_xlabel('Time (hours)', fontweight='bold', color=COLORS['text'])
    ax_soc.set_ylabel('SOC (%)', fontweight='bold', color=COLORS['text'])
    ax_soc.set_title('Battery Discharge Simulation', fontsize=12, fontweight='bold',
                     color=COLORS['text'], pad=10)
    ax_soc.legend(loc='lower left', frameon=True, facecolor=COLORS['bg_card'],
                  edgecolor=COLORS['border'], labelcolor=COLORS['text'])
    ax_soc.set_ylim(0, 105)
    ax_soc.set_xlim(0, 6)
    ax_soc.grid(True, alpha=0.2)
    ax_soc.spines['top'].set_visible(False)
    ax_soc.spines['right'].set_visible(False)
    ax_soc.tick_params(colors=COLORS['text_secondary'])
    
    # 方程面板
    ax_eq = fig.add_subplot(gs[2, 2:])
    ax_eq.set_xlim(0, 10)
    ax_eq.set_ylim(0, 10)
    ax_eq.axis('off')
    
    # 背景
    bg_box = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                             boxstyle="round,pad=0.1,rounding_size=0.3",
                             facecolor=COLORS['bg_card'],
                             edgecolor=COLORS['border'], linewidth=1)
    ax_eq.add_patch(bg_box)
    
    ax_eq.text(5, 9, 'Core Equations', fontsize=12, fontweight='bold',
               ha='center', color=GRADIENT_BLUE[0])
    
    equations = [
        (r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}} - k_{sd} \cdot SOC$', 'SOC Dynamics'),
        (r'$P_{total} = P_{net} + P_{bt} + P_{bg}$', 'Total Power'),
        (r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}} + P_{sleep}$', 'BLE Advertising'),
        (r'$P_{DVFS} = \alpha C V^2 f \propto f^{2.5}$', 'CPU Power'),
    ]
    
    y = 7.2
    for eq, label in equations:
        ax_eq.text(1, y, f'{label}:', fontsize=9, color=COLORS['text_secondary'])
        ax_eq.text(3.5, y, eq, fontsize=11, color=COLORS['text'])
        y -= 1.7
    
    # 底部引用
    fig.text(0.5, 0.01, 'Data: Iontech Repository | Model: Continuous-time ODE | References: Huang (2012), Carroll (2010)',
             fontsize=8, ha='center', color=COLORS['text_secondary'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 7. 立体柱状对比图
# ============================================================================

def plot_3d_bar_comparison(save_path=None):
    """
    立体柱状对比图 - 技术功耗对比
    """
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111, facecolor=COLORS['bg_card'])
    ax.set_xlim(-1, 12)
    ax.set_ylim(0, 4000)
    
    # 数据
    categories = ['WiFi\nIdle', 'WiFi\nActive', 'LTE\nDRX', 'LTE\nActive', '5G\nActive',
                  'BLE\nStandby', 'BLE\nActive', 'BT\nSBC', 'BT\nLDAC', 'CPU\nIdle', 'CPU\nActive']
    values = [10, 650, 150, 1500, 3000, 0.5, 25, 45, 68, 15, 200]
    
    gradients = [
        GRADIENT_BLUE, GRADIENT_BLUE, GRADIENT_ORANGE, GRADIENT_ORANGE, GRADIENT_RED,
        GRADIENT_PURPLE, GRADIENT_PURPLE, GRADIENT_PURPLE, GRADIENT_PURPLE,
        GRADIENT_GREEN, GRADIENT_GREEN
    ]
    
    bar_width = 0.7
    
    for i, (cat, val, grad) in enumerate(zip(categories, values, gradients)):
        # 阴影
        shadow = Rectangle((i - bar_width/2 + 0.1, 0), bar_width, val,
                           facecolor='black', alpha=0.2)
        ax.add_patch(shadow)
        
        # 渐变柱
        n_segments = 30
        for j in range(n_segments):
            ratio = j / n_segments
            color = [c1 + (c2 - c1) * ratio for c1, c2 in 
                     zip(to_rgba(grad[0])[:3], to_rgba(grad[1])[:3])]
            segment_height = val / n_segments
            rect = Rectangle((i - bar_width/2, j * segment_height), 
                             bar_width, segment_height,
                             facecolor=color + [0.9], edgecolor='none')
            ax.add_patch(rect)
        
        # 顶部高光
        highlight = Rectangle((i - bar_width/2, val - val*0.1), bar_width, val*0.1,
                              facecolor='white', alpha=0.3)
        ax.add_patch(highlight)
        
        # 边框
        border = Rectangle((i - bar_width/2, 0), bar_width, val,
                           facecolor='none', edgecolor='white', linewidth=1, alpha=0.5)
        ax.add_patch(border)
        
        # 数值标签
        if val > 100:
            ax.text(i, val + 80, f'{val}', ha='center', fontsize=10, fontweight='bold',
                   color=COLORS['text'])
        else:
            ax.text(i, val + 80, f'{val}', ha='center', fontsize=9,
                   color=COLORS['text_secondary'])
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9, color=COLORS['text'])
    ax.set_ylabel('Power (mW)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('Technology Power Comparison - 3D Bar Visualization',
                 fontsize=16, fontweight='bold', color=COLORS['text'], pad=20)
    
    # 分组标注
    ax.axvspan(-0.5, 4.5, alpha=0.05, color=GRADIENT_BLUE[0])
    ax.axvspan(4.5, 8.5, alpha=0.05, color=GRADIENT_PURPLE[0])
    ax.axvspan(8.5, 10.5, alpha=0.05, color=GRADIENT_GREEN[0])
    
    ax.text(2, 3700, 'Network', ha='center', fontsize=11, fontweight='bold',
            color=GRADIENT_BLUE[0])
    ax.text(6.5, 3700, 'Bluetooth', ha='center', fontsize=11, fontweight='bold',
            color=GRADIENT_PURPLE[0])
    ax.text(9.5, 3700, 'CPU', ha='center', fontsize=11, fontweight='bold',
            color=GRADIENT_GREEN[0])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y')
    ax.tick_params(colors=COLORS['text_secondary'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 8. 科技感信息图
# ============================================================================

def plot_tech_infographic(save_path=None):
    """
    科技感信息图 - 综合展示
    """
    fig = plt.figure(figsize=(18, 14), facecolor=COLORS['bg'])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 主标题
    ax.text(50, 97, 'SMARTPHONE BATTERY POWER MODEL', fontsize=24, fontweight='bold',
            ha='center', va='top', color=COLORS['text'],
            path_effects=[pe.withStroke(linewidth=3, foreground=GRADIENT_BLUE[0])])
    ax.text(50, 93, 'Continuous-Time Mathematical Modeling System', fontsize=12,
            ha='center', va='top', color=COLORS['text_secondary'])
    
    # 中央电池图标
    battery_x, battery_y = 50, 55
    
    # 电池外框
    battery_body = FancyBboxPatch((battery_x-12, battery_y-15), 24, 30,
                                   boxstyle="round,pad=0.02,rounding_size=1",
                                   facecolor=COLORS['bg_card'],
                                   edgecolor=GRADIENT_CYAN[0], linewidth=3)
    ax.add_patch(battery_body)
    
    # 电池正极
    battery_top = FancyBboxPatch((battery_x-4, battery_y+15), 8, 4,
                                  boxstyle="round,pad=0.02,rounding_size=0.5",
                                  facecolor=GRADIENT_CYAN[0], edgecolor='none')
    ax.add_patch(battery_top)
    
    # 电量填充 (渐变)
    fill_height = 25
    for i in range(30):
        ratio = i / 30
        color = [c1 + (c2 - c1) * ratio for c1, c2 in 
                 zip(to_rgba(GRADIENT_GREEN[1])[:3], to_rgba(GRADIENT_GREEN[0])[:3])]
        rect = Rectangle((battery_x-10, battery_y-13 + i * fill_height/30), 
                         20, fill_height/30,
                         facecolor=color + [0.9], edgecolor='none')
        ax.add_patch(rect)
    
    ax.text(battery_x, battery_y, '87%', fontsize=18, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(battery_x, battery_y-8, '4500 mAh', fontsize=10,
            ha='center', va='center', color=COLORS['text_secondary'])
    
    # 子系统卡片
    subsystems = [
        (15, 75, 'NETWORK', '120 mW', GRADIENT_BLUE, ['WiFi: 10-650 mW', 'LTE: 150-1500 mW', '5G: 3000 mW']),
        (85, 75, 'BLUETOOTH', '45 mW', GRADIENT_PURPLE, ['BLE: 0.5-25 mW', 'Audio: 45-68 mW', 'Multi-device +10%']),
        (15, 30, 'BACKGROUND', '85 mW', GRADIENT_GREEN, ['CPU DVFS: f^2.5', 'Memory: 20 mW', 'Services: 15 mW']),
        (85, 30, 'DISPLAY', '150 mW', GRADIENT_ORANGE, ['Brightness: 0-100%', 'Refresh: 60-120Hz', 'Resolution: 2K']),
    ]
    
    for x, y, title, power, grad, details in subsystems:
        # 卡片
        card = FancyBboxPatch((x-12, y-12), 24, 24,
                               boxstyle="round,pad=0.02,rounding_size=0.8",
                               facecolor=COLORS['bg_card'],
                               edgecolor=grad[0], linewidth=2)
        ax.add_patch(card)
        
        # 顶部色条
        for i in range(24):
            ratio = i / 24
            color = [c1 + (c2 - c1) * ratio for c1, c2 in 
                     zip(to_rgba(grad[0])[:3], to_rgba(grad[1])[:3])]
            rect = Rectangle((x-12 + i, y+9), 1, 3,
                             facecolor=color + [0.9], edgecolor='none')
            ax.add_patch(rect)
        
        ax.text(x, y+10.5, title, fontsize=10, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(x, y+5, power, fontsize=16, fontweight='bold',
                ha='center', va='center', color=grad[0])
        
        for i, detail in enumerate(details):
            ax.text(x, y - i*3, detail, fontsize=8, ha='center', va='center',
                   color=COLORS['text_secondary'])
        
        # 连接线到电池
        if x < 50:
            ax.annotate('', xy=(battery_x-13, battery_y + (10 if y > 50 else -10)),
                       xytext=(x+12, y),
                       arrowprops=dict(arrowstyle='->', color=grad[0], lw=2,
                                      connectionstyle='arc3,rad=0.2'))
        else:
            ax.annotate('', xy=(battery_x+13, battery_y + (10 if y > 50 else -10)),
                       xytext=(x-12, y),
                       arrowprops=dict(arrowstyle='->', color=grad[0], lw=2,
                                      connectionstyle='arc3,rad=-0.2'))
    
    # 底部方程
    eq_y = 8
    equations = [
        (r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}}$', 'SOC Model'),
        (r'$P = \alpha C V^2 f$', 'CMOS Power'),
        (r'$P_{adv} = P_{tx} \cdot \delta$', 'BLE Advertising'),
    ]
    
    for i, (eq, label) in enumerate(equations):
        x = 20 + i * 30
        box = FancyBboxPatch((x-12, eq_y-3), 24, 6,
                              boxstyle="round,pad=0.02,rounding_size=0.3",
                              facecolor=COLORS['bg_card'],
                              edgecolor=COLORS['border'], linewidth=1)
        ax.add_patch(box)
        ax.text(x, eq_y+0.5, eq, fontsize=11, ha='center', va='center', color=COLORS['text'])
        ax.text(x, eq_y-1.5, label, fontsize=8, ha='center', va='center', color=COLORS['text_secondary'])
    
    # 底部引用
    ax.text(50, 2, 'Data: Iontech Repository | Bluetooth Core Spec 5.3 | 3GPP TS 36.321',
            fontsize=8, ha='center', color=COLORS['text_secondary'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 主函数
# ============================================================================

def generate_premium_visualizations():
    """生成所有高级可视化图表"""
    print("="*70)
    print("生成高级美学可视化图表")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/8] 层叠环形图...")
    plot_layered_donut(os.path.join(script_dir, 'premium_donut.png'))
    
    print("[2/8] 流体渐变曲线...")
    plot_fluid_curves(os.path.join(script_dir, 'premium_fluid_curves.png'))
    
    print("[3/8] 蜂窝热力图...")
    plot_honeycomb_heatmap(os.path.join(script_dir, 'premium_honeycomb.png'))
    
    print("[4/8] 动态时间轴...")
    plot_dynamic_timeline(os.path.join(script_dir, 'premium_timeline.png'))
    
    print("[5/8] 辐射雷达图...")
    plot_radial_comparison(os.path.join(script_dir, 'premium_radar.png'))
    
    print("[6/8] 层级仪表盘...")
    plot_layered_dashboard(os.path.join(script_dir, 'premium_dashboard.png'))
    
    print("[7/8] 立体柱状图...")
    plot_3d_bar_comparison(os.path.join(script_dir, 'premium_3d_bars.png'))
    
    print("[8/8] 科技信息图...")
    plot_tech_infographic(os.path.join(script_dir, 'premium_infographic.png'))
    
    print("\n" + "="*70)
    print("高级美学可视化图表生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_premium_visualizations()
