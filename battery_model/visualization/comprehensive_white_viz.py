#!/usr/bin/env python3
"""
综合功耗分析可视化 - 白色背景版

涵盖四大子模块:
1. 网络连接 (WiFi/LTE/5G)
2. 蓝牙 (BLE/Classic/Audio)
3. 后台任务 (CPU/Memory/Services)
4. GPS (多模式/多星座)

设计风格: 现代、清新、层次分明
背景: 纯白色

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Circle, Wedge, Rectangle, 
                                 Polygon, Arc, Ellipse, FancyArrowPatch, PathPatch)
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
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
# 配色系统 - 白色背景专用
# ============================================================================

COLORS = {
    # 背景
    'bg': '#FFFFFF',
    'card': '#F8FAFC',
    'card_alt': '#F1F5F9',
    
    # 文字
    'text': '#1E293B',
    'text_secondary': '#64748B',
    'text_muted': '#94A3B8',
    
    # 边框与网格
    'border': '#E2E8F0',
    'grid': '#F1F5F9',
    
    # 四大子模块主色
    'network': '#2563EB',         # 蓝色
    'network_light': '#DBEAFE',
    'network_gradient': ['#60A5FA', '#2563EB'],
    
    'bluetooth': '#7C3AED',       # 紫色
    'bluetooth_light': '#EDE9FE',
    'bluetooth_gradient': ['#A78BFA', '#7C3AED'],
    
    'background': '#059669',      # 绿色
    'background_light': '#D1FAE5',
    'background_gradient': ['#34D399', '#059669'],
    
    'gps': '#DC2626',             # 红色
    'gps_light': '#FEE2E2',
    'gps_gradient': ['#F87171', '#DC2626'],
    
    # 辅助色
    'battery': '#0891B2',
    'accent': '#F59E0B',
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444',
}

# 子模块图标标记
ICONS = {
    'network': 'NET',
    'bluetooth': 'BT',
    'background': 'BG',
    'gps': 'GPS',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': COLORS['border'],
    'axes.labelcolor': COLORS['text'],
    'axes.titlecolor': COLORS['text'],
    'xtick.color': COLORS['text_secondary'],
    'ytick.color': COLORS['text_secondary'],
    'text.color': COLORS['text'],
    'grid.color': COLORS['grid'],
    'grid.alpha': 0.8,
})


def smooth_curve(x, y, num_points=200):
    """平滑曲线"""
    if len(x) < 4:
        return x, y
    x_new = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)
    y_new = spl(x_new)
    return x_new, y_new


# ============================================================================
# 1. 四模块总览环形图
# ============================================================================

def plot_four_module_overview(save_path=None):
    """
    四模块总览 - 环形图+详情卡片
    """
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    
    # 标题
    fig.text(0.5, 0.96, 'Power Consumption by Subsystem', fontsize=22,
             fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.92, 'Network | Bluetooth | Background | GPS', fontsize=12,
             ha='center', color=COLORS['text_secondary'])
    
    gs = GridSpec(1, 2, figure=fig, wspace=0.05, left=0.05, right=0.95,
                  top=0.88, bottom=0.08)
    
    # 左侧: 双层环形图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 数据
    modules = [
        ('Network', 150, COLORS['network'], COLORS['network_light']),
        ('Bluetooth', 55, COLORS['bluetooth'], COLORS['bluetooth_light']),
        ('Background', 95, COLORS['background'], COLORS['background_light']),
        ('GPS', 80, COLORS['gps'], COLORS['gps_light']),
    ]
    
    total = sum(m[1] for m in modules)
    
    # 外环
    start_angle = 90
    outer_r, mid_r, inner_r = 2.0, 1.4, 1.0
    
    for name, value, color, light in modules:
        extent = value / total * 360
        
        # 外环
        wedge_outer = Wedge((0, 0), outer_r, start_angle - extent, start_angle,
                            width=outer_r - mid_r,
                            facecolor=color, edgecolor='white', linewidth=3, alpha=0.9)
        ax1.add_patch(wedge_outer)
        
        # 内环 (浅色)
        wedge_inner = Wedge((0, 0), mid_r, start_angle - extent, start_angle,
                            width=mid_r - inner_r,
                            facecolor=light, edgecolor='white', linewidth=2)
        ax1.add_patch(wedge_inner)
        
        # 标签
        mid_angle = np.radians(start_angle - extent/2)
        label_r = (outer_r + mid_r) / 2
        lx, ly = label_r * np.cos(mid_angle), label_r * np.sin(mid_angle)
        ax1.text(lx, ly, f'{value/total*100:.0f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        
        start_angle -= extent
    
    # 中心
    center = Circle((0, 0), inner_r - 0.05, facecolor='white', 
                    edgecolor=COLORS['border'], linewidth=2)
    ax1.add_patch(center)
    ax1.text(0, 0.15, f'{total}', fontsize=28, fontweight='bold', 
             ha='center', va='center', color=COLORS['text'])
    ax1.text(0, -0.3, 'mW', fontsize=12, ha='center', color=COLORS['text_secondary'])
    
    # 右侧: 详情卡片
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    
    details = [
        ('Network', 150, COLORS['network'], COLORS['network_light'],
         'WiFi/LTE/5G', ['WiFi: 10-800 mW', 'LTE: 45-1500 mW', '5G: 80-3500 mW']),
        ('Bluetooth', 55, COLORS['bluetooth'], COLORS['bluetooth_light'],
         'BLE/Classic', ['BLE: 0.5-25 mW', 'Audio: 45-68 mW', 'Multi: +10%/dev']),
        ('Background', 95, COLORS['background'], COLORS['background_light'],
         'CPU/Memory', ['CPU DVFS: 5-200 mW', 'Memory: 20 mW', 'Services: 15 mW']),
        ('GPS', 80, COLORS['gps'], COLORS['gps_light'],
         'GNSS', ['Cold: 150 mW', 'Tracking: 50 mW', 'A-GPS: 30 mW']),
    ]
    
    y = 11
    for name, value, color, light, subtitle, items in details:
        # 卡片
        card = FancyBboxPatch((0, y-2.5), 10, 2.7,
                               boxstyle="round,pad=0.03,rounding_size=0.15",
                               facecolor=light, edgecolor=color, linewidth=2)
        ax2.add_patch(card)
        
        # 色块
        block = FancyBboxPatch((0.3, y-2.2), 0.7, 2.1,
                                boxstyle="round,pad=0.02,rounding_size=0.1",
                                facecolor=color, edgecolor='none')
        ax2.add_patch(block)
        
        # 标题
        ax2.text(1.3, y-0.7, name, fontsize=13, fontweight='bold', 
                 va='center', color=COLORS['text'])
        ax2.text(1.3, y-1.5, subtitle, fontsize=9, va='center', 
                 color=COLORS['text_secondary'])
        
        # 功耗值
        ax2.text(9.7, y-1.1, f'{value} mW', fontsize=14, fontweight='bold',
                 va='center', ha='right', color=color)
        
        # 详情
        for i, item in enumerate(items):
            ax2.text(4 + i*2, y-2.2, item, fontsize=8, va='center',
                    color=COLORS['text_muted'])
        
        y -= 3
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 2. 四模块功耗曲线对比
# ============================================================================

def plot_four_module_curves(save_path=None):
    """
    四模块功耗曲线 - 参数敏感性
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    fig.text(0.5, 0.97, 'Power Consumption Curves by Subsystem', fontsize=20,
             fontweight='bold', ha='center', color=COLORS['text'])
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25,
                  top=0.92, bottom=0.08, left=0.06, right=0.94)
    
    model = IntegratedPowerModel()
    
    # 1. 网络功耗
    ax1 = fig.add_subplot(gs[0, 0])
    
    rssi = np.linspace(-90, -40, 100)
    wifi = np.array([model.network.wifi_power(50, r, True) * 1000 for r in rssi])
    lte = np.array([model.network.lte_power(50, r, NetworkState.ACTIVE_RX) * 1000 for r in rssi])
    
    rssi_s, wifi_s = smooth_curve(rssi, wifi)
    _, lte_s = smooth_curve(rssi, lte)
    
    ax1.fill_between(rssi_s, wifi_s, alpha=0.3, color=COLORS['network_light'])
    ax1.fill_between(rssi_s, lte_s, alpha=0.2, color=COLORS['accent'])
    ax1.plot(rssi_s, wifi_s, color=COLORS['network'], linewidth=3, label='WiFi')
    ax1.plot(rssi_s, lte_s, color=COLORS['accent'], linewidth=3, label='LTE')
    
    ax1.set_xlabel('Signal Strength (dBm)', fontweight='bold')
    ax1.set_ylabel('Power (mW)', fontweight='bold')
    ax1.set_title('Network Power vs Signal', fontsize=12, fontweight='bold', 
                  color=COLORS['network'], pad=10)
    ax1.legend(loc='upper right', frameon=True, facecolor='white')
    ax1.grid(True, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 2. 蓝牙功耗
    ax2 = fig.add_subplot(gs[0, 1])
    
    intervals = np.linspace(20, 2000, 100)
    adv = np.array([model.bluetooth.ble_advertising_power(i) * 1000 for i in intervals])
    
    conn_intervals = np.linspace(7.5, 500, 100)
    conn_idle = np.array([model.bluetooth.ble_connected_power(ci, 0.01) * 1000 for ci in conn_intervals])
    conn_active = np.array([model.bluetooth.ble_connected_power(ci, 0.5) * 1000 for ci in conn_intervals])
    
    ax2.fill_between(intervals, adv, alpha=0.3, color=COLORS['bluetooth_light'])
    ax2.plot(intervals, adv, color=COLORS['bluetooth'], linewidth=3, label='Advertising')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(conn_intervals, conn_idle, '--', color=COLORS['bluetooth'], 
                  linewidth=2, alpha=0.7, label='Conn Idle')
    ax2_twin.plot(conn_intervals, conn_active, ':', color=COLORS['bluetooth'],
                  linewidth=2, alpha=0.7, label='Conn Active')
    ax2_twin.set_ylabel('Connected Power (mW)', color=COLORS['bluetooth'])
    
    ax2.set_xlabel('Interval (ms)', fontweight='bold')
    ax2.set_ylabel('Advertising Power (mW)', fontweight='bold')
    ax2.set_title('Bluetooth Power vs Interval', fontsize=12, fontweight='bold',
                  color=COLORS['bluetooth'], pad=10)
    ax2.legend(loc='upper right', frameon=True, facecolor='white')
    ax2.grid(True, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 3. 后台功耗
    ax3 = fig.add_subplot(gs[1, 0])
    
    freq = np.linspace(0.5, 2.8, 100)
    loads = [0.25, 0.5, 0.75, 1.0]
    colors_load = ['#34D399', '#10B981', '#059669', '#047857']
    
    for load, color in zip(loads, colors_load):
        cpu = np.array([model.background.dvfs_power(f * 1e9, load) * 1000 for f in freq])
        freq_s, cpu_s = smooth_curve(freq, cpu)
        ax3.fill_between(freq_s, cpu_s, alpha=0.1, color=color)
        ax3.plot(freq_s, cpu_s, color=color, linewidth=2.5, label=f'{int(load*100)}% Load')
    
    ax3.set_xlabel('CPU Frequency (GHz)', fontweight='bold')
    ax3.set_ylabel('Power (mW)', fontweight='bold')
    ax3.set_title('CPU DVFS Power Scaling', fontsize=12, fontweight='bold',
                  color=COLORS['background'], pad=10)
    ax3.legend(loc='upper left', frameon=True, facecolor='white')
    ax3.grid(True, alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 添加方程
    ax3.text(0.95, 0.95, r'$P_{DVFS} \propto f^{2.5}$', transform=ax3.transAxes,
             fontsize=11, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['border']))
    
    # 4. GPS功耗
    ax4 = fig.add_subplot(gs[1, 1])
    
    # GPS模式功耗
    modes = ['Cold\nStart', 'Warm\nStart', 'Hot\nStart', 'Tracking', 'A-GPS']
    gps_power = [180, 120, 80, 50, 30]
    ttff = [45, 15, 2, 0, 1]  # 首次定位时间(秒)
    
    x = np.arange(len(modes))
    bars = ax4.bar(x, gps_power, color=COLORS['gps'], alpha=0.8,
                   edgecolor='white', linewidth=2, width=0.6)
    
    for bar, power, t in zip(bars, gps_power, ttff):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{power}', ha='center', fontsize=10, fontweight='bold', 
                color=COLORS['gps'])
        if t > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    f'TTFF\n{t}s', ha='center', va='center', fontsize=8, 
                    color='white', fontweight='bold')
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(modes, fontsize=9)
    ax4.set_ylabel('Power (mW)', fontweight='bold')
    ax4.set_title('GPS Mode Power Comparison', fontsize=12, fontweight='bold',
                  color=COLORS['gps'], pad=10)
    ax4.set_ylim(0, 220)
    ax4.grid(True, alpha=0.5, axis='y')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 3. 24小时功耗热力图
# ============================================================================

def plot_24h_heatmap(save_path=None):
    """
    24小时功耗热力图 - 四模块
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    fig.text(0.5, 0.97, '24-Hour Power Consumption Heatmap', fontsize=20,
             fontweight='bold', ha='center', color=COLORS['text'])
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.25, wspace=0.2,
                  top=0.92, bottom=0.08, left=0.06, right=0.94)
    
    hours = np.arange(24)
    
    # 生成功耗数据
    def get_usage_pattern(module):
        base = np.zeros(24)
        if module == 'network':
            base[0:7] = 0.05
            base[7:9] = 0.7
            base[9:12] = 0.4
            base[12:14] = 0.8
            base[14:18] = 0.35
            base[18:20] = 0.75
            base[20:23] = 0.6
            base[23:24] = 0.1
        elif module == 'bluetooth':
            base[0:7] = 0.02
            base[7:9] = 0.9
            base[9:12] = 0.1
            base[12:14] = 0.15
            base[14:18] = 0.1
            base[18:20] = 0.85
            base[20:23] = 0.7
            base[23:24] = 0.05
        elif module == 'background':
            base[:] = 0.3
            base[7:9] = 0.5
            base[12:14] = 0.6
            base[20:23] = 0.4
        elif module == 'gps':
            base[0:7] = 0
            base[7:9] = 0.8
            base[9:12] = 0.1
            base[12:14] = 0.3
            base[14:18] = 0.1
            base[18:20] = 0.9
            base[20:23] = 0.2
            base[23:24] = 0
        return base
    
    modules_data = [
        ('Network', 'network', COLORS['network']),
        ('Bluetooth', 'bluetooth', COLORS['bluetooth']),
        ('Background', 'background', COLORS['background']),
        ('GPS', 'gps', COLORS['gps']),
    ]
    
    for idx, (title, module, color) in enumerate(modules_data):
        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        pattern = get_usage_pattern(module)
        
        # 创建渐变色图
        cmap = LinearSegmentedColormap.from_list('custom', ['white', color], N=256)
        
        # 绘制条形热力图
        bars = ax.bar(hours, pattern, color=[cmap(p) for p in pattern],
                      edgecolor='white', linewidth=1, width=0.9)
        
        # 添加数值
        for h, p in enumerate(pattern):
            if p > 0.1:
                ax.text(h, p + 0.03, f'{p*100:.0f}%', ha='center', fontsize=7,
                       color=COLORS['text_muted'])
        
        ax.set_xlim(-0.5, 23.5)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=8)
        ax.set_xlabel('Hour of Day', fontweight='bold')
        ax.set_ylabel('Usage Level', fontweight='bold')
        ax.set_title(f'{title} Usage Pattern', fontsize=12, fontweight='bold',
                     color=color, pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 时段标记
        ax.axvspan(-0.5, 6.5, alpha=0.05, color=COLORS['text_muted'])
        ax.axvspan(22.5, 23.5, alpha=0.05, color=COLORS['text_muted'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 4. 堆叠河流图 - 24小时
# ============================================================================

def plot_stacked_stream(save_path=None):
    """
    堆叠河流图 - 24小时四模块功耗
    """
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    ax = fig.add_subplot(111)
    
    model = IntegratedPowerModel()
    
    # 生成数据
    hours = np.linspace(0, 24, 288)
    
    network, bluetooth, background, gps = [], [], [], []
    
    for h in hours:
        t = h * 3600
        
        # 网络
        if 0 <= h < 7:
            net = 10
        elif 7 <= h < 9:
            net = 600 + 100 * np.sin(h)
        elif 9 <= h < 12:
            net = 300 + 50 * np.random.random()
        elif 12 <= h < 14:
            net = 700
        elif 14 <= h < 18:
            net = 250
        elif 18 <= h < 20:
            net = 650
        elif 20 <= h < 23:
            net = 500
        else:
            net = 30
        
        # 蓝牙
        if 7 <= h < 9 or 18 <= h < 20:
            bt = 50
        elif 20 <= h < 23:
            bt = 40
        else:
            bt = 2
        
        # GPS
        if 7 <= h < 9 or 18 <= h < 20:
            g = 80
        elif 12 <= h < 14:
            g = 30
        else:
            g = 0
        
        network.append(max(net, 0))
        bluetooth.append(bt)
        background.append(model.background.total_power(t) * 1000)
        gps.append(g)
    
    # 平滑
    network = gaussian_filter1d(network, sigma=3)
    bluetooth = gaussian_filter1d(bluetooth, sigma=2)
    gps = gaussian_filter1d(gps, sigma=2)
    
    # 堆叠
    ax.stackplot(hours, 
                 network, bluetooth, background, gps,
                 labels=['Network', 'Bluetooth', 'Background', 'GPS'],
                 colors=[COLORS['network'], COLORS['bluetooth'], 
                        COLORS['background'], COLORS['gps']],
                 alpha=0.85)
    
    # 时段标记
    periods = [(0, 7, 'Sleep'), (7, 9, 'Commute AM'), (9, 12, 'Work'),
               (12, 14, 'Lunch'), (14, 18, 'Work'), (18, 20, 'Commute PM'),
               (20, 23, 'Evening'), (23, 24, 'Night')]
    
    y_max = max(np.array(network) + np.array(bluetooth) + 
                np.array(background) + np.array(gps))
    
    for start, end, label in periods:
        ax.axvline(x=start, color=COLORS['border'], linestyle='-', alpha=0.5)
        ax.text((start + end) / 2, y_max * 1.02, label, ha='center', fontsize=9,
                color=COLORS['text_secondary'], fontweight='bold')
    
    ax.set_xlim(0, 24)
    ax.set_ylim(0, y_max * 1.08)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (mW)', fontsize=12, fontweight='bold')
    ax.set_title('24-Hour Power Consumption by Subsystem', fontsize=16, 
                 fontweight='bold', pad=20)
    
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    
    ax.legend(loc='upper left', frameon=True, facecolor='white',
              edgecolor=COLORS['border'])
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 5. 雷达图 - 场景对比
# ============================================================================

def plot_scenario_radar(save_path=None):
    """
    雷达图 - 不同场景下四模块功耗
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    
    fig.text(0.5, 0.96, 'Usage Scenario Power Profile', fontsize=20,
             fontweight='bold', ha='center', color=COLORS['text'])
    
    # 数据
    dimensions = ['Network', 'Bluetooth', 'Background', 'GPS']
    n_dims = 4
    
    scenarios = {
        'Idle': ([0.05, 0.02, 0.2, 0], COLORS['background']),
        'Music (BT)': ([0.15, 0.8, 0.25, 0], COLORS['bluetooth']),
        'Social Media': ([0.7, 0.1, 0.4, 0.05], COLORS['network']),
        'Navigation': ([0.4, 0.1, 0.35, 1.0], COLORS['gps']),
        'Fitness': ([0.2, 0.7, 0.3, 0.8], COLORS['accent']),
    }
    
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = fig.add_subplot(111, projection='polar')
    
    for scenario, (values, color) in scenarios.items():
        values_plot = values + values[:1]
        
        ax.plot(angles, values_plot, 'o-', color=color, linewidth=2.5,
                label=scenario, markersize=8)
        ax.fill(angles, values_plot, color=color, alpha=0.12)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9,
                       color=COLORS['text_secondary'])
    ax.grid(True, alpha=0.4)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=True,
              facecolor='white', edgecolor=COLORS['border'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 6. 技术卡片网格
# ============================================================================

def plot_tech_grid(save_path=None):
    """
    技术卡片网格 - 详细功耗参数
    """
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    
    fig.text(0.5, 0.97, 'Technology Power Specifications', fontsize=22,
             fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.94, 'Detailed breakdown by technology and mode',
             fontsize=11, ha='center', color=COLORS['text_secondary'])
    
    gs = GridSpec(2, 4, figure=fig, hspace=0.2, wspace=0.15,
                  top=0.90, bottom=0.05, left=0.03, right=0.97)
    
    # 卡片数据
    cards = [
        # 网络
        ('WiFi', COLORS['network'], COLORS['network_light'],
         [('Idle', 10), ('PSM', 15), ('Rx', 350), ('Tx', 650), ('Active', 800)]),
        ('LTE', COLORS['network'], COLORS['network_light'],
         [('Idle', 45), ('DRX', 150), ('Rx', 850), ('Tx', 1200), ('Active', 1500)]),
        # 蓝牙
        ('BLE', COLORS['bluetooth'], COLORS['bluetooth_light'],
         [('Off', 0), ('Standby', 0.5), ('Adv', 0.6), ('Scan', 12), ('Active', 25)]),
        ('BT Audio', COLORS['bluetooth'], COLORS['bluetooth_light'],
         [('SBC', 45), ('AAC', 52), ('aptX', 56), ('aptX HD', 63), ('LDAC', 68)]),
        # 后台
        ('CPU', COLORS['background'], COLORS['background_light'],
         [('C4', 5), ('C2', 15), ('C1', 30), ('50%', 100), ('100%', 200)]),
        ('Memory', COLORS['background'], COLORS['background_light'],
         [('Idle', 10), ('Read', 20), ('Write', 25), ('Active', 40), ('Peak', 60)]),
        # GPS
        ('GPS', COLORS['gps'], COLORS['gps_light'],
         [('Cold', 180), ('Warm', 120), ('Hot', 80), ('Track', 50), ('A-GPS', 30)]),
        ('Multi-GNSS', COLORS['gps'], COLORS['gps_light'],
         [('GPS', 50), ('+GLO', 70), ('+GAL', 85), ('+BDS', 100), ('All', 120)]),
    ]
    
    for idx, (title, color, light, data) in enumerate(cards):
        row, col = idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 卡片背景
        card = FancyBboxPatch((0.1, 0.1), 9.8, 9.8,
                               boxstyle="round,pad=0.03,rounding_size=0.2",
                               facecolor='white', edgecolor=COLORS['border'],
                               linewidth=1)
        ax.add_patch(card)
        
        # 标题栏
        title_bar = FancyBboxPatch((0.1, 8.2), 9.8, 1.7,
                                    boxstyle="round,pad=0,rounding_size=0.2",
                                    facecolor=color, edgecolor='none')
        ax.add_patch(title_bar)
        ax.text(5, 9, title, fontsize=13, fontweight='bold', ha='center',
                va='center', color='white')
        
        # 数据条
        max_val = max(d[1] for d in data) if data else 1
        y_pos = 7.3
        
        for name, value in data:
            ax.text(0.5, y_pos, name, fontsize=9, va='center', color=COLORS['text'])
            
            val_str = f'{value}' if value >= 1 else f'{value:.1f}'
            ax.text(9.5, y_pos, val_str, fontsize=9, fontweight='bold',
                   va='center', ha='right', color=color)
            
            # 进度条
            bar_bg = FancyBboxPatch((2.3, y_pos-0.25), 5.2, 0.5,
                                     boxstyle="round,pad=0.01,rounding_size=0.08",
                                     facecolor=light, edgecolor='none')
            ax.add_patch(bar_bg)
            
            bar_width = 5.2 * (value / max_val) if max_val > 0 else 0
            bar = FancyBboxPatch((2.3, y_pos-0.25), bar_width, 0.5,
                                  boxstyle="round,pad=0.01,rounding_size=0.08",
                                  facecolor=color, edgecolor='none', alpha=0.75)
            ax.add_patch(bar)
            
            y_pos -= 1.35
        
        ax.text(5, 0.4, 'mW', fontsize=8, ha='center', color=COLORS['text_muted'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 7. 功耗流向桑基图
# ============================================================================

def plot_power_flow(save_path=None):
    """
    功耗流向图 - 从电池到各子系统
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 标题
    ax.text(50, 97, 'Power Flow from Battery to Subsystems', fontsize=20,
            fontweight='bold', ha='center', color=COLORS['text'])
    
    # 电池 (左侧)
    battery = FancyBboxPatch((5, 35), 15, 30,
                              boxstyle="round,pad=0.02,rounding_size=0.5",
                              facecolor=COLORS['battery'], edgecolor='white',
                              linewidth=3, alpha=0.9)
    ax.add_patch(battery)
    
    # 电池正极
    cap = FancyBboxPatch((8, 65), 9, 4,
                          boxstyle="round,pad=0.02,rounding_size=0.3",
                          facecolor=COLORS['battery'], edgecolor='white', linewidth=2)
    ax.add_patch(cap)
    
    ax.text(12.5, 50, '380\nmW', fontsize=16, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(12.5, 42, 'Total', fontsize=10, ha='center', color='white', alpha=0.8)
    
    # 四个子系统 (右侧)
    subsystems = [
        ('Network', 150, 75, COLORS['network']),
        ('Bluetooth', 55, 55, COLORS['bluetooth']),
        ('Background', 95, 35, COLORS['background']),
        ('GPS', 80, 15, COLORS['gps']),
    ]
    
    total = sum(s[1] for s in subsystems)
    
    for name, power, y_pos, color in subsystems:
        # 子系统框
        height = power / total * 50
        box = FancyBboxPatch((75, y_pos - height/2), 20, height,
                              boxstyle="round,pad=0.02,rounding_size=0.3",
                              facecolor=color, edgecolor='white',
                              linewidth=2, alpha=0.9)
        ax.add_patch(box)
        
        ax.text(85, y_pos, f'{name}\n{power} mW', fontsize=11, fontweight='bold',
                ha='center', va='center', color='white')
        
        # 流线
        flow_width = power / total * 20
        
        # 贝塞尔曲线路径
        verts = [
            (20, 50),
            (35, 50),
            (60, y_pos),
            (75, y_pos),
        ]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        
        patch = PathPatch(path, facecolor='none', edgecolor=color,
                         linewidth=flow_width, alpha=0.6, capstyle='round')
        ax.add_patch(patch)
        
        # 功耗百分比
        pct = power / total * 100
        ax.text(48, y_pos + 2, f'{pct:.0f}%', fontsize=10, fontweight='bold',
                ha='center', color=color)
    
    # 图例框
    legend_box = FancyBboxPatch((30, 5), 40, 15,
                                 boxstyle="round,pad=0.03,rounding_size=0.2",
                                 facecolor=COLORS['card'], edgecolor=COLORS['border'],
                                 linewidth=1)
    ax.add_patch(legend_box)
    
    ax.text(50, 17, 'Power Distribution', fontsize=11, fontweight='bold',
            ha='center', color=COLORS['text'])
    
    for i, (name, power, _, color) in enumerate(subsystems):
        x = 33 + i * 10
        dot = Circle((x, 10), 1.5, facecolor=color, edgecolor='white')
        ax.add_patch(dot)
        ax.text(x, 7, f'{name[:3]}', fontsize=8, ha='center', color=COLORS['text_secondary'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 8. SOC模拟与预测
# ============================================================================

def plot_soc_prediction(save_path=None):
    """
    SOC模拟与电池寿命预测
    """
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    
    fig.text(0.5, 0.97, 'Battery SOC Simulation & Life Prediction', fontsize=20,
             fontweight='bold', ha='center', color=COLORS['text'])
    
    gs = GridSpec(1, 3, figure=fig, wspace=0.2,
                  top=0.90, bottom=0.1, left=0.05, right=0.95,
                  width_ratios=[2, 1, 1])
    
    model = IntegratedPowerModel()
    
    # 左侧: SOC曲线
    ax1 = fig.add_subplot(gs[0, 0])
    
    scenarios = [
        ('Standby', 'idle', COLORS['background'], 48),
        ('Light Use', 'light_use', COLORS['battery'], 14),
        ('Active', 'social_browsing', COLORS['network'], 7),
        ('Heavy', 'gaming', COLORS['gps'], 3.5),
    ]
    
    for name, scenario, color, expected_life in scenarios:
        power_func = create_scenario_power_func(model, scenario)
        t, soc = model.simulate_soc(min(expected_life * 1.2, 12) * 3600, power_func)
        
        t_h = t / 3600
        soc_pct = soc * 100
        
        t_s, soc_s = smooth_curve(t_h, soc_pct)
        
        ax1.fill_between(t_s, soc_s, alpha=0.15, color=color)
        ax1.plot(t_s, soc_s, color=color, linewidth=3, label=name)
    
    ax1.axhline(y=20, color=COLORS['accent'], linestyle='--', alpha=0.7, linewidth=2)
    ax1.axhline(y=5, color=COLORS['danger'], linestyle='--', alpha=0.7, linewidth=2)
    
    ax1.text(0.3, 22, 'Low Battery (20%)', fontsize=9, color=COLORS['accent'])
    ax1.text(0.3, 7, 'Critical (5%)', fontsize=9, color=COLORS['danger'])
    
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Discharge Curves', fontsize=12, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', frameon=True, facecolor='white')
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, 12)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 中间: 电池寿命条形图
    ax2 = fig.add_subplot(gs[0, 1])
    
    names = ['Standby', 'Light', 'Active', 'Heavy']
    life_hours = [48, 14, 7, 3.5]
    colors = [COLORS['background'], COLORS['battery'], COLORS['network'], COLORS['gps']]
    
    bars = ax2.barh(names, life_hours, color=colors, edgecolor='white',
                    linewidth=2, height=0.6)
    
    for bar, hours in zip(bars, life_hours):
        ax2.text(hours + 0.5, bar.get_y() + bar.get_height()/2,
                f'{hours}h', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Battery Life (hours)', fontweight='bold')
    ax2.set_title('Predicted Life', fontsize=12, fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 右侧: 关键参数
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    ax3.text(5, 9.5, 'Key Parameters', fontsize=12, fontweight='bold',
             ha='center', color=COLORS['text'])
    
    params = [
        ('Capacity', '4500 mAh', COLORS['battery']),
        ('Voltage', '3.85 V', COLORS['network']),
        ('Energy', '17.3 Wh', COLORS['background']),
        ('Self-discharge', '1%/month', COLORS['gps']),
    ]
    
    y = 8
    for label, value, color in params:
        card = FancyBboxPatch((0.2, y-1.5), 9.6, 1.8,
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=COLORS['card'],
                               edgecolor=color, linewidth=2)
        ax3.add_patch(card)
        ax3.text(0.8, y-0.6, label, fontsize=10, va='center', color=COLORS['text'])
        ax3.text(9.3, y-0.6, value, fontsize=11, fontweight='bold',
                va='center', ha='right', color=color)
        y -= 2.2
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 9. 综合信息图
# ============================================================================

def plot_comprehensive_infographic(save_path=None):
    """
    综合信息图 - 四模块全览
    """
    fig = plt.figure(figsize=(22, 16), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 主标题
    ax.text(50, 98, 'Smartphone Battery Power Consumption Model', fontsize=26,
            fontweight='bold', ha='center', color=COLORS['text'])
    ax.text(50, 94, 'Network | Bluetooth | Background | GPS — Continuous-Time Analysis',
            fontsize=13, ha='center', color=COLORS['text_secondary'])
    
    # 中央电池
    battery_x, battery_y = 50, 50
    
    # 电池阴影
    shadow = FancyBboxPatch((battery_x-9.5, battery_y-14.5), 19, 29,
                             boxstyle="round,pad=0.02,rounding_size=0.8",
                             facecolor=COLORS['border'], alpha=0.3)
    ax.add_patch(shadow)
    
    # 电池主体
    battery = FancyBboxPatch((battery_x-10, battery_y-15), 20, 30,
                              boxstyle="round,pad=0.02,rounding_size=0.8",
                              facecolor='white', edgecolor=COLORS['battery'],
                              linewidth=4)
    ax.add_patch(battery)
    
    # 电池正极
    cap = FancyBboxPatch((battery_x-3.5, battery_y+15), 7, 3.5,
                          boxstyle="round,pad=0.02,rounding_size=0.4",
                          facecolor=COLORS['battery'], edgecolor='none')
    ax.add_patch(cap)
    
    # 电量填充
    for i in range(24):
        ratio = i / 24
        r1, g1, b1 = to_rgba(COLORS['background'])[:3]
        r2, g2, b2 = to_rgba(COLORS['battery'])[:3]
        color = [r1 + (r2-r1)*ratio, g1 + (g2-g1)*ratio, b1 + (b2-b1)*ratio, 0.85]
        rect = Rectangle((battery_x-8, battery_y-13 + i), 16, 1,
                         facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    ax.text(battery_x, battery_y+3, '85%', fontsize=20, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(battery_x, battery_y-5, '4500 mAh', fontsize=11,
            ha='center', va='center', color='white')
    
    # 四个子模块卡片
    modules = [
        (15, 78, 'Network', '150 mW', COLORS['network'], COLORS['network_light'],
         ['WiFi: 10-800 mW', 'LTE: 45-1500 mW', '5G: 80-3500 mW']),
        (85, 78, 'Bluetooth', '55 mW', COLORS['bluetooth'], COLORS['bluetooth_light'],
         ['BLE: 0.5-25 mW', 'Audio: 45-68 mW', 'Multi: +10%/dev']),
        (15, 22, 'Background', '95 mW', COLORS['background'], COLORS['background_light'],
         ['CPU: 5-200 mW', 'Memory: 10-60 mW', 'Services: 15 mW']),
        (85, 22, 'GPS', '80 mW', COLORS['gps'], COLORS['gps_light'],
         ['Cold: 180 mW', 'Tracking: 50 mW', 'A-GPS: 30 mW']),
    ]
    
    for x, y, title, power, color, light, details in modules:
        # 阴影
        shadow = FancyBboxPatch((x-11.5, y-11.5), 23, 23,
                                 boxstyle="round,pad=0.02,rounding_size=0.5",
                                 facecolor=COLORS['border'], alpha=0.2)
        ax.add_patch(shadow)
        
        # 卡片
        card = FancyBboxPatch((x-12, y-12), 24, 24,
                               boxstyle="round,pad=0.02,rounding_size=0.5",
                               facecolor='white', edgecolor=color, linewidth=3)
        ax.add_patch(card)
        
        # 标题栏
        title_bar = FancyBboxPatch((x-12, y+9), 24, 3,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=color, edgecolor='none')
        ax.add_patch(title_bar)
        
        ax.text(x, y+10.5, title, fontsize=12, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(x, y+4, power, fontsize=18, fontweight='bold',
                ha='center', va='center', color=color)
        
        for i, detail in enumerate(details):
            ax.text(x, y - i*3.5 - 1, detail, fontsize=9, ha='center',
                   color=COLORS['text_secondary'])
        
        # 连接线到电池
        if x < 50:
            ax.annotate('', xy=(battery_x-12, battery_y + (10 if y > 50 else -10)),
                       xytext=(x+12, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                      connectionstyle='arc3,rad=0.15'))
        else:
            ax.annotate('', xy=(battery_x+12, battery_y + (10 if y > 50 else -10)),
                       xytext=(x-12, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                      connectionstyle='arc3,rad=-0.15'))
    
    # 底部方程
    equations = [
        (r'$\frac{dSOC}{dt} = -\frac{I}{Q_{eff}} - k_{sd} \cdot SOC$', 'Battery Model'),
        (r'$P_{net} = P_{base} \cdot 10^{\Delta RSSI \cdot k/10}$', 'Network'),
        (r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}}$', 'Bluetooth'),
        (r'$P_{DVFS} \propto f^{2.5}$', 'CPU'),
    ]
    
    eq_y = 5
    for i, (eq, label) in enumerate(equations):
        x = 12.5 + i * 25
        box = FancyBboxPatch((x-10, eq_y-3), 20, 6,
                              boxstyle="round,pad=0.02,rounding_size=0.2",
                              facecolor=COLORS['card'],
                              edgecolor=COLORS['border'], linewidth=1)
        ax.add_patch(box)
        ax.text(x, eq_y+0.8, eq, fontsize=11, ha='center', color=COLORS['text'])
        ax.text(x, eq_y-1.5, label, fontsize=8, ha='center', color=COLORS['text_muted'])
    
    # 数据来源
    ax.text(50, 0.5, 'Data: Iontech Repository | Bluetooth Core Spec 5.3 | 3GPP TS 36.321 | u-blox GNSS',
            fontsize=8, ha='center', color=COLORS['text_muted'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 10. 方程与参数汇总
# ============================================================================

def plot_equations_summary(save_path=None):
    """
    方程与参数汇总 - 四模块
    """
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    
    fig.text(0.5, 0.97, 'Mathematical Model Equations Summary', fontsize=22,
             fontweight='bold', ha='center', color=COLORS['text'])
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.2, wspace=0.15,
                  top=0.92, bottom=0.05, left=0.03, right=0.97)
    
    modules = [
        ('Network Power Model', COLORS['network'], COLORS['network_light'], [
            (r'$P_{tx} = P_{base} \cdot 10^{(RSSI_{ref} - RSSI) \cdot k / 10}$', 'TX Power adjustment'),
            (r'$P_{DRX} = P_{idle} + (P_{active} - P_{idle}) \cdot \frac{T_{on}}{T_{cycle}}$', 'DRX average'),
            (r'$P_{WiFi} = P_{idle} + P_{BB}(R) + P_{RF}(RSSI)$', 'WiFi total'),
            (r'$P_{5G} \approx P_{LTE} \cdot k_{BW} \cdot k_{MIMO} \approx 3.9 \times P_{LTE}$', '5G scaling'),
        ]),
        ('Bluetooth Power Model', COLORS['bluetooth'], COLORS['bluetooth_light'], [
            (r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}} + P_{sleep} \cdot (1 - \frac{T_{tx}}{T_{int}})$', 'Advertising'),
            (r'$P_{conn} = P_{idle} + \frac{P_{active} \cdot T_{CE}}{T_{CI}}$', 'Connected'),
            (r'$P_{audio} = P_{base} \cdot k_{codec}$', 'A2DP Audio'),
            (r'$P_{multi} = N \cdot P_{single} \cdot (1 + 0.1(N-1))$', 'Multi-device'),
        ]),
        ('Background Tasks Model', COLORS['background'], COLORS['background_light'], [
            (r'$P_{dyn} = \alpha \cdot C \cdot V^2 \cdot f$', 'CMOS Dynamic'),
            (r'$P_{DVFS} \propto f^{2.5}$ (with $V \propto f$)', 'DVFS Scaling'),
            (r'$P_{task} = P_{active} \cdot \frac{\tau}{T} + P_{idle} \cdot (1 - \frac{\tau}{T})$', 'Periodic Task'),
            (r'$P_{CPU} = \sum_{state} P_{state} \cdot t_{state} / T$', 'C-state Average'),
        ]),
        ('GPS/GNSS Power Model', COLORS['gps'], COLORS['gps_light'], [
            (r'$P_{startup} = P_{acquisition} \cdot T_{TTFF} / T_{session}$', 'Startup amortized'),
            (r'$P_{tracking} = P_{RF} + P_{correlator} + P_{baseband}$', 'Tracking'),
            (r'$P_{AGNSS} \ll P_{cold}$ (reduced TTFF)', 'Assisted GPS'),
            (r'$P_{multi} = P_{GPS} + \Delta P_{constellation}$', 'Multi-constellation'),
        ]),
    ]
    
    for idx, (title, color, light, equations) in enumerate(modules):
        row, col = idx // 2, idx % 2
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 背景
        bg = FancyBboxPatch((0.1, 0.1), 9.8, 9.8,
                             boxstyle="round,pad=0.03,rounding_size=0.2",
                             facecolor=light, edgecolor=color, linewidth=2)
        ax.add_patch(bg)
        
        ax.text(5, 9.2, title, fontsize=13, fontweight='bold',
                ha='center', color=color)
        
        y = 7.8
        for eq, desc in equations:
            ax.text(5, y, eq, fontsize=10, ha='center', color=COLORS['text'])
            ax.text(5, y-0.7, f'({desc})', fontsize=8, ha='center',
                   color=COLORS['text_secondary'], style='italic')
            y -= 2
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 主函数
# ============================================================================

def generate_comprehensive_visualizations():
    """生成所有综合可视化"""
    print("="*70)
    print("生成四模块综合可视化 (白色背景)")
    print("Network | Bluetooth | Background | GPS")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/10] 四模块总览...")
    plot_four_module_overview(os.path.join(script_dir, 'comp_overview.png'))
    
    print("[2/10] 功耗曲线对比...")
    plot_four_module_curves(os.path.join(script_dir, 'comp_curves.png'))
    
    print("[3/10] 24小时热力图...")
    plot_24h_heatmap(os.path.join(script_dir, 'comp_24h_heatmap.png'))
    
    print("[4/10] 堆叠河流图...")
    plot_stacked_stream(os.path.join(script_dir, 'comp_stream.png'))
    
    print("[5/10] 场景雷达图...")
    plot_scenario_radar(os.path.join(script_dir, 'comp_radar.png'))
    
    print("[6/10] 技术卡片网格...")
    plot_tech_grid(os.path.join(script_dir, 'comp_tech_grid.png'))
    
    print("[7/10] 功耗流向图...")
    plot_power_flow(os.path.join(script_dir, 'comp_flow.png'))
    
    print("[8/10] SOC模拟预测...")
    plot_soc_prediction(os.path.join(script_dir, 'comp_soc.png'))
    
    print("[9/10] 综合信息图...")
    plot_comprehensive_infographic(os.path.join(script_dir, 'comp_infographic.png'))
    
    print("[10/10] 方程汇总...")
    plot_equations_summary(os.path.join(script_dir, 'comp_equations.png'))
    
    print("\n" + "="*70)
    print("四模块综合可视化生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_comprehensive_visualizations()
