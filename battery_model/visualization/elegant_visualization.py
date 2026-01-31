#!/usr/bin/env python3
"""
优雅白色主题可视化 - 层次感与现代美学

设计特点:
1. 清新白色背景
2. 柔和渐变与阴影
3. 精致的层次结构
4. 信息图表风格
5. 高对比度可读性

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Circle, Wedge, Rectangle, 
                                 Polygon, PathPatch, Arc, Ellipse, Shadow)
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
# 优雅配色系统 (白色主题)
# ============================================================================

COLORS = {
    'bg': '#FFFFFF',
    'bg_card': '#F8FAFC',
    'bg_accent': '#F1F5F9',
    'text': '#1E293B',
    'text_secondary': '#64748B',
    'border': '#E2E8F0',
    'shadow': '#94A3B8',
    
    # 功能色 (柔和)
    'network': '#3B82F6',
    'network_light': '#DBEAFE',
    'bluetooth': '#8B5CF6',
    'bluetooth_light': '#EDE9FE',
    'background_task': '#10B981',
    'background_light': '#D1FAE5',
    'display': '#F59E0B',
    'display_light': '#FEF3C7',
    'battery': '#06B6D4',
    'battery_light': '#CFFAFE',
    
    # 渐变
    'grad_blue': ['#60A5FA', '#3B82F6'],
    'grad_purple': ['#A78BFA', '#8B5CF6'],
    'grad_green': ['#34D399', '#10B981'],
    'grad_orange': ['#FBBF24', '#F59E0B'],
    'grad_cyan': ['#22D3EE', '#06B6D4'],
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
    'grid.color': COLORS['border'],
    'grid.alpha': 0.5,
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
# 1. 现代卡片式环形图
# ============================================================================

def plot_modern_ring_chart(save_path=None):
    """
    现代卡片式环形图 - 功耗分布
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    
    # 主标题
    fig.text(0.5, 0.95, 'Power Consumption Distribution', fontsize=20, 
             fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.91, 'Component-level Analysis', fontsize=12,
             ha='center', color=COLORS['text_secondary'])
    
    gs = GridSpec(1, 2, figure=fig, wspace=0.1, left=0.05, right=0.95, 
                  top=0.85, bottom=0.1)
    
    # 左侧: 环形图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 数据
    data = [
        ('Network', 120, COLORS['network'], COLORS['network_light']),
        ('Bluetooth', 45, COLORS['bluetooth'], COLORS['bluetooth_light']),
        ('Background', 85, COLORS['background_task'], COLORS['background_light']),
        ('Display', 150, COLORS['display'], COLORS['display_light']),
        ('Other', 50, COLORS['shadow'], COLORS['bg_accent']),
    ]
    
    total = sum(d[1] for d in data)
    
    # 绘制环形
    start_angle = 90
    outer_r, inner_r = 1.5, 0.9
    
    for name, value, color, light_color in data:
        extent = value / total * 360
        
        # 外环 (带阴影效果)
        wedge = Wedge((0, 0), outer_r, start_angle - extent, start_angle, 
                      width=outer_r - inner_r,
                      facecolor=color, edgecolor='white', linewidth=3)
        ax1.add_patch(wedge)
        
        # 内发光
        inner_wedge = Wedge((0, 0), inner_r + 0.1, start_angle - extent, start_angle,
                            width=0.1, facecolor=light_color, alpha=0.5)
        ax1.add_patch(inner_wedge)
        
        start_angle -= extent
    
    # 中心
    center = Circle((0, 0), inner_r - 0.05, facecolor='white', edgecolor=COLORS['border'],
                    linewidth=2)
    ax1.add_patch(center)
    
    ax1.text(0, 0.15, f'{total}', fontsize=32, fontweight='bold', ha='center', va='center',
             color=COLORS['text'])
    ax1.text(0, -0.25, 'mW Total', fontsize=12, ha='center', va='center',
             color=COLORS['text_secondary'])
    
    # 右侧: 详情卡片
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    
    y_pos = 11
    for name, value, color, light_color in data:
        pct = value / total * 100
        
        # 卡片背景
        card = FancyBboxPatch((0, y_pos - 1.8), 10, 2,
                               boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor=light_color, edgecolor=color,
                               linewidth=2, alpha=0.8)
        ax2.add_patch(card)
        
        # 色块
        color_block = FancyBboxPatch((0.3, y_pos - 1.5), 0.8, 1.4,
                                      boxstyle="round,pad=0.02,rounding_size=0.1",
                                      facecolor=color, edgecolor='none')
        ax2.add_patch(color_block)
        
        # 文字
        ax2.text(1.5, y_pos - 0.5, name, fontsize=12, fontweight='bold',
                 va='center', color=COLORS['text'])
        ax2.text(1.5, y_pos - 1.2, f'{value} mW', fontsize=10,
                 va='center', color=COLORS['text_secondary'])
        
        # 百分比
        ax2.text(9.5, y_pos - 0.8, f'{pct:.1f}%', fontsize=14, fontweight='bold',
                 va='center', ha='right', color=color)
        
        # 进度条
        bar_width = 4 * (pct / 100)
        bar_bg = FancyBboxPatch((4, y_pos - 1.3), 4, 0.6,
                                 boxstyle="round,pad=0.01,rounding_size=0.1",
                                 facecolor=COLORS['bg_accent'], edgecolor='none')
        ax2.add_patch(bar_bg)
        
        bar = FancyBboxPatch((4, y_pos - 1.3), bar_width, 0.6,
                              boxstyle="round,pad=0.01,rounding_size=0.1",
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax2.add_patch(bar)
        
        y_pos -= 2.3
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 2. 层叠面积图 - 24小时功耗
# ============================================================================

def plot_layered_area_chart(save_path=None):
    """
    层叠面积图 - 24小时功耗趋势
    """
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    ax = fig.add_subplot(111)
    
    model = IntegratedPowerModel()
    
    # 生成数据
    hours = np.linspace(0, 24, 288)
    
    network = []
    bluetooth = []
    background = []
    display = []
    
    for h in hours:
        if 0 <= h < 7:
            net, bt, disp = 10, 0.5, 0
        elif 7 <= h < 9:
            net, bt, disp = 500 + 100*np.sin(h), 45, 200
        elif 9 <= h < 12:
            net, bt, disp = 300 + 50*np.random.random(), 5, 150
        elif 12 <= h < 14:
            net, bt, disp = 700 + 100*np.random.random(), 10, 350
        elif 14 <= h < 18:
            net, bt, disp = 250 + 50*np.random.random(), 5, 150
        elif 18 <= h < 20:
            net, bt, disp = 600 + 100*np.random.random(), 45, 300
        elif 20 <= h < 23:
            net, bt, disp = 500 + 100*np.random.random(), 40, 400
        else:
            net, bt, disp = 30, 1, 0
        
        network.append(max(net, 0))
        bluetooth.append(bt)
        background.append(model.background.total_power(h * 3600) * 1000)
        display.append(disp)
    
    # 平滑
    network = gaussian_filter1d(network, sigma=3)
    bluetooth = gaussian_filter1d(bluetooth, sigma=2)
    display = gaussian_filter1d(display, sigma=3)
    
    # 堆叠面积
    ax.fill_between(hours, 0, display, color=COLORS['display'], alpha=0.8, label='Display')
    ax.fill_between(hours, display, np.array(display) + np.array(network),
                    color=COLORS['network'], alpha=0.8, label='Network')
    ax.fill_between(hours, np.array(display) + np.array(network),
                    np.array(display) + np.array(network) + np.array(bluetooth),
                    color=COLORS['bluetooth'], alpha=0.8, label='Bluetooth')
    ax.fill_between(hours, np.array(display) + np.array(network) + np.array(bluetooth),
                    np.array(display) + np.array(network) + np.array(bluetooth) + np.array(background),
                    color=COLORS['background_task'], alpha=0.8, label='Background')
    
    # 时段标记
    periods = [(0, 7, 'Sleep'), (7, 9, 'Morning'), (9, 12, 'Work'),
               (12, 14, 'Lunch'), (14, 18, 'Work'), (18, 20, 'Commute'),
               (20, 23, 'Evening'), (23, 24, 'Night')]
    
    for start, end, label in periods:
        ax.axvline(x=start, color=COLORS['border'], linestyle='-', alpha=0.5)
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.95, label, 
                ha='center', fontsize=9, color=COLORS['text_secondary'],
                fontweight='bold')
    
    ax.set_xlim(0, 24)
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (mW)', fontsize=12, fontweight='bold')
    ax.set_title('24-Hour Power Consumption Profile', fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(0, 25, 3)])
    
    ax.legend(loc='upper left', frameon=True, facecolor='white',
              edgecolor=COLORS['border'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 3. 对比卡片组 - 技术功耗
# ============================================================================

def plot_comparison_cards(save_path=None):
    """
    对比卡片组 - 各技术功耗对比
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    fig.text(0.5, 0.96, 'Technology Power Comparison', fontsize=20,
             fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.93, 'Detailed breakdown by wireless technology and mode',
             fontsize=11, ha='center', color=COLORS['text_secondary'])
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2,
                  top=0.88, bottom=0.08, left=0.05, right=0.95)
    
    # 卡片数据
    cards = [
        ('WiFi', COLORS['network'], COLORS['network_light'],
         [('Idle', 10), ('PSM', 15), ('Rx', 350), ('Tx', 650), ('Active', 800)]),
        ('LTE', COLORS['display'], COLORS['display_light'],
         [('Idle', 45), ('DRX', 150), ('Rx', 850), ('Tx', 1200), ('Active', 1500)]),
        ('5G NR', '#EF4444', '#FEE2E2',
         [('Idle', 80), ('DRX', 300), ('Rx', 1800), ('Tx', 2500), ('Active', 3500)]),
        ('BLE', COLORS['bluetooth'], COLORS['bluetooth_light'],
         [('Off', 0), ('Standby', 0.5), ('Adv', 0.6), ('Scan', 12), ('Active', 25)]),
        ('BT Audio', '#EC4899', '#FCE7F3',
         [('SBC', 45), ('AAC', 52), ('aptX', 56), ('aptX HD', 63), ('LDAC', 68)]),
        ('CPU', COLORS['background_task'], COLORS['background_light'],
         [('C4', 5), ('C2', 15), ('C1', 30), ('C0 50%', 100), ('C0 100%', 200)]),
    ]
    
    for idx, (title, color, light_color, data) in enumerate(cards):
        row, col = idx // 3, idx % 3
        ax = fig.add_subplot(gs[row, col])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # 卡片背景
        card_bg = FancyBboxPatch((0.1, 0.1), 9.8, 9.8,
                                  boxstyle="round,pad=0.05,rounding_size=0.3",
                                  facecolor='white', edgecolor=COLORS['border'],
                                  linewidth=1)
        ax.add_patch(card_bg)
        
        # 标题栏
        title_bar = FancyBboxPatch((0.1, 8.2), 9.8, 1.7,
                                    boxstyle="round,pad=0,rounding_size=0.3",
                                    facecolor=color, edgecolor='none')
        ax.add_patch(title_bar)
        ax.text(5, 9, title, fontsize=14, fontweight='bold', ha='center',
                va='center', color='white')
        
        # 数据条
        max_val = max(d[1] for d in data) if data else 1
        y_pos = 7.2
        
        for name, value in data:
            # 标签
            ax.text(0.5, y_pos, name, fontsize=9, va='center', color=COLORS['text'])
            
            # 数值
            ax.text(9.5, y_pos, f'{value}' if value >= 1 else f'{value:.1f}',
                    fontsize=9, fontweight='bold', va='center', ha='right', color=color)
            
            # 进度条背景
            bar_bg = FancyBboxPatch((2.5, y_pos - 0.25), 5.5, 0.5,
                                     boxstyle="round,pad=0.01,rounding_size=0.1",
                                     facecolor=light_color, edgecolor='none')
            ax.add_patch(bar_bg)
            
            # 进度条
            bar_width = 5.5 * (value / max_val) if max_val > 0 else 0
            bar = FancyBboxPatch((2.5, y_pos - 0.25), bar_width, 0.5,
                                  boxstyle="round,pad=0.01,rounding_size=0.1",
                                  facecolor=color, edgecolor='none', alpha=0.8)
            ax.add_patch(bar)
            
            y_pos -= 1.35
        
        # 单位
        ax.text(5, 0.5, 'mW', fontsize=9, ha='center', color=COLORS['text_secondary'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 4. 渐变曲线组 - 参数敏感性
# ============================================================================

def plot_gradient_curves(save_path=None):
    """
    渐变曲线组 - 参数敏感性分析
    """
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    model = IntegratedPowerModel()
    
    # 曲线配置
    configs = [
        (gs[0, 0], 'WiFi Power vs Signal', np.linspace(-90, -40, 100),
         lambda x: [model.network.wifi_power(50, r, True) * 1000 for r in x],
         'RSSI (dBm)', 'Power (mW)', COLORS['network'], COLORS['network_light']),
        (gs[0, 1], 'BLE Advertising Power', np.linspace(20, 2000, 100),
         lambda x: [model.bluetooth.ble_advertising_power(i) * 1000 for i in x],
         'Interval (ms)', 'Power (mW)', COLORS['bluetooth'], COLORS['bluetooth_light']),
        (gs[1, 0], 'CPU DVFS Power (75% Load)', np.linspace(0.5, 2.8, 100),
         lambda x: [model.background.dvfs_power(f * 1e9, 0.75) * 1000 for f in x],
         'Frequency (GHz)', 'Power (mW)', COLORS['background_task'], COLORS['background_light']),
        (gs[1, 1], 'LTE Power vs Signal', np.linspace(-110, -60, 100),
         lambda x: [model.network.lte_power(50, r, NetworkState.ACTIVE_RX) * 1000 for r in x],
         'RSRP (dBm)', 'Power (mW)', COLORS['display'], COLORS['display_light']),
    ]
    
    for gs_pos, title, x_data, y_func, xlabel, ylabel, color, light_color in configs:
        ax = fig.add_subplot(gs_pos)
        
        y_data = np.array(y_func(x_data))
        x_smooth, y_smooth = smooth_curve(x_data, y_data)
        
        # 填充区域
        ax.fill_between(x_smooth, y_smooth, alpha=0.3, color=light_color)
        ax.fill_between(x_smooth, y_smooth * 0.8, alpha=0.2, color=color)
        
        # 主曲线
        ax.plot(x_smooth, y_smooth, color=color, linewidth=3)
        
        # 数据点
        step = len(x_data) // 8
        ax.scatter(x_data[::step], y_data[::step], color=color, s=60, 
                   edgecolor='white', linewidth=2, zorder=5)
        
        ax.set_xlabel(xlabel, fontweight='bold')
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 5. 现代雷达图
# ============================================================================

def plot_modern_radar(save_path=None):
    """
    现代雷达图 - 场景对比
    """
    fig = plt.figure(figsize=(14, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    
    dimensions = ['Network', 'Bluetooth', 'CPU', 'Display', 'GPS', 'Memory']
    n_dims = len(dimensions)
    
    scenarios = {
        'Idle': ([0.05, 0.02, 0.1, 0, 0, 0.05], COLORS['background_task']),
        'Music Streaming': ([0.2, 0.8, 0.2, 0.1, 0, 0.1], COLORS['bluetooth']),
        'Social Media': ([0.7, 0.1, 0.4, 0.8, 0.1, 0.3], COLORS['network']),
        'Gaming': ([0.5, 0.6, 0.9, 1.0, 0.1, 0.8], '#EF4444'),
        'Navigation': ([0.4, 0.2, 0.5, 0.9, 1.0, 0.4], COLORS['display']),
    }
    
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]
    
    for scenario, (values, color) in scenarios.items():
        values_plot = values + values[:1]
        
        ax.plot(angles, values_plot, 'o-', color=color, linewidth=2.5,
                label=scenario, markersize=8)
        ax.fill(angles, values_plot, color=color, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9, color=COLORS['text_secondary'])
    ax.grid(True, alpha=0.4)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True,
              facecolor='white', edgecolor=COLORS['border'])
    
    plt.title('Usage Scenario Power Profile', fontsize=16, fontweight='bold', pad=20, y=1.1)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 6. 综合信息图
# ============================================================================

def plot_elegant_infographic(save_path=None):
    """
    优雅信息图 - 综合展示
    """
    fig = plt.figure(figsize=(20, 14), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 标题
    ax.text(50, 97, 'Smartphone Battery Power Model', fontsize=24,
            fontweight='bold', ha='center', color=COLORS['text'])
    ax.text(50, 93, 'Continuous-Time Mathematical Framework', fontsize=13,
            ha='center', color=COLORS['text_secondary'])
    
    # 中央电池
    battery_x, battery_y = 50, 52
    
    # 电池外框 (阴影)
    shadow = FancyBboxPatch((battery_x-11.5, battery_y-14.5), 23, 29,
                             boxstyle="round,pad=0.02,rounding_size=1",
                             facecolor=COLORS['shadow'], alpha=0.2)
    ax.add_patch(shadow)
    
    # 电池主体
    battery = FancyBboxPatch((battery_x-12, battery_y-15), 24, 30,
                              boxstyle="round,pad=0.02,rounding_size=1",
                              facecolor='white', edgecolor=COLORS['battery'],
                              linewidth=3)
    ax.add_patch(battery)
    
    # 电池正极
    cap = FancyBboxPatch((battery_x-4, battery_y+15), 8, 4,
                          boxstyle="round,pad=0.02,rounding_size=0.5",
                          facecolor=COLORS['battery'], edgecolor='none')
    ax.add_patch(cap)
    
    # 电量填充
    for i in range(25):
        ratio = i / 25
        r1, g1, b1 = to_rgba(COLORS['background_task'])[:3]
        r2, g2, b2 = to_rgba(COLORS['battery'])[:3]
        color = [r1 + (r2 - r1) * ratio, g1 + (g2 - g1) * ratio, b1 + (b2 - b1) * ratio, 0.8]
        rect = Rectangle((battery_x-10, battery_y-13 + i), 20, 1,
                         facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    ax.text(battery_x, battery_y + 2, '85%', fontsize=20, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(battery_x, battery_y - 5, '4500 mAh', fontsize=11,
            ha='center', va='center', color='white')
    
    # 四个子系统卡片
    cards = [
        (15, 75, 'Network', '120 mW', COLORS['network'], COLORS['network_light'],
         ['WiFi: 10-800 mW', 'LTE: 45-1500 mW', '5G: 80-3500 mW']),
        (85, 75, 'Bluetooth', '45 mW', COLORS['bluetooth'], COLORS['bluetooth_light'],
         ['BLE: 0.5-25 mW', 'A2DP: 45-68 mW', 'Multi: +10%/dev']),
        (15, 28, 'Background', '85 mW', COLORS['background_task'], COLORS['background_light'],
         ['CPU: 5-200 mW', 'Memory: 20 mW', 'Services: 15 mW']),
        (85, 28, 'Display', '150 mW', COLORS['display'], COLORS['display_light'],
         ['Brightness: var', '60-120 Hz', 'Resolution: 2K']),
    ]
    
    for x, y, title, power, color, light_color, details in cards:
        # 卡片阴影
        shadow = FancyBboxPatch((x-11.5, y-11.5), 23, 23,
                                 boxstyle="round,pad=0.02,rounding_size=0.5",
                                 facecolor=COLORS['shadow'], alpha=0.15)
        ax.add_patch(shadow)
        
        # 卡片主体
        card = FancyBboxPatch((x-12, y-12), 24, 24,
                               boxstyle="round,pad=0.02,rounding_size=0.5",
                               facecolor='white', edgecolor=color,
                               linewidth=2)
        ax.add_patch(card)
        
        # 顶部色条
        title_bar = FancyBboxPatch((x-12, y+9), 24, 3,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    facecolor=color, edgecolor='none')
        ax.add_patch(title_bar)
        
        ax.text(x, y+10.5, title, fontsize=11, fontweight='bold',
                ha='center', va='center', color='white')
        ax.text(x, y+5, power, fontsize=18, fontweight='bold',
                ha='center', va='center', color=color)
        
        for i, detail in enumerate(details):
            ax.text(x, y - i*3, detail, fontsize=9, ha='center', va='center',
                   color=COLORS['text_secondary'])
        
        # 连接线
        if x < 50:
            ax.annotate('', xy=(battery_x-14, battery_y + (8 if y > 50 else -8)),
                       xytext=(x+14, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                      connectionstyle='arc3,rad=0.15'))
        else:
            ax.annotate('', xy=(battery_x+14, battery_y + (8 if y > 50 else -8)),
                       xytext=(x-14, y),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                      connectionstyle='arc3,rad=-0.15'))
    
    # 底部方程框
    eq_y = 6
    equations = [
        (r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}} - k_{sd} \cdot SOC$', 'Battery Dynamics'),
        (r'$P_{DVFS} = \alpha C V^2 f \propto f^{2.5}$', 'CPU Scaling'),
        (r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}}$', 'BLE Advertising'),
    ]
    
    for i, (eq, label) in enumerate(equations):
        x = 20 + i * 30
        box = FancyBboxPatch((x-13, eq_y-3.5), 26, 7,
                              boxstyle="round,pad=0.02,rounding_size=0.3",
                              facecolor=COLORS['bg_accent'],
                              edgecolor=COLORS['border'], linewidth=1)
        ax.add_patch(box)
        ax.text(x, eq_y+1, eq, fontsize=12, ha='center', va='center', color=COLORS['text'])
        ax.text(x, eq_y-2, label, fontsize=9, ha='center', va='center',
                color=COLORS['text_secondary'])
    
    ax.text(50, 1, 'Data Sources: Iontech Repository | Bluetooth Core Spec 5.3 | 3GPP TS 36.321 | ARM Cortex TRM',
            fontsize=8, ha='center', color=COLORS['text_secondary'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 7. SOC模拟仪表板
# ============================================================================

def plot_soc_dashboard(save_path=None):
    """
    SOC模拟仪表板
    """
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    fig.text(0.5, 0.97, 'Battery SOC Simulation Dashboard', fontsize=20,
             fontweight='bold', ha='center', color=COLORS['text'])
    
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.25,
                  top=0.92, bottom=0.08, left=0.06, right=0.94)
    
    model = IntegratedPowerModel()
    
    # 左侧大图: SOC曲线
    ax_main = fig.add_subplot(gs[:, :2])
    
    scenarios = [
        ('Standby', 'idle', COLORS['background_task']),
        ('Light Use', 'light_use', COLORS['battery']),
        ('Active', 'social_browsing', COLORS['network']),
        ('Heavy', 'gaming', '#EF4444'),
    ]
    
    for name, scenario, color in scenarios:
        power_func = create_scenario_power_func(model, scenario)
        t, soc = model.simulate_soc(8 * 3600, power_func)
        
        t_h = t / 3600
        soc_pct = soc * 100
        
        t_smooth, soc_smooth = smooth_curve(t_h, soc_pct)
        
        ax_main.fill_between(t_smooth, soc_smooth, alpha=0.15, color=color)
        ax_main.plot(t_smooth, soc_smooth, color=color, linewidth=3, label=name)
    
    ax_main.axhline(y=20, color=COLORS['display'], linestyle='--', 
                    alpha=0.7, linewidth=2, label='Low Battery')
    ax_main.axhline(y=5, color='#EF4444', linestyle='--',
                    alpha=0.7, linewidth=2, label='Critical')
    
    ax_main.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold')
    ax_main.set_title('Battery Discharge Curves', fontsize=14, fontweight='bold', pad=10)
    ax_main.legend(loc='upper right', frameon=True, facecolor='white')
    ax_main.set_ylim(0, 105)
    ax_main.set_xlim(0, 8)
    ax_main.grid(True, alpha=0.3)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    
    # 右上: 电池寿命预测
    ax_life = fig.add_subplot(gs[0, 2])
    
    scenario_names = ['Standby', 'Light', 'Active', 'Heavy']
    life_hours = [48, 12, 6, 3]
    colors_life = [COLORS['background_task'], COLORS['battery'], 
                   COLORS['network'], '#EF4444']
    
    bars = ax_life.barh(scenario_names, life_hours, color=colors_life, 
                        edgecolor='white', linewidth=2, height=0.6)
    
    for bar, hours in zip(bars, life_hours):
        ax_life.text(hours + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{hours}h', va='center', fontsize=10, fontweight='bold')
    
    ax_life.set_xlabel('Battery Life (hours)', fontweight='bold')
    ax_life.set_title('Predicted Battery Life', fontsize=12, fontweight='bold', pad=10)
    ax_life.spines['top'].set_visible(False)
    ax_life.spines['right'].set_visible(False)
    
    # 右下: 关键指标
    ax_kpi = fig.add_subplot(gs[1, 2])
    ax_kpi.set_xlim(0, 10)
    ax_kpi.set_ylim(0, 10)
    ax_kpi.axis('off')
    
    kpis = [
        ('Battery Capacity', '4500 mAh', COLORS['battery']),
        ('Nominal Voltage', '3.85 V', COLORS['network']),
        ('Energy', '17.3 Wh', COLORS['background_task']),
        ('Avg Power (Active)', '350 mW', COLORS['display']),
    ]
    
    y_pos = 9
    for label, value, color in kpis:
        card = FancyBboxPatch((0.2, y_pos-1.8), 9.6, 2,
                               boxstyle="round,pad=0.02,rounding_size=0.2",
                               facecolor=COLORS['bg_accent'],
                               edgecolor=color, linewidth=2)
        ax_kpi.add_patch(card)
        ax_kpi.text(0.8, y_pos-0.8, label, fontsize=10, va='center', color=COLORS['text'])
        ax_kpi.text(9.3, y_pos-0.8, value, fontsize=11, fontweight='bold',
                   va='center', ha='right', color=color)
        y_pos -= 2.4
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 主函数
# ============================================================================

def generate_elegant_visualizations():
    """生成优雅白色主题可视化"""
    print("="*70)
    print("生成优雅白色主题可视化图表")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/7] 现代环形图...")
    plot_modern_ring_chart(os.path.join(script_dir, 'elegant_ring_chart.png'))
    
    print("[2/7] 层叠面积图...")
    plot_layered_area_chart(os.path.join(script_dir, 'elegant_area_chart.png'))
    
    print("[3/7] 对比卡片组...")
    plot_comparison_cards(os.path.join(script_dir, 'elegant_comparison_cards.png'))
    
    print("[4/7] 渐变曲线图...")
    plot_gradient_curves(os.path.join(script_dir, 'elegant_gradient_curves.png'))
    
    print("[5/7] 现代雷达图...")
    plot_modern_radar(os.path.join(script_dir, 'elegant_radar.png'))
    
    print("[6/7] 综合信息图...")
    plot_elegant_infographic(os.path.join(script_dir, 'elegant_infographic.png'))
    
    print("[7/7] SOC仪表板...")
    plot_soc_dashboard(os.path.join(script_dir, 'elegant_soc_dashboard.png'))
    
    print("\n" + "="*70)
    print("优雅白色主题可视化图表生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_elegant_visualizations()
