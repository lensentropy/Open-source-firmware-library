#!/usr/bin/env python3
"""
Iontech集成模型 - 创新可视化模块

基于Iontech电池数据特性的网络连接、蓝牙与后台功耗可视化

图表类型:
1. 径向条形图 - 技术对比
2. 和弦图风格 - 组件关联
3. 等高线图 - 参数空间
4. 时序热力图 - 功耗分布
5. 桑基流向图 - 能量分配
6. 极坐标气泡图 - 场景对比

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iontech_integrated_model import (
    IntegratedPowerModel, NetworkType, NetworkState,
    BluetoothMode, CPUState, create_scenario_power_func
)


# ============================================================================
# 配色方案
# ============================================================================

COLORS = {
    'bg_dark': '#0D1117',
    'bg_card': '#161B22',
    'text': '#E6EDF3',
    'text_muted': '#7D8590',
    'accent': '#58A6FF',
    'network': '#3FB950',
    'bluetooth': '#A371F7', 
    'background': '#F0883E',
    'battery': '#79C0FF',
    'wifi': '#56D364',
    'lte': '#DB6D28',
    'nr5g': '#F85149',
    'ble': '#D2A8FF',
    'bt_classic': '#8B5CF6',
    'success': '#3FB950',
    'warning': '#D29922',
    'danger': '#F85149',
}


def create_custom_cmap(color1, color2):
    """创建双色渐变"""
    c1 = to_rgba(color1)
    c2 = to_rgba(color2)
    return LinearSegmentedColormap.from_list('custom', [c1, c2])


# ============================================================================
# 1. 径向条形图 - 无线技术功耗对比
# ============================================================================

def plot_radial_bar_comparison(save_path=None):
    """
    径向条形图 - WiFi/LTE/5G/BLE/BT功耗对比
    
    创新点: 使用极坐标展示不同无线技术的功耗范围
    """
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg_dark'])
    
    # 左侧: 径向条形图
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_facecolor(COLORS['bg_dark'])
    
    # 数据: (技术, 最小功耗, 最大功耗, 颜色)
    technologies = [
        ('WiFi\nIdle', 10, 10, COLORS['wifi']),
        ('WiFi\nRx', 350, 350, COLORS['wifi']),
        ('WiFi\nTx', 650, 1100, COLORS['wifi']),
        ('LTE\nDRX', 150, 150, COLORS['lte']),
        ('LTE\nActive', 850, 2200, COLORS['lte']),
        ('5G NR\nDRX', 350, 350, COLORS['nr5g']),
        ('5G NR\nActive', 1200, 4500, COLORS['nr5g']),
        ('BLE\nIdle', 3, 3, COLORS['ble']),
        ('BLE\nActive', 25, 25, COLORS['ble']),
        ('BT\nAudio', 45, 65, COLORS['bt_classic']),
    ]
    
    n = len(technologies)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    width = 2*np.pi / n * 0.75
    
    # 绘制条形
    for i, (name, p_min, p_max, color) in enumerate(technologies):
        # 外部条形 (最大功耗)
        bar_max = ax1.bar(angles[i], np.log10(p_max + 1) * 0.8, width=width,
                          bottom=0.5, color=color, alpha=0.8, edgecolor='white', linewidth=1)
        
        # 内部条形 (最小功耗, 如果不同)
        if p_min != p_max:
            bar_min = ax1.bar(angles[i], np.log10(p_min + 1) * 0.8, width=width,
                              bottom=0.5, color=color, alpha=0.4, edgecolor='none')
    
    # 设置刻度
    ax1.set_xticks(angles)
    ax1.set_xticklabels([t[0] for t in technologies], fontsize=9, color=COLORS['text'])
    
    # 添加功耗刻度环
    power_levels = [10, 100, 1000]
    for level in power_levels:
        circle = plt.Circle((0, 0), 0.5 + np.log10(level + 1) * 0.8, 
                            fill=False, color=COLORS['text_muted'], 
                            linestyle='--', linewidth=0.5, alpha=0.3)
        ax1.add_patch(circle)
        ax1.text(np.pi/4, 0.5 + np.log10(level + 1) * 0.8, f'{level}mW',
                fontsize=7, color=COLORS['text_muted'], ha='center')
    
    ax1.set_ylim(0, 4)
    ax1.set_yticklabels([])
    ax1.spines['polar'].set_color(COLORS['text_muted'])
    ax1.grid(True, alpha=0.2, color=COLORS['text_muted'])
    
    ax1.set_title('Wireless Technology Power Comparison\n(Radial Log Scale)', 
                  fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    # 右侧: 功耗范围柱状图
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(COLORS['bg_dark'])
    
    categories = ['Network\n(WiFi)', 'Network\n(LTE)', 'Network\n(5G)', 
                  'Bluetooth\n(BLE)', 'Bluetooth\n(Classic)']
    
    min_powers = [10, 45, 80, 0.5, 5]
    max_powers = [1100, 2200, 4500, 25, 65]
    colors = [COLORS['wifi'], COLORS['lte'], COLORS['nr5g'], 
              COLORS['ble'], COLORS['bt_classic']]
    
    y_pos = np.arange(len(categories))
    
    # 绘制范围条
    for i, (cat, pmin, pmax, color) in enumerate(zip(categories, min_powers, max_powers, colors)):
        # 最小到最大的范围
        ax2.barh(i, pmax - pmin, left=pmin, height=0.6, color=color, alpha=0.6,
                edgecolor='white', linewidth=1)
        # 最小值标记
        ax2.plot(pmin, i, 'o', color=color, markersize=10, markeredgecolor='white')
        # 最大值标记
        ax2.plot(pmax, i, 's', color=color, markersize=10, markeredgecolor='white')
        
        # 标注
        ax2.text(pmin - 50, i, f'{pmin}', va='center', ha='right', 
                fontsize=9, color=COLORS['text'])
        ax2.text(pmax + 50, i, f'{pmax}', va='center', ha='left',
                fontsize=9, color=COLORS['text'])
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=10, color=COLORS['text'])
    ax2.set_xlabel('Power Consumption (mW)', fontsize=11, 
                   fontweight='bold', color=COLORS['text'])
    ax2.set_title('Power Range by Technology', fontsize=14, 
                  fontweight='bold', color=COLORS['text'], pad=15)
    ax2.set_xlim(-100, 5000)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['text_muted'])
    ax2.spines['bottom'].set_color(COLORS['text_muted'])
    ax2.tick_params(colors=COLORS['text_muted'])
    ax2.xaxis.label.set_color(COLORS['text'])
    
    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['text_muted'],
                   markersize=8, label='Min Power'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS['text_muted'],
                   markersize=8, label='Max Power'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', 
               facecolor=COLORS['bg_card'], edgecolor=COLORS['text_muted'],
               labelcolor=COLORS['text'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 2. 等高线图 - 网络参数空间
# ============================================================================

def plot_network_contour_analysis(save_path=None):
    """
    等高线图 - 信号强度/数据速率功耗映射
    """
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg_dark'])
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    model = IntegratedPowerModel()
    
    # 参数网格
    signal_range = np.linspace(-95, -40, 40)
    rate_range = np.linspace(0, 100, 40)
    SIGNAL, RATE = np.meshgrid(signal_range, rate_range)
    
    # 1. WiFi功耗等高线
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['bg_dark'])
    
    wifi_power = np.zeros_like(SIGNAL)
    for i in range(len(rate_range)):
        for j in range(len(signal_range)):
            wifi_power[i, j] = model.network.wifi_power(rate_range[i], signal_range[j], True) * 1000
    
    cf1 = ax1.contourf(SIGNAL, RATE, wifi_power, levels=20, cmap='Greens')
    cs1 = ax1.contour(SIGNAL, RATE, wifi_power, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax1.clabel(cs1, inline=True, fontsize=7, fmt='%.0f')
    
    cbar1 = plt.colorbar(cf1, ax=ax1)
    cbar1.set_label('Power (mW)', color=COLORS['text'])
    cbar1.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=COLORS['text'])
    
    ax1.set_xlabel('RSSI (dBm)', color=COLORS['text'], fontweight='bold')
    ax1.set_ylabel('Data Rate (Mbps)', color=COLORS['text'], fontweight='bold')
    ax1.set_title('WiFi Power Contour', fontsize=12, fontweight='bold', 
                  color=COLORS['wifi'], pad=10)
    ax1.tick_params(colors=COLORS['text_muted'])
    
    # 2. LTE功耗等高线
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['bg_dark'])
    
    lte_power = np.zeros_like(SIGNAL)
    for i in range(len(rate_range)):
        for j in range(len(signal_range)):
            lte_power[i, j] = model.network.lte_power(rate_range[i], signal_range[j], 
                                                       NetworkState.ACTIVE_RX) * 1000
    
    cf2 = ax2.contourf(SIGNAL, RATE, lte_power, levels=20, cmap='Oranges')
    cs2 = ax2.contour(SIGNAL, RATE, lte_power, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax2.clabel(cs2, inline=True, fontsize=7, fmt='%.0f')
    
    cbar2 = plt.colorbar(cf2, ax=ax2)
    cbar2.set_label('Power (mW)', color=COLORS['text'])
    cbar2.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color=COLORS['text'])
    
    ax2.set_xlabel('RSRP (dBm)', color=COLORS['text'], fontweight='bold')
    ax2.set_ylabel('Data Rate (Mbps)', color=COLORS['text'], fontweight='bold')
    ax2.set_title('LTE Power Contour', fontsize=12, fontweight='bold', 
                  color=COLORS['lte'], pad=10)
    ax2.tick_params(colors=COLORS['text_muted'])
    
    # 3. 5G功耗等高线
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(COLORS['bg_dark'])
    
    rate_5g = np.linspace(0, 500, 40)
    SIGNAL_5G, RATE_5G = np.meshgrid(signal_range, rate_5g)
    
    nr_power = np.zeros_like(SIGNAL_5G)
    for i in range(len(rate_5g)):
        for j in range(len(signal_range)):
            nr_power[i, j] = model.network.nr_5g_power(rate_5g[i], signal_range[j],
                                                        NetworkState.ACTIVE_RX) * 1000
    
    cf3 = ax3.contourf(SIGNAL_5G, RATE_5G, nr_power, levels=20, cmap='Reds')
    cs3 = ax3.contour(SIGNAL_5G, RATE_5G, nr_power, levels=10, colors='white', linewidths=0.5, alpha=0.5)
    ax3.clabel(cs3, inline=True, fontsize=7, fmt='%.0f')
    
    cbar3 = plt.colorbar(cf3, ax=ax3)
    cbar3.set_label('Power (mW)', color=COLORS['text'])
    cbar3.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color=COLORS['text'])
    
    ax3.set_xlabel('SS-RSRP (dBm)', color=COLORS['text'], fontweight='bold')
    ax3.set_ylabel('Data Rate (Mbps)', color=COLORS['text'], fontweight='bold')
    ax3.set_title('5G NR Power Contour', fontsize=12, fontweight='bold', 
                  color=COLORS['nr5g'], pad=10)
    ax3.tick_params(colors=COLORS['text_muted'])
    
    # 4-6. 蓝牙和后台功耗分析
    # 4. 蓝牙模式对比
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor(COLORS['bg_dark'])
    
    bt_modes = ['Standby', 'Advertising', 'Scanning', 'Connected\nIdle', 
                'Connected\nActive', 'Audio\n(SBC)', 'Audio\n(LDAC)']
    bt_powers = [
        model.bluetooth.get_power(BluetoothMode.STANDBY) * 1000,
        model.bluetooth.ble_advertising_power() * 1000,
        model.bluetooth.ble_scanning_power() * 1000,
        model.bluetooth.get_power(BluetoothMode.CONNECTED_IDLE) * 1000,
        model.bluetooth.get_power(BluetoothMode.CONNECTED_ACTIVE, data_rate=100) * 1000,
        model.bluetooth.bt_audio_power('SBC') * 1000,
        model.bluetooth.bt_audio_power('LDAC') * 1000,
    ]
    
    colors_bt = [COLORS['ble']] * 5 + [COLORS['bt_classic']] * 2
    bars = ax4.barh(bt_modes, bt_powers, color=colors_bt, alpha=0.8,
                    edgecolor='white', linewidth=1)
    
    for bar, power in zip(bars, bt_powers):
        ax4.text(power + 1, bar.get_y() + bar.get_height()/2,
                 f'{power:.1f}', va='center', fontsize=9, color=COLORS['text'])
    
    ax4.set_xlabel('Power (mW)', color=COLORS['text'], fontweight='bold')
    ax4.set_title('Bluetooth Power by Mode', fontsize=12, fontweight='bold',
                  color=COLORS['bluetooth'], pad=10)
    ax4.tick_params(colors=COLORS['text_muted'])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color(COLORS['text_muted'])
    ax4.spines['bottom'].set_color(COLORS['text_muted'])
    
    # 5. BLE连接数影响
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(COLORS['bg_dark'])
    
    devices = np.arange(1, 9)
    conn_powers = [model.bluetooth.ble_connected_power(50, 50, d) * 1000 for d in devices]
    
    ax5.plot(devices, conn_powers, 'o-', color=COLORS['ble'], linewidth=2, markersize=10,
             markeredgecolor='white', markeredgewidth=2)
    ax5.fill_between(devices, conn_powers, alpha=0.3, color=COLORS['ble'])
    
    ax5.set_xlabel('Number of Connected Devices', color=COLORS['text'], fontweight='bold')
    ax5.set_ylabel('Power (mW)', color=COLORS['text'], fontweight='bold')
    ax5.set_title('BLE Multi-Device Power Scaling', fontsize=12, fontweight='bold',
                  color=COLORS['ble'], pad=10)
    ax5.tick_params(colors=COLORS['text_muted'])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['left'].set_color(COLORS['text_muted'])
    ax5.spines['bottom'].set_color(COLORS['text_muted'])
    ax5.grid(True, alpha=0.2, color=COLORS['text_muted'])
    
    # 6. 后台任务时序
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(COLORS['bg_dark'])
    
    t = np.linspace(0, 1800, 300)  # 30分钟
    bg_power = [model.background.total_power(ti) * 1000 for ti in t]
    
    ax6.fill_between(t/60, bg_power, alpha=0.5, color=COLORS['background'])
    ax6.plot(t/60, bg_power, color=COLORS['background'], linewidth=1)
    
    # 移动平均
    window = 15
    moving_avg = np.convolve(bg_power, np.ones(window)/window, mode='valid')
    ax6.plot(t[window-1:]/60, moving_avg, color=COLORS['danger'], linewidth=2,
             label='Moving Avg')
    
    ax6.set_xlabel('Time (minutes)', color=COLORS['text'], fontweight='bold')
    ax6.set_ylabel('Power (mW)', color=COLORS['text'], fontweight='bold')
    ax6.set_title('Background Tasks Power Pattern', fontsize=12, fontweight='bold',
                  color=COLORS['background'], pad=10)
    ax6.legend(facecolor=COLORS['bg_card'], edgecolor=COLORS['text_muted'],
               labelcolor=COLORS['text'])
    ax6.tick_params(colors=COLORS['text_muted'])
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.spines['left'].set_color(COLORS['text_muted'])
    ax6.spines['bottom'].set_color(COLORS['text_muted'])
    
    plt.suptitle('Network, Bluetooth & Background Power Analysis\nBased on Iontech Battery Dataset Characteristics',
                 fontsize=16, fontweight='bold', color=COLORS['text'], y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 3. 场景对比气泡图
# ============================================================================

def plot_scenario_bubble_comparison(save_path=None):
    """
    气泡图 - 使用场景功耗对比
    
    X轴: 网络功耗, Y轴: 蓝牙功耗, 气泡大小: 后台功耗
    """
    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg_dark'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(COLORS['bg_dark'])
    
    model = IntegratedPowerModel()
    
    # 场景数据
    scenarios = [
        ('Idle\n(Screen Off)', NetworkType.WIFI, NetworkState.DRX, 0, -65,
         BluetoothMode.CONNECTED_IDLE, 1, COLORS['success']),
        ('Music\nStreaming', NetworkType.WIFI, NetworkState.ACTIVE_RX, 1.5, -60,
         BluetoothMode.AUDIO_STREAMING, 1, COLORS['bluetooth']),
        ('Social\nBrowsing', NetworkType.LTE, NetworkState.ACTIVE_RX, 10, -75,
         BluetoothMode.CONNECTED_IDLE, 2, COLORS['lte']),
        ('Video\nStreaming', NetworkType.WIFI, NetworkState.ACTIVE_RX, 8, -55,
         BluetoothMode.OFF, 0, COLORS['wifi']),
        ('5G Download', NetworkType.NR_5G, NetworkState.ACTIVE_RX, 200, -80,
         BluetoothMode.OFF, 0, COLORS['nr5g']),
        ('Fitness\nTracking', NetworkType.OFF, NetworkState.IDLE, 0, 0,
         BluetoothMode.CONNECTED_ACTIVE, 3, COLORS['ble']),
        ('Gaming\n(Online)', NetworkType.WIFI, NetworkState.ACTIVE_TX, 5, -50,
         BluetoothMode.AUDIO_STREAMING, 1, COLORS['danger']),
    ]
    
    for name, net_type, net_state, net_rate, signal, bt_mode, bt_devices, color in scenarios:
        # 计算各组件功耗
        if net_type == NetworkType.WIFI:
            net_power = model.network.wifi_power(net_rate, signal, 
                                                  net_state == NetworkState.ACTIVE_TX) * 1000
        elif net_type == NetworkType.LTE:
            net_power = model.network.lte_power(net_rate, signal, net_state) * 1000
        elif net_type == NetworkType.NR_5G:
            net_power = model.network.nr_5g_power(net_rate, signal, net_state) * 1000
        else:
            net_power = 0
        
        bt_power = model.bluetooth.get_power(bt_mode, devices=bt_devices) * 1000
        
        # 后台功耗 (平均)
        bg_power = np.mean([model.background.total_power(t) * 1000 for t in range(100)])
        
        # 绘制气泡
        size = bg_power * 30 + 200
        scatter = ax.scatter(net_power, bt_power, s=size, c=color, alpha=0.7,
                            edgecolors='white', linewidths=2)
        
        # 标签
        ax.annotate(name, (net_power, bt_power), 
                   textcoords="offset points", xytext=(0, np.sqrt(size)/2 + 10),
                   ha='center', fontsize=9, color=COLORS['text'], fontweight='bold')
    
    ax.set_xlabel('Network Power (mW)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_ylabel('Bluetooth Power (mW)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax.set_title('Usage Scenario Power Comparison\n(Bubble Size = Background Power)', 
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    
    ax.tick_params(colors=COLORS['text_muted'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['text_muted'])
    ax.spines['bottom'].set_color(COLORS['text_muted'])
    ax.grid(True, alpha=0.2, color=COLORS['text_muted'])
    
    # 添加象限标签
    ax.axhline(y=30, color=COLORS['text_muted'], linestyle='--', alpha=0.3)
    ax.axvline(x=500, color=COLORS['text_muted'], linestyle='--', alpha=0.3)
    
    ax.text(100, 60, 'Low Network\nHigh BT', fontsize=9, color=COLORS['text_muted'],
            alpha=0.7, ha='center')
    ax.text(1500, 60, 'High Network\nHigh BT', fontsize=9, color=COLORS['text_muted'],
            alpha=0.7, ha='center')
    ax.text(100, 5, 'Low Network\nLow BT', fontsize=9, color=COLORS['text_muted'],
            alpha=0.7, ha='center')
    ax.text(1500, 5, 'High Network\nLow BT', fontsize=9, color=COLORS['text_muted'],
            alpha=0.7, ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 4. 时序热力图 - 24小时功耗分布
# ============================================================================

def plot_24h_heatmap(save_path=None):
    """
    时序热力图 - 24小时功耗热力分布
    """
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg_dark'])
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.5, 1], hspace=0.3, wspace=0.25)
    
    model = IntegratedPowerModel()
    
    # 模拟24小时功耗数据
    hours = 24
    resolution = 60  # 每分钟
    t_total = np.linspace(0, hours * 3600, hours * resolution)
    
    # 生成各组件功耗
    network_power = np.zeros(len(t_total))
    bluetooth_power = np.zeros(len(t_total))
    background_power = np.zeros(len(t_total))
    
    for i, t in enumerate(t_total):
        hour = (t / 3600) % 24
        
        # 基于时间的使用模式
        if 0 <= hour < 7:  # 睡眠
            network_power[i] = model.network.wifi_power(0, -70, False) * 1000
            bluetooth_power[i] = model.bluetooth.get_power(BluetoothMode.STANDBY) * 1000
        elif 7 <= hour < 9:  # 早高峰
            network_power[i] = model.network.lte_power(15, -75, NetworkState.ACTIVE_RX) * 1000
            bluetooth_power[i] = model.bluetooth.bt_audio_power('AAC') * 1000
        elif 9 <= hour < 12:  # 工作
            network_power[i] = model.network.wifi_power(5, -55, False) * 1000
            bluetooth_power[i] = model.bluetooth.get_power(BluetoothMode.CONNECTED_IDLE) * 1000
        elif 12 <= hour < 14:  # 午餐
            network_power[i] = model.network.lte_power(20, -70, NetworkState.ACTIVE_RX) * 1000
            bluetooth_power[i] = model.bluetooth.get_power(BluetoothMode.CONNECTED_ACTIVE, data_rate=50) * 1000
        elif 14 <= hour < 18:  # 工作
            network_power[i] = model.network.wifi_power(3, -55, False) * 1000
            bluetooth_power[i] = model.bluetooth.get_power(BluetoothMode.CONNECTED_IDLE) * 1000
        elif 18 <= hour < 22:  # 晚间
            network_power[i] = model.network.wifi_power(10, -50, False) * 1000
            bluetooth_power[i] = model.bluetooth.bt_audio_power('SBC') * 1000
        else:  # 深夜
            network_power[i] = model.network.wifi_power(0, -70, False) * 1000
            bluetooth_power[i] = model.bluetooth.get_power(BluetoothMode.STANDBY) * 1000
        
        background_power[i] = model.background.total_power(t) * 1000
    
    # 重塑为24x60矩阵
    network_matrix = network_power.reshape(hours, resolution)
    bluetooth_matrix = bluetooth_power.reshape(hours, resolution)
    background_matrix = background_power.reshape(hours, resolution)
    total_matrix = network_matrix + bluetooth_matrix + background_matrix + 50
    
    # 1. 总功耗热力图
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS['bg_dark'])
    
    im1 = ax1.imshow(total_matrix, aspect='auto', cmap='hot', origin='lower',
                     extent=[0, 60, 0, 24])
    
    ax1.set_xlabel('Minutes in Hour', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax1.set_ylabel('Hour of Day', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax1.set_title('24-Hour Total Power Consumption Heatmap', fontsize=14, fontweight='bold',
                  color=COLORS['text'], pad=15)
    ax1.tick_params(colors=COLORS['text_muted'])
    
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', shrink=0.8)
    cbar1.set_label('Power (mW)', color=COLORS['text'])
    cbar1.ax.yaxis.set_tick_params(color=COLORS['text'])
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color=COLORS['text'])
    
    # 添加时段标注
    periods = [(0, 7, 'Sleep'), (7, 9, 'Commute'), (9, 12, 'Work'),
               (12, 14, 'Lunch'), (14, 18, 'Work'), (18, 22, 'Evening'), (22, 24, 'Night')]
    for start, end, label in periods:
        ax1.axhline(y=start, color='white', linestyle='-', linewidth=0.5, alpha=0.3)
        ax1.text(62, (start + end) / 2, label, fontsize=8, color=COLORS['text'], va='center')
    
    # 2. 组件分解曲线
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['bg_dark'])
    
    hours_avg = np.arange(24)
    net_avg = network_matrix.mean(axis=1)
    bt_avg = bluetooth_matrix.mean(axis=1)
    bg_avg = background_matrix.mean(axis=1)
    
    ax2.fill_between(hours_avg, 0, net_avg, alpha=0.7, color=COLORS['network'], label='Network')
    ax2.fill_between(hours_avg, net_avg, net_avg + bt_avg, alpha=0.7, 
                     color=COLORS['bluetooth'], label='Bluetooth')
    ax2.fill_between(hours_avg, net_avg + bt_avg, net_avg + bt_avg + bg_avg, 
                     alpha=0.7, color=COLORS['background'], label='Background')
    
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax2.set_ylabel('Power (mW)', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax2.set_title('Hourly Power Breakdown', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    ax2.legend(facecolor=COLORS['bg_card'], edgecolor=COLORS['text_muted'],
               labelcolor=COLORS['text'])
    ax2.tick_params(colors=COLORS['text_muted'])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['text_muted'])
    ax2.spines['bottom'].set_color(COLORS['text_muted'])
    ax2.set_xlim(0, 23)
    
    # 3. 时段功耗饼图
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLORS['bg_dark'])
    
    # 计算各时段能量
    sleep_energy = total_matrix[:7, :].sum()
    active_energy = total_matrix[7:22, :].sum()
    night_energy = total_matrix[22:, :].sum()
    
    energies = [sleep_energy, active_energy, night_energy]
    labels = ['Sleep (0-7h)', 'Active (7-22h)', 'Night (22-24h)']
    colors_pie = ['#1F6FEB', '#F0883E', '#8B949E']
    explode = (0, 0.05, 0)
    
    wedges, texts, autotexts = ax3.pie(energies, explode=explode, labels=labels, 
                                        colors=colors_pie, autopct='%1.1f%%',
                                        textprops={'color': COLORS['text']},
                                        wedgeprops={'edgecolor': COLORS['bg_dark'], 'linewidth': 2})
    
    ax3.set_title('Energy Distribution by Period', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 5. SOC仿真与预测
# ============================================================================

def plot_soc_simulation_analysis(save_path=None):
    """
    SOC仿真分析图
    """
    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg_dark'])
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    model = IntegratedPowerModel()
    
    # 不同场景仿真
    scenarios = [
        ('idle', 'Idle (Screen Off)', COLORS['success']),
        ('music_streaming', 'Music Streaming', COLORS['bluetooth']),
        ('social_browsing', 'Social Browsing', COLORS['lte']),
        ('fitness_tracking', 'Fitness Tracking', COLORS['ble']),
    ]
    
    # 1. 多场景SOC曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(COLORS['bg_dark'])
    
    duration = 4 * 3600  # 4小时
    
    for scenario_name, label, color in scenarios:
        power_func = create_scenario_power_func(model, scenario_name)
        t, soc = model.simulate_soc(duration, power_func, initial_soc=1.0)
        ax1.plot(t / 3600, soc * 100, label=label, color=color, linewidth=2.5)
    
    ax1.axhline(y=20, color=COLORS['warning'], linestyle='--', alpha=0.7, label='Low Battery')
    ax1.axhline(y=5, color=COLORS['danger'], linestyle='--', alpha=0.7, label='Critical')
    
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax1.set_ylabel('State of Charge (%)', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax1.set_title('Battery SOC Simulation by Usage Scenario', fontsize=14, fontweight='bold',
                  color=COLORS['text'], pad=15)
    ax1.legend(loc='upper right', ncol=3, facecolor=COLORS['bg_card'], 
               edgecolor=COLORS['text_muted'], labelcolor=COLORS['text'])
    ax1.tick_params(colors=COLORS['text_muted'])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['text_muted'])
    ax1.spines['bottom'].set_color(COLORS['text_muted'])
    ax1.grid(True, alpha=0.2, color=COLORS['text_muted'])
    ax1.set_ylim(0, 105)
    ax1.set_xlim(0, 4)
    
    # 2. 功耗分解饼图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(COLORS['bg_dark'])
    
    # 计算social_browsing场景的功耗分解
    power_func = create_scenario_power_func(model, 'social_browsing')
    t_sample = np.linspace(0, 600, 100)
    
    net_total = sum(model.network.lte_power(10, -75, NetworkState.ACTIVE_RX) 
                    if (ti % 20) < 5 else model.network.lte_power(0, -75, NetworkState.DRX) 
                    for ti in t_sample) * 1000
    bt_total = sum(model.bluetooth.get_power(BluetoothMode.CONNECTED_IDLE, devices=2) 
                   for _ in t_sample) * 1000
    bg_total = sum(model.background.total_power(ti) for ti in t_sample) * 1000
    base_total = 50 * len(t_sample)
    
    values = [net_total, bt_total, bg_total, base_total]
    labels = ['Network', 'Bluetooth', 'Background', 'Base']
    colors_pie = [COLORS['network'], COLORS['bluetooth'], COLORS['background'], COLORS['text_muted']]
    
    wedges, texts, autotexts = ax2.pie(values, labels=labels, colors=colors_pie,
                                        autopct='%1.1f%%',
                                        textprops={'color': COLORS['text']},
                                        wedgeprops={'edgecolor': COLORS['bg_dark'], 'linewidth': 2})
    ax2.set_title('Power Distribution (Social Browsing)', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    
    # 3. 续航时间预测
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(COLORS['bg_dark'])
    
    scenario_names = ['Idle', 'Music', 'Social', 'Fitness']
    battery_life_hours = []
    
    for scenario_name, _, _ in scenarios:
        power_func = create_scenario_power_func(model, scenario_name)
        # 估算续航时间
        avg_power = np.mean([power_func(t) for t in np.linspace(0, 600, 100)])
        battery_wh = 4500 * 3.85 / 1000  # 17.3Wh
        life_hours = battery_wh / avg_power * 0.95  # 95%可用
        battery_life_hours.append(min(life_hours, 100))  # 限制最大显示
    
    colors_bar = [COLORS['success'], COLORS['bluetooth'], COLORS['lte'], COLORS['ble']]
    bars = ax3.bar(scenario_names, battery_life_hours, color=colors_bar, 
                   edgecolor='white', linewidth=2)
    
    for bar, hours in zip(bars, battery_life_hours):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{hours:.1f}h', ha='center', fontsize=11, fontweight='bold',
                 color=COLORS['text'])
    
    ax3.set_ylabel('Estimated Battery Life (hours)', fontsize=11, fontweight='bold',
                   color=COLORS['text'])
    ax3.set_title('Predicted Battery Life by Scenario', fontsize=12, fontweight='bold',
                  color=COLORS['text'], pad=10)
    ax3.tick_params(colors=COLORS['text_muted'])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(COLORS['text_muted'])
    ax3.spines['bottom'].set_color(COLORS['text_muted'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 6. 综合报告仪表板
# ============================================================================

def plot_comprehensive_dashboard(save_path=None):
    """
    综合仪表板 - 所有关键指标
    """
    fig = plt.figure(figsize=(20, 14), facecolor=COLORS['bg_dark'])
    
    # 标题
    fig.text(0.5, 0.97, 'Smartphone Battery Power Model Dashboard', 
             fontsize=20, fontweight='bold', ha='center', color=COLORS['text'])
    fig.text(0.5, 0.94, 'Integrated Analysis: Network | Bluetooth | Background Tasks',
             fontsize=12, ha='center', color=COLORS['text_muted'])
    
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3,
                  top=0.90, bottom=0.05, left=0.05, right=0.95)
    
    model = IntegratedPowerModel()
    
    # ===== 第一行: 关键指标卡片 =====
    metrics = [
        ('Network Range', '10 - 4500 mW', 'WiFi/LTE/5G', COLORS['network']),
        ('Bluetooth Range', '0.5 - 65 mW', 'BLE/Classic', COLORS['bluetooth']),
        ('Background Avg', '~80 mW', 'System Tasks', COLORS['background']),
        ('Battery Capacity', '4500 mAh', '17.3 Wh', COLORS['battery']),
    ]
    
    for i, (title, value, subtitle, color) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_facecolor(COLORS['bg_dark'])
        
        # 卡片
        card = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                               boxstyle="round,pad=0.1,rounding_size=0.5",
                               facecolor=COLORS['bg_card'], edgecolor=color, linewidth=3)
        ax.add_patch(card)
        
        # 顶部色条
        bar = FancyBboxPatch((0.2, 8.5), 9.6, 1.3,
                              boxstyle="round,pad=0,rounding_size=0.5",
                              facecolor=color, edgecolor='none', alpha=0.8)
        ax.add_patch(bar)
        
        ax.text(5, 9.1, title, fontsize=11, fontweight='bold', ha='center', 
                color='white')
        ax.text(5, 5.5, value, fontsize=16, fontweight='bold', ha='center',
                color=COLORS['text'])
        ax.text(5, 2.5, subtitle, fontsize=10, ha='center', color=COLORS['text_muted'])
    
    # ===== 第二行: 技术对比图 =====
    
    # 网络功耗条形图
    ax_net = fig.add_subplot(gs[1, :2])
    ax_net.set_facecolor(COLORS['bg_dark'])
    
    net_techs = ['WiFi Idle', 'WiFi Active', 'LTE DRX', 'LTE Active', '5G Active']
    net_powers = [10, 650, 150, 1200, 2500]
    net_colors = [COLORS['wifi']] * 2 + [COLORS['lte']] * 2 + [COLORS['nr5g']]
    
    bars = ax_net.barh(net_techs, net_powers, color=net_colors, alpha=0.8,
                       edgecolor='white', linewidth=1)
    
    for bar, power in zip(bars, net_powers):
        ax_net.text(power + 50, bar.get_y() + bar.get_height()/2,
                    f'{power}', va='center', fontsize=10, color=COLORS['text'])
    
    ax_net.set_xlabel('Power (mW)', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax_net.set_title('Network Technology Power', fontsize=12, fontweight='bold',
                     color=COLORS['text'], pad=10)
    ax_net.tick_params(colors=COLORS['text_muted'])
    ax_net.spines['top'].set_visible(False)
    ax_net.spines['right'].set_visible(False)
    ax_net.spines['left'].set_color(COLORS['text_muted'])
    ax_net.spines['bottom'].set_color(COLORS['text_muted'])
    
    # 蓝牙功耗条形图
    ax_bt = fig.add_subplot(gs[1, 2:])
    ax_bt.set_facecolor(COLORS['bg_dark'])
    
    bt_modes = ['BLE Standby', 'BLE Advertising', 'BLE Connected', 'BLE Active', 
                'BT Audio SBC', 'BT Audio LDAC']
    bt_powers = [0.5, 8, 3, 25, 45, 68]
    bt_colors = [COLORS['ble']] * 4 + [COLORS['bt_classic']] * 2
    
    bars = ax_bt.barh(bt_modes, bt_powers, color=bt_colors, alpha=0.8,
                      edgecolor='white', linewidth=1)
    
    for bar, power in zip(bars, bt_powers):
        ax_bt.text(power + 1, bar.get_y() + bar.get_height()/2,
                   f'{power}', va='center', fontsize=10, color=COLORS['text'])
    
    ax_bt.set_xlabel('Power (mW)', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax_bt.set_title('Bluetooth Mode Power', fontsize=12, fontweight='bold',
                    color=COLORS['text'], pad=10)
    ax_bt.tick_params(colors=COLORS['text_muted'])
    ax_bt.spines['top'].set_visible(False)
    ax_bt.spines['right'].set_visible(False)
    ax_bt.spines['left'].set_color(COLORS['text_muted'])
    ax_bt.spines['bottom'].set_color(COLORS['text_muted'])
    
    # ===== 第三行: SOC曲线和方程 =====
    
    # SOC曲线
    ax_soc = fig.add_subplot(gs[2, :2])
    ax_soc.set_facecolor(COLORS['bg_dark'])
    
    scenarios = [
        ('idle', 'Idle', COLORS['success']),
        ('social_browsing', 'Active Use', COLORS['lte']),
    ]
    
    for name, label, color in scenarios:
        power_func = create_scenario_power_func(model, name)
        t, soc = model.simulate_soc(4 * 3600, power_func)
        ax_soc.plot(t / 3600, soc * 100, label=label, color=color, linewidth=2.5)
        ax_soc.fill_between(t / 3600, soc * 100, alpha=0.2, color=color)
    
    ax_soc.axhline(y=20, color=COLORS['warning'], linestyle='--', alpha=0.5)
    ax_soc.set_xlabel('Time (hours)', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax_soc.set_ylabel('SOC (%)', fontsize=11, fontweight='bold', color=COLORS['text'])
    ax_soc.set_title('Battery Discharge Curves', fontsize=12, fontweight='bold',
                     color=COLORS['text'], pad=10)
    ax_soc.legend(facecolor=COLORS['bg_card'], edgecolor=COLORS['text_muted'],
                  labelcolor=COLORS['text'])
    ax_soc.tick_params(colors=COLORS['text_muted'])
    ax_soc.spines['top'].set_visible(False)
    ax_soc.spines['right'].set_visible(False)
    ax_soc.spines['left'].set_color(COLORS['text_muted'])
    ax_soc.spines['bottom'].set_color(COLORS['text_muted'])
    ax_soc.set_ylim(0, 105)
    ax_soc.grid(True, alpha=0.2, color=COLORS['text_muted'])
    
    # 方程卡片
    ax_eq = fig.add_subplot(gs[2, 2:])
    ax_eq.set_xlim(0, 10)
    ax_eq.set_ylim(0, 10)
    ax_eq.axis('off')
    ax_eq.set_facecolor(COLORS['bg_dark'])
    
    # 背景卡片
    card = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.1,rounding_size=0.3",
                           facecolor=COLORS['bg_card'], edgecolor=COLORS['accent'], linewidth=2)
    ax_eq.add_patch(card)
    
    ax_eq.text(5, 9, 'Core Model Equations', fontsize=12, fontweight='bold',
               ha='center', color=COLORS['accent'])
    
    equations = [
        ('SOC:', r'$\frac{dSOC}{dt} = -\frac{I(t)}{Q_{eff}} - k_{sd} \cdot SOC$'),
        ('Total:', r'$P = P_{net} + P_{bt} + P_{bg} + P_{base}$'),
        ('Network:', r'$P_{tx} = P_0 \cdot 10^{(\Delta RSSI \cdot k)/10}$'),
        ('BLE:', r'$P_{avg} = P_{tx} \cdot \frac{T_{event}}{T_{interval}}$'),
        ('CPU:', r'$P_{dyn} = \alpha \cdot C \cdot V^2 \cdot f$'),
    ]
    
    y = 7.5
    for label, eq in equations:
        ax_eq.text(1, y, label, fontsize=10, color=COLORS['text_muted'])
        ax_eq.text(2.5, y, eq, fontsize=11, color=COLORS['text'])
        y -= 1.4
    
    # 底部参考
    fig.text(0.5, 0.01, 'Data Sources: Iontech Battery Repository | Huang et al. (2012) | Bluetooth SIG Core Spec | ARM TRM',
             fontsize=9, ha='center', color=COLORS['text_muted'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=COLORS['bg_dark'])
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 主函数
# ============================================================================

def generate_all_visualizations():
    """生成所有可视化图表"""
    print("="*70)
    print("Iontech集成模型 - 生成创新可视化图表")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/5] 径向条形图 - 技术对比...")
    plot_radial_bar_comparison(os.path.join(script_dir, 'iontech_radial_comparison.png'))
    
    print("[2/5] 等高线图 - 参数空间分析...")
    plot_network_contour_analysis(os.path.join(script_dir, 'iontech_contour_analysis.png'))
    
    print("[3/5] 气泡图 - 场景对比...")
    plot_scenario_bubble_comparison(os.path.join(script_dir, 'iontech_scenario_bubble.png'))
    
    print("[4/5] 时序热力图 - 24小时分布...")
    plot_24h_heatmap(os.path.join(script_dir, 'iontech_24h_heatmap.png'))
    
    print("[5/5] SOC仿真分析...")
    plot_soc_simulation_analysis(os.path.join(script_dir, 'iontech_soc_analysis.png'))
    
    print("\n[Bonus] 综合仪表板...")
    plot_comprehensive_dashboard(os.path.join(script_dir, 'iontech_dashboard.png'))
    
    print("\n" + "="*70)
    print("可视化图表生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_all_visualizations()
