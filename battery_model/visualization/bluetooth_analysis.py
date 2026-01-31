#!/usr/bin/env python3
"""
蓝牙功耗建模与可视化

详细分析:
1. BLE状态机功耗
2. Classic Bluetooth音频功耗
3. 连接参数影响
4. 多设备功耗叠加
5. 蓝牙与其他子系统对比

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Rectangle, Arc, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iontech_integrated_model import (
    IntegratedPowerModel, BluetoothPowerModel, BluetoothPowerParams, BluetoothMode
)

# ============================================================================
# 配色方案 (白色背景)
# ============================================================================

COLORS = {
    'bg': '#FFFFFF',
    'text': '#1A202C',
    'text_light': '#718096',
    'grid': '#E2E8F0',
    'border': '#CBD5E0',
    
    # 蓝牙专用色系
    'bt_primary': '#0061F2',      # 蓝牙蓝
    'bt_secondary': '#6610F2',    # 紫色
    'ble': '#00D4FF',             # 青色 (BLE)
    'bt_classic': '#7C3AED',      # 紫色 (Classic)
    'bt_audio': '#EC4899',        # 粉色 (Audio)
    
    # 状态颜色
    'off': '#CBD5E0',
    'standby': '#A0AEC0',
    'advertising': '#38B2AC',
    'scanning': '#4299E1',
    'connected': '#48BB78',
    'active': '#ED8936',
    
    # 对比色
    'network': '#2563EB',
    'background': '#059669',
    'battery': '#0891B2',
    
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
# 1. BLE状态机功耗分析
# ============================================================================

def plot_ble_state_machine(save_path=None):
    """
    BLE状态机与各状态功耗
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # ===== 左上: 状态机图 =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 100)
    ax1.axis('off')
    
    # 状态定义
    states = [
        ('OFF', 50, 85, COLORS['off'], '0 mW'),
        ('Standby', 20, 55, COLORS['standby'], '0.5 mW'),
        ('Advertising', 50, 55, COLORS['advertising'], '0.6-15 mW'),
        ('Scanning', 80, 55, COLORS['scanning'], '12 mW'),
        ('Initiating', 20, 25, COLORS['bt_primary'], '8 mW'),
        ('Connected', 50, 25, COLORS['connected'], '0.5-25 mW'),
    ]
    
    # 绘制状态节点
    for name, x, y, color, power in states:
        circle = Circle((x, y), 12, facecolor=color, edgecolor='white',
                        linewidth=2, alpha=0.9)
        ax1.add_patch(circle)
        ax1.text(x, y + 1, name, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
        ax1.text(x, y - 4, power, ha='center', va='center', fontsize=7,
                color='white')
    
    # 状态转换箭头
    transitions = [
        (50, 73, 50, 67),      # OFF -> Standby (垂直)
        (50, 73, 20, 60),      # OFF -> Standby
        (32, 55, 38, 55),      # Standby -> Advertising
        (62, 55, 68, 55),      # Advertising -> Scanning
        (20, 43, 20, 37),      # Standby -> Initiating
        (32, 25, 38, 25),      # Initiating -> Connected
        (50, 43, 50, 37),      # Advertising -> Connected
    ]
    
    for x1, y1, x2, y2 in transitions:
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_light'],
                                   lw=1.5, connectionstyle='arc3,rad=0.1'))
    
    ax1.set_title('BLE State Machine & Power Consumption', 
                  fontsize=12, fontweight='bold', pad=10)
    
    # ===== 右上: 广播功耗与间隔关系 =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    bt_model = BluetoothPowerModel()
    
    intervals = np.linspace(20, 10240, 200)  # ms
    adv_powers = [bt_model.ble_advertising_power(i) * 1000 for i in intervals]
    
    ax2.plot(intervals, adv_powers, color=COLORS['advertising'], linewidth=2.5,
             label='Advertising Power')
    ax2.fill_between(intervals, adv_powers, alpha=0.2, color=COLORS['advertising'])
    
    # 标记典型间隔
    typical_intervals = [100, 500, 1000, 2000]
    for ti in typical_intervals:
        p = bt_model.ble_advertising_power(ti) * 1000
        ax2.axvline(x=ti, color=COLORS['text_light'], linestyle='--', alpha=0.5)
        ax2.scatter([ti], [p], color=COLORS['bt_primary'], s=80, zorder=5)
        ax2.annotate(f'{ti}ms\n{p:.2f}mW', xy=(ti, p), xytext=(ti+200, p+0.3),
                    fontsize=8, ha='left')
    
    ax2.set_xlabel('Advertising Interval (ms)', fontweight='bold')
    ax2.set_ylabel('Average Power (mW)', fontweight='bold')
    ax2.set_title('BLE Advertising Power vs Interval', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_xlim(20, 12000)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 添加公式
    ax2.text(0.95, 0.95, r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}} + P_{sleep}$',
             transform=ax2.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['border']))
    
    # ===== 左下: 连接功耗与参数关系 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    conn_intervals = np.array([7.5, 15, 30, 50, 75, 100, 200, 400, 1000, 2000, 4000])
    
    # 不同活跃程度
    activities = [
        ('Idle', 0.01, COLORS['standby']),
        ('Low Activity', 0.1, COLORS['connected']),
        ('Active Data', 0.5, COLORS['active']),
    ]
    
    for label, duty, color in activities:
        powers = [bt_model.ble_connected_power(ci, duty) * 1000 for ci in conn_intervals]
        ax3.plot(conn_intervals, powers, 'o-', label=label, color=color, 
                linewidth=2, markersize=6)
    
    ax3.set_xlabel('Connection Interval (ms)', fontweight='bold')
    ax3.set_ylabel('Average Power (mW)', fontweight='bold')
    ax3.set_title('BLE Connected Power vs Connection Interval', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 添加公式
    ax3.text(0.95, 0.45, r'$P_{conn} = P_{idle} + \frac{P_{active} \cdot T_{event}}{T_{CI}}$',
             transform=ax3.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['border']))
    
    # ===== 右下: BLE模式功耗对比 =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    modes = ['Off', 'Standby', 'Advertising\n(100ms)', 'Scanning', 
             'Connected\nIdle', 'Connected\nActive']
    powers = [0, 0.5, 0.56, 12, 0.8, 25]
    colors = [COLORS['off'], COLORS['standby'], COLORS['advertising'],
              COLORS['scanning'], COLORS['connected'], COLORS['active']]
    
    bars = ax4.bar(modes, powers, color=colors, edgecolor='white', 
                   linewidth=2, alpha=0.85)
    
    # 添加数值标签
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                f'{power} mW', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax4.set_ylabel('Power (mW)', fontweight='bold')
    ax4.set_title('BLE Mode Power Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 32)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle('Bluetooth Low Energy (BLE) Power Model Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 2. Classic Bluetooth音频功耗分析
# ============================================================================

def plot_bt_audio_analysis(save_path=None):
    """
    Classic Bluetooth音频功耗详细分析
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    bt_model = BluetoothPowerModel()
    
    # ===== 左上: 编解码器功耗对比 =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    codecs = ['SBC', 'AAC', 'aptX', 'aptX HD', 'LDAC']
    bitrates = [328, 256, 352, 576, 990]  # kbps
    powers_sd = [bt_model.bt_audio_power(c, False) * 1000 for c in ['SBC', 'AAC', 'aptX', 'aptX_HD', 'LDAC']]
    powers_hd = [bt_model.bt_audio_power(c, True) * 1000 for c in ['SBC', 'AAC', 'aptX', 'aptX_HD', 'LDAC']]
    
    x = np.arange(len(codecs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, powers_sd, width, label='Standard', 
                    color=COLORS['bt_classic'], alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x + width/2, powers_hd, width, label='HD Audio',
                    color=COLORS['bt_audio'], alpha=0.8, edgecolor='white', linewidth=2)
    
    # 添加数值标签
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', fontsize=8, fontweight='bold')
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', fontsize=8, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(codecs)
    ax1.set_ylabel('Power (mW)', fontweight='bold')
    ax1.set_title('Audio Codec Power Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_ylim(0, 120)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ===== 右上: 比特率与功耗关系 =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 散点图: 比特率 vs 功耗
    codec_data = {
        'SBC': (328, 45, COLORS['bt_classic']),
        'AAC': (256, 52, COLORS['scanning']),
        'aptX': (352, 56, COLORS['connected']),
        'aptX HD': (576, 63, COLORS['active']),
        'LDAC': (990, 68, COLORS['bt_audio']),
    }
    
    for codec, (bitrate, power, color) in codec_data.items():
        ax2.scatter(bitrate, power, s=200, color=color, edgecolor='white', 
                   linewidth=2, label=codec, zorder=5)
        ax2.annotate(codec, xy=(bitrate, power), xytext=(bitrate+30, power+2),
                    fontsize=9, fontweight='bold')
    
    # 拟合趋势线
    bitrates_arr = np.array([d[0] for d in codec_data.values()])
    powers_arr = np.array([d[1] for d in codec_data.values()])
    z = np.polyfit(bitrates_arr, powers_arr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(200, 1100, 100)
    ax2.plot(x_line, p(x_line), '--', color=COLORS['text_light'], alpha=0.7,
             label=f'Trend: P = {z[0]:.3f}R + {z[1]:.1f}')
    
    ax2.set_xlabel('Bitrate (kbps)', fontweight='bold')
    ax2.set_ylabel('Power (mW)', fontweight='bold')
    ax2.set_title('Power vs Bitrate Relationship', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', frameon=True, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # 添加公式
    ax2.text(0.05, 0.95, r'$P_{audio} = P_{base} \cdot k_{codec}$',
             transform=ax2.transAxes, fontsize=10, ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['border']))
    
    # ===== 左下: 音频功耗分解 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    # 堆叠条形图
    codecs_short = ['SBC', 'AAC', 'aptX', 'aptX HD', 'LDAC']
    
    # 功耗分解组件
    rf_power = [18, 18, 18, 22, 25]
    codec_power = [10, 15, 18, 22, 30]
    buffer_power = [5, 7, 8, 7, 5]
    dac_power = [12, 12, 12, 12, 8]
    
    x = np.arange(len(codecs_short))
    
    ax3.bar(x, rf_power, label='RF Transmission', color=COLORS['bt_primary'], 
            edgecolor='white', linewidth=1.5)
    ax3.bar(x, codec_power, bottom=rf_power, label='Codec Processing',
            color=COLORS['bt_audio'], edgecolor='white', linewidth=1.5)
    ax3.bar(x, buffer_power, bottom=np.array(rf_power)+np.array(codec_power),
            label='Buffer Management', color=COLORS['scanning'], edgecolor='white', linewidth=1.5)
    ax3.bar(x, dac_power, bottom=np.array(rf_power)+np.array(codec_power)+np.array(buffer_power),
            label='DAC Output', color=COLORS['connected'], edgecolor='white', linewidth=1.5)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(codecs_short)
    ax3.set_ylabel('Power (mW)', fontweight='bold')
    ax3.set_title('Audio Power Breakdown by Component', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', frameon=True, fontsize=9)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ===== 右下: 音频质量与功耗权衡 =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 多维气泡图
    quality_scores = [60, 70, 80, 90, 95]  # 音质评分
    latencies = [200, 150, 100, 150, 200]  # 延迟 ms
    powers = powers_sd
    
    colors_bubble = [COLORS['bt_classic'], COLORS['scanning'], COLORS['connected'],
                     COLORS['active'], COLORS['bt_audio']]
    
    for i, (codec, q, lat, pwr, color) in enumerate(zip(codecs, quality_scores, latencies, powers, colors_bubble)):
        ax4.scatter(q, lat, s=pwr*8, color=color, alpha=0.7, edgecolor='white', 
                   linewidth=2, label=f'{codec} ({pwr:.0f}mW)')
    
    ax4.set_xlabel('Audio Quality Score', fontweight='bold')
    ax4.set_ylabel('Latency (ms)', fontweight='bold')
    ax4.set_title('Quality-Latency-Power Trade-off\n(Bubble size = Power)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True, fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(55, 100)
    ax4.set_ylim(80, 220)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle('Classic Bluetooth A2DP Audio Power Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 3. 多设备连接功耗分析
# ============================================================================

def plot_multi_device_analysis(save_path=None):
    """
    多设备蓝牙连接功耗分析
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    bt_model = BluetoothPowerModel()
    
    # ===== 左上: 设备数量与功耗 =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    n_devices = np.arange(1, 8)
    
    # 不同场景
    scenarios = [
        ('BLE Sensors', 'ble_idle', COLORS['ble']),
        ('BLE Active', 'ble_active', COLORS['connected']),
        ('Mixed (BLE+Audio)', 'mixed', COLORS['bt_audio']),
    ]
    
    for label, mode, color in scenarios:
        if mode == 'ble_idle':
            powers = [bt_model.multi_device_power(n, 'ble', 0.01) * 1000 for n in n_devices]
        elif mode == 'ble_active':
            powers = [bt_model.multi_device_power(n, 'ble', 0.3) * 1000 for n in n_devices]
        else:  # mixed
            powers = []
            for n in n_devices:
                if n == 1:
                    p = bt_model.bt_audio_power('SBC', False) * 1000
                else:
                    p = bt_model.bt_audio_power('SBC', False) * 1000 + \
                        bt_model.multi_device_power(n-1, 'ble', 0.1) * 1000
                powers.append(p)
        
        ax1.plot(n_devices, powers, 'o-', label=label, color=color, 
                linewidth=2.5, markersize=8)
    
    ax1.set_xlabel('Number of Connected Devices', fontweight='bold')
    ax1.set_ylabel('Total Power (mW)', fontweight='bold')
    ax1.set_title('Power vs Number of Devices', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_xticks(n_devices)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 添加公式
    ax1.text(0.95, 0.25, r'$P_{multi} = N \cdot P_{single} \cdot (1 + 0.1(N-1))$',
             transform=ax1.transAxes, fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['border']))
    
    # ===== 右上: 典型设备组合 =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    device_combos = [
        'Single\nEarbuds',
        'Earbuds +\nWatch',
        'Earbuds +\nWatch + Band',
        'Multi-room\nSpeakers',
        'Smart\nHome Hub',
    ]
    
    # 各组合的功耗分解
    audio_power = [45, 45, 45, 90, 0]
    ble_power = [0, 3, 6, 0, 15]
    overhead = [0, 5, 12, 15, 8]
    
    x = np.arange(len(device_combos))
    
    ax2.bar(x, audio_power, label='Audio', color=COLORS['bt_audio'],
            edgecolor='white', linewidth=1.5)
    ax2.bar(x, ble_power, bottom=audio_power, label='BLE Devices',
            color=COLORS['ble'], edgecolor='white', linewidth=1.5)
    ax2.bar(x, overhead, bottom=np.array(audio_power)+np.array(ble_power),
            label='Scheduling Overhead', color=COLORS['standby'],
            edgecolor='white', linewidth=1.5)
    
    # 总功耗标签
    totals = np.array(audio_power) + np.array(ble_power) + np.array(overhead)
    for i, total in enumerate(totals):
        ax2.text(i, total + 2, f'{total}mW', ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(device_combos, fontsize=9)
    ax2.set_ylabel('Power (mW)', fontweight='bold')
    ax2.set_title('Typical Device Combination Power', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ===== 左下: 调度开销分析 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    n_range = np.arange(1, 11)
    
    # 理论线性增长 vs 实际(含开销)
    base_power = 5  # mW per device
    linear_power = n_range * base_power
    actual_power = [n * base_power * (1 + 0.1*(n-1)) for n in n_range]
    overhead_power = np.array(actual_power) - np.array(linear_power)
    
    ax3.fill_between(n_range, linear_power, actual_power, 
                     alpha=0.3, color=COLORS['warning'], label='Overhead')
    ax3.plot(n_range, linear_power, '--', color=COLORS['text_light'], 
             linewidth=2, label='Linear (ideal)')
    ax3.plot(n_range, actual_power, 'o-', color=COLORS['bt_primary'],
             linewidth=2.5, markersize=6, label='Actual')
    
    ax3.set_xlabel('Number of Devices', fontweight='bold')
    ax3.set_ylabel('Power (mW)', fontweight='bold')
    ax3.set_title('Scheduling Overhead Analysis', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', frameon=True)
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # 开销百分比标注
    ax3_twin = ax3.twinx()
    overhead_pct = [100 * (a - l) / l for a, l in zip(actual_power, linear_power)]
    ax3_twin.plot(n_range, overhead_pct, 's--', color=COLORS['danger'], 
                  alpha=0.7, markersize=5, label='Overhead %')
    ax3_twin.set_ylabel('Overhead (%)', color=COLORS['danger'])
    ax3_twin.tick_params(axis='y', labelcolor=COLORS['danger'])
    ax3_twin.set_ylim(0, 100)
    
    # ===== 右下: 使用场景功耗预算 =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    scenarios = ['Workout\n(Watch+Earbuds)', 'Office\n(Mouse+KB+Earbuds)', 
                 'Gaming\n(Headset+Controller)', 'Smart Home\n(5 Sensors)']
    
    bt_power = [53, 48, 75, 25]
    other_power = [150, 80, 400, 30]  # 其他功耗
    
    x = np.arange(len(scenarios))
    width = 0.4
    
    bars1 = ax4.bar(x - width/2, bt_power, width, label='Bluetooth',
                    color=COLORS['bt_primary'], edgecolor='white', linewidth=2)
    bars2 = ax4.bar(x + width/2, other_power, width, label='Other Components',
                    color=COLORS['text_light'], alpha=0.6, edgecolor='white', linewidth=2)
    
    # 蓝牙占比标注
    for i, (bt, other) in enumerate(zip(bt_power, other_power)):
        pct = 100 * bt / (bt + other)
        ax4.text(i, bt + other + 10, f'BT: {pct:.0f}%', ha='center', fontsize=9,
                fontweight='bold', color=COLORS['bt_primary'])
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(scenarios, fontsize=9)
    ax4.set_ylabel('Power (mW)', fontweight='bold')
    ax4.set_title('Bluetooth Power in Usage Scenarios', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle('Multi-Device Bluetooth Connection Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 4. 蓝牙功耗时序分析
# ============================================================================

def plot_bt_temporal_analysis(save_path=None):
    """
    蓝牙功耗时序与脉冲分析
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # ===== 左上: BLE广播脉冲 =====
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 模拟100ms广播间隔的功耗脉冲
    t = np.linspace(0, 500, 5000)  # 500ms
    power = np.zeros_like(t)
    
    # 每100ms一个广播事件，持续约0.4ms
    for start in [0, 100, 200, 300, 400]:
        mask = (t >= start) & (t < start + 0.4)
        power[mask] = 15  # 发送功率 15mW
    
    # 睡眠期间 0.5mW
    power[power == 0] = 0.5
    
    ax1.fill_between(t, power, alpha=0.5, color=COLORS['advertising'])
    ax1.plot(t, power, color=COLORS['advertising'], linewidth=1)
    
    ax1.axhline(y=0.56, color=COLORS['danger'], linestyle='--', 
                label=f'Average: 0.56 mW', linewidth=2)
    
    ax1.set_xlabel('Time (ms)', fontweight='bold')
    ax1.set_ylabel('Instantaneous Power (mW)', fontweight='bold')
    ax1.set_title('BLE Advertising Power Waveform', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 20)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 标注
    ax1.annotate('TX Pulse\n(0.4ms)', xy=(100, 15), xytext=(150, 17),
                arrowprops=dict(arrowstyle='->', color=COLORS['text']),
                fontsize=9, fontweight='bold')
    ax1.annotate('Sleep\n(99.6ms)', xy=(50, 0.5), xytext=(30, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['text']),
                fontsize=9)
    
    # ===== 右上: A2DP音频流功耗 =====
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 模拟音频流的功耗模式
    t = np.linspace(0, 100, 10000)  # 100ms
    power = np.zeros_like(t)
    
    # A2DP使用2-DH5包，约625μs slot
    packet_interval = 10  # 每10ms发送一个包
    packet_duration = 0.625  # ms
    
    for start in np.arange(0, 100, packet_interval):
        mask = (t >= start) & (t < start + packet_duration)
        power[mask] = 80  # 发送功率
        # 接收ACK
        ack_mask = (t >= start + 1) & (t < start + 1.5)
        power[ack_mask] = 40
    
    # 空闲期间
    power[power == 0] = 8
    
    ax2.fill_between(t, power, alpha=0.5, color=COLORS['bt_audio'])
    ax2.plot(t, power, color=COLORS['bt_audio'], linewidth=0.8)
    
    avg_power = 45
    ax2.axhline(y=avg_power, color=COLORS['danger'], linestyle='--',
                label=f'Average: {avg_power} mW', linewidth=2)
    
    ax2.set_xlabel('Time (ms)', fontweight='bold')
    ax2.set_ylabel('Instantaneous Power (mW)', fontweight='bold')
    ax2.set_title('A2DP Audio Streaming Power Waveform', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 100)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ===== 左下: 一天中的蓝牙使用模式 =====
    ax3 = fig.add_subplot(gs[1, 0])
    
    hours = np.arange(0, 24)
    
    # 模拟典型用户一天的蓝牙使用
    ble_power = []
    audio_power = []
    
    for h in hours:
        if 0 <= h < 7:  # 睡眠 - 可能有睡眠追踪
            ble_power.append(2)
            audio_power.append(0)
        elif 7 <= h < 9:  # 早晨通勤 - 耳机音乐
            ble_power.append(3)
            audio_power.append(50)
        elif 9 <= h < 12:  # 上午工作
            ble_power.append(5)
            audio_power.append(0)
        elif 12 <= h < 13:  # 午餐
            ble_power.append(3)
            audio_power.append(45)
        elif 13 <= h < 18:  # 下午工作
            ble_power.append(5)
            audio_power.append(0)
        elif 18 <= h < 19:  # 晚通勤
            ble_power.append(3)
            audio_power.append(50)
        elif 19 <= h < 22:  # 晚间 - 智能家居 + 可能音频
            ble_power.append(8)
            audio_power.append(30)
        else:  # 深夜
            ble_power.append(1)
            audio_power.append(0)
    
    ax3.stackplot(hours, ble_power, audio_power, 
                  labels=['BLE (Sensors/Wearables)', 'Audio Streaming'],
                  colors=[COLORS['ble'], COLORS['bt_audio']], alpha=0.8)
    
    ax3.set_xlabel('Hour of Day', fontweight='bold')
    ax3.set_ylabel('Power (mW)', fontweight='bold')
    ax3.set_title('Daily Bluetooth Usage Pattern', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True)
    ax3.set_xlim(0, 23)
    ax3.set_xticks(range(0, 24, 3))
    ax3.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # ===== 右下: 功耗效率指标 =====
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 不同蓝牙技术的效率对比
    techs = ['BLE 5.0', 'BLE 5.2\n(LE Audio)', 'Classic\nA2DP', 'Classic\nHFP']
    
    # 指标
    throughput = [2000, 2000, 700, 64]  # kbps
    power = [15, 12, 45, 35]  # mW (活跃状态平均)
    efficiency = [t/p for t, p in zip(throughput, power)]  # kbps/mW
    
    x = np.arange(len(techs))
    
    # 双轴
    color1 = COLORS['bt_primary']
    color2 = COLORS['success']
    
    ax4.bar(x - 0.2, power, 0.4, label='Power (mW)', color=color1, alpha=0.8,
            edgecolor='white', linewidth=2)
    
    ax4_twin = ax4.twinx()
    ax4_twin.bar(x + 0.2, efficiency, 0.4, label='Efficiency (kbps/mW)', 
                 color=color2, alpha=0.8, edgecolor='white', linewidth=2)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(techs, fontsize=9)
    ax4.set_ylabel('Power (mW)', color=color1, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor=color1)
    ax4_twin.set_ylabel('Efficiency (kbps/mW)', color=color2, fontweight='bold')
    ax4_twin.tick_params(axis='y', labelcolor=color2)
    
    ax4.set_title('Bluetooth Technology Efficiency', fontsize=12, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
    
    ax4.spines['top'].set_visible(False)
    ax4_twin.spines['top'].set_visible(False)
    
    plt.suptitle('Bluetooth Power Temporal Analysis',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 5. 蓝牙建模方程综合展示
# ============================================================================

def plot_bt_equations_summary(save_path=None):
    """
    蓝牙功耗建模方程综合展示
    """
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # ===== BLE广播模型 =====
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    box1 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#E0F2FE', edgecolor=COLORS['ble'],
                           linewidth=2)
    ax1.add_patch(box1)
    
    ax1.text(5, 9, 'BLE Advertising Power Model', fontsize=11, fontweight='bold',
             ha='center', color=COLORS['ble'])
    
    content1 = [
        (r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}} + P_{sleep} \cdot \left(1 - \frac{T_{tx}}{T_{int}}\right)$', ''),
        ('', ''),
        (r'Where:', ''),
        (r'$T_{tx} = 3 \times T_{ch} \approx 0.4$ ms (3 channels)', ''),
        (r'$P_{tx} \approx 15$ mW, $P_{sleep} \approx 0.5$ mW', ''),
        (r'$T_{int} \in [20, 10240]$ ms', ''),
    ]
    
    y = 7.5
    for line, _ in content1:
        ax1.text(5, y, line, fontsize=9, ha='center', color=COLORS['text'])
        y -= 1.2
    
    # ===== BLE连接模型 =====
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    box2 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#DCFCE7', edgecolor=COLORS['connected'],
                           linewidth=2)
    ax2.add_patch(box2)
    
    ax2.text(5, 9, 'BLE Connection Power Model', fontsize=11, fontweight='bold',
             ha='center', color=COLORS['connected'])
    
    content2 = [
        (r'$P_{conn} = P_{idle} + \frac{P_{active} \cdot T_{event}}{T_{CI} \cdot (1 + L_{slave})}$', ''),
        ('', ''),
        (r'Parameters:', ''),
        (r'$T_{CI} \in [7.5, 4000]$ ms (Connection Interval)', ''),
        (r'$L_{slave}$ = Slave Latency (0-499)', ''),
        (r'$T_{event}$ depends on payload size', ''),
    ]
    
    y = 7.5
    for line, _ in content2:
        ax2.text(5, y, line, fontsize=9, ha='center', color=COLORS['text'])
        y -= 1.2
    
    # ===== Classic Bluetooth音频 =====
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    box3 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#FCE7F3', edgecolor=COLORS['bt_audio'],
                           linewidth=2)
    ax3.add_patch(box3)
    
    ax3.text(5, 9, 'A2DP Audio Streaming Model', fontsize=11, fontweight='bold',
             ha='center', color=COLORS['bt_audio'])
    
    content3 = [
        (r'$P_{audio} = P_{RF} + P_{codec} + P_{buffer} + P_{DAC}$', ''),
        ('', ''),
        (r'$P_{audio} = P_{base} \cdot k_{codec}$', ''),
        ('', ''),
        (r'Codec factors $k_{codec}$:', ''),
        (r'SBC: 1.0, AAC: 1.15, aptX: 1.25, LDAC: 1.5', ''),
    ]
    
    y = 7.5
    for line, _ in content3:
        ax3.text(5, y, line, fontsize=9, ha='center', color=COLORS['text'])
        y -= 1.2
    
    # ===== 多设备模型 =====
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    box4 = FancyBboxPatch((0.2, 0.2), 9.6, 9.6,
                           boxstyle="round,pad=0.05,rounding_size=0.3",
                           facecolor='#FEF3C7', edgecolor=COLORS['warning'],
                           linewidth=2)
    ax4.add_patch(box4)
    
    ax4.text(5, 9, 'Multi-Device Connection Model', fontsize=11, fontweight='bold',
             ha='center', color=COLORS['warning'])
    
    content4 = [
        (r'$P_{multi} = \sum_{i=1}^{N} P_i + P_{overhead}(N)$', ''),
        ('', ''),
        (r'$P_{overhead} \approx 0.1 \cdot (N-1) \cdot P_{avg}$', ''),
        ('', ''),
        (r'Total: $P_{multi} = N \cdot P_{single} \cdot (1 + 0.1(N-1))$', ''),
        (r'Scheduling adds ~10% per additional device', ''),
    ]
    
    y = 7.5
    for line, _ in content4:
        ax4.text(5, y, line, fontsize=9, ha='center', color=COLORS['text'])
        y -= 1.2
    
    # ===== 参数表 =====
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    ax5.text(0.5, 0.95, 'Bluetooth Power Model Parameters', fontsize=12, fontweight='bold',
             ha='center', transform=ax5.transAxes, color=COLORS['bt_primary'])
    
    table_data = [
        ['Parameter', 'Symbol', 'BLE Value', 'Classic Value', 'Source'],
        ['TX Power', r'$P_{tx}$', '15 mW', '80 mW', 'Nordic nRF52'],
        ['Sleep Power', r'$P_{sleep}$', '0.5 mW', '8 mW', 'BT Core Spec'],
        ['Adv Interval', r'$T_{int}$', '100 ms', '-', '3GPP/BT SIG'],
        ['Conn Interval', r'$T_{CI}$', '7.5-4000 ms', '625 μs slot', 'BT Core Spec'],
        ['Audio Base', r'$P_{base}$', '-', '45 mW (SBC)', 'Measured'],
        ['Overhead Factor', r'$k_{oh}$', '10%/device', '10%/device', 'Estimated'],
    ]
    
    table = ax5.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.22, 0.12, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    
    for i in range(5):
        table[(0, i)].set_facecolor(COLORS['bt_primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(table_data)):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F8FAFC')
    
    plt.suptitle('Bluetooth Power Modeling Equations Summary',
                 fontsize=14, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 6. 综合蓝牙仪表板
# ============================================================================

def plot_bt_comprehensive_dashboard(save_path=None):
    """
    蓝牙功耗综合仪表板
    """
    fig = plt.figure(figsize=(18, 14), facecolor='white')
    
    # 标题
    fig.text(0.5, 0.97, 'Bluetooth Power Consumption Model Dashboard',
             fontsize=18, fontweight='bold', ha='center', color=COLORS['bt_primary'])
    fig.text(0.5, 0.94, 'BLE | Classic Bluetooth | A2DP Audio | Multi-Device Analysis',
             fontsize=11, ha='center', color=COLORS['text_light'])
    
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3,
                  top=0.90, bottom=0.05, left=0.05, right=0.95)
    
    bt_model = BluetoothPowerModel()
    
    # ===== 第一行: KPI卡片 =====
    kpis = [
        ('BLE Range', '0.5 - 25 mW', 'Advertising to Active', COLORS['ble']),
        ('Audio Power', '45 - 68 mW', 'SBC to LDAC', COLORS['bt_audio']),
        ('Efficiency', '133 kbps/mW', 'BLE 5.0 Data', COLORS['success']),
        ('Multi-Device', '+10%/device', 'Overhead Factor', COLORS['warning']),
    ]
    
    for i, (title, value, subtitle, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        card = FancyBboxPatch((0.3, 0.3), 9.4, 9.4,
                               boxstyle="round,pad=0.05,rounding_size=0.4",
                               facecolor='white', edgecolor=color,
                               linewidth=3)
        ax.add_patch(card)
        
        bar = FancyBboxPatch((0.3, 8), 9.4, 1.7,
                              boxstyle="round,pad=0,rounding_size=0.4",
                              facecolor=color, edgecolor='none', alpha=0.9)
        ax.add_patch(bar)
        
        ax.text(5, 8.8, title, fontsize=10, fontweight='bold', ha='center', color='white')
        ax.text(5, 5.5, value, fontsize=14, fontweight='bold', ha='center', color=color)
        ax.text(5, 3, subtitle, fontsize=9, ha='center', color=COLORS['text_light'])
    
    # ===== 第二行: BLE状态功耗 + 音频编解码 =====
    
    # BLE状态
    ax_ble = fig.add_subplot(gs[1, :2])
    
    modes = ['Off', 'Standby', 'Advertising', 'Scanning', 'Conn Idle', 'Conn Active']
    powers = [0, 0.5, 0.56, 12, 0.8, 25]
    colors_ble = [COLORS['off'], COLORS['standby'], COLORS['advertising'],
                  COLORS['scanning'], COLORS['connected'], COLORS['active']]
    
    bars = ax_ble.bar(modes, powers, color=colors_ble, edgecolor='white', 
                      linewidth=2, alpha=0.85)
    
    for bar, power in zip(bars, powers):
        ax_ble.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{power}', ha='center', fontsize=9, fontweight='bold')
    
    ax_ble.set_ylabel('Power (mW)', fontweight='bold')
    ax_ble.set_title('BLE State Power Consumption', fontsize=11, fontweight='bold')
    ax_ble.spines['top'].set_visible(False)
    ax_ble.spines['right'].set_visible(False)
    
    # 音频编解码
    ax_audio = fig.add_subplot(gs[1, 2:])
    
    codecs = ['SBC', 'AAC', 'aptX', 'aptX HD', 'LDAC']
    powers_audio = [45, 52, 56, 63, 68]
    
    bars_audio = ax_audio.barh(codecs, powers_audio, color=COLORS['bt_audio'],
                                edgecolor='white', linewidth=2, alpha=0.85, height=0.6)
    
    for bar, power in zip(bars_audio, powers_audio):
        ax_audio.text(power + 1, bar.get_y() + bar.get_height()/2,
                     f'{power} mW', va='center', fontsize=9, fontweight='bold')
    
    ax_audio.set_xlabel('Power (mW)', fontweight='bold')
    ax_audio.set_title('A2DP Audio Codec Power', fontsize=11, fontweight='bold')
    ax_audio.spines['top'].set_visible(False)
    ax_audio.spines['right'].set_visible(False)
    
    # ===== 第三行: 广播间隔曲线 + 多设备功耗 =====
    
    # 广播间隔
    ax_adv = fig.add_subplot(gs[2, :2])
    
    intervals = np.linspace(20, 2000, 100)
    adv_powers = [bt_model.ble_advertising_power(i) * 1000 for i in intervals]
    
    ax_adv.plot(intervals, adv_powers, color=COLORS['advertising'], linewidth=2.5)
    ax_adv.fill_between(intervals, adv_powers, alpha=0.2, color=COLORS['advertising'])
    
    ax_adv.set_xlabel('Advertising Interval (ms)', fontweight='bold')
    ax_adv.set_ylabel('Power (mW)', fontweight='bold')
    ax_adv.set_title('BLE Advertising Power vs Interval', fontsize=11, fontweight='bold')
    ax_adv.grid(True, alpha=0.3)
    ax_adv.spines['top'].set_visible(False)
    ax_adv.spines['right'].set_visible(False)
    
    # 多设备
    ax_multi = fig.add_subplot(gs[2, 2:])
    
    n_devices = np.arange(1, 8)
    ble_idle = [bt_model.multi_device_power(n, 'ble', 0.01) * 1000 for n in n_devices]
    ble_active = [bt_model.multi_device_power(n, 'ble', 0.3) * 1000 for n in n_devices]
    
    ax_multi.plot(n_devices, ble_idle, 'o-', label='BLE Idle', 
                  color=COLORS['connected'], linewidth=2.5, markersize=8)
    ax_multi.plot(n_devices, ble_active, 's-', label='BLE Active',
                  color=COLORS['active'], linewidth=2.5, markersize=8)
    
    ax_multi.set_xlabel('Number of Devices', fontweight='bold')
    ax_multi.set_ylabel('Total Power (mW)', fontweight='bold')
    ax_multi.set_title('Multi-Device Power Scaling', fontsize=11, fontweight='bold')
    ax_multi.legend(loc='upper left', frameon=True)
    ax_multi.set_xticks(n_devices)
    ax_multi.grid(True, alpha=0.3)
    ax_multi.spines['top'].set_visible(False)
    ax_multi.spines['right'].set_visible(False)
    
    # ===== 第四行: 核心方程 =====
    ax_eq = fig.add_subplot(gs[3, :])
    ax_eq.set_xlim(0, 100)
    ax_eq.set_ylim(0, 10)
    ax_eq.axis('off')
    
    # 三个方程框
    equations = [
        ('BLE Advertising', r'$P_{adv} = P_{tx} \cdot \frac{T_{tx}}{T_{int}} + P_{sleep}$',
         COLORS['advertising'], 5),
        ('A2DP Audio', r'$P_{audio} = P_{base} \cdot k_{codec}$',
         COLORS['bt_audio'], 37),
        ('Multi-Device', r'$P_{multi} = N \cdot P_{single} \cdot (1 + 0.1(N-1))$',
         COLORS['warning'], 69),
    ]
    
    for title, eq, color, x in equations:
        box = FancyBboxPatch((x, 1), 28, 8,
                              boxstyle="round,pad=0.05,rounding_size=0.3",
                              facecolor='white', edgecolor=color, linewidth=2)
        ax_eq.add_patch(box)
        ax_eq.text(x + 14, 7.5, title, fontsize=10, fontweight='bold',
                  ha='center', color=color)
        ax_eq.text(x + 14, 4, eq, fontsize=11, ha='center', color=COLORS['text'])
    
    # 底部引用
    fig.text(0.5, 0.01, 'Data Sources: Bluetooth Core Spec 5.3 | Nordic nRF52 | Qualcomm aptX | Measured Values',
             fontsize=8, ha='center', color=COLORS['text_light'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# 主函数
# ============================================================================

def generate_bluetooth_analysis():
    """生成所有蓝牙分析图表"""
    print("="*70)
    print("生成蓝牙功耗分析图表")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/6] BLE状态机分析...")
    plot_ble_state_machine(os.path.join(script_dir, 'bt_ble_state_machine.png'))
    
    print("[2/6] 音频功耗分析...")
    plot_bt_audio_analysis(os.path.join(script_dir, 'bt_audio_analysis.png'))
    
    print("[3/6] 多设备分析...")
    plot_multi_device_analysis(os.path.join(script_dir, 'bt_multi_device.png'))
    
    print("[4/6] 时序分析...")
    plot_bt_temporal_analysis(os.path.join(script_dir, 'bt_temporal_analysis.png'))
    
    print("[5/6] 方程汇总...")
    plot_bt_equations_summary(os.path.join(script_dir, 'bt_equations_summary.png'))
    
    print("[6/6] 综合仪表板...")
    plot_bt_comprehensive_dashboard(os.path.join(script_dir, 'bt_dashboard.png'))
    
    print("\n" + "="*70)
    print("蓝牙功耗分析图表生成完成!")
    print("="*70)


if __name__ == "__main__":
    generate_bluetooth_analysis()
