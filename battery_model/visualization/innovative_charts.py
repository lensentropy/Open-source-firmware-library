"""
创新可视化模块 - 独特图表设计

包含:
1. 环形仪表盘 - 电池状态
2. 瀑布图 - 功耗分解
3. 甘特图风格 - 任务时间线
4. 极坐标热力图 - 24小时功耗分布
5. 网络图 - 组件关系
6. 动画风格静态图 - 能量流动

作者: Battery Modeling Team
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge, Circle, Rectangle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_battery_model import ContinuousTimeBatteryModel
from submodules.network_model import NetworkPowerModel, NetworkType, NetworkState
from submodules.gps_model import GPSPowerModel, GNSSMode, GNSSConstellation
from submodules.background_tasks_model import BackgroundPowerModel, BackgroundScenarios

# 现代配色
PALETTE = {
    'bg': '#0F172A',           # 深蓝黑背景
    'card': '#1E293B',         # 卡片背景
    'text': '#F8FAFC',         # 主文字
    'muted': '#94A3B8',        # 次要文字
    'network': '#38BDF8',      # 天蓝
    'gps': '#4ADE80',          # 绿色
    'background': '#C084FC',   # 紫色
    'base': '#FB923C',         # 橙色
    'accent': '#F472B6',       # 粉色
    'success': '#22C55E',
    'warning': '#FBBF24',
    'danger': '#EF4444',
    'gradient1': '#6366F1',
    'gradient2': '#EC4899',
}


def plot_battery_gauge_dashboard(soc=75, save_path=None):
    """
    环形仪表盘 - 电池状态可视化
    
    类似智能手表/汽车仪表盘风格
    """
    fig = plt.figure(figsize=(16, 8), facecolor=PALETTE['bg'])
    
    # 左侧: 主仪表盘
    ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8], aspect='equal')
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis('off')
    ax1.set_facecolor(PALETTE['bg'])
    
    # 外圈装饰
    outer_ring = Circle((0, 0), 1.3, fill=False, color=PALETTE['muted'], linewidth=2, alpha=0.3)
    ax1.add_patch(outer_ring)
    
    # 刻度
    for angle in np.linspace(135, -135, 11):
        rad = np.radians(angle)
        x1, y1 = 1.15 * np.cos(rad), 1.15 * np.sin(rad)
        x2, y2 = 1.25 * np.cos(rad), 1.25 * np.sin(rad)
        ax1.plot([x1, x2], [y1, y2], color=PALETTE['muted'], linewidth=2, alpha=0.5)
    
    # 主进度环 (多层渐变效果)
    # 背景环
    bg_arc = Wedge((0, 0), 1.1, -135, 135, width=0.15, 
                    facecolor=PALETTE['card'], edgecolor='none')
    ax1.add_patch(bg_arc)
    
    # SOC进度环
    soc_angle = -135 + (soc / 100) * 270
    
    # 渐变色环 (分段绘制模拟渐变)
    n_segments = 50
    for i in range(int(soc / 100 * n_segments)):
        seg_start = -135 + i * (270 / n_segments)
        seg_end = seg_start + (270 / n_segments) + 1
        
        # 颜色渐变
        if soc > 50:
            color = PALETTE['success'] if i < n_segments * 0.7 else PALETTE['warning']
        elif soc > 20:
            color = PALETTE['warning']
        else:
            color = PALETTE['danger']
        
        alpha = 0.5 + 0.5 * (i / n_segments)
        seg = Wedge((0, 0), 1.1, seg_start, seg_end, width=0.15, 
                    facecolor=color, edgecolor='none', alpha=alpha)
        ax1.add_patch(seg)
    
    # 中心圆
    center_circle = Circle((0, 0), 0.85, facecolor=PALETTE['card'], edgecolor=PALETTE['muted'], linewidth=1)
    ax1.add_patch(center_circle)
    
    # SOC文字
    ax1.text(0, 0.15, f'{soc}', fontsize=60, fontweight='bold', 
             ha='center', va='center', color=PALETTE['text'])
    ax1.text(0, -0.25, '%', fontsize=24, ha='center', va='center', color=PALETTE['muted'])
    ax1.text(0, -0.55, 'State of Charge', fontsize=12, ha='center', va='center', color=PALETTE['muted'])
    
    # 状态指示
    if soc > 50:
        status = 'GOOD'
        status_color = PALETTE['success']
    elif soc > 20:
        status = 'LOW'
        status_color = PALETTE['warning']
    else:
        status = 'CRITICAL'
        status_color = PALETTE['danger']
    
    ax1.text(0, -0.85, status, fontsize=14, fontweight='bold', 
             ha='center', va='center', color=status_color)
    
    # 右侧: 详细信息面板
    ax2 = fig.add_axes([0.5, 0.1, 0.45, 0.8])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_facecolor(PALETTE['bg'])
    
    # 标题
    ax2.text(5, 9.5, 'Battery Status Dashboard', fontsize=18, fontweight='bold',
             ha='center', color=PALETTE['text'])
    
    # 信息卡片
    info_cards = [
        ('Remaining Time', '~12.5h', PALETTE['network'], 8),
        ('Current Draw', '320 mW', PALETTE['gps'], 6.5),
        ('Temperature', '28°C', PALETTE['background'], 5),
        ('Health', '94%', PALETTE['success'], 3.5),
    ]
    
    for label, value, color, y in info_cards:
        # 卡片背景
        card = FancyBboxPatch((0.5, y - 0.5), 9, 1.2,
                               boxstyle="round,pad=0.05,rounding_size=0.2",
                               facecolor=PALETTE['card'], edgecolor=color, linewidth=2)
        ax2.add_patch(card)
        
        # 左侧色条
        bar = Rectangle((0.6, y - 0.4), 0.15, 1.0, facecolor=color)
        ax2.add_patch(bar)
        
        ax2.text(1.2, y, label, fontsize=11, va='center', color=PALETTE['muted'])
        ax2.text(8.5, y, value, fontsize=14, fontweight='bold', va='center', 
                 ha='right', color=PALETTE['text'])
    
    # 功耗分布小环图
    ax3 = fig.add_axes([0.75, 0.15, 0.2, 0.25], aspect='equal')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.axis('off')
    ax3.set_facecolor(PALETTE['bg'])
    
    # 绘制小环图
    sizes = [35, 25, 25, 15]  # Network, GPS, BG, Base
    colors = [PALETTE['network'], PALETTE['gps'], PALETTE['background'], PALETTE['base']]
    
    start_angle = 90
    for size, color in zip(sizes, colors):
        end_angle = start_angle - size * 3.6
        wedge = Wedge((0, 0), 1, end_angle, start_angle, width=0.4, 
                      facecolor=color, edgecolor=PALETTE['bg'], linewidth=2)
        ax3.add_patch(wedge)
        start_angle = end_angle
    
    ax3.text(0, 0, 'Power\nMix', fontsize=8, ha='center', va='center', color=PALETTE['muted'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_waterfall_breakdown(save_path=None):
    """
    瀑布图 - 功耗逐项分解
    
    展示从零到总功耗的累积过程
    """
    fig = plt.figure(figsize=(14, 8), facecolor=PALETTE['bg'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(PALETTE['bg'])
    
    # 功耗项目
    categories = ['Base\nSystem', 'CPU\nIdle', 'Memory', 'WiFi\nIdle', 'Push\nService', 
                  'Email\nSync', 'Location\nUpdate', 'Social\nMedia', 'GPS\nBackground', 'Total']
    values = [25, 15, 50, 14.5, 8, 12, 15, 20, 5, None]  # None for total
    
    # 计算累积值
    cumulative = [0]
    for v in values[:-1]:
        cumulative.append(cumulative[-1] + v)
    
    total = cumulative[-1]
    values[-1] = total
    
    # 颜色映射
    colors = [PALETTE['base'], PALETTE['background'], PALETTE['background'], 
              PALETTE['network'], PALETTE['background'], PALETTE['background'],
              PALETTE['gps'], PALETTE['network'], PALETTE['gps'], PALETTE['accent']]
    
    x = np.arange(len(categories))
    
    # 绘制瀑布图
    for i, (cat, val, cum, color) in enumerate(zip(categories, values, cumulative + [0], colors)):
        if i < len(categories) - 1:
            # 中间柱子
            bar = ax.bar(i, val, bottom=cum, color=color, alpha=0.8, 
                        edgecolor='white', linewidth=1.5, width=0.6)
            
            # 连接线
            if i < len(categories) - 2:
                ax.plot([i + 0.3, i + 0.7], [cum + val, cum + val], 
                       color=PALETTE['muted'], linewidth=1, linestyle='--', alpha=0.5)
        else:
            # 总计柱子
            bar = ax.bar(i, val, color=color, alpha=0.9, 
                        edgecolor='white', linewidth=2, width=0.6)
    
    # 添加数值标签
    for i, (val, cum) in enumerate(zip(values, cumulative + [0])):
        if i < len(categories) - 1:
            y_pos = cum + val / 2
            ax.text(i, y_pos, f'+{val:.1f}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
        else:
            ax.text(i, val / 2, f'{val:.1f}\nmW', ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')
    
    # 样式设置
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, color=PALETTE['text'])
    ax.set_ylabel('Power Consumption (mW)', fontsize=12, fontweight='bold', color=PALETTE['text'])
    ax.set_title('Power Consumption Waterfall Breakdown', fontsize=16, fontweight='bold', 
                 color=PALETTE['text'], pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['muted'])
    ax.spines['bottom'].set_color(PALETTE['muted'])
    ax.tick_params(colors=PALETTE['muted'])
    ax.yaxis.label.set_color(PALETTE['text'])
    
    ax.set_ylim(0, total * 1.15)
    ax.axhline(y=total, color=PALETTE['accent'], linestyle='--', alpha=0.5, linewidth=1)
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=PALETTE['base'], label='Base System'),
        mpatches.Patch(color=PALETTE['background'], label='CPU/Memory/Services'),
        mpatches.Patch(color=PALETTE['network'], label='Network'),
        mpatches.Patch(color=PALETTE['gps'], label='GPS'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', facecolor=PALETTE['card'],
              edgecolor=PALETTE['muted'], labelcolor=PALETTE['text'])
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_polar_heatmap_24h(save_path=None):
    """
    极坐标热力图 - 24小时功耗分布
    
    类似时钟的功耗可视化
    """
    fig = plt.figure(figsize=(14, 7), facecolor=PALETTE['bg'])
    
    # 左侧: 极坐标热力图
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_facecolor(PALETTE['bg'])
    
    # 生成24小时功耗数据
    hours = np.linspace(0, 24, 49)[:-1]  # 48个半小时段
    
    # 模拟真实使用模式
    base_power = 100
    network_pattern = np.zeros(48)
    gps_pattern = np.zeros(48)
    bg_pattern = np.zeros(48)
    
    # 睡眠时间 (0-7点): 低功耗
    # 早高峰 (7-9点): 高网络使用
    # 工作时间 (9-18点): 中等
    # 晚高峰 (18-22点): 高使用
    # 深夜 (22-24点): 低
    
    for i, h in enumerate(hours):
        if 0 <= h < 7:
            network_pattern[i] = 20
            bg_pattern[i] = 30
        elif 7 <= h < 9:
            network_pattern[i] = 300
            gps_pattern[i] = 100
            bg_pattern[i] = 80
        elif 9 <= h < 12:
            network_pattern[i] = 150
            bg_pattern[i] = 70
        elif 12 <= h < 14:
            network_pattern[i] = 400
            gps_pattern[i] = 50
            bg_pattern[i] = 100
        elif 14 <= h < 18:
            network_pattern[i] = 200
            bg_pattern[i] = 60
        elif 18 <= h < 22:
            network_pattern[i] = 500
            gps_pattern[i] = 150
            bg_pattern[i] = 120
        else:
            network_pattern[i] = 100
            bg_pattern[i] = 50
    
    total_power = base_power + network_pattern + gps_pattern + bg_pattern
    
    # 转换为极坐标
    theta = np.linspace(0, 2 * np.pi, 49)[:-1]
    r = np.ones(48) * 0.3  # 内圈
    width = 2 * np.pi / 48
    
    # 绘制堆叠环
    # 基础功耗
    bars1 = ax1.bar(theta, np.ones(48) * 0.2, width=width, bottom=0.3, 
                    color=PALETTE['base'], alpha=0.7, edgecolor='none')
    
    # 网络功耗
    network_norm = network_pattern / 500 * 0.3
    bars2 = ax1.bar(theta, network_norm, width=width, bottom=0.5, 
                    color=PALETTE['network'], alpha=0.8, edgecolor='none')
    
    # GPS功耗
    gps_norm = gps_pattern / 200 * 0.15
    bars3 = ax1.bar(theta, gps_norm, width=width, bottom=0.5 + network_norm, 
                    color=PALETTE['gps'], alpha=0.8, edgecolor='none')
    
    # 后台功耗
    bg_norm = bg_pattern / 150 * 0.15
    bars4 = ax1.bar(theta, bg_norm, width=width, bottom=0.5 + network_norm + gps_norm, 
                    color=PALETTE['background'], alpha=0.8, edgecolor='none')
    
    # 设置时钟刻度
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
    ax1.set_xticklabels([f'{i}:00' for i in range(24)], fontsize=8, color=PALETTE['muted'])
    ax1.set_yticks([])
    ax1.set_ylim(0, 1.2)
    
    # 中心标签
    ax1.text(0, 0, '24h\nPower\nPattern', ha='center', va='center', 
             fontsize=10, fontweight='bold', color=PALETTE['text'])
    
    ax1.spines['polar'].set_color(PALETTE['muted'])
    ax1.set_title('Daily Power Consumption Pattern', fontsize=14, fontweight='bold',
                  color=PALETTE['text'], pad=20)
    
    # 右侧: 时段统计
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(PALETTE['bg'])
    
    time_periods = ['Night\n(0-7h)', 'Morning\n(7-12h)', 'Afternoon\n(12-18h)', 'Evening\n(18-24h)']
    period_powers = [
        np.mean(total_power[:14]),
        np.mean(total_power[14:24]),
        np.mean(total_power[24:36]),
        np.mean(total_power[36:])
    ]
    
    colors = ['#1E40AF', '#F59E0B', '#EA580C', '#7C3AED']
    
    bars = ax2.barh(time_periods, period_powers, color=colors, alpha=0.8,
                    edgecolor='white', linewidth=2, height=0.6)
    
    # 添加数值和图标
    for bar, power in zip(bars, period_powers):
        ax2.text(power + 10, bar.get_y() + bar.get_height()/2,
                 f'{power:.0f} mW', va='center', fontsize=11, 
                 fontweight='bold', color=PALETTE['text'])
    
    ax2.set_xlabel('Average Power (mW)', fontsize=12, fontweight='bold', color=PALETTE['text'])
    ax2.set_title('Power by Time Period', fontsize=14, fontweight='bold',
                  color=PALETTE['text'], pad=20)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['muted'])
    ax2.spines['bottom'].set_color(PALETTE['muted'])
    ax2.tick_params(colors=PALETTE['muted'])
    
    # 图例
    legend_patches = [
        mpatches.Patch(color=PALETTE['base'], label='Base'),
        mpatches.Patch(color=PALETTE['network'], label='Network'),
        mpatches.Patch(color=PALETTE['gps'], label='GPS'),
        mpatches.Patch(color=PALETTE['background'], label='Background'),
    ]
    ax2.legend(handles=legend_patches, loc='lower right', facecolor=PALETTE['card'],
               edgecolor=PALETTE['muted'], labelcolor=PALETTE['text'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_network_topology(save_path=None):
    """
    网络拓扑图 - 组件关系可视化
    
    展示电池模型各组件之间的关系
    """
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['bg'])
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(PALETTE['bg'])
    
    # 标题
    ax.text(8, 9.5, 'Battery Power Model Architecture', fontsize=18, fontweight='bold',
            ha='center', color=PALETTE['text'])
    
    # 中心: 电池
    battery_circle = Circle((8, 5), 1.2, facecolor=PALETTE['card'], 
                             edgecolor=PALETTE['accent'], linewidth=4)
    ax.add_patch(battery_circle)
    ax.text(8, 5.3, 'Li-ion', fontsize=12, fontweight='bold', ha='center', color=PALETTE['text'])
    ax.text(8, 4.9, 'Battery', fontsize=10, ha='center', color=PALETTE['muted'])
    ax.text(8, 4.4, '4000mAh', fontsize=9, ha='center', color=PALETTE['accent'])
    
    # 主要组件节点
    components = [
        ('Network\nModule', 3, 7.5, PALETTE['network'], ['WiFi', 'LTE', '5G']),
        ('GPS\nModule', 13, 7.5, PALETTE['gps'], ['Tracking', 'A-GPS', 'Multi-GNSS']),
        ('Background\nTasks', 3, 2.5, PALETTE['background'], ['Sync', 'Push', 'Services']),
        ('Core\nSystem', 13, 2.5, PALETTE['base'], ['CPU', 'Memory', 'I/O']),
    ]
    
    for name, x, y, color, subs in components:
        # 主节点
        node = Circle((x, y), 0.9, facecolor=PALETTE['card'], 
                       edgecolor=color, linewidth=3)
        ax.add_patch(node)
        ax.text(x, y, name, fontsize=10, fontweight='bold', ha='center', va='center',
                color=PALETTE['text'])
        
        # 子节点
        sub_angles = np.linspace(0, 2*np.pi, len(subs), endpoint=False)
        for sub, angle in zip(subs, sub_angles):
            # 调整子节点位置 (远离中心)
            if x < 8:
                sx = x - 1.5 + 0.5 * np.cos(angle + np.pi/2)
            else:
                sx = x + 1.5 + 0.5 * np.cos(angle - np.pi/2)
            sy = y + 1.2 * np.sin(angle)
            
            sub_node = Circle((sx, sy), 0.4, facecolor=color, alpha=0.3,
                              edgecolor=color, linewidth=1.5)
            ax.add_patch(sub_node)
            ax.text(sx, sy, sub, fontsize=7, ha='center', va='center', 
                    color=PALETTE['text'])
            
            # 连接线
            ax.annotate('', xy=(x + 0.9 * np.cos(np.arctan2(sy-y, sx-x)), 
                               y + 0.9 * np.sin(np.arctan2(sy-y, sx-x))),
                       xytext=(sx - 0.4 * np.cos(np.arctan2(sy-y, sx-x)),
                              sy - 0.4 * np.sin(np.arctan2(sy-y, sx-x))),
                       arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, lw=1))
        
        # 到电池的连接
        ax.annotate('', xy=(8 + 1.2 * np.cos(np.arctan2(y-5, x-8)),
                           5 + 1.2 * np.sin(np.arctan2(y-5, x-8))),
                   xytext=(x - 0.9 * np.cos(np.arctan2(y-5, x-8)),
                          y - 0.9 * np.sin(np.arctan2(y-5, x-8))),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2.5,
                                  connectionstyle='arc3,rad=0.1'))
    
    # 功耗流向标注
    flow_labels = [
        (5.5, 6.8, '35%', PALETTE['network']),
        (10.5, 6.8, '15%', PALETTE['gps']),
        (5.5, 3.2, '25%', PALETTE['background']),
        (10.5, 3.2, '25%', PALETTE['base']),
    ]
    
    for x, y, pct, color in flow_labels:
        ax.text(x, y, pct, fontsize=11, fontweight='bold', ha='center',
                color=color, bbox=dict(boxstyle='round', facecolor=PALETTE['card'],
                                       edgecolor=color, alpha=0.8))
    
    # 方程式卡片
    eq_box = FancyBboxPatch((5.5, 0.3), 5, 1.2,
                             boxstyle="round,pad=0.1,rounding_size=0.2",
                             facecolor=PALETTE['card'], edgecolor=PALETTE['muted'], linewidth=2)
    ax.add_patch(eq_box)
    ax.text(8, 1.1, 'dSOC/dt = -P_total(t) / (V × Q_eff)', fontsize=11,
            ha='center', color=PALETTE['text'], family='monospace')
    ax.text(8, 0.6, 'P_total = P_net + P_gps + P_bg + P_base', fontsize=10,
            ha='center', color=PALETTE['muted'], family='monospace')
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"Saved: {save_path}")
    
    return fig


def plot_comparison_matrix(save_path=None):
    """
    对比矩阵图 - 不同场景和组件的功耗对比
    """
    fig = plt.figure(figsize=(14, 10), facecolor=PALETTE['bg'])
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. 网络技术对比 (条形图)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PALETTE['bg'])
    
    techs = ['WiFi\nPSM', 'WiFi\nActive', 'LTE\nDRX', 'LTE\nActive', '5G\nActive']
    powers = [10, 450, 100, 1500, 2500]
    alphas = [0.3, 0.6, 0.4, 0.7, 0.9]
    
    # 绘制每个柱子单独设置alpha
    for i, (tech, power, alpha) in enumerate(zip(techs, powers, alphas)):
        ax1.bar(i, power, color=PALETTE['network'], alpha=alpha, edgecolor='white', linewidth=2)
    
    ax1.set_xticks(range(len(techs)))
    ax1.set_xticklabels(techs)
    bars = ax1.patches  # 获取所有柱子
    
    for bar, power in zip(bars, powers):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{power}', ha='center', fontsize=10, fontweight='bold', color=PALETTE['text'])
    
    ax1.set_ylabel('Power (mW)', fontsize=11, fontweight='bold', color=PALETTE['text'])
    ax1.set_title('Network Technology Comparison', fontsize=13, fontweight='bold',
                  color=PALETTE['text'], pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(PALETTE['muted'])
    ax1.spines['bottom'].set_color(PALETTE['muted'])
    ax1.tick_params(colors=PALETTE['muted'])
    
    # 2. GPS模式对比 (水平条形图)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE['bg'])
    
    modes = ['Standby', 'GPS Only', 'GPS+GLONASS', 'Multi-GNSS', 'Navigation']
    gps_powers = [0.5, 30, 45, 65, 180]
    
    bars2 = ax2.barh(modes, gps_powers, color=PALETTE['gps'], alpha=0.7,
                     edgecolor='white', linewidth=2, height=0.6)
    
    for bar, power in zip(bars2, gps_powers):
        ax2.text(power + 5, bar.get_y() + bar.get_height()/2,
                 f'{power}mW', va='center', fontsize=10, fontweight='bold', color=PALETTE['text'])
    
    ax2.set_xlabel('Power (mW)', fontsize=11, fontweight='bold', color=PALETTE['text'])
    ax2.set_title('GPS Mode Comparison', fontsize=13, fontweight='bold',
                  color=PALETTE['text'], pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['muted'])
    ax2.spines['bottom'].set_color(PALETTE['muted'])
    ax2.tick_params(colors=PALETTE['muted'])
    
    # 3. CPU频率-功耗关系 (曲线图)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PALETTE['bg'])
    
    freq = np.linspace(0.3, 2.8, 50)
    
    bg_model = BackgroundPowerModel()
    power_50 = [bg_model.dvfs_power(f * 1e9, 0.5) * 1000 for f in freq]
    power_100 = [bg_model.dvfs_power(f * 1e9, 1.0) * 1000 for f in freq]
    
    ax3.fill_between(freq, power_50, power_100, alpha=0.3, color=PALETTE['background'])
    ax3.plot(freq, power_50, linewidth=2.5, color=PALETTE['background'], label='50% Load')
    ax3.plot(freq, power_100, linewidth=2.5, color=PALETTE['accent'], label='100% Load')
    
    ax3.set_xlabel('CPU Frequency (GHz)', fontsize=11, fontweight='bold', color=PALETTE['text'])
    ax3.set_ylabel('Power (mW)', fontsize=11, fontweight='bold', color=PALETTE['text'])
    ax3.set_title('CPU DVFS Power Scaling', fontsize=13, fontweight='bold',
                  color=PALETTE['text'], pad=15)
    ax3.legend(facecolor=PALETTE['card'], edgecolor=PALETTE['muted'], labelcolor=PALETTE['text'])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color(PALETTE['muted'])
    ax3.spines['bottom'].set_color(PALETTE['muted'])
    ax3.tick_params(colors=PALETTE['muted'])
    ax3.grid(True, alpha=0.2, color=PALETTE['muted'])
    
    # 4. 场景对比 (气泡图)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PALETTE['bg'])
    
    scenarios = ['Sleep', 'Idle', 'Browsing', 'Navigation', 'Streaming', 'Gaming']
    battery_life = [80, 50, 12, 6, 8, 4]  # 小时
    avg_power = [50, 150, 400, 800, 600, 1200]  # mW
    
    colors = [PALETTE['success'], PALETTE['success'], PALETTE['network'],
              PALETTE['gps'], PALETTE['network'], PALETTE['danger']]
    sizes = np.array(avg_power) / 3
    
    scatter = ax4.scatter(range(len(scenarios)), battery_life, s=sizes, c=colors,
                          alpha=0.7, edgecolors='white', linewidths=2)
    
    for i, (scen, life, power) in enumerate(zip(scenarios, battery_life, avg_power)):
        ax4.annotate(f'{power}mW', (i, life + 3), ha='center', fontsize=8,
                     color=PALETTE['muted'])
    
    ax4.set_xticks(range(len(scenarios)))
    ax4.set_xticklabels(scenarios, fontsize=10, color=PALETTE['text'])
    ax4.set_ylabel('Battery Life (hours)', fontsize=11, fontweight='bold', color=PALETTE['text'])
    ax4.set_title('Usage Scenario Impact', fontsize=13, fontweight='bold',
                  color=PALETTE['text'], pad=15)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color(PALETTE['muted'])
    ax4.spines['bottom'].set_color(PALETTE['muted'])
    ax4.tick_params(colors=PALETTE['muted'])
    
    ax4.text(0.95, 0.05, 'Bubble size = Power consumption', transform=ax4.transAxes,
             fontsize=8, ha='right', color=PALETTE['muted'], style='italic')
    
    plt.suptitle('Comprehensive Power Consumption Analysis', fontsize=16, fontweight='bold',
                 color=PALETTE['text'], y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=PALETTE['bg'])
        print(f"Saved: {save_path}")
    
    return fig


def generate_innovative_charts():
    """生成所有创新图表"""
    print("="*60)
    print("生成创新可视化图表...")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n[1/5] 电池仪表盘...")
    plot_battery_gauge_dashboard(75, os.path.join(script_dir, 'innovative_gauge_dashboard.png'))
    
    print("[2/5] 功耗瀑布图...")
    plot_waterfall_breakdown(os.path.join(script_dir, 'innovative_waterfall.png'))
    
    print("[3/5] 24小时极坐标热力图...")
    plot_polar_heatmap_24h(os.path.join(script_dir, 'innovative_polar_24h.png'))
    
    print("[4/5] 组件网络拓扑图...")
    plot_network_topology(os.path.join(script_dir, 'innovative_topology.png'))
    
    print("[5/5] 对比矩阵图...")
    plot_comparison_matrix(os.path.join(script_dir, 'innovative_comparison.png'))
    
    print("\n" + "="*60)
    print("创新可视化图表生成完成!")
    print("="*60)


if __name__ == "__main__":
    generate_innovative_charts()
