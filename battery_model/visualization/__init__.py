"""
功耗分析与可视化模块

提供三类可视化风格:

1. 基础分析 (power_analysis.py):
   - PowerAnalyzer: 综合功耗分析器
   - 标准科学图表

2. 高级可视化 (advanced_visualization.py):
   - 雷达图对比
   - 桑基图能量流
   - 热力图敏感性分析
   - 3D功耗曲面

3. 创新图表 (innovative_charts.py):
   - 仪表盘风格
   - 瀑布图分解
   - 极坐标24小时分布
   - 网络拓扑图
"""

from .power_analysis import (
    PowerAnalyzer,
    plot_network_analysis,
    plot_gps_analysis,
    plot_background_analysis,
    plot_soc_simulation,
    generate_comprehensive_report
)

from .advanced_visualization import (
    plot_radar_comparison,
    plot_energy_flow_sankey,
    plot_heatmap_sensitivity,
    plot_dynamic_timeline,
    plot_infographic_summary,
    plot_3d_power_surface,
    generate_advanced_visualizations
)

from .innovative_charts import (
    plot_battery_gauge_dashboard,
    plot_waterfall_breakdown,
    plot_polar_heatmap_24h,
    plot_network_topology,
    plot_comparison_matrix,
    generate_innovative_charts
)

__all__ = [
    # 基础分析
    'PowerAnalyzer',
    'plot_network_analysis',
    'plot_gps_analysis',
    'plot_background_analysis',
    'plot_soc_simulation',
    'generate_comprehensive_report',
    # 高级可视化
    'plot_radar_comparison',
    'plot_energy_flow_sankey',
    'plot_heatmap_sensitivity',
    'plot_dynamic_timeline',
    'plot_infographic_summary',
    'plot_3d_power_surface',
    'generate_advanced_visualizations',
    # 创新图表
    'plot_battery_gauge_dashboard',
    'plot_waterfall_breakdown',
    'plot_polar_heatmap_24h',
    'plot_network_topology',
    'plot_comparison_matrix',
    'generate_innovative_charts'
]
