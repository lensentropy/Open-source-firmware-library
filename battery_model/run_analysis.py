#!/usr/bin/env python3
"""
智能手机电池功耗模型 - 主运行脚本

运行完整的功耗分析并生成可视化图表

使用方法:
    python run_analysis.py

输出:
    - battery_model/visualization/network_power_analysis.png
    - battery_model/visualization/gps_power_analysis.png
    - battery_model/visualization/background_power_analysis.png
    - battery_model/visualization/soc_simulation.png
"""

import os
import sys

# 确保模块路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualization.power_analysis import generate_comprehensive_report
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def main():
    """主函数"""
    print("\n" + "="*70)
    print("   智能手机锂离子电池连续时间数学模型")
    print("   Continuous-Time Mathematical Model for Smartphone Li-ion Battery")
    print("="*70)
    
    print("\n子模块:")
    print("  1. 网络连接功耗 (WiFi/LTE/5G)")
    print("  2. GPS/GNSS功耗")
    print("  3. 后台任务功耗")
    
    print("\n开始分析...\n")
    
    # 运行综合分析
    results = generate_comprehensive_report()
    
    print("\n分析完成!")
    print("\n生成的图表:")
    print("  - network_power_analysis.png  (网络功耗分析)")
    print("  - gps_power_analysis.png      (GPS功耗分析)")
    print("  - background_power_analysis.png (后台任务功耗分析)")
    print("  - soc_simulation.png          (SOC仿真结果)")
    
    return results


if __name__ == "__main__":
    main()
