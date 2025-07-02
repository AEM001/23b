#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略效果对比可视化
展示保守延伸策略相对于之前策略的改进效果
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_plot():
    """创建策略对比图"""
    
    # 数据对比
    strategies = ['智能延伸策略', '保守延伸策略']
    total_length = [263.79, 293.82]  # 海里
    miss_rate = [3.2512, 0.9450]  # 百分比
    coverage_rate = [96.7488, 99.0550]  # 百分比
    overlap_excess = [8.15, 9.02]  # 海里
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 测线总长度对比
    bars1 = ax1.bar(strategies, total_length, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax1.set_title('测线总长度对比', fontsize=14, weight='bold')
    ax1.set_ylabel('长度 (海里)', fontsize=12)
    for i, v in enumerate(total_length):
        ax1.text(i, v + 5, f'{v:.1f}', ha='center', va='bottom', fontsize=11, weight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 漏测率对比
    bars2 = ax2.bar(strategies, miss_rate, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax2.set_title('漏测率对比', fontsize=14, weight='bold')
    ax2.set_ylabel('漏测率 (%)', fontsize=12)
    for i, v in enumerate(miss_rate):
        ax2.text(i, v + 0.1, f'{v:.3f}%', ha='center', va='bottom', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 覆盖率对比
    bars3 = ax3.bar(strategies, coverage_rate, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax3.set_title('覆盖率对比', fontsize=14, weight='bold')
    ax3.set_ylabel('覆盖率 (%)', fontsize=12)
    ax3.set_ylim([95, 100])
    for i, v in enumerate(coverage_rate):
        ax3.text(i, v + 0.1, f'{v:.2f}%', ha='center', va='bottom', fontsize=11, weight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 重叠率超过20%部分的总长度对比
    bars4 = ax4.bar(strategies, overlap_excess, color=['#3498db', '#e74c3c'], alpha=0.7)
    ax4.set_title('重叠率超过20%部分总长度', fontsize=14, weight='bold')
    ax4.set_ylabel('长度 (海里)', fontsize=12)
    for i, v in enumerate(overlap_excess):
        ax4.text(i, v + 0.2, f'{v:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加改进标注
    improvement_text = f"漏测率降低: {miss_rate[0] - miss_rate[1]:.3f}% (降幅{((miss_rate[0] - miss_rate[1])/miss_rate[0]*100):.1f}%)"
    fig.suptitle(f'保守延伸策略效果对比\n{improvement_text}', 
                 fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
    print("策略对比图已保存至: strategy_comparison.png")

def print_comparison_summary():
    """打印对比总结"""
    print("="*60)
    print("保守延伸策略改进效果总结")
    print("="*60)
    print("策略对比:")
    print("  智能延伸策略 → 保守延伸策略")
    print()
    print("关键指标改进:")
    print(f"  • 漏测率: 3.251% → 0.945% (降低70.9%)")
    print(f"  • 覆盖率: 96.75% → 99.06% (提升2.31%)")
    print(f"  • 测线总长: 263.79海里 → 293.82海里 (增加30.03海里)")
    print(f"  • 重叠超限: 8.15海里 → 9.02海里 (略微增加)")
    print()
    print("技术改进:")
    print("  • 基础延伸距离从半宽度提升到完整覆盖宽度")
    print("  • 增加角度修正机制")
    print("  • 增加20%安全边距")
    print("  • 高精度覆盖检测")
    print("="*60)

if __name__ == '__main__':
    create_comparison_plot()
    print_comparison_summary() 