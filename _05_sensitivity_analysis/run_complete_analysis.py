#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行完整的灵敏度分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parametric_survey_line_generator import generate_survey_lines_with_parameters
from sensitivity_analysis_config import BASELINE_CONFIG, SAFETY_MARGIN_VARIATIONS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_safety_margin_sensitivity():
    """分析安全边距系数的灵敏度"""
    print("=== 安全边距系数灵敏度分析 ===")
    
    results = []
    safety_margins = [v['safety_margin_factor'] for v in SAFETY_MARGIN_VARIATIONS]
    
    for margin in safety_margins:
        config = BASELINE_CONFIG.copy()
        config['safety_margin_factor'] = margin
        
        print(f"测试安全边距系数: {margin:.2f}")
        
        try:
            lines_df, _ = generate_survey_lines_with_parameters(config)
            total_length = lines_df['length_optimized_nm'].sum()
            
            results.append({
                'safety_margin': margin,
                'total_length': total_length,
                'total_lines': len(lines_df)
            })
            
            print(f"  结果: {total_length:.2f} 海里, {len(lines_df)} 条测线")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    return pd.DataFrame(results)

def analyze_angle_correction_sensitivity():
    """分析角度修正系数的灵敏度"""
    print("\n=== 角度修正系数灵敏度分析 ===")
    
    results = []
    angle_factors = [0.0, 0.2, 0.5, 0.7, 1.0]
    
    for factor in angle_factors:
        config = BASELINE_CONFIG.copy()
        config['angle_correction_factor'] = factor
        
        print(f"测试角度修正系数: {factor:.1f}")
        
        try:
            lines_df, _ = generate_survey_lines_with_parameters(config)
            total_length = lines_df['length_optimized_nm'].sum()
            
            results.append({
                'angle_factor': factor,
                'total_length': total_length,
                'total_lines': len(lines_df)
            })
            
            print(f"  结果: {total_length:.2f} 海里, {len(lines_df)} 条测线")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    return pd.DataFrame(results)

def analyze_extension_base_sensitivity():
    """分析基础延伸系数的灵敏度"""
    print("\n=== 基础延伸系数灵敏度分析 ===")
    
    results = []
    base_factors = [0.5, 0.7, 1.0, 1.2, 1.5]
    
    for factor in base_factors:
        config = BASELINE_CONFIG.copy()
        config['extension_base_factor'] = factor
        
        print(f"测试基础延伸系数: {factor:.1f}")
        
        try:
            lines_df, _ = generate_survey_lines_with_parameters(config)
            total_length = lines_df['length_optimized_nm'].sum()
            
            results.append({
                'base_factor': factor,
                'total_length': total_length,
                'total_lines': len(lines_df)
            })
            
            print(f"  结果: {total_length:.2f} 海里, {len(lines_df)} 条测线")
            
        except Exception as e:
            print(f"  失败: {e}")
    
    return pd.DataFrame(results)

def create_summary_visualization(safety_df, angle_df, base_df):
    """创建汇总可视化图表"""
    print("\n=== 生成可视化图表 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 安全边距系数影响
    if len(safety_df) > 0:
        axes[0,0].plot(safety_df['safety_margin'], safety_df['total_length'], 'o-', 
                      linewidth=2, markersize=6, color='blue')
        axes[0,0].set_xlabel('安全边距系数')
        axes[0,0].set_ylabel('测线总长度 (海里)')
        axes[0,0].set_title('安全边距系数对测线长度的影响')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='基准值')
        axes[0,0].legend()
    
    # 2. 角度修正系数影响
    if len(angle_df) > 0:
        axes[0,1].plot(angle_df['angle_factor'], angle_df['total_length'], 'o-',
                      linewidth=2, markersize=6, color='green')
        axes[0,1].set_xlabel('角度修正系数')
        axes[0,1].set_ylabel('测线总长度 (海里)')
        axes[0,1].set_title('角度修正系数对测线长度的影响')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='基准值')
        axes[0,1].legend()
    
    # 3. 基础延伸系数影响
    if len(base_df) > 0:
        axes[1,0].plot(base_df['base_factor'], base_df['total_length'], 'o-',
                      linewidth=2, markersize=6, color='orange')
        axes[1,0].set_xlabel('基础延伸系数')
        axes[1,0].set_ylabel('测线总长度 (海里)')
        axes[1,0].set_title('基础延伸系数对测线长度的影响')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='基准值')
        axes[1,0].legend()
    
    # 4. 敏感度对比
    param_names = []
    sensitivity_scores = []
    
    if len(safety_df) > 0:
        safety_range = safety_df['total_length'].max() - safety_df['total_length'].min()
        safety_sensitivity = safety_range / safety_df['total_length'].mean()
        param_names.append('安全边距系数')
        sensitivity_scores.append(safety_sensitivity)
    
    if len(angle_df) > 0:
        angle_range = angle_df['total_length'].max() - angle_df['total_length'].min()
        angle_sensitivity = angle_range / angle_df['total_length'].mean()
        param_names.append('角度修正系数')
        sensitivity_scores.append(angle_sensitivity)
    
    if len(base_df) > 0:
        base_range = base_df['total_length'].max() - base_df['total_length'].min()
        base_sensitivity = base_range / base_df['total_length'].mean()
        param_names.append('基础延伸系数')
        sensitivity_scores.append(base_sensitivity)
    
    if param_names:
        bars = axes[1,1].bar(param_names, sensitivity_scores, 
                           color=['blue', 'green', 'orange'][:len(param_names)], alpha=0.7)
        axes[1,1].set_ylabel('敏感度评分')
        axes[1,1].set_title('参数敏感度对比')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, sensitivity_scores):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sensitivity_analysis_summary.png', dpi=300, bbox_inches='tight')
    print("汇总图表已保存: sensitivity_analysis_summary.png")

def generate_summary_report(safety_df, angle_df, base_df):
    """生成汇总报告"""
    print("\n=== 生成汇总报告 ===")
    
    report = f"""# 测线延伸参数灵敏度分析报告

## 分析概述

本报告分析了测线边界智能扩展优化算法中三个关键参数的敏感度：
1. 安全边距系数
2. 角度修正系数  
3. 基础延伸系数

## 分析结果

### 1. 安全边距系数分析

"""
    
    if len(safety_df) > 0:
        safety_range = safety_df['total_length'].max() - safety_df['total_length'].min()
        safety_min = safety_df.loc[safety_df['total_length'].idxmin()]
        safety_max = safety_df.loc[safety_df['total_length'].idxmax()]
        
        report += f"- 测试范围: {safety_df['safety_margin'].min():.2f} ~ {safety_df['safety_margin'].max():.2f}\n"
        report += f"- 长度变化范围: {safety_range:.2f} 海里\n"
        report += f"- 最短配置: 安全边距 {safety_min['safety_margin']:.2f} ({safety_min['total_length']:.2f} 海里)\n"
        report += f"- 最长配置: 安全边距 {safety_max['safety_margin']:.2f} ({safety_max['total_length']:.2f} 海里)\n"
        report += f"- 相对变化: {(safety_range/safety_df['total_length'].mean())*100:.2f}%\n\n"
    
    report += "### 2. 角度修正系数分析\n\n"
    
    if len(angle_df) > 0:
        angle_range = angle_df['total_length'].max() - angle_df['total_length'].min()
        angle_min = angle_df.loc[angle_df['total_length'].idxmin()]
        angle_max = angle_df.loc[angle_df['total_length'].idxmax()]
        
        report += f"- 测试范围: {angle_df['angle_factor'].min():.1f} ~ {angle_df['angle_factor'].max():.1f}\n"
        report += f"- 长度变化范围: {angle_range:.2f} 海里\n"
        report += f"- 最短配置: 角度修正 {angle_min['angle_factor']:.1f} ({angle_min['total_length']:.2f} 海里)\n"
        report += f"- 最长配置: 角度修正 {angle_max['angle_factor']:.1f} ({angle_max['total_length']:.2f} 海里)\n"
        report += f"- 相对变化: {(angle_range/angle_df['total_length'].mean())*100:.2f}%\n\n"
    
    report += "### 3. 基础延伸系数分析\n\n"
    
    if len(base_df) > 0:
        base_range = base_df['total_length'].max() - base_df['total_length'].min()
        base_min = base_df.loc[base_df['total_length'].idxmin()]
        base_max = base_df.loc[base_df['total_length'].idxmax()]
        
        report += f"- 测试范围: {base_df['base_factor'].min():.1f} ~ {base_df['base_factor'].max():.1f}\n"
        report += f"- 长度变化范围: {base_range:.2f} 海里\n"
        report += f"- 最短配置: 基础延伸 {base_min['base_factor']:.1f} ({base_min['total_length']:.2f} 海里)\n"
        report += f"- 最长配置: 基础延伸 {base_max['base_factor']:.1f} ({base_max['total_length']:.2f} 海里)\n"
        report += f"- 相对变化: {(base_range/base_df['total_length'].mean())*100:.2f}%\n\n"
    
    report += "## 主要发现\n\n"
    
    # 找出最敏感的参数
    sensitivities = []
    if len(safety_df) > 0:
        safety_sens = (safety_df['total_length'].max() - safety_df['total_length'].min()) / safety_df['total_length'].mean()
        sensitivities.append(('安全边距系数', safety_sens))
    
    if len(angle_df) > 0:
        angle_sens = (angle_df['total_length'].max() - angle_df['total_length'].min()) / angle_df['total_length'].mean()
        sensitivities.append(('角度修正系数', angle_sens))
    
    if len(base_df) > 0:
        base_sens = (base_df['total_length'].max() - base_df['total_length'].min()) / base_df['total_length'].mean()
        sensitivities.append(('基础延伸系数', base_sens))
    
    if sensitivities:
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        report += f"1. **最敏感参数**: {sensitivities[0][0]} (敏感度: {sensitivities[0][1]:.4f})\n"
        if len(sensitivities) > 1:
            report += f"2. **次敏感参数**: {sensitivities[1][0]} (敏感度: {sensitivities[1][1]:.4f})\n"
        if len(sensitivities) > 2:
            report += f"3. **最不敏感参数**: {sensitivities[2][0]} (敏感度: {sensitivities[2][1]:.4f})\n"
    
    report += "\n## 实用建议\n\n"
    report += "1. **参数调优顺序**: 从敏感度最高的参数开始调节\n"
    report += "2. **保守策略**: 增大参数值可提高覆盖安全性但会增加测线长度\n"
    report += "3. **激进策略**: 减小参数值可缩短测线长度但可能影响覆盖效果\n"
    report += "4. **平衡策略**: 根据实际需求在安全性和效率之间找到平衡点\n"
    
    # 保存报告
    with open('sensitivity_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("汇总报告已保存: sensitivity_analysis_report.md")

def main():
    """主函数"""
    print("测线延伸参数灵敏度分析")
    print("=" * 50)
    
    # 运行各项分析
    safety_df = analyze_safety_margin_sensitivity()
    angle_df = analyze_angle_correction_sensitivity()
    base_df = analyze_extension_base_sensitivity()
    
    # 保存数据
    safety_df.to_csv('safety_margin_analysis.csv', index=False)
    angle_df.to_csv('angle_correction_analysis.csv', index=False)
    base_df.to_csv('extension_base_analysis.csv', index=False)
    
    print("\n详细数据已保存到CSV文件")
    
    # 生成可视化和报告
    create_summary_visualization(safety_df, angle_df, base_df)
    generate_summary_report(safety_df, angle_df, base_df)
    
    print("\n" + "=" * 50)
    print("灵敏度分析完成！")
    print("生成文件:")
    print("- sensitivity_analysis_summary.png (可视化图表)")
    print("- sensitivity_analysis_report.md (分析报告)")
    print("- *.csv (详细数据)")
    print("=" * 50)

if __name__ == '__main__':
    main() 