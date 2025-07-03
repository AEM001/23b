#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的测线延伸参数灵敏度分析
运行更多配置并生成详细的分析报告和可视化
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from parametric_survey_line_generator import generate_survey_lines_with_parameters
from sensitivity_analysis_config import (
    get_all_test_configs, 
    get_config_description,
    BASELINE_CONFIG,
    SAFETY_MARGIN_VARIATIONS,
    ANGLE_CORRECTION_VARIATIONS,
    EXTENSION_BASE_VARIATIONS
)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_comprehensive_analysis(max_configs=20):
    """运行全面的灵敏度分析"""
    print("=" * 70)
    print("完整的测线延伸参数灵敏度分析")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = "comprehensive_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取配置
    all_configs = get_all_test_configs()[:max_configs]
    print(f"将测试 {len(all_configs)} 个参数配置\n")
    
    results = []
    
    # 运行分析
    for i, (config_name, config) in enumerate(all_configs, 1):
        print(f"[{i}/{len(all_configs)}] 分析配置: {config_name}")
        
        try:
            lines_df, param_summary = generate_survey_lines_with_parameters(config)
            total_length = lines_df['length_optimized_nm'].sum()
            
            result = {
                'config_name': config_name,
                'description': get_config_description(config_name, config),
                'total_length_nm': total_length,
                'total_length_km': total_length * 1.852,
                'total_lines': len(lines_df),
                **config,
                'success': True
            }
            
            print(f"  ✓ 成功: {total_length:.2f} 海里")
            
        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            result = {
                'config_name': config_name,
                'description': get_config_description(config_name, config),
                'total_length_nm': np.nan,
                'total_length_km': np.nan,
                'total_lines': np.nan,
                **config,
                'success': False
            }
        
        results.append(result)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    success_df = results_df[results_df['success'] == True].copy()
    
    # 保存详细结果
    results_df.to_csv(f"{output_dir}/detailed_results.csv", index=False, encoding='utf-8-sig')
    
    print(f"\n成功配置: {len(success_df)}/{len(results_df)}")
    
    return success_df, output_dir

def analyze_parameter_effects(success_df, output_dir):
    """分析各参数的影响"""
    print("\n=== 参数影响分析 ===")
    
    # 找到基准配置
    baseline = success_df[success_df['config_name'] == 'baseline']
    if len(baseline) == 0:
        baseline_length = success_df['total_length_nm'].median()
    else:
        baseline_length = baseline['total_length_nm'].iloc[0]
    
    print(f"基准长度: {baseline_length:.2f} 海里")
    
    # 分析安全边距系数的影响
    safety_analysis = analyze_single_parameter(
        success_df, 'safety_margin_factor', 
        [v['safety_margin_factor'] for v in SAFETY_MARGIN_VARIATIONS],
        baseline_length
    )
    
    # 分析角度修正系数的影响
    angle_analysis = analyze_single_parameter(
        success_df, 'angle_correction_factor',
        [v['angle_correction_factor'] for v in ANGLE_CORRECTION_VARIATIONS],
        baseline_length
    )
    
    # 分析基础延伸系数的影响
    extension_analysis = analyze_single_parameter(
        success_df, 'extension_base_factor',
        [v['extension_base_factor'] for v in EXTENSION_BASE_VARIATIONS],
        baseline_length
    )
    
    # 生成参数分析报告
    analysis_results = {
        'safety_margin_factor': safety_analysis,
        'angle_correction_factor': angle_analysis,
        'extension_base_factor': extension_analysis
    }
    
    generate_parameter_report(analysis_results, baseline_length, output_dir)
    
    return analysis_results

def analyze_single_parameter(df, param_name, param_values, baseline_length):
    """分析单个参数的影响"""
    param_data = []
    
    for param_value in param_values:
        # 找到符合条件的配置（该参数为指定值，其他参数为基准值）
        matches = df[abs(df[param_name] - param_value) < 1e-6]
        
        # 进一步筛选：其他参数应该是基准值
        filtered_matches = []
        for _, row in matches.iterrows():
            is_single_param_change = True
            for other_param, baseline_val in BASELINE_CONFIG.items():
                if other_param != param_name and abs(row[other_param] - baseline_val) > 1e-6:
                    is_single_param_change = False
                    break
            
            if is_single_param_change:
                filtered_matches.append(row)
        
        if filtered_matches:
            row = filtered_matches[0]  # 取第一个匹配的
            length = row['total_length_nm']
            change_percent = (length - baseline_length) / baseline_length * 100
            
            param_data.append({
                'param_value': param_value,
                'total_length': length,
                'change_percent': change_percent
            })
    
    if param_data:
        param_df = pd.DataFrame(param_data).sort_values('param_value')
        
        # 计算敏感度指标
        length_range = param_df['total_length'].max() - param_df['total_length'].min()
        param_range = param_df['param_value'].max() - param_df['param_value'].min()
        sensitivity_score = length_range / baseline_length if baseline_length > 0 else 0
        max_change = param_df['change_percent'].abs().max()
        
        return {
            'data': param_df,
            'sensitivity_score': sensitivity_score,
            'length_range': length_range,
            'param_range': param_range,
            'max_change_percent': max_change
        }
    
    return None

def generate_parameter_report(analysis_results, baseline_length, output_dir):
    """生成参数分析报告"""
    report_content = f"""# 测线延伸参数灵敏度分析报告

## 分析概述

**分析时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
**基准配置测线长度**: {baseline_length:.2f} 海里

## 参数敏感度排序

"""
    
    # 按敏感度排序
    valid_analyses = {k: v for k, v in analysis_results.items() if v is not None}
    sorted_params = sorted(valid_analyses.items(), 
                          key=lambda x: x[1]['sensitivity_score'], reverse=True)
    
    for i, (param_name, analysis) in enumerate(sorted_params, 1):
        param_display = {
            'safety_margin_factor': '安全边距系数',
            'angle_correction_factor': '角度修正系数',
            'extension_base_factor': '基础延伸系数'
        }.get(param_name, param_name)
        
        report_content += f"### {i}. {param_display}\n"
        report_content += f"- **敏感度评分**: {analysis['sensitivity_score']:.4f}\n"
        report_content += f"- **最大变化**: ±{analysis['max_change_percent']:.1f}%\n"
        report_content += f"- **长度变化范围**: {analysis['length_range']:.1f} 海里\n"
        report_content += f"- **参数测试范围**: {analysis['param_range']:.3f}\n\n"
    
    report_content += "## 详细分析结果\n\n"
    
    for param_name, analysis in valid_analyses.items():
        param_display = {
            'safety_margin_factor': '安全边距系数',
            'angle_correction_factor': '角度修正系数', 
            'extension_base_factor': '基础延伸系数'
        }.get(param_name, param_name)
        
        report_content += f"### {param_display}详细数据\n\n"
        
        # 添加数据表格
        if 'data' in analysis and len(analysis['data']) > 0:
            data_table = analysis['data'][['param_value', 'total_length', 'change_percent']].copy()
            data_table.columns = ['参数值', '总长度(海里)', '变化(%)']
            data_table['总长度(海里)'] = data_table['总长度(海里)'].round(2)
            data_table['变化(%)'] = data_table['变化(%)'].round(2)
            
            report_content += data_table.to_markdown(index=False)
            report_content += "\n\n"
    
    report_content += "## 参数调优建议\n\n"
    
    if sorted_params:
        most_sensitive = sorted_params[0]
        least_sensitive = sorted_params[-1]
        
        report_content += f"1. **最敏感参数**: {most_sensitive[0]}，建议优先调节\n"
        report_content += f"2. **最不敏感参数**: {least_sensitive[0]}，影响较小\n"
        report_content += f"3. **调节建议**: 从敏感度高的参数开始逐步调节\n"
        report_content += f"4. **实际应用**: 根据海域特点选择合适的参数组合\n"
    
    # 保存报告
    with open(f"{output_dir}/parameter_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"参数分析报告已保存: {output_dir}/parameter_analysis_report.md")

def create_visualizations(analysis_results, output_dir):
    """生成可视化图表"""
    print("\n=== 生成可视化图表 ===")
    
    valid_analyses = {k: v for k, v in analysis_results.items() if v is not None}
    
    if not valid_analyses:
        print("没有有效的分析结果，跳过可视化")
        return
    
    # 1. 敏感度对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    param_names = []
    sensitivity_scores = []
    max_changes = []
    
    for param_name, analysis in valid_analyses.items():
        param_display = {
            'safety_margin_factor': '安全边距系数',
            'angle_correction_factor': '角度修正系数',
            'extension_base_factor': '基础延伸系数'
        }.get(param_name, param_name)
        
        param_names.append(param_display)
        sensitivity_scores.append(analysis['sensitivity_score'])
        max_changes.append(analysis['max_change_percent'])
    
    # 敏感度评分
    bars1 = ax1.bar(param_names, sensitivity_scores, color='skyblue', alpha=0.7)
    ax1.set_ylabel('敏感度评分')
    ax1.set_title('参数敏感度评分对比')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars1, sensitivity_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{score:.4f}', ha='center', va='bottom')
    
    # 最大变化
    bars2 = ax2.bar(param_names, max_changes, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('最大长度变化 (%)')
    ax2.set_title('参数引起的最大测线长度变化')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, change in zip(bars2, max_changes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{change:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_comparison.png", dpi=300, bbox_inches='tight')
    print(f"敏感度对比图已保存: {output_dir}/sensitivity_comparison.png")
    
    # 2. 参数变化曲线
    fig, axes = plt.subplots(1, len(valid_analyses), figsize=(5*len(valid_analyses), 5))
    if len(valid_analyses) == 1:
        axes = [axes]
    
    for i, (param_name, analysis) in enumerate(valid_analyses.items()):
        param_display = {
            'safety_margin_factor': '安全边距系数',
            'angle_correction_factor': '角度修正系数',
            'extension_base_factor': '基础延伸系数'
        }.get(param_name, param_name)
        
        data = analysis['data']
        
        axes[i].plot(data['param_value'], data['total_length'], 'o-', 
                    linewidth=2, markersize=6, color='darkblue')
        axes[i].set_xlabel(param_display)
        axes[i].set_ylabel('测线总长度 (海里)')
        axes[i].set_title(f'{param_display}影响分析')
        axes[i].grid(True, alpha=0.3)
        
        # 标记基准值
        baseline_value = BASELINE_CONFIG[param_name]
        axes[i].axvline(x=baseline_value, color='red', linestyle='--', alpha=0.7, label='基准值')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/parameter_curves.png", dpi=300, bbox_inches='tight')
    print(f"参数变化曲线已保存: {output_dir}/parameter_curves.png")
    
    plt.close('all')

def main():
    """主函数"""
    # 运行分析
    success_df, output_dir = run_comprehensive_analysis(max_configs=25)
    
    if len(success_df) == 0:
        print("没有成功的配置，无法进行分析")
        return
    
    # 分析参数影响
    analysis_results = analyze_parameter_effects(success_df, output_dir)
    
    # 生成可视化
    create_visualizations(analysis_results, output_dir)
    
    print("\n" + "=" * 70)
    print("完整灵敏度分析完成！")
    print(f"所有结果已保存到: {output_dir}/")
    print("=" * 70)

if __name__ == '__main__':
    main() 