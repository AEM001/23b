#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测线延伸参数灵敏度分析主程序
系统性地测试不同参数组合对测线方案性能的影响
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from parametric_survey_line_generator import generate_survey_lines_with_parameters
from sensitivity_analysis_config import (
    get_all_test_configs, 
    get_config_description,
    BASELINE_CONFIG,
    SAFETY_MARGIN_VARIATIONS,
    ANGLE_CORRECTION_VARIATIONS,
    MAX_ANGLE_FACTOR_VARIATIONS,
    EXTENSION_BASE_VARIATIONS,
    MIN_DIRECTION_VARIATIONS
)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SensitivityAnalyzer:
    """灵敏度分析器"""
    
    def __init__(self, output_dir="sensitivity_results"):
        """初始化分析器"""
        self.output_dir = output_dir
        self.results = []
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"灵敏度分析结果将保存到: {output_dir}")
    
    def run_single_analysis(self, config_name, config, save_lines=False):
        """
        运行单个配置的分析
        
        参数:
        - config_name: 配置名称
        - config: 参数配置字典
        - save_lines: 是否保存测线数据
        
        返回:
        - 结果字典
        """
        try:
            print(f"  正在分析配置: {config_name}")
            print(f"  参数: {config}")
            
            # 生成测线
            lines_df, param_summary = generate_survey_lines_with_parameters(
                config,
                output_filename=f"{self.output_dir}/lines_{config_name}.csv" if save_lines else None
            )
            
            # 计算基础指标
            total_length = lines_df['length_optimized_nm'].sum()
            total_lines = len(lines_df)
            
            # 准备结果
            result = {
                'config_name': config_name,
                'description': get_config_description(config_name, config),
                'total_length_nm': total_length,
                'total_length_km': total_length * 1.852,
                'total_lines': total_lines,
                **config,  # 包含所有参数
                'generation_success': True,
                'error_message': None
            }
            
            print(f"  ✓ 成功生成 {total_lines} 条测线，总长度 {total_length:.2f} 海里")
            return result
            
        except Exception as e:
            print(f"  ✗ 配置 {config_name} 分析失败: {str(e)}")
            result = {
                'config_name': config_name,
                'description': get_config_description(config_name, config),
                'total_length_nm': np.nan,
                'total_length_km': np.nan,
                'total_lines': np.nan,
                **config,
                'generation_success': False,
                'error_message': str(e)
            }
            return result
    
    def run_comprehensive_analysis(self, max_configs=None, save_individual_lines=False):
        """
        运行全面的灵敏度分析
        
        参数:
        - max_configs: 最大测试配置数（用于快速测试）
        - save_individual_lines: 是否保存每个配置的测线数据
        """
        print("=== 开始测线延伸参数灵敏度分析 ===")
        
        # 获取所有测试配置
        all_configs = get_all_test_configs()
        if max_configs:
            all_configs = all_configs[:max_configs]
            print(f"快速测试模式：只测试前 {max_configs} 个配置")
        
        print(f"总共将测试 {len(all_configs)} 个参数配置\n")
        
        # 逐个运行分析
        for i, (config_name, config) in enumerate(all_configs, 1):
            print(f"[{i}/{len(all_configs)}] 配置: {config_name}")
            
            result = self.run_single_analysis(
                config_name, 
                config, 
                save_lines=save_individual_lines
            )
            self.results.append(result)
            print()
        
        # 转换为DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        # 保存原始结果
        output_file = f"{self.output_dir}/sensitivity_analysis_results.csv"
        self.results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"原始分析结果已保存到: {output_file}")
        
        return self.results_df
    
    def analyze_parameter_sensitivity(self):
        """分析各参数的灵敏度"""
        if not hasattr(self, 'results_df'):
            print("请先运行 run_comprehensive_analysis()")
            return
        
        print("=== 参数灵敏度分析 ===")
        
        # 筛选成功的结果
        success_df = self.results_df[self.results_df['generation_success'] == True].copy()
        
        if len(success_df) == 0:
            print("没有成功的配置，无法进行灵敏度分析")
            return
        
        # 找到基准配置结果
        baseline_result = success_df[success_df['config_name'] == 'baseline']
        if len(baseline_result) == 0:
            print("警告：未找到基准配置结果")
            baseline_length = success_df['total_length_nm'].median()
        else:
            baseline_length = baseline_result['total_length_nm'].iloc[0]
        
        print(f"基准配置测线长度: {baseline_length:.2f} 海里")
        
        # 计算相对变化
        success_df['length_change_ratio'] = (success_df['total_length_nm'] - baseline_length) / baseline_length
        success_df['length_change_percent'] = success_df['length_change_ratio'] * 100
        
        # 分析各参数的影响
        sensitivity_analysis = {}
        
        parameter_groups = {
            'safety_margin_factor': [v['safety_margin_factor'] for v in SAFETY_MARGIN_VARIATIONS],
            'angle_correction_factor': [v['angle_correction_factor'] for v in ANGLE_CORRECTION_VARIATIONS],
            'max_angle_factor': [v['max_angle_factor'] for v in MAX_ANGLE_FACTOR_VARIATIONS],
            'extension_base_factor': [v['extension_base_factor'] for v in EXTENSION_BASE_VARIATIONS],
            'min_direction_component': [v['min_direction_component'] for v in MIN_DIRECTION_VARIATIONS]
        }
        
        for param_name, param_values in parameter_groups.items():
            # 找到这个参数的所有变化配置
            param_configs = []
            for _, row in success_df.iterrows():
                if any(abs(row[param_name] - v) < 1e-6 for v in param_values):
                    # 检查其他参数是否为基准值
                    is_single_param_variation = True
                    for other_param, baseline_val in BASELINE_CONFIG.items():
                        if other_param != param_name and abs(row[other_param] - baseline_val) > 1e-6:
                            is_single_param_variation = False
                            break
                    
                    if is_single_param_variation:
                        param_configs.append({
                            'param_value': row[param_name],
                            'total_length': row['total_length_nm'],
                            'length_change_percent': row['length_change_percent']
                        })
            
            if param_configs:
                param_df = pd.DataFrame(param_configs).sort_values('param_value')
                
                # 计算敏感度指标
                length_range = param_df['total_length'].max() - param_df['total_length'].min()
                param_range = param_df['param_value'].max() - param_df['param_value'].min()
                
                sensitivity_score = length_range / baseline_length if baseline_length > 0 else 0
                
                sensitivity_analysis[param_name] = {
                    'sensitivity_score': sensitivity_score,
                    'length_range_nm': length_range,
                    'param_range': param_range,
                    'max_change_percent': param_df['length_change_percent'].abs().max(),
                    'param_configs': param_df
                }
                
                print(f"\n{param_name}:")
                print(f"  敏感度评分: {sensitivity_score:.4f}")
                print(f"  最大长度变化: ±{param_df['length_change_percent'].abs().max():.1f}%")
                print(f"  长度范围: {length_range:.1f} 海里")
        
        self.sensitivity_analysis = sensitivity_analysis
        return sensitivity_analysis
    
    def generate_visualizations(self):
        """生成可视化图表"""
        if not hasattr(self, 'results_df') or not hasattr(self, 'sensitivity_analysis'):
            print("请先运行分析")
            return
        
        print("=== 生成可视化图表 ===")
        
        # 筛选成功的结果
        success_df = self.results_df[self.results_df['generation_success'] == True].copy()
        
        # 1. 参数敏感度条形图
        plt.figure(figsize=(12, 8))
        
        param_names = list(self.sensitivity_analysis.keys())
        sensitivity_scores = [self.sensitivity_analysis[p]['sensitivity_score'] for p in param_names]
        max_changes = [self.sensitivity_analysis[p]['max_change_percent'] for p in param_names]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 敏感度评分
        bars1 = ax1.bar(range(len(param_names)), sensitivity_scores, color='skyblue', alpha=0.7)
        ax1.set_xlabel('参数')
        ax1.set_ylabel('敏感度评分')
        ax1.set_title('各参数敏感度评分对比')
        ax1.set_xticks(range(len(param_names)))
        ax1.set_xticklabels([self._format_param_name(p) for p in param_names], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, score in zip(bars1, sensitivity_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 最大变化百分比
        bars2 = ax2.bar(range(len(param_names)), max_changes, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('参数')
        ax2.set_ylabel('最大长度变化 (%)')
        ax2.set_title('各参数导致的最大测线长度变化')
        ax2.set_xticks(range(len(param_names)))
        ax2.set_xticklabels([self._format_param_name(p) for p in param_names], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, change in zip(bars2, max_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{change:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/parameter_sensitivity_comparison.png", dpi=300, bbox_inches='tight')
        print(f"参数敏感度对比图已保存到: {self.output_dir}/parameter_sensitivity_comparison.png")
        
        # 2. 各参数的详细变化曲线
        self._plot_parameter_curves()
        
        # 3. 配置方案对比
        self._plot_config_comparison()
        
        plt.close('all')  # 关闭所有图形以释放内存
    
    def _plot_parameter_curves(self):
        """绘制各参数的详细变化曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        param_names = list(self.sensitivity_analysis.keys())
        
        for i, param_name in enumerate(param_names):
            if i >= len(axes):
                break
                
            param_data = self.sensitivity_analysis[param_name]['param_configs']
            
            axes[i].plot(param_data['param_value'], param_data['total_length'], 
                        'o-', linewidth=2, markersize=6, color='darkblue')
            axes[i].set_xlabel(self._format_param_name(param_name))
            axes[i].set_ylabel('测线总长度 (海里)')
            axes[i].set_title(f'{self._format_param_name(param_name)} 影响分析')
            axes[i].grid(True, alpha=0.3)
            
            # 标记基准值
            baseline_value = BASELINE_CONFIG[param_name]
            baseline_idx = param_data['param_value'].sub(baseline_value).abs().idxmin()
            baseline_length = param_data.loc[baseline_idx, 'total_length']
            axes[i].axvline(x=baseline_value, color='red', linestyle='--', alpha=0.7, label='基准值')
            axes[i].axhline(y=baseline_length, color='red', linestyle='--', alpha=0.7)
            axes[i].legend()
        
        # 隐藏多余的子图
        for i in range(len(param_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/parameter_curves_detailed.png", dpi=300, bbox_inches='tight')
        print(f"参数变化曲线图已保存到: {self.output_dir}/parameter_curves_detailed.png")
    
    def _plot_config_comparison(self):
        """绘制关键配置方案对比"""
        success_df = self.results_df[self.results_df['generation_success'] == True].copy()
        
        # 选择关键配置进行对比
        key_configs = ['baseline', 'conservative', 'aggressive', 'balanced']
        key_data = success_df[success_df['config_name'].isin(key_configs)].copy()
        
        if len(key_data) == 0:
            print("未找到关键配置数据，跳过配置对比图")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 创建分组条形图
        x_pos = np.arange(len(key_data))
        
        plt.bar(x_pos, key_data['total_length_nm'], color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
        plt.xlabel('配置方案')
        plt.ylabel('测线总长度 (海里)')
        plt.title('关键配置方案测线长度对比')
        plt.xticks(x_pos, [get_config_description(name, {}) for name in key_data['config_name']])
        
        # 添加数值标签
        for i, (_, row) in enumerate(key_data.iterrows()):
            plt.text(i, row['total_length_nm'] + 2, f"{row['total_length_nm']:.1f}", 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/key_configs_comparison.png", dpi=300, bbox_inches='tight')
        print(f"关键配置对比图已保存到: {self.output_dir}/key_configs_comparison.png")
    
    def _format_param_name(self, param_name):
        """格式化参数名称用于显示"""
        name_mapping = {
            'safety_margin_factor': '安全边距系数',
            'angle_correction_factor': '角度修正系数', 
            'max_angle_factor': '最大角度系数',
            'extension_base_factor': '基础延伸系数',
            'min_direction_component': '最小方向分量'
        }
        return name_mapping.get(param_name, param_name)
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        if not hasattr(self, 'results_df') or not hasattr(self, 'sensitivity_analysis'):
            print("请先运行分析")
            return
        
        print("=== 生成综合分析报告 ===")
        
        success_df = self.results_df[self.results_df['generation_success'] == True].copy()
        
        # 创建报告内容
        report_content = f"""# 测线延伸参数灵敏度分析报告

## 分析概述

本报告对测线边界智能扩展优化算法中的关键参数进行了系统性的灵敏度分析。

**分析时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
**测试配置总数**: {len(self.results_df)}
**成功配置数**: {len(success_df)}
**失败配置数**: {len(self.results_df) - len(success_df)}

## 参数敏感度排序

根据敏感度评分（参数变化对测线长度影响的相对大小），参数重要性排序如下：

"""
        
        # 按敏感度评分排序
        sorted_params = sorted(self.sensitivity_analysis.items(), 
                             key=lambda x: x[1]['sensitivity_score'], reverse=True)
        
        for i, (param_name, analysis) in enumerate(sorted_params, 1):
            report_content += f"{i}. **{self._format_param_name(param_name)}**\n"
            report_content += f"   - 敏感度评分: {analysis['sensitivity_score']:.4f}\n"
            report_content += f"   - 最大长度变化: ±{analysis['max_change_percent']:.1f}%\n"
            report_content += f"   - 长度变化范围: {analysis['length_range_nm']:.1f} 海里\n\n"
        
        # 关键发现
        report_content += "## 关键发现\n\n"
        
        most_sensitive = sorted_params[0]
        least_sensitive = sorted_params[-1]
        
        report_content += f"1. **最敏感参数**: {self._format_param_name(most_sensitive[0])}，敏感度评分 {most_sensitive[1]['sensitivity_score']:.4f}\n"
        report_content += f"2. **最不敏感参数**: {self._format_param_name(least_sensitive[0])}，敏感度评分 {least_sensitive[1]['sensitivity_score']:.4f}\n"
        
        # 基准配置性能
        baseline_result = success_df[success_df['config_name'] == 'baseline']
        if len(baseline_result) > 0:
            baseline_length = baseline_result['total_length_nm'].iloc[0]
            report_content += f"3. **基准配置性能**: 测线总长度 {baseline_length:.2f} 海里\n"
        
        # 最优和最差配置
        best_config = success_df.loc[success_df['total_length_nm'].idxmin()]
        worst_config = success_df.loc[success_df['total_length_nm'].idxmax()]
        
        report_content += f"4. **最短测线配置**: {best_config['config_name']} ({best_config['total_length_nm']:.2f} 海里)\n"
        report_content += f"5. **最长测线配置**: {worst_config['config_name']} ({worst_config['total_length_nm']:.2f} 海里)\n"
        report_content += f"6. **长度差异**: {worst_config['total_length_nm'] - best_config['total_length_nm']:.2f} 海里 ({((worst_config['total_length_nm'] - best_config['total_length_nm'])/best_config['total_length_nm']*100):.1f}%)\n\n"
        
        # 参数建议
        report_content += "## 参数调优建议\n\n"
        report_content += "基于灵敏度分析结果，对各参数提出以下调优建议：\n\n"
        
        for param_name, analysis in sorted_params:
            formatted_name = self._format_param_name(param_name)
            param_data = analysis['param_configs']
            optimal_config = param_data.loc[param_data['total_length'].idxmin()]
            
            report_content += f"### {formatted_name}\n"
            report_content += f"- **敏感度**: {'高' if analysis['sensitivity_score'] > 0.05 else '中' if analysis['sensitivity_score'] > 0.02 else '低'}\n"
            report_content += f"- **最优值**: {optimal_config['param_value']:.3f}\n"
            report_content += f"- **基准值**: {BASELINE_CONFIG[param_name]:.3f}\n"
            report_content += f"- **建议**: {'需要重点关注和精细调节' if analysis['sensitivity_score'] > 0.05 else '可以适度调节' if analysis['sensitivity_score'] > 0.02 else '对结果影响较小，可保持默认值'}\n\n"
        
        # 实用性总结
        report_content += "## 实用性总结\n\n"
        report_content += "1. **高敏感度参数**应优先进行精细调节，对测线方案优化效果显著\n"
        report_content += "2. **低敏感度参数**可以保持默认值，避免过度优化\n"
        report_content += "3. **参数组合效应**可能存在，建议在关键参数基础上进行组合测试\n"
        report_content += "4. **实际应用**中建议根据具体海域特征和作业要求选择合适的参数配置\n\n"
        
        # 数据表格
        report_content += "## 详细数据表\n\n"
        
        # 生成成功配置的简化表格
        summary_df = success_df[['config_name', 'description', 'total_length_nm', 'total_lines']].copy()
        summary_df['total_length_nm'] = summary_df['total_length_nm'].round(2)
        summary_df.columns = ['配置名称', '配置描述', '测线长度(海里)', '测线数量']
        
        report_content += summary_df.to_markdown(index=False)
        report_content += "\n\n"
        
        # 保存报告
        report_file = f"{self.output_dir}/sensitivity_analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"综合分析报告已保存到: {report_file}")
        
        return report_content

def main():
    """主函数"""
    print("=" * 60)
    print("测线延伸参数灵敏度分析系统")
    print("=" * 60)
    
    # 创建分析器
    analyzer = SensitivityAnalyzer()
    
    try:
        # 获取所有测试配置
        all_configs = get_all_test_configs()
        print(f"总共将测试 {len(all_configs)} 个参数配置\n")
        
        # 逐个运行分析
        for i, (config_name, config) in enumerate(all_configs[:5], 1):  # 先测试前5个
            print(f"[{i}/5] 配置: {config_name}")
            
            result = analyzer.run_single_analysis(config_name, config, save_lines=False)
            analyzer.results.append(result)
            print()
        
        # 转换为DataFrame并保存
        results_df = pd.DataFrame(analyzer.results)
        output_file = f"{analyzer.output_dir}/sensitivity_analysis_results.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"分析结果已保存到: {output_file}")
        
        print("\n" + "=" * 60)
        print("初步灵敏度分析完成！")
        print(f"结果已保存到目录: {analyzer.output_dir}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 