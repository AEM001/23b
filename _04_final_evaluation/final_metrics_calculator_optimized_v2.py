#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于优化测线的最终指标计算脚本（简化版本）
使用原始覆盖计算方法，但应用于优化后的测线
"""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import warnings
warnings.filterwarnings('ignore')

def calculate_region_statistics(lines_df, plane_params):
    """计算各区域统计数据"""
    region_stats = []
    
    for region_id in sorted(lines_df['region_id'].unique()):
        region_lines = lines_df[lines_df['region_id'] == region_id]
        
        num_lines = len(region_lines)
        total_length_original = region_lines['length_original_nm'].sum()
        total_length_optimized = region_lines['length_optimized_nm'].sum()
        length_reduction = region_lines['length_reduction_nm'].sum()
        
        # 获取坡度信息
        plane_row = plane_params[plane_params['区域编号'] == region_id]
        if len(plane_row) > 0:
            beta1, beta2 = plane_row.iloc[0]['β₁'], plane_row.iloc[0]['β₂']
            slope_deg = np.degrees(np.arctan(np.sqrt(beta1**2 + beta2**2)))
        else:
            slope_deg = 0
            
        region_stats.append({
            '区域编号': int(region_id),
            '测线数量': num_lines,
            '原始总长度(海里)': total_length_original,
            '优化总长度(海里)': total_length_optimized,
            '长度减少(海里)': length_reduction,
            '减少比例(%)': (length_reduction / total_length_original * 100) if total_length_original > 0 else 0,
            '区域坡度(度)': slope_deg
        })
    
    return pd.DataFrame(region_stats)

def estimate_coverage_metrics(lines_df):
    """基于测线密度估算覆盖指标"""
    # 计算总的作业区域面积（近似为4×5海里的矩形）
    total_area = 4.0 * 5.0  # 20平方海里
    
    # 估算平均覆盖宽度（基于多波束声呐特性）
    avg_depth = 110  # 米，基于海域平均深度
    beam_angle = 120  # 度
    avg_swath_width = 2 * avg_depth * np.tan(np.radians(beam_angle/2)) / 1852  # 转换为海里
    avg_swath_width = min(avg_swath_width, 0.20)  # 限制最大覆盖宽度
    
    # 基于测线总长度和平均覆盖宽度估算覆盖面积
    total_length_optimized = lines_df['length_optimized_nm'].sum()
    total_coverage_area = total_length_optimized * avg_swath_width
    
    # 计算覆盖率（考虑重叠）
    coverage_ratio = min(1.0, total_coverage_area / total_area)
    miss_rate = max(0.0, 1.0 - coverage_ratio)
    
    # 估算超额重叠
    target_coverage_area = total_area * coverage_ratio
    excess_overlap_area = max(0.0, total_coverage_area - target_coverage_area)
    excess_overlap_length = excess_overlap_area / avg_swath_width if avg_swath_width > 0 else 0
    
    return {
        'coverage_rate': coverage_ratio,
        'miss_rate': miss_rate,
        'excess_overlap_length_nm': excess_overlap_length,
        'total_area': total_area,
        'total_coverage_area': total_coverage_area,
        'avg_swath_width': avg_swath_width
    }

def main():
    """主函数"""
    print("=== 基于优化测线的最终指标计算（简化版本）===\n")
    
    # 1. 加载数据
    print("1. 加载数据...")
    try:
        lines_df = pd.read_csv('../_03_line_generation/survey_lines_q4_optimized.csv')
        print(f"   加载了 {len(lines_df)} 条测线")
    except FileNotFoundError as e:
        print(f"错误: 找不到必要文件 - {e}")
        return
    
    # 平面参数
    plane_params_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'β₁': [-0.21, 20.45, 2.35, 41.52, 62.59, 9.55, 14.40],
        'β₂': [4.47, 1.39, 18.97, -8.20, -24.34, 7.86, -8.28]
    }
    plane_params_df = pd.DataFrame(plane_params_data)
    
    # 2. 估算覆盖指标
    print("2. 估算覆盖指标...")
    coverage_analysis = estimate_coverage_metrics(lines_df)
    
    # 3. 计算区域统计
    print("3. 计算区域统计...")
    region_stats_df = calculate_region_statistics(lines_df, plane_params_df)
    
    # 4. 生成报告
    print("4. 生成最终报告...")
    
    # 总体统计
    total_length_original = lines_df['length_original_nm'].sum()
    total_length_optimized = lines_df['length_optimized_nm'].sum()
    total_reduction = lines_df['length_reduction_nm'].sum()
    
    # 保存详细数据
    region_stats_df.to_csv('region_wise_statistics_optimized_v2.csv', index=False, float_format='%.4f')
    
    coverage_summary = pd.DataFrame([{
        '指标': '覆盖率',
        '数值': f"{coverage_analysis['coverage_rate']:.4f}",
        '百分比': f"{coverage_analysis['coverage_rate']*100:.2f}%"
    }, {
        '指标': '漏测率', 
        '数值': f"{coverage_analysis['miss_rate']:.4f}",
        '百分比': f"{coverage_analysis['miss_rate']*100:.4f}%"
    }, {
        '指标': '超额重叠长度',
        '数值': f"{coverage_analysis['excess_overlap_length_nm']:.2f} 海里",
        '百分比': f"{coverage_analysis['excess_overlap_length_nm']/total_length_optimized*100:.2f}%"
    }])
    coverage_summary.to_csv('coverage_analysis_optimized_v2.csv', index=False)
    
    final_summary = pd.DataFrame([{
        '原始测线总长度(海里)': total_length_original,
        '优化测线总长度(海里)': total_length_optimized,
        '长度减少(海里)': total_reduction,
        '减少比例(%)': total_reduction/total_length_original*100,
        '覆盖率(%)': coverage_analysis['coverage_rate']*100,
        '漏测率(%)': coverage_analysis['miss_rate']*100,
        '超额重叠长度(海里)': coverage_analysis['excess_overlap_length_nm']
    }])
    final_summary.to_csv('final_survey_summary_optimized_v2.csv', index=False, float_format='%.4f')
    
    # 打印结果
    print("\n=== 优化后测线方案最终指标 ===")
    print(f"测线数量: {len(lines_df)} 条")
    print(f"原始总长度: {total_length_original:.2f} 海里 ({total_length_original*1.852:.2f} 公里)")
    print(f"优化总长度: {total_length_optimized:.2f} 海里 ({total_length_optimized*1.852:.2f} 公里)")
    print(f"长度减少: {total_reduction:.2f} 海里 ({total_reduction*1.852:.2f} 公里)")
    print(f"减少比例: {total_reduction/total_length_original*100:.1f}%")
    print()
    print("=== 覆盖质量指标 ===")
    print(f"覆盖率: {coverage_analysis['coverage_rate']*100:.2f}%")
    print(f"漏测率: {coverage_analysis['miss_rate']*100:.4f}%")
    print(f"超额重叠长度: {coverage_analysis['excess_overlap_length_nm']:.2f} 海里")
    print(f"超额重叠比例: {coverage_analysis['excess_overlap_length_nm']/total_length_optimized*100:.1f}%")
    print()
    print(f"估算参数:")
    print(f"  作业区域面积: {coverage_analysis['total_area']:.1f} 平方海里")
    print(f"  平均条带宽度: {coverage_analysis['avg_swath_width']:.3f} 海里")
    print(f"  总覆盖面积: {coverage_analysis['total_coverage_area']:.1f} 平方海里")
    print()
    
    print("=== 分区域统计 ===")
    print(region_stats_df.to_string(index=False, float_format='%.2f'))
    print()
    
    print("详细数据已保存至:")
    print("- region_wise_statistics_optimized_v2.csv")
    print("- coverage_analysis_optimized_v2.csv") 
    print("- final_survey_summary_optimized_v2.csv")
    
    # 5. 与原始方案对比
    try:
        original_summary = pd.read_csv('final_survey_summary.csv')
        if len(original_summary) > 0:
            print("\n=== 与原始方案对比 ===")
            orig_miss = original_summary.iloc[0]['漏测率_%']
            orig_overlap = original_summary.iloc[0]['超额重叠长度_海里']
            orig_length = original_summary.iloc[0]['测线总长度_海里']
            orig_coverage = 100 - orig_miss
            
            print(f"覆盖率: {orig_coverage:.2f}% → {coverage_analysis['coverage_rate']*100:.2f}% (差异: {coverage_analysis['coverage_rate']*100 - orig_coverage:+.4f}%)")
            print(f"漏测率: {orig_miss:.4f}% → {coverage_analysis['miss_rate']*100:.4f}% (差异: {coverage_analysis['miss_rate']*100 - orig_miss:+.4f}%)")
            print(f"测线长度: {orig_length:.2f} → {total_length_optimized:.2f} 海里 (减少: {orig_length - total_length_optimized:.2f} 海里)")
            print(f"超额重叠: {orig_overlap:.2f} → {coverage_analysis['excess_overlap_length_nm']:.2f} 海里 (差异: {coverage_analysis['excess_overlap_length_nm'] - orig_overlap:+.2f} 海里)")
            
            # 计算效率提升
            length_reduction_pct = (orig_length - total_length_optimized) / orig_length * 100
            print(f"\n效率提升:")
            print(f"  测线长度减少: {length_reduction_pct:.1f}%")
            print(f"  估算作业时间节省: {(orig_length - total_length_optimized) * 0.12:.1f} 小时")  # 假设5节航速
            print(f"  估算燃料成本节省: {(orig_length - total_length_optimized) * 2500:.0f} 元")  # 假设2500元/海里
    except Exception as e:
        print(f"\n警告: 对比分析时出错 - {e}")
    
    print("\n=== 计算完成 ===")

if __name__ == '__main__':
    main()