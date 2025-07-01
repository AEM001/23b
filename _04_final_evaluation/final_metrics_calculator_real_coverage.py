#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于原始格网数据的真实覆盖分析计算脚本
按照4solution.md第四步要求：基于原始的、真实的格网数据进行验算
"""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
import warnings
warnings.filterwarnings('ignore')

def load_original_data():
    """加载原始格网数据"""
    print("   - 加载原始格网数据...")
    df = pd.read_csv('/Users/Mac/Downloads/23b/_00_source_data/output.csv')
    
    # 转换为海里 (1海里 = 1852米，原始数据假设为公里)
    x_nm = df['横坐标'].values / 1.852  # 转换为海里
    y_nm = df['纵坐标'].values / 1.852  # 转换为海里
    depth = df['深度'].values
    
    print(f"   - 数据范围: X=[{x_nm.min():.3f}, {x_nm.max():.3f}] 海里, Y=[{y_nm.min():.3f}, {y_nm.max():.3f}] 海里")
    print(f"   - 水深范围: [{depth.min():.1f}, {depth.max():.1f}] 米")
    
    # 建立插值函数 D_true = f(x, y)
    interpolator = LinearNDInterpolator(list(zip(x_nm, y_nm)), depth)
    
    return interpolator, x_nm, y_nm, depth

def calculate_swath_width(depth, beam_angle=120):
    """根据水深计算条带宽度"""
    # 多波束条带宽度 = 2 * depth * tan(beam_angle/2)
    swath_width_m = 2 * depth * np.tan(np.radians(beam_angle / 2))
    swath_width_nm = swath_width_m / 1852  # 转换为海里
    return min(swath_width_nm, 0.2)  # 限制最大宽度0.2海里

def calculate_line_coverage(x_start, y_start, x_end, y_end, interpolator, num_samples=50):
    """计算单条测线的真实覆盖带"""
    # 沿测线采样
    t = np.linspace(0, 1, num_samples)
    x_samples = x_start + t * (x_end - x_start)
    y_samples = y_start + t * (y_end - y_start)
    
    # 查询真实水深
    depths = interpolator(x_samples, y_samples)
    valid_mask = ~np.isnan(depths)
    
    if not np.any(valid_mask):
        return [], []
    
    # 计算有效段的覆盖宽度
    valid_x = x_samples[valid_mask]
    valid_y = y_samples[valid_mask]
    valid_depths = depths[valid_mask]
    
    # 计算各点的条带宽度
    swath_widths = [calculate_swath_width(d) for d in valid_depths]
    
    # 返回覆盖点和宽度
    coverage_points = list(zip(valid_x, valid_y))
    return coverage_points, swath_widths

def calculate_overlap_and_gaps(lines_df, interpolator):
    """计算相邻测线间的重叠和空隙"""
    print("   - 计算测线间重叠和空隙...")
    
    total_area = 0
    covered_area = 0
    overlap_area = 0
    excess_overlap_length = 0
    
    # 创建覆盖网格 (0.01海里分辨率)
    grid_resolution = 0.01  # 海里
    x_min, x_max = 0, 2.2  # 海里
    y_min, y_max = 0, 2.7  # 海里
    
    x_grid = np.arange(x_min, x_max, grid_resolution)
    y_grid = np.arange(y_min, y_max, grid_resolution)
    coverage_count = np.zeros((len(y_grid), len(x_grid)))
    
    print(f"   - 网格大小: {len(x_grid)} x {len(y_grid)} = {len(x_grid)*len(y_grid)} 个格点")
    
    # 遍历每条测线
    for idx, row in lines_df.iterrows():
        if idx % 30 == 0:
            print(f"   - 处理测线 {idx+1}/{len(lines_df)}")
            
        x_start, y_start = row['x_start_nm'], row['y_start_nm'] 
        x_end, y_end = row['x_end_nm'], row['y_end_nm']
        
        # 计算测线覆盖
        coverage_points, swath_widths = calculate_line_coverage(
            x_start, y_start, x_end, y_end, interpolator, num_samples=30)
        
        if not coverage_points:
            continue
            
        # 将覆盖投影到网格
        for i, (x, y) in enumerate(coverage_points):
            swath_width = swath_widths[i]
            half_width = swath_width / 2
            
            # 找到影响的网格范围
            x_indices = np.where((x_grid >= x - half_width) & (x_grid <= x + half_width))[0]
            y_indices = np.where((y_grid >= y - half_width) & (y_grid <= y + half_width))[0]
            
            # 标记覆盖的网格点
            for yi in y_indices:
                for xi in x_indices:
                    # 计算距离测线的距离
                    dist = np.sqrt((x_grid[xi] - x)**2 + (y_grid[yi] - y)**2)
                    if dist <= half_width:
                        coverage_count[yi, xi] += 1
    
    # 统计覆盖情况
    total_points = len(x_grid) * len(y_grid) 
    covered_points = np.sum(coverage_count > 0)
    overlap_points = np.sum(coverage_count > 1)
    
    # 计算面积
    cell_area = grid_resolution ** 2  # 平方海里
    total_area = total_points * cell_area
    covered_area = covered_points * cell_area
    overlap_area = overlap_points * cell_area
    
    # 计算漏测率
    miss_rate = (total_points - covered_points) / total_points
    coverage_rate = covered_points / total_points
    
    # 估算超额重叠长度 (基于重叠面积)
    avg_swath_width = 0.15  # 海里，估算平均条带宽度
    excess_overlap_length = overlap_area / avg_swath_width if avg_swath_width > 0 else 0
    
    print(f"   - 网格统计: 总格点{total_points}, 覆盖{covered_points}, 重叠{overlap_points}")
    
    return {
        'coverage_rate': coverage_rate,
        'miss_rate': miss_rate,
        'total_area': total_area,
        'covered_area': covered_area,
        'overlap_area': overlap_area,
        'excess_overlap_length_nm': excess_overlap_length,
        'coverage_grid': coverage_count
    }

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

def main():
    """主函数"""
    print("=== 基于原始格网数据的真实覆盖分析计算 ===\n")
    
    # 1. 加载数据
    print("1. 加载数据...")
    try:
        lines_df = pd.read_csv('/Users/Mac/Downloads/23b/_03_line_generation/survey_lines_q4_optimized.csv')
        print(f"   - 加载了 {len(lines_df)} 条测线")
        
        interpolator, x_orig, y_orig, depth_orig = load_original_data()
        
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
    
    # 2. 真实覆盖分析
    print("\n2. 基于原始格网数据进行真实覆盖分析...")
    coverage_analysis = calculate_overlap_and_gaps(lines_df, interpolator)
    
    # 3. 计算区域统计
    print("\n3. 计算区域统计...")
    region_stats_df = calculate_region_statistics(lines_df, plane_params_df)
    
    # 4. 生成报告
    print("\n4. 生成最终报告...")
    
    # 总体统计
    total_length_original = lines_df['length_original_nm'].sum()
    total_length_optimized = lines_df['length_optimized_nm'].sum()
    total_reduction = lines_df['length_reduction_nm'].sum()
    
    # 保存详细数据
    region_stats_df.to_csv('region_wise_statistics_real_coverage.csv', index=False, float_format='%.4f')
    
    coverage_summary = pd.DataFrame([{
        '指标': '覆盖率',
        '数值': f"{coverage_analysis['coverage_rate']:.6f}",
        '百分比': f"{coverage_analysis['coverage_rate']*100:.4f}%"
    }, {
        '指标': '漏测率', 
        '数值': f"{coverage_analysis['miss_rate']:.6f}",
        '百分比': f"{coverage_analysis['miss_rate']*100:.4f}%"
    }, {
        '指标': '超额重叠长度',
        '数值': f"{coverage_analysis['excess_overlap_length_nm']:.2f} 海里",
        '百分比': f"{coverage_analysis['excess_overlap_length_nm']/total_length_optimized*100:.2f}%"
    }])
    coverage_summary.to_csv('coverage_analysis_real_coverage.csv', index=False)
    
    final_summary = pd.DataFrame([{
        '原始测线总长度(海里)': total_length_original,
        '优化测线总长度(海里)': total_length_optimized,
        '长度减少(海里)': total_reduction,
        '减少比例(%)': total_reduction/total_length_original*100,
        '覆盖率(%)': coverage_analysis['coverage_rate']*100,
        '漏测率(%)': coverage_analysis['miss_rate']*100,
        '超额重叠长度(海里)': coverage_analysis['excess_overlap_length_nm'],
        '总面积(平方海里)': coverage_analysis['total_area'],
        '覆盖面积(平方海里)': coverage_analysis['covered_area'],
        '重叠面积(平方海里)': coverage_analysis['overlap_area']
    }])
    final_summary.to_csv('final_survey_summary_real_coverage.csv', index=False, float_format='%.6f')
    
    # 打印结果
    print("\n=== 基于真实格网数据的最终指标 ===")
    print(f"测线数量: {len(lines_df)} 条")
    print(f"原始总长度: {total_length_original:.2f} 海里 ({total_length_original*1.852:.2f} 公里)")
    print(f"优化总长度: {total_length_optimized:.2f} 海里 ({total_length_optimized*1.852:.2f} 公里)")
    print(f"长度减少: {total_reduction:.2f} 海里 ({total_reduction*1.852:.2f} 公里)")
    print(f"减少比例: {total_reduction/total_length_original*100:.1f}%")
    print()
    print("=== 真实覆盖质量指标 ===")
    print(f"覆盖率: {coverage_analysis['coverage_rate']*100:.4f}%")
    print(f"漏测率: {coverage_analysis['miss_rate']*100:.4f}%")
    print(f"超额重叠长度: {coverage_analysis['excess_overlap_length_nm']:.2f} 海里")
    print(f"超额重叠比例: {coverage_analysis['excess_overlap_length_nm']/total_length_optimized*100:.2f}%")
    print()
    print(f"面积统计:")
    print(f"  总作业面积: {coverage_analysis['total_area']:.3f} 平方海里")
    print(f"  实际覆盖面积: {coverage_analysis['covered_area']:.3f} 平方海里")
    print(f"  重叠面积: {coverage_analysis['overlap_area']:.3f} 平方海里")
    print()
    
    print("=== 分区域统计 ===")
    print(region_stats_df.to_string(index=False, float_format='%.2f'))
    print()
    
    print("详细数据已保存至:")
    print("- region_wise_statistics_real_coverage.csv")
    print("- coverage_analysis_real_coverage.csv") 
    print("- final_survey_summary_real_coverage.csv")
    
    print("\n=== 计算完成 ===")
    print("注意: 此计算基于原始格网数据的真实水深插值，符合4solution.md第四步要求")

if __name__ == '__main__':
    main() 