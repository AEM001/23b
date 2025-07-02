#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第四问最终指标计算脚本 - 智能延伸版本
基于智能延伸后的测线数据计算测线总长度、漏测率、重叠率超过20%部分的总长度
采用更精确的覆盖率计算算法，考虑测线延伸策略的影响
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
    
    # 原始数据中坐标已经是海里单位，深度是米单位
    x_nm = np.array(df['横坐标'].values, dtype=float)  # 坐标已经是海里
    y_nm = np.array(df['纵坐标'].values, dtype=float)  # 坐标已经是海里
    depth = np.array(df['深度'].values, dtype=float)  # 深度是米
    
    print(f"   - 数据范围: X=[{x_nm.min():.3f}, {x_nm.max():.3f}] 海里, Y=[{y_nm.min():.3f}, {y_nm.max():.3f}] 海里")
    print(f"   - 水深范围: [{depth.min():.1f}, {depth.max():.1f}] 米")
    print(f"   - 海域尺寸: 东西{x_nm.max()-x_nm.min():.1f}海里 × 南北{y_nm.max()-y_nm.min():.1f}海里")
    
    # 建立插值函数 D_true = f(x, y)
    interpolator = LinearNDInterpolator(list(zip(x_nm, y_nm)), depth)
    
    return interpolator

def calculate_swath_width(depth, beam_angle=120):
    """根据水深计算条带宽度"""
    # 多波束条带宽度 = 2 * depth * tan(beam_angle/2)
    swath_width_m = 2 * depth * np.tan(np.radians(beam_angle / 2))
    swath_width_nm = swath_width_m / 1852  # 转换为海里
    return min(swath_width_nm, 0.2)  # 限制最大宽度0.2海里

def is_point_in_region(x, y):
    """判断点是否在有效海域内（不包括延伸部分）"""
    return 0 <= x <= 4.0 and 0 <= y <= 5.0

def calculate_line_coverage_smart(x_start, y_start, x_end, y_end, interpolator, num_samples=50):
    """计算单条测线的智能覆盖带，只考虑有效海域内的覆盖，增强边界覆盖检测"""
    # 沿测线采样（增加采样密度）
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
    
    # 只保留在有效海域内的点，但包含边界上的点
    in_region_mask = [is_point_in_region_inclusive(x, y) for x, y in zip(valid_x, valid_y)]
    
    if not any(in_region_mask):
        return [], []
    
    region_x = valid_x[in_region_mask]
    region_y = valid_y[in_region_mask] 
    region_depths = valid_depths[in_region_mask]
    
    # 计算各点的条带宽度
    swath_widths = [calculate_swath_width(d) for d in region_depths]
    
    # 返回覆盖点和宽度
    coverage_points = list(zip(region_x, region_y))
    return coverage_points, swath_widths

def is_point_in_region_inclusive(x, y):
    """判断点是否在有效海域内（包括边界）"""
    return 0 <= x <= 4.0 and 0 <= y <= 5.0

def calculate_coverage_metrics(lines_df, interpolator):
    """计算覆盖指标：覆盖率、漏测率、重叠率超过20%部分的总长度"""
    print("   - 计算智能延伸后的覆盖指标（高精度模式）...")
    
    # 创建覆盖网格 (0.006海里分辨率，更精细)
    grid_resolution = 0.006  # 海里，约11米
    x_min, x_max = 0, 4.0  # 海里 (东西宽4海里)
    y_min, y_max = 0, 5.0  # 海里 (南北长5海里)
    
    x_grid = np.arange(x_min, x_max, grid_resolution)
    y_grid = np.arange(y_min, y_max, grid_resolution)
    coverage_count = np.zeros((len(y_grid), len(x_grid)))
    
    print(f"   - 超高精度网格大小: {len(x_grid)} x {len(y_grid)} = {len(x_grid)*len(y_grid)} 个格点")
    
    # 遍历每条测线计算覆盖
    total_lines = len(lines_df)
    for idx, row in lines_df.iterrows():
        if idx % 15 == 0:
            print(f"   - 处理测线 {idx+1}/{total_lines}")
            
        x_start, y_start = row['x_start_nm'], row['y_start_nm'] 
        x_end, y_end = row['x_end_nm'], row['y_end_nm']
        
        # 计算测线覆盖（只考虑有效海域内）
        coverage_points, swath_widths = calculate_line_coverage_smart(
            x_start, y_start, x_end, y_end, interpolator)
        
        if not coverage_points:
            continue
            
        # 将覆盖投影到网格（使用更精细的覆盖计算）
        for i, (x, y) in enumerate(coverage_points):
            swath_width = swath_widths[i]
            half_width = swath_width / 2
            
            # 找到影响的网格范围（略微扩大以确保边界覆盖）
            buffer = grid_resolution * 0.5  # 增加小缓冲区
            x_indices = np.where((x_grid >= x - half_width - buffer) & 
                               (x_grid <= x + half_width + buffer))[0]
            y_indices = np.where((y_grid >= y - half_width - buffer) & 
                               (y_grid <= y + half_width + buffer))[0]
            
            # 标记覆盖的网格点
            for yi in y_indices:
                for xi in x_indices:
                    # 计算距离测线的距离
                    dist = np.sqrt((x_grid[xi] - x)**2 + (y_grid[yi] - y)**2)
                    # 使用略微宽松的判断条件以确保边界覆盖
                    if dist <= half_width + buffer:
                        coverage_count[yi, xi] += 1
    
    # 统计覆盖情况
    total_points = len(x_grid) * len(y_grid) 
    covered_points = np.sum(coverage_count > 0)
    
    # 计算覆盖率和漏测率
    coverage_rate = covered_points / total_points
    miss_rate = (total_points - covered_points) / total_points
    
    # 计算重叠率超过20%部分的总长度
    print("   - 计算重叠率超过20%的测线段...")
    excess_overlap_length = calculate_excess_overlap_length_smart(lines_df, interpolator)
    
    return {
        'coverage_rate': coverage_rate,
        'miss_rate': miss_rate,
        'excess_overlap_length_nm': excess_overlap_length
    }

def calculate_excess_overlap_length_smart(lines_df, interpolator):
    """计算重叠率超过20%部分的总长度 - 智能版本"""
    excess_length_total = 0.0
    target_overlap_rate = 0.20  # 20%阈值
    
    # 按区域分组处理
    for region_id in sorted(lines_df['region_id'].unique()):
        region_lines = lines_df[lines_df['region_id'] == region_id].sort_values('line_id')
        
        if len(region_lines) < 2:
            continue
            
        region_lines_list = region_lines.to_dict('records')
        
        print(f"   - 处理区域 {region_id}，共 {len(region_lines_list)} 条测线")
        
        # 遍历相邻测线对
        for i in range(len(region_lines_list) - 1):
            line1 = region_lines_list[i]
            line2 = region_lines_list[i + 1]
            
            # 计算相邻测线间的重叠率和长度（只考虑有效海域内）
            segment_excess_length = calculate_pairwise_excess_overlap_smart(
                line1, line2, interpolator, target_overlap_rate)
            
            excess_length_total += segment_excess_length
    
    return excess_length_total

def calculate_pairwise_excess_overlap_smart(line1, line2, interpolator, target_overlap_rate):
    """计算两条相邻测线间重叠率超过阈值部分的长度 - 智能版本"""
    # 获取测线参数
    x1_start, y1_start = line1['x_start_nm'], line1['y_start_nm']
    x1_end, y1_end = line1['x_end_nm'], line1['y_end_nm']
    x2_start, y2_start = line2['x_start_nm'], line2['y_start_nm']
    x2_end, y2_end = line2['x_end_nm'], line2['y_end_nm']
    
    # 沿测线采样分析重叠情况
    num_samples = 25
    t_values = np.linspace(0, 1, num_samples)
    
    excess_length = 0.0
    
    for i in range(len(t_values) - 1):
        t1, t2 = t_values[i], t_values[i + 1]
        
        # 计算当前段中点的坐标
        t_mid = (t1 + t2) / 2
        
        # 测线1上的点
        x1_mid = x1_start + t_mid * (x1_end - x1_start)
        y1_mid = y1_start + t_mid * (y1_end - y1_start)
        
        # 测线2上的点
        x2_mid = x2_start + t_mid * (x2_end - x2_start)
        y2_mid = y2_start + t_mid * (y2_end - y2_start)
        
        # 只考虑在有效海域内的部分
        if not (is_point_in_region(x1_mid, y1_mid) and is_point_in_region(x2_mid, y2_mid)):
            continue
            
        # 查询真实水深
        depth1 = interpolator(x1_mid, y1_mid)
        depth2 = interpolator(x2_mid, y2_mid)
        
        if np.isnan(depth1) or np.isnan(depth2):
            continue
            
        # 计算各自的条带宽度
        swath1_width = calculate_swath_width(depth1)
        swath2_width = calculate_swath_width(depth2)
        
        # 计算测线间距（垂直距离）
        line_distance = np.sqrt((x2_mid - x1_mid)**2 + (y2_mid - y1_mid)**2)
        
        # 计算重叠宽度
        half_swath1 = swath1_width / 2
        half_swath2 = swath2_width / 2
        
        # 重叠宽度 = 两个半宽度之和 - 测线间距
        overlap_width = half_swath1 + half_swath2 - line_distance
        
        if overlap_width > 0:
            # 计算重叠率（以较小条带宽度为基准）
            base_width = min(swath1_width, swath2_width)
            overlap_rate = overlap_width / base_width if base_width > 0 else 0
            
            # 如果重叠率超过阈值，累计该段长度
            if overlap_rate > target_overlap_rate:
                # 计算该段在有效海域内的实际长度
                # 只计算在有效海域内的线段长度
                segment_length1 = np.sqrt((x1_end - x1_start)**2 + (y1_end - y1_start)**2) / num_samples
                segment_length2 = np.sqrt((x2_end - x2_start)**2 + (y2_end - y2_start)**2) / num_samples
                avg_segment_length = (segment_length1 + segment_length2) / 2
                
                excess_length += avg_segment_length
    
    return excess_length

def main():
    """主函数"""
    print("=== 第四问最终指标计算（保守延伸策略版本） ===\n")
    
    # 1. 加载数据
    print("1. 加载数据...")
    try:
        lines_df = pd.read_csv('/Users/Mac/Downloads/23b/_03_line_generation/survey_lines_q4_optimized.csv')
        print(f"   - 加载了 {len(lines_df)} 条保守延伸优化测线")
        
        interpolator = load_original_data()
        
    except FileNotFoundError as e:
        print(f"错误: 找不到必要文件 - {e}")
        return
    
    # 2. 计算最终指标
    print("\n2. 计算最终指标...")
    coverage_metrics = calculate_coverage_metrics(lines_df, interpolator)
    
    # 3. 计算测线总长度（使用智能优化后的长度）
    total_length_nm = lines_df['length_optimized_nm'].sum()
    total_length_km = total_length_nm * 1.852
    
    # 4. 输出最终结果
    print("\n" + "="*60)
    print("第四问最终答案（保守边界延伸策略）")
    print("="*60)
    print(f"(1) 测线的总长度: {total_length_nm:.2f} 海里 ({total_length_km:.2f} 公里)")
    print(f"(2) 漏测海区占总待测海域面积的百分比: {coverage_metrics['miss_rate']*100:.4f}%")
    print(f"(3) 重叠率超过20%部分的总长度: {coverage_metrics['excess_overlap_length_nm']:.2f} 海里")
    print("="*60)
    print("说明: 采用保守边界延伸策略 + 高精度覆盖检测")
    print("     基础延伸=完整覆盖宽度 + 角度修正 + 20%安全边距")
    print("     最大化减少边界处的漏测问题")
    print("="*60)
    
    # 5. 保存结果到文件
    final_results = pd.DataFrame([{
        '指标': '测线总长度(海里)',
        '数值': f"{total_length_nm:.2f}"
    }, {
        '指标': '测线总长度(公里)', 
        '数值': f"{total_length_km:.2f}"
    }, {
        '指标': '漏测率(%)',
        '数值': f"{coverage_metrics['miss_rate']*100:.4f}"
    }, {
        '指标': '覆盖率(%)',
        '数值': f"{coverage_metrics['coverage_rate']*100:.4f}"
    }, {
        '指标': '重叠率超过20%部分总长度(海里)',
        '数值': f"{coverage_metrics['excess_overlap_length_nm']:.2f}"
    }])
    
    final_results.to_csv('第四问最终答案.csv', index=False)
    print(f"\n结果已保存至: 第四问最终答案.csv")
    print("保守边界延伸策略计算完成！")

if __name__ == '__main__':
    main() 