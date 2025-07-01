#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的测线长度优化脚本
基于现有成功的测线生成，对测线长度进行边界裁剪优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_line_region_intersections(x_start, y_start, x_end, y_end, x_min, x_max, y_min, y_max):
    """
    计算测线与矩形区域边界的交点，返回裁剪后的线段
    使用Cohen-Sutherland直线裁剪算法
    """
    # Define region codes
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def get_code(x, y):
        code = INSIDE
        if x < x_min:
            code |= LEFT
        elif x > x_max:
            code |= RIGHT
        if y < y_min:
            code |= BOTTOM
        elif y > y_max:
            code |= TOP
        return code

    x1, y1, x2, y2 = x_start, y_start, x_end, y_end
    code1 = get_code(x1, y1)
    code2 = get_code(x2, y2)
    accept = False

    while True:
        if code1 == 0 and code2 == 0:  # Both endpoints inside
            accept = True
            break
        elif (code1 & code2) != 0:  # Both endpoints outside in same region
            break
        else:
            # Find intersection point
            x, y = 0, 0
            code_out = code1 if code1 != 0 else code2
            
            if code_out & TOP:
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & BOTTOM:
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & RIGHT:
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & LEFT:
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            if code_out == code1:
                x1, y1 = x, y
                code1 = get_code(x1, y1)
            else:
                x2, y2 = x, y
                code2 = get_code(x2, y2)
    
    if accept:
        return x1, y1, x2, y2
    else:
        return None

def optimize_survey_lines(lines_df, region_boundaries):
    """
    对现有的测线进行长度优化
    """
    optimized_lines = []
    
    for _, line in lines_df.iterrows():
        region_id = line['region_id']
        
        # 获取区域边界
        region_bounds = region_boundaries[region_boundaries['区域编号'] == region_id].iloc[0]
        x_min, x_max = region_bounds['X_min'], region_bounds['X_max']
        y_min, y_max = region_bounds['Y_min'], region_bounds['Y_max']
        
        # 对测线进行边界裁剪
        result = find_line_region_intersections(
            line['x_start_nm'], line['y_start_nm'],
            line['x_end_nm'], line['y_end_nm'],
            x_min, x_max, y_min, y_max
        )
        
        if result is not None:
            x1, y1, x2, y2 = result
            # 计算优化后的长度
            optimized_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            optimized_line = line.copy()
            optimized_line['x_start_nm'] = x1
            optimized_line['y_start_nm'] = y1
            optimized_line['x_end_nm'] = x2
            optimized_line['y_end_nm'] = y2
            optimized_line['length_optimized_nm'] = optimized_length
            optimized_line['length_original_nm'] = line['length_nm']
            optimized_line['length_reduction_nm'] = line['length_nm'] - optimized_length
            
            optimized_lines.append(optimized_line)
    
    return pd.DataFrame(optimized_lines)

def visualize_comparison(lines_df_original, lines_df_optimized, region_boundaries, grid_data, 
                        output_path="survey_plan_comparison.png"):
    """Creates a comparison plot of original vs optimized survey plans."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # 原始方案
    ax1.scatter(grid_data['横坐标'], grid_data['纵坐标'], c=grid_data['深度'], 
                cmap='ocean_r', s=1, alpha=0.6, label='海深数据点')
    
    for _, line in lines_df_original.iterrows():
        ax1.plot([line['x_start_nm'], line['x_end_nm']], [line['y_start_nm'], line['y_end_nm']], 
                'r-', lw=0.7, alpha=0.8)

    for _, region in region_boundaries.iterrows():
        x_min, x_max = region['X_min'], region['X_max']
        y_min, y_max = region['Y_min'], region['Y_max']
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--')
        ax1.add_patch(rect)
        ax1.text(x_min + 0.05, y_min + 0.05, str(region['区域编号']), 
                fontsize=12, weight='bold', color='black')

    ax1.set_xlabel('东西方向坐标 (海里)')
    ax1.set_ylabel('南北方向坐标 (海里)')
    ax1.set_title('原始方案：固定长度测线')
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 优化方案
    ax2.scatter(grid_data['横坐标'], grid_data['纵坐标'], c=grid_data['深度'], 
                cmap='ocean_r', s=1, alpha=0.6, label='海深数据点')
    
    for _, line in lines_df_optimized.iterrows():
        ax2.plot([line['x_start_nm'], line['x_end_nm']], [line['y_start_nm'], line['y_end_nm']], 
                'b-', lw=0.7, alpha=0.8)

    for _, region in region_boundaries.iterrows():
        x_min, x_max = region['X_min'], region['X_max']
        y_min, y_max = region['Y_min'], region['Y_max']
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--')
        ax2.add_patch(rect)
        ax2.text(x_min + 0.05, y_min + 0.05, str(region['区域编号']), 
                fontsize=12, weight='bold', color='black')

    ax2.set_xlabel('东西方向坐标 (海里)')
    ax2.set_ylabel('南北方向坐标 (海里)')
    ax2.set_title('优化方案：边界裁剪测线')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比可视化图像已保存至 {output_path}")

def main():
    """Main function to run the optimized survey line planning pipeline."""
    # 1. Load data
    print("正在加载数据...")
    try:
        lines_df_original = pd.read_csv("survey_lines_q4.csv")
        grid_df = pd.read_csv('../_00_source_data/output.csv')
    except FileNotFoundError as e:
        print(f"错误: 必需的数据文件未找到 - {e}")
        return

    # Region boundaries
    region_boundaries_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'X_min': [0.00, 0.98, 0.00, 1.99, 2.99, 1.99, 2.99],
        'X_max': [0.98, 1.99, 1.99, 2.99, 4.00, 2.99, 4.00],
        'Y_min': [0.00, 0.00, 2.49, 0.00, 0.00, 2.49, 2.49],
        'Y_max': [2.49, 2.49, 5.00, 2.49, 2.49, 5.00, 5.00]
    }
    region_boundaries_df = pd.DataFrame(region_boundaries_data)

    # 2. Optimize survey lines
    print("正在优化测线长度...")
    lines_df_optimized = optimize_survey_lines(lines_df_original, region_boundaries_df)
    lines_df_optimized.to_csv("survey_lines_q4_optimized.csv", index=False, float_format='%.4f')
    print("优化的测线数据已保存至 survey_lines_q4_optimized.csv")

    # 3. Generate comparison report
    print("正在生成优化报告...")
    total_length_original = lines_df_original['length_nm'].sum()
    total_length_optimized = lines_df_optimized['length_optimized_nm'].sum()
    length_reduction = total_length_original - total_length_optimized
    reduction_percentage = (length_reduction / total_length_original) * 100

    report_content = "# 测线长度优化报告\n\n"
    report_content += "## 优化思路\n\n"
    report_content += "**问题识别**: 原始方案中，每条测线都从区域的局部坐标系边界延伸，但由于测线方向与矩形区域边界不平行，"
    report_content += "实际上很多测线段延伸到了区域边界之外，造成了不必要的测量长度。\n\n"
    report_content += "**优化方法**: 使用Cohen-Sutherland直线裁剪算法，对每条测线进行精确的边界裁剪，只保留在区域内的有效测线段。"
    report_content += "这样既保持了测线的平行性和覆盖效果，又减少了无效的测量长度。\n\n"
    report_content += "## 优化结果对比\n\n"
    
    summary_data = []
    for region_id in lines_df_optimized['region_id'].unique():
        region_lines = lines_df_optimized[lines_df_optimized['region_id'] == region_id]
        
        length_original = region_lines['length_original_nm'].sum()
        length_optimized = region_lines['length_optimized_nm'].sum()
        reduction = length_original - length_optimized
        reduction_pct = (reduction / length_original) * 100 if length_original > 0 else 0
        
        summary_data.append({
            '区域编号': region_id,
            '测线数量': len(region_lines),
            '原始长度(海里)': length_original,
            '优化长度(海里)': length_optimized,
            '减少长度(海里)': reduction,
            '减少比例(%)': reduction_pct
        })
    
    summary_df = pd.DataFrame(summary_data)
    table_md = summary_df.to_markdown(index=False, floatfmt='.2f')
    if table_md:
        report_content += table_md

    report_content += f"\n\n**总体优化效果**:\n"
    report_content += f"- 原始总长度: {total_length_original:.2f} 海里 ({total_length_original * 1.852:.2f} 公里)\n"
    report_content += f"- 优化总长度: {total_length_optimized:.2f} 海里 ({total_length_optimized * 1.852:.2f} 公里)\n"
    report_content += f"- 减少长度: {length_reduction:.2f} 海里 ({length_reduction * 1.852:.2f} 公里)\n"
    report_content += f"- 减少比例: {reduction_percentage:.1f}%\n\n"
    
    report_content += "**重要说明**: \n"
    report_content += "1. 优化后的测线依然保持在各区域内严格平行\n"
    report_content += "2. 测线的间距和布置逻辑完全不变，确保漏测率和重叠率不受影响\n"
    report_content += "3. 优化仅针对测线长度，减少了超出区域边界的无效测量部分\n"
    report_content += "4. 这种优化在实际作业中可显著提高效率，减少作业时间和成本\n\n"
    
    with open("survey_optimization_report.md", "w", encoding='utf-8') as f:
        f.write(report_content)
    print("优化报告已生成: survey_optimization_report.md")

    # 4. Create comparison visualization
    print("正在生成对比可视化...")
    visualize_comparison(lines_df_original, lines_df_optimized, region_boundaries_df, grid_df)
    
    print(f"\n=== 优化总结 ===")
    print(f"测线总长度减少: {length_reduction:.2f} 海里 ({reduction_percentage:.1f}%)")
    print(f"这相当于减少了 {length_reduction * 1.852:.2f} 公里的测量工作量")
    print(f"优化前总测线数: {len(lines_df_original)}")
    print(f"优化后总测线数: {len(lines_df_optimized)}")

if __name__ == '__main__':
    main() 