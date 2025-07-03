import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# 设置字体为支持中文字符的本地可用字体
mpl.rcParams['font.sans-serif'] = ['Heiti TC']
mpl.rcParams['axes.unicode_minus'] = False

def design_survey_lines():
    """
    计算问题3的最优测线布置并生成结果。
    """
    # 1. 常量和参数
    D_0 = 110  # 中心水深 (米)
    alpha_deg = 1.5  # 坡度 (度)
    theta_deg = 120  # 换能器开角 (度)
    eta = 0.10  # 目标重叠率 (10%)
    L_ew_nm = 4  # 东西方向长度 (海里)
    L_ns_nm = 2  # 南北方向长度 (海里)
    NM_TO_M = 1852  # 海里到米的转换系数

    L_ew_m = L_ew_nm * NM_TO_M
    L_ns_m = L_ns_nm * NM_TO_M
    
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)

    # 2. 坐标系与初始计算
    # x=0 在西边缘，向东为正。
    # 中心处的水深 (x = L_ew_m / 2) 为 D_0。
    # 西边缘 (x=0) 的水深为最大水深。
    D_max = D_0 + (L_ew_m / 2) * np.tan(alpha)

    # 3. 迭代布线
    # 第一条测线的位置被计算为覆盖西侧边界 (x=0)。
    x_1 = D_max * np.tan(theta / 2)
    
    lines_positions = [x_1]
    x_k = x_1
    
    while True:
        # 当前测线位置的水深
        D_k = D_max - x_k * np.tan(alpha)
        
        # 检查当前测线的覆盖范围是否已覆盖整个区域。
        W_k_right_proj = (D_k * np.sin(theta / 2) / np.cos(theta / 2 - alpha)) * np.cos(alpha)
        if (x_k + W_k_right_proj) >= L_ew_m:
            break

        # 使用推导出的公式计算到下一条测线的距离 d_k
        W_k_left = D_k * np.sin(theta / 2) / np.cos(theta / 2 + alpha)
        W_k_right = D_k * np.sin(theta / 2) / np.cos(theta / 2 - alpha)
        
        numerator = (W_k_left + W_k_right) * (1 - eta) * np.cos(alpha)
        denominator = 1 + (1 - eta) * np.sin(alpha) * np.sin(theta / 2) / np.cos(theta / 2 + alpha)
        d_k = numerator / denominator
        
        # 下一条测线的位置
        x_k_plus_1 = x_k + d_k
        lines_positions.append(x_k_plus_1)
        
        x_k = x_k_plus_1

    # 4. 结果处理和输出
    results_data = []
    for i, pos in enumerate(lines_positions):
        depth = D_max - pos * np.tan(alpha)
        W_left_proj = (depth * np.sin(theta / 2) / np.cos(theta / 2 + alpha)) * np.cos(alpha)
        W_right_proj = (depth * np.sin(theta / 2) / np.cos(theta / 2 - alpha)) * np.cos(alpha)
        
        left_edge = pos - W_left_proj
        right_edge = pos + W_right_proj
        
        dist_to_next = lines_positions[i+1] - pos if i < len(lines_positions) - 1 else np.nan
        
        # 计算与上一条测线的实际重叠率
        if i > 0:
            prev_pos = lines_positions[i-1]
            prev_depth = D_max - prev_pos * np.tan(alpha)
            prev_W_right_proj = (prev_depth * np.sin(theta/2) / np.cos(theta/2 - alpha)) * np.cos(alpha)
            prev_right_edge = prev_pos + prev_W_right_proj
            
            overlap_dist = prev_right_edge - left_edge
            
            # 使用IHO定义计算重叠率分母：两条相邻测线外边缘之间的距离
            # denominator_w = right_edge - prev_pos + prev_W_right_proj
            # 更简单的方法是使用当前测线的测绘带宽度
            total_width = W_left_proj + W_right_proj
            overlap_perc = (overlap_dist / total_width) * 100 if total_width > 0 else 0
        else:
            overlap_perc = np.nan

        results_data.append([i + 1, pos, depth, dist_to_next, left_edge, right_edge, overlap_perc])

    df = pd.DataFrame(results_data, columns=[
        '测线号', '离西侧距离 (m)', '水深 (m)', 
        '与下一条测线间距 (m)', '左覆盖边界 (m)', '右覆盖边界 (m)', 
        '与上一条重叠率 (%)'
    ])

    total_lines = len(df)
    total_length_km = (total_lines * L_ns_m) / 1000

    print("="*20 + " 测线设计结果 " + "="*20)
    print(f"海域东西宽度: {L_ew_m:.2f} m")
    print(f"所需测线总数: {total_lines}")
    print(f"测线总长度: {total_length_km:.2f} km")
    print("\n详细测线参数:")
    print(df.to_string())
    
    # 将结果保存到文件
    output_path = "3/result3.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n结果已保存至 {output_path}")
    
    return df, D_max, L_ew_m, alpha

def visualize_results(df, D_max, L_ew_m, alpha):
    """
    生成并保存测线设计图。
    """
    # 1. 绘制海底剖面图
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x_coords = np.linspace(0, L_ew_m, 500)
    y_coords_depth = D_max - x_coords * np.tan(alpha)
    ax1.plot(x_coords, y_coords_depth, label='海底剖面')
    ax1.invert_yaxis()  # 深度向下增加
    ax1.set_xlabel('离西侧距离 (m)')
    ax1.set_ylabel('水深 (m)')
    ax1.set_title('待测海域东西向海底剖面图')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    plt.tight_layout()
    plt.savefig("3/seabed_profile.png")
    print("海底剖面图已保存至 3/seabed_profile.png")

    # 2. 绘制覆盖示意图
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    
    # 绘制海域边界
    ax2.axvline(0, color='k', linestyle='--', label='海域西边界')
    ax2.axvline(L_ew_m, color='k', linestyle='--', label='海域东边界')

    for i, row in df.iterrows():
        line_pos = row['离西侧距离 (m)']
        left_edge = row['左覆盖边界 (m)']
        right_edge = row['右覆盖边界 (m)']
        width = right_edge - left_edge
        
        # 绘制主要测绘带
        rect = patches.Rectangle((left_edge, i + 0.5), width, 0.5, 
                                 edgecolor='black', facecolor='skyblue', alpha=0.6)
        ax2.add_patch(rect)
        # 标记测线
        ax2.plot([line_pos, line_pos], [i + 0.5, i + 1.0], color='red', lw=2)

    ax2.set_yticks(np.arange(len(df)) + 0.75)
    ax2.set_yticklabels(df['测线号'])
    ax2.set_xlabel('离西侧距离 (m)')
    ax2.set_ylabel('测线号')
    ax2.set_title('测线布设与覆盖范围示意图')
    ax2.set_xlim(-500, L_ew_m + 500)
    ax2.set_ylim(0, len(df) + 1)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("3/survey_coverage_diagram.png")
    print("测线覆盖示意图已保存至 3/survey_coverage_diagram.png")


if __name__ == '__main__':
    results_df, D_max, L_ew_m, alpha = design_survey_lines()
    visualize_results(results_df, D_max, L_ew_m, alpha) 