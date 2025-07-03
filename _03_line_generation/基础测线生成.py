import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.lines import Line2D

# 设置字体为支持中文字符的本地可用字体
mpl.rcParams['font.sans-serif'] = ['Heiti TC']
mpl.rcParams['axes.unicode_minus'] = False


# --- 常量定义 ---
NM_TO_M = 1852
THETA_DEG = 120
ETA_TARGET = 0.10  # 目标重叠率 10%

def get_region_slope(beta1, beta2):
    """根据平面系数计算区域的坡度角。"""
    tan_alpha = np.sqrt((beta1 / NM_TO_M)**2 + (beta2 / NM_TO_M)**2)
    alpha_rad = np.arctan(tan_alpha)
    return alpha_rad

def uv_to_xy(u, v, theta_rad):
    """将点从局部坐标系(u,v)转换到全局坐标系(x,y)。"""
    x = u * np.cos(theta_rad) - v * np.sin(theta_rad)
    y = u * np.sin(theta_rad) + v * np.cos(theta_rad)
    return x, y

def get_avg_depth_on_line(v_coord, u_min, u_max, theta_rad, interpolator, num_samples=20):
    """计算局部坐标系中某条测线上的平均深度。"""
    u_samples = np.linspace(u_min, u_max, num_samples)
    v_samples = np.full_like(u_samples, v_coord)
    
    x_coords, y_coords = uv_to_xy(u_samples, v_samples, theta_rad)
    
    depths = interpolator(x_coords, y_coords)
    valid_depths = depths[~np.isnan(depths)]
    
    if len(valid_depths) == 0:
        print(f"警告: 无法在 v={v_coord:.2f} 处找到深度数据。回退使用100m深度。")
        return 100.0 
        
    return np.mean(valid_depths)

def calculate_line_distance(D_k, alpha_rad, theta_rad, eta):
    """根据深度和坡度计算到下一条测线的距离。"""
    W_k_left = D_k * np.sin(theta_rad / 2) / np.cos(theta_rad / 2 + alpha_rad)
    W_k_right = D_k * np.sin(theta_rad / 2) / np.cos(theta_rad / 2 - alpha_rad)
    
    numerator = (W_k_left + W_k_right) * (1 - eta) * np.cos(alpha_rad)
    denominator = 1 + (1 - eta) * np.sin(alpha_rad) * np.sin(theta_rad / 2) / np.cos(theta_rad / 2 + alpha_rad)
    
    d_k_meters = numerator / denominator
    d_k_nm = d_k_meters / NM_TO_M
    return d_k_nm

def plan_lines_for_region(region_params, plane_params, interpolator):
    """为单个子区域规划所有测线。"""
    region_id = region_params['区域编号']
    u_min, u_max = region_params['u_min'], region_params['u_max']
    v_min, v_max = region_params['v_min'], region_params['v_max']
    theta_rad = region_params['主测线方向_rad']
    
    beta1 = plane_params.loc[plane_params['区域编号'] == region_id, 'β₁'].iloc[0]
    beta2 = plane_params.loc[plane_params['区域编号'] == region_id, 'β₂'].iloc[0]
    
    alpha_rad = get_region_slope(beta1, beta2)
    theta_rad_const = np.radians(THETA_DEG)

    print(f"    区域 {region_id}: 主测线方向 {np.degrees(theta_rad):.1f}°, 坡度 {np.degrees(alpha_rad):.1f}°")

    # 计算第一条测线位置：确保覆盖v_min边界
    D_bdy = get_avg_depth_on_line(v_min, u_min, u_max, theta_rad, interpolator)
    W_left_proj_m = (D_bdy * np.sin(theta_rad_const / 2) / np.cos(theta_rad_const / 2 + alpha_rad)) * np.cos(alpha_rad)
    W_left_proj_nm = W_left_proj_m / NM_TO_M
    
    # 第一条测线的v坐标
    v_1 = v_min + W_left_proj_nm
    
    survey_line_vs = [v_1]
    current_v = v_1
    
    # 迭代生成后续测线
    for iteration in range(200):  # 安全限制
        D_k = get_avg_depth_on_line(current_v, u_min, u_max, theta_rad, interpolator)
        
        # 计算当前测线的右覆盖范围
        W_right_proj_m = (D_k * np.sin(theta_rad_const / 2) / np.cos(theta_rad_const / 2 - alpha_rad)) * np.cos(alpha_rad)
        W_right_proj_nm = W_right_proj_m / NM_TO_M
        
        # 检查是否已覆盖到v_max边界
        if current_v + W_right_proj_nm >= v_max:
            break
            
        # 计算到下一条测线的距离
        d_k_nm = calculate_line_distance(D_k, alpha_rad, theta_rad_const, ETA_TARGET)
        
        if d_k_nm <= 1e-4:
            print(f"    警告: 区域 {region_id} 中计算出的间距过小 ({d_k_nm:.5f} nm)。停止迭代。")
            break

        # 下一条测线的v坐标
        current_v += d_k_nm
        survey_line_vs.append(current_v)
    else:
        print(f"    警告: 区域 {region_id} 达到最大迭代次数。")

    print(f"    区域 {region_id}: 生成了 {len(survey_line_vs)} 条测线")

    # 生成测线数据：每条测线都是在局部坐标系中v=constant的水平线
    lines_data = []
    for i, v_coord in enumerate(survey_line_vs):
        # 在局部坐标系中定义测线的两个端点
        u_start, u_end = u_min, u_max
        
        # 转换到全局坐标系
        x_start, y_start = uv_to_xy(u_start, v_coord, theta_rad)
        x_end, y_end = uv_to_xy(u_end, v_coord, theta_rad)
        
        # 计算测线长度
        length_nm = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        
        lines_data.append({
            'region_id': int(region_id),
            'line_id': i + 1,
            'x_start_nm': x_start,
            'y_start_nm': y_start,
            'x_end_nm': x_end,
            'y_end_nm': y_end,
            'length_nm': length_nm,
            'v_coordinate': v_coord,
            'u_start': u_start,
            'u_end': u_end
        })
        
    return lines_data

def cohen_sutherland_clip(x1, y1, x2, y2, xmin, ymax, xmax, ymin):
    """使用Cohen-Sutherland算法裁剪线段。"""
    # 注意：ymin/ymax交换是因为典型的图形上下文中y向下递增，
    # 但我们的地理坐标y向上递增。算法本身与坐标系无关。
    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def _get_code(x, y):
        code = INSIDE
        if x < xmin: code |= LEFT
        elif x > xmax: code |= RIGHT
        if y < ymin: code |= BOTTOM
        elif y > ymax: code |= TOP
        return code

    code1 = _get_code(x1, y1)
    code2 = _get_code(x2, y2)
    accept = False

    while True:
        if code1 == 0 and code2 == 0: # 两个端点都在内部
            accept = True
            break
        elif (code1 & code2) != 0: # 两个端点都在外部且在同一区域
            break
        else:
            x, y = 0, 0
            code_out = code1 if code1 != 0 else code2
            
            if code_out & TOP:
                x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
                y = ymax
            elif code_out & BOTTOM:
                x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
                y = ymin
            elif code_out & RIGHT:
                x = x1 + (x2 - x1) * (xmax - x1) / (x2 - x1)
                y = xmax
            elif code_out & LEFT:
                x = x1 + (x2 - x1) * (xmin - x1) / (x2 - x1)
                y = xmin

            if code_out == code1:
                x1, y1 = x, y
                code1 = _get_code(x1, y1)
            else:
                x2, y2 = x, y
                code2 = _get_code(x2, y2)
    
    if accept:
        return x1, y1, x2, y2
    else:
        return None

def visualize_plan(lines_df, region_boundaries, grid_data, output_path="survey_plan_q4.png"):
    """创建测线规划的可视化图表。"""
    fig, ax = plt.subplots(figsize=(10, 12.5))
    
    ax.scatter(grid_data['横坐标'], grid_data['纵坐标'], c=grid_data['深度'], cmap='ocean_r', s=1, alpha=0.6, label='原始海深数据点')

    for _, line in lines_df.iterrows():
        ax.plot([line['x_start_nm'], line['x_end_nm']], [line['y_start_nm'], line['y_end_nm']], 'r-', lw=0.7)

    for _, region in region_boundaries.iterrows():
        x_min, x_max = region['X_min'], region['X_max']
        y_min, y_max = region['Y_min'], region['Y_max']
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--',
                                 label=f"区域 {region['区域编号']}")
        ax.add_patch(rect)
        ax.text(x_min + 0.05, y_min + 0.05, str(region['区域编号']), fontsize=12, weight='bold', color='black')

    ax.set_xlabel('东西方向坐标 (海里)')
    ax.set_ylabel('南北方向坐标 (海里)')
    ax.set_title('第四问：各子区域测线规划结果')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.5)
    # 自定义图例
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles)) # 移除重复标签
    red_line = Line2D([0], [0], color='r', lw=2, label='规划测线')
    unique_labels['规划测线'] = red_line
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', bbox_to_anchor=(1.35, 1.0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可视化图像已保存至 {output_path}")

def main():
    """运行测线规划流水线的主函数。"""
    # 1. 加载数据
    print("正在加载数据...")
    try:
        # 首先尝试新的组织结构，然后回退到当前结构
        try:
            params_df = pd.read_csv('../_02_coord_transform/local_coord_params.csv')
            grid_df = pd.read_csv('../_00_source_data/output.csv')
        except FileNotFoundError:
            # 回退到当前结构
            try:
                params_df = pd.read_csv('local_coord_params.csv')
            except FileNotFoundError:
                params_df = pd.read_csv('局部坐标系划分/local_coord_params.csv')
            grid_df = pd.read_csv('output.csv')
    except FileNotFoundError as e:
        print(f"错误: 必需的数据文件未找到 - {e}。请确保相关数据文件存在。")
        return

    # 从之前步骤硬编码的平面拟合和边界参数
    plane_params_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'β₁': [-0.21, 20.45, 2.35, 41.52, 62.59, 9.55, 14.40],
        'β₂': [4.47, 1.39, 18.97, -8.20, -24.34, 7.86, -8.28]
    }
    plane_params_df = pd.DataFrame(plane_params_data)

    region_boundaries_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'X_min': [0.00, 0.98, 0.00, 1.99, 2.99, 1.99, 2.99],
        'X_max': [0.98, 1.99, 1.99, 2.99, 4.00, 2.99, 4.00],
        'Y_min': [0.00, 0.00, 2.49, 0.00, 0.00, 2.49, 2.49],
        'Y_max': [2.49, 2.49, 5.00, 2.49, 2.49, 5.00, 5.00]
    }
    region_boundaries_df = pd.DataFrame(region_boundaries_data)

    # 2. 准备插值器
    print("正在创建深度插值器...")
    interpolator = LinearNDInterpolator(points=grid_df[['横坐标', '纵坐标']].values, values=grid_df['深度'].values)

    # 3. 为所有区域规划测线
    print("正在为所有区域规划测线...")
    all_lines = []
    for _, region_row in params_df.iterrows():
        region_id = int(region_row['区域编号'])
        print(f"  - 处理区域 {region_id}...")
        
        # 为此区域生成测线（不进行裁剪）
        region_lines = plan_lines_for_region(region_row, plane_params_df, interpolator)
        all_lines.extend(region_lines)

    lines_df = pd.DataFrame(all_lines)
    lines_df.to_csv("survey_lines_q4.csv", index=False, float_format='%.4f')
    print("\n所有测线数据已保存至 survey_lines_q4.csv")

    # 4. 生成报告
    print("\n--- 测线规划结果摘要 ---")

    summary_data = []
    total_length = 0
    for region_id in lines_df['region_id'].unique():
        region_lines_df = lines_df[lines_df['region_id'] == region_id]
        num_lines = len(region_lines_df)
        length_nm = region_lines_df['length_nm'].sum()
        total_length += length_nm
        summary_data.append({
            '区域编号': region_id,
            '测线数量': num_lines,
            '总长度 (海里)': length_nm,
            '总长度 (公里)': length_nm * 1.852
        })
    summary_df = pd.DataFrame(summary_data)
    
    print("\n| 区域编号 | 测线数量 | 总长度 (海里) | 总长度 (公里) |")
    print("|:---:|:---:|:---:|:---:|")
    for index, row in summary_df.iterrows():
        print(f"| {int(row['区域编号'])} "
              f"| {int(row['测线数量'])} "
              f"| {row['总长度 (海里)']:.2f} "
              f"| {row['总长度 (公里)']:.2f} |")

    print(f"\n所有区域测线总长度为: {total_length:.2f} 海里 ({total_length * 1.852:.2f} 公里)。")

    # 5. 可视化结果
    print("\n正在生成可视化图像...")
    visualize_plan(lines_df, region_boundaries_df, grid_df)

if __name__ == '__main__':
    main() 