import pandas as pd
import numpy as np

def calculate_local_coordinates():
    """
    根据区域划分和平面拟合结果，计算每个子区域的局部坐标系参数。
    """
    # 从 region_division_report.md 中提取的数据
    boundaries_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'X 范围 (min, max)': ['(0.00, 0.98)', '(0.98, 1.99)', '(0.00, 1.99)', '(1.99, 2.99)', '(2.99, 4.00)', '(1.99, 2.99)', '(2.99, 4.00)'],
        'Y 范围 (min, max)': ['(0.00, 2.49)', '(0.00, 2.49)', '(2.49, 5.00)', '(0.00, 2.49)', '(0.00, 2.49)', '(2.49, 5.00)', '(2.49, 5.00)']
    }
    boundaries_df = pd.DataFrame(boundaries_data)

    params_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'β₁': [-0.21, 20.45, 2.35, 41.52, 62.59, 9.55, 14.40],
        'β₂': [4.47, 1.39, 18.97, -8.20, -24.34, 7.86, -8.28]
    }
    params_df = pd.DataFrame(params_data)

    df = pd.merge(boundaries_df, params_df, on='区域编号')

    def parse_range(s):
        s = s.strip().replace('(', '').replace(')', '')
        return map(float, s.split(','))

    results = []
    for index, row in df.iterrows():
        region_id = row['区域编号']
        x_min, x_max = parse_range(row['X 范围 (min, max)'])
        y_min, y_max = parse_range(row['Y 范围 (min, max)'])
        beta1 = row['β₁']
        beta2 = row['β₂']

        # 核心计算步骤
        # 1. 确定方向角
        phi_k_rad = np.arctan2(-beta2, -beta1)
        theta_k_rad = phi_k_rad + np.pi / 2
        
        # 2. 定义区域的四个角点
        corners_xy = [
            (x_min, y_min), (x_min, y_max),
            (x_max, y_min), (x_max, y_max)
        ]

        # 3. 将角点变换到 (u,v) 坐标系
        corners_uv = []
        for x, y in corners_xy:
            u = x * np.cos(theta_k_rad) + y * np.sin(theta_k_rad)
            v = -x * np.sin(theta_k_rad) + y * np.cos(theta_k_rad)
            corners_uv.append((u, v))

        # 4. 找到 (u,v) 坐标的边界
        u_coords = [p[0] for p in corners_uv]
        v_coords = [p[1] for p in corners_uv]
        u_min, u_max = min(u_coords), max(u_coords)
        v_min, v_max = min(v_coords), max(v_coords)

        results.append({
            '区域编号': region_id,
            '坡向_rad': phi_k_rad,
            '坡向_deg': np.rad2deg(phi_k_rad),
            '主测线方向_rad': theta_k_rad,
            '主测线方向_deg': np.rad2deg(theta_k_rad),
            'u_min': u_min,
            'u_max': u_max,
            'v_min': v_min,
            'v_max': v_max
        })

    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    local_coord_params_df = calculate_local_coordinates()
    
    # 保存为 CSV 文件，供后续计算使用
    output_csv_path = 'local_coord_params.csv'
    local_coord_params_df.to_csv(output_csv_path, index=False, float_format='%.4f')
    
    print(f"计算完成，结果已保存到 {output_csv_path}")

    print("\n--- 各子区域局部坐标系参数 ---")
    print("| 区域编号 | 坡向 φ (°) | 主测线方向 θ (°) | u_min | u_max | v_min | v_max |")
    print("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for index, row in local_coord_params_df.iterrows():
        print(f"| {int(row['区域编号'])} "
              f"| {np.rad2deg(row['坡向_rad']):.2f} "
              f"| {np.rad2deg(row['主测线方向_rad']):.2f} "
              f"| {row['u_min']:.2f} "
              f"| {row['u_max']:.2f} "
              f"| {row['v_min']:.2f} "
              f"| {row['v_max']:.2f} |") 