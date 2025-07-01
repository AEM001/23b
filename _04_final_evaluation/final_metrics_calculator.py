import pandas as pd
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

# --- Constants ---
NM_TO_M = 1852
THETA_RAD = np.radians(120)

# --- Grid Parameters ---
X_MIN_NM, X_MAX_NM = 0, 4
Y_MIN_NM, Y_MAX_NM = 0, 5
GRID_RESOLUTION_M = 25 # meters, a balance between accuracy and computation time

def get_region_slope(beta1, beta2):
    """Calculates the region's slope angle from plane coefficients."""
    tan_alpha = np.sqrt((beta1 / NM_TO_M)**2 + (beta2 / NM_TO_M)**2)
    return np.arctan(tan_alpha)

def main():
    """Main function to calculate final metrics."""
    print("开始最终指标计算...")

    # 1. Load all necessary data
    print("1/5: 加载数据...")
    try:
        lines_df = pd.read_csv('../_03_line_generation/survey_lines_q4.csv')
        params_df = pd.read_csv('../_02_coord_transform/local_coord_params.csv')
        grid_df = pd.read_csv('../_00_source_data/output.csv')
    except FileNotFoundError as e:
        print(f"错误: 必需的数据文件未找到 - {e}。请确保脚本在 '_04_final_evaluation' 目录中运行，并且其他数据目录存在。")
        return
        
    plane_params_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'β₁': [-0.21, 20.45, 2.35, 41.52, 62.59, 9.55, 14.40],
        'β₂': [4.47, 1.39, 18.97, -8.20, -24.34, 7.86, -8.28]
    }
    plane_params_df = pd.DataFrame(plane_params_data)

    # Prepare detailed analysis results storage
    region_stats = []
    overlap_details = []

    # 2. Prepare Interpolator and Grid
    print("2/5: 创建深度插值器和分析格网...")
    depth_interpolator = LinearNDInterpolator(points=grid_df[['横坐标', '纵坐标']].values, values=grid_df['深度'].values)
    
    x_coords_m = np.arange(X_MIN_NM * NM_TO_M, X_MAX_NM * NM_TO_M, GRID_RESOLUTION_M)
    y_coords_m = np.arange(Y_MIN_NM * NM_TO_M, Y_MAX_NM * NM_TO_M, GRID_RESOLUTION_M)
    grid_x_m, grid_y_m = np.meshgrid(x_coords_m, y_coords_m)
    
    coverage_grid = np.zeros_like(grid_x_m, dtype=np.int16)
    region_coverage_grids = {}
    
    # 3. Rasterize Swaths (Calculate Coverage)
    print("3/5: 光栅化所有测线覆盖范围 (这可能需要几分钟)...")
    for _, line in tqdm(lines_df.iterrows(), total=len(lines_df), desc="处理测线"):
        region_id = line['region_id']
        if region_id not in region_coverage_grids:
            region_coverage_grids[region_id] = np.zeros_like(grid_x_m, dtype=np.int16)
            
        region_params = params_df[params_df['区域编号'] == region_id].iloc[0]
        plane_params = plane_params_df[plane_params_df['区域编号'] == region_id].iloc[0]
        
        alpha_rad = get_region_slope(plane_params['β₁'], plane_params['β₂'])

        p1 = np.array([line['x_start_nm'], line['y_start_nm']]) * NM_TO_M
        p2 = np.array([line['x_end_nm'], line['y_end_nm']]) * NM_TO_M

        line_vec = p2 - p1
        line_len_sq = np.sum(line_vec**2)
        if line_len_sq == 0: continue

        grid_points_vec = np.stack([grid_x_m.ravel() - p1[0], grid_y_m.ravel() - p1[1]], axis=1)
        t = np.dot(grid_points_vec, line_vec) / line_len_sq
        t = np.clip(t, 0, 1)

        proj_points = p1 + t[:, np.newaxis] * line_vec
        dist_perp = np.linalg.norm(np.stack([grid_x_m.ravel(), grid_y_m.ravel()], axis=1) - proj_points, axis=1)

        depths_on_line = depth_interpolator(proj_points / NM_TO_M)
        nan_depths = np.isnan(depths_on_line)
        if np.all(nan_depths): continue
        depths_on_line[nan_depths] = np.nanmean(depths_on_line) # Fill gaps

        W_left = (depths_on_line * np.sin(THETA_RAD / 2) / np.cos(THETA_RAD / 2 + alpha_rad))
        W_right = (depths_on_line * np.sin(THETA_RAD / 2) / np.cos(THETA_RAD / 2 - alpha_rad))
        
        # Determine if points are to the left or right of the directed line
        cross_product = np.cross(line_vec, grid_points_vec)
        is_left = cross_product > 0
        
        covered_mask = (is_left & (dist_perp <= W_left)) | (~is_left & (dist_perp <= W_right))
        coverage_grid += covered_mask.reshape(grid_x_m.shape)
        region_coverage_grids[region_id] += covered_mask.reshape(grid_x_m.shape)

    # 4. Calculate Leakage Rate and Region Statistics
    print("4/5: 计算漏测率和分区统计...")
    
    # Overall statistics
    uncovered_cells = np.sum(coverage_grid == 0)
    total_cells = grid_x_m.size
    leakage_rate = (uncovered_cells / total_cells) * 100
    
    print(f"  - 总网格单元数: {total_cells}")
    print(f"  - 未覆盖单元数: {uncovered_cells}")
    print(f"  - 漏测率: {leakage_rate:.4f}%")
    
    # Region-wise statistics
    for region_id in sorted(region_coverage_grids.keys()):
        region_lines = lines_df[lines_df['region_id'] == region_id]
        region_grid = region_coverage_grids[region_id]
        
        region_uncovered = np.sum(region_grid == 0)
        region_single_covered = np.sum(region_grid == 1)
        region_multi_covered = np.sum(region_grid > 1)
        region_total = region_grid.size
        
        region_length = region_lines['length_nm'].sum()
        region_line_count = len(region_lines)
        
        region_params = params_df[params_df['区域编号'] == region_id].iloc[0]
        region_area = (region_params['u_max'] - region_params['u_min']) * (region_params['v_max'] - region_params['v_min'])
        
        region_stats.append({
            '区域编号': int(region_id),
            '测线数量': region_line_count,
            '测线总长度_海里': region_length,
            '区域面积_平方海里': region_area,
            '测线密度_海里每平方海里': region_length / region_area if region_area > 0 else 0,
            '网格总单元数': region_total,
            '未覆盖单元数': region_uncovered,
            '单次覆盖单元数': region_single_covered,
            '多次覆盖单元数': region_multi_covered,
            '区域漏测率_%': (region_uncovered / region_total) * 100 if region_total > 0 else 0,
            '多次覆盖率_%': (region_multi_covered / region_total) * 100 if region_total > 0 else 0
        })
    
    # 5. Calculate Excess Overlap Length
    print("5/5: 计算超额重叠长度...")
    total_excess_overlap_length_nm = 0

    lines_by_region = lines_df.groupby('region_id')

    for region_id, lines in lines_by_region:
        lines = lines.sort_values('line_id').reset_index()
        
        region_params = params_df[params_df['区域编号'] == region_id].iloc[0]
        plane_params = plane_params_df[plane_params_df['区域编号'] == region_id].iloc[0]
        alpha_rad = get_region_slope(plane_params['β₁'], plane_params['β₂'])

        for i in range(1, len(lines)):
            line_curr = lines.iloc[i]
            line_prev = lines.iloc[i-1]

            # Sample points along the current line
            p1_curr = np.array([line_curr['x_start_nm'], line_curr['y_start_nm']])
            p2_curr = np.array([line_curr['x_end_nm'], line_curr['y_end_nm']])
            
            p1_prev = np.array([line_prev['x_start_nm'], line_prev['y_start_nm']])
            p2_prev = np.array([line_prev['x_end_nm'], line_prev['y_end_nm']])

            num_samples = int(line_curr['length_nm'] * NM_TO_M / GRID_RESOLUTION_M)
            if num_samples < 2: num_samples = 2

            excess_segments = 0
            for j in range(num_samples):
                t = j / (num_samples - 1)
                sample_point_curr_nm = p1_curr + t * (p2_curr - p1_curr)
                
                # Find corresponding point on previous line
                line_prev_vec_nm = p2_prev - p1_prev
                t_proj = np.dot(sample_point_curr_nm - p1_prev, line_prev_vec_nm) / np.dot(line_prev_vec_nm, line_prev_vec_nm)
                sample_point_prev_nm = p1_prev + np.clip(t_proj, 0, 1) * line_prev_vec_nm

                d_m = np.linalg.norm(sample_point_curr_nm - sample_point_prev_nm) * NM_TO_M

                D_curr = depth_interpolator(sample_point_curr_nm)
                D_prev = depth_interpolator(sample_point_prev_nm)
                if np.isnan(D_curr) or np.isnan(D_prev): continue

                W_curr_left_proj = (D_curr * np.sin(THETA_RAD/2) / np.cos(THETA_RAD/2 + alpha_rad)) * np.cos(alpha_rad)
                W_prev_right_proj = (D_prev * np.sin(THETA_RAD/2) / np.cos(THETA_RAD/2 - alpha_rad)) * np.cos(alpha_rad)

                overlap_dist_m = W_curr_left_proj + W_prev_right_proj - d_m
                
                W_curr_right_proj = (D_curr * np.sin(THETA_RAD/2) / np.cos(THETA_RAD/2 - alpha_rad)) * np.cos(alpha_rad)
                total_width_m = W_curr_left_proj + W_curr_right_proj

                if total_width_m > 0:
                    overlap_rate = overlap_dist_m / total_width_m
                    if overlap_rate > 0.20:
                        segment_len_nm = line_curr['length_nm'] / num_samples
                        total_excess_overlap_length_nm += segment_len_nm
                        excess_segments += 1
                        
            if excess_segments > 0:
                overlap_details.append({
                    '区域编号': int(region_id),
                    '当前测线编号': int(line_curr['line_id']),
                    '前一测线编号': int(line_prev['line_id']),
                    '测线长度_海里': line_curr['length_nm'],
                    '超额重叠段数': excess_segments,
                    '总采样段数': num_samples,
                    '超额重叠长度_海里': (excess_segments / num_samples) * line_curr['length_nm'],
                    '超额重叠比例_%': (excess_segments / num_samples) * 100
                })
                        
    print(f"  - 重叠率超20%部分的总长度: {total_excess_overlap_length_nm:.4f} 海里")

    # 6. Save detailed results to files
    print("6/6: 保存详细分析结果到文件...")
    
    # Save region statistics
    region_stats_df = pd.DataFrame(region_stats)
    region_stats_df.to_csv('region_wise_statistics.csv', index=False, float_format='%.4f')
    print("  - 区域统计数据已保存至: region_wise_statistics.csv")
    
    # Save overlap details
    if overlap_details:
        overlap_details_df = pd.DataFrame(overlap_details)
        overlap_details_df.to_csv('excess_overlap_details.csv', index=False, float_format='%.4f')
        print("  - 超额重叠详情已保存至: excess_overlap_details.csv")
    
    # Save coverage analysis
    coverage_analysis = {
        '总网格单元数': total_cells,
        '未覆盖单元数': uncovered_cells,
        '单次覆盖单元数': np.sum(coverage_grid == 1),
        '双次覆盖单元数': np.sum(coverage_grid == 2),
        '三次及以上覆盖单元数': np.sum(coverage_grid >= 3),
        '最大覆盖次数': int(np.max(coverage_grid)),
        '平均覆盖次数': float(np.mean(coverage_grid)),
        '漏测率_%': leakage_rate
    }
    
    coverage_df = pd.DataFrame([coverage_analysis])
    coverage_df.to_csv('coverage_analysis.csv', index=False, float_format='%.4f')
    print("  - 覆盖分析数据已保存至: coverage_analysis.csv")

    # Final summary
    final_summary = {
        '测线总数量': len(lines_df),
        '测线总长度_海里': lines_df['length_nm'].sum(),
        '测线总长度_公里': lines_df['length_nm'].sum() * 1.852,
        '漏测率_%': leakage_rate,
        '超额重叠长度_海里': total_excess_overlap_length_nm,
        '超额重叠比例_%': (total_excess_overlap_length_nm / lines_df['length_nm'].sum()) * 100,
        '网格分辨率_米': GRID_RESOLUTION_M,
        '分析区域面积_平方海里': (X_MAX_NM - X_MIN_NM) * (Y_MAX_NM - Y_MIN_NM),
        '平均测线密度_海里每平方海里': lines_df['length_nm'].sum() / ((X_MAX_NM - X_MIN_NM) * (Y_MAX_NM - Y_MIN_NM))
    }
    
    summary_df = pd.DataFrame([final_summary])
    summary_df.to_csv('final_survey_summary.csv', index=False, float_format='%.4f')
    print("  - 最终测量总结已保存至: final_survey_summary.csv")

    # Final Report
    print("\n--- 最终指标计算结果 ---")
    final_length = lines_df['length_nm'].sum()
    print(f"测线总长度: {final_length:.2f} 海里 ({final_length * 1.852:.2f} 公里)")
    print(f"漏测海区占总待测海域面积的百分比: {leakage_rate:.4f}%")
    print(f"重叠率超过20%部分的总长度: {total_excess_overlap_length_nm:.2f} 海里")
    print("详细分析结果已保存到相应的CSV文件中")
    print("------------------------")

if __name__ == '__main__':
    main() 