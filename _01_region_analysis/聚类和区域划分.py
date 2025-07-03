import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def setup_chinese_font():
    """简洁设置matplotlib中文字体。"""
    import matplotlib
    fonts = ['PingFang SC', 'Microsoft YaHei']
    for font in fonts:
        if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False

# --- 分析模块 1: K-Means 聚类分析 ---
def analyze_and_plot_clustering(grid_x, grid_y, grid_z, slope, aspect, points, depths, n_clusters):
    """执行K-Means聚类分析并可视化结果。"""
    print("\n--- 步骤 1: 执行 K-Means 聚类分析 ---")
    
    # 准备聚类特征
    features = np.vstack([
        grid_x.flatten(),
        grid_y.flatten(),
        np.nan_to_num(slope.flatten()),
        np.nan_to_num(aspect.flatten())
    ]).T
    features_scaled = StandardScaler().fit_transform(features)

    # 执行K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features_scaled)
    labels_grid = labels.reshape(grid_x.shape)
    labels_grid[np.isnan(grid_z)] = -1

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
    fig.suptitle('分析步骤 1: 基于地形特征的聚类 (K-Means)', fontsize=16)
    grid_z_plot = np.nan_to_num(grid_z, nan=np.nanmean(depths))

    # a) 等深线图
    axes[0].contourf(grid_x, grid_y, grid_z_plot, levels=40, cmap='viridis')
    axes[0].contour(grid_x, grid_y, grid_z_plot, levels=40, colors='white', linewidths=0.5, alpha=0.6)
    axes[0].set_title('等深线图')
    axes[0].set_aspect('equal', 'box')

    # b) 聚类结果图
    unique_labels = np.unique(labels_grid)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    cmap_discrete = mcolors.ListedColormap([colors(i) for i in range(len(unique_labels))])
    bounds = np.arange(len(unique_labels)+1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap_discrete.N)
    axes[1].imshow(labels_grid, extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                   origin='lower', cmap=cmap_discrete, norm=norm, interpolation='none')
    axes[1].set_title('聚类结果 (重叠区域)')
    axes[1].set_aspect('equal', 'box')

    # c) 聚类矩形划分图
    axes[2].contourf(grid_x, grid_y, grid_z_plot, levels=40, cmap='viridis', alpha=0.7)
    axes[2].set_title('聚类边界框 (有重叠)')
    axes[2].set_aspect('equal', 'box')
    for cluster_id in range(n_clusters):
        points_in_cluster = np.argwhere(labels_grid == cluster_id)
        if points_in_cluster.size == 0: continue
        y_indices, x_indices = points_in_cluster.T
        x_coords, y_coords = grid_x[0, x_indices], grid_y[y_indices, 0]
        rect = patches.Rectangle((x_coords.min(), y_coords.min()), x_coords.max() - x_coords.min(), y_coords.max() - y_coords.min(),
                                 linewidth=2, edgecolor=colors(cluster_id % 20), facecolor='none')
        axes[2].add_patch(rect)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('clustering_analysis.png')
    print("聚类分析图已保存至 'clustering_analysis.png'")
    plt.show(block=True)
    return labels_grid

# --- 分析模块 2: 递归分割 ---
def recursive_split(bounds, aspect_grid, grid_x_coords, grid_y_coords, depth, max_depth, aspect_variance_threshold, min_cells):
    """根据坡向方差，递归地将区域分割成不重叠的矩形。"""
    x_start_idx, x_end_idx, y_start_idx, y_end_idx = bounds
    if depth >= max_depth:
        return [(grid_x_coords[x_start_idx], grid_x_coords[x_end_idx], grid_y_coords[y_start_idx], grid_y_coords[y_end_idx])]

    aspect_slice = aspect_grid[y_start_idx:y_end_idx+1, x_start_idx:x_end_idx+1]
    valid_aspects = aspect_slice[~np.isnan(aspect_slice)]

    if valid_aspects.size < min_cells or np.var(valid_aspects) < aspect_variance_threshold:
        return [(grid_x_coords[x_start_idx], grid_x_coords[x_end_idx], grid_y_coords[y_start_idx], grid_y_coords[y_end_idx])]

    width, height = x_end_idx - x_start_idx, y_end_idx - y_start_idx
    if width >= height:
        split_idx = x_start_idx + width // 2
        bounds1, bounds2 = (x_start_idx, split_idx, y_start_idx, y_end_idx), (split_idx, x_end_idx, y_start_idx, y_end_idx)
    else:
        split_idx = y_start_idx + height // 2
        bounds1, bounds2 = (x_start_idx, x_end_idx, y_start_idx, split_idx), (x_start_idx, x_end_idx, split_idx, y_end_idx)
    
    return recursive_split(bounds1, aspect_grid, grid_x_coords, grid_y_coords, depth + 1, max_depth, aspect_variance_threshold, min_cells) + \
           recursive_split(bounds2, aspect_grid, grid_x_coords, grid_y_coords, depth + 1, max_depth, aspect_variance_threshold, min_cells)

def analyze_and_plot_split(grid_x, grid_y, grid_z, aspect, depths, params):
    """执行递归分割分析并可视化结果。"""
    print("\n--- 步骤 2: 执行递归分割获取最终方案 ---")
    grid_x_coords = grid_x[0, :]
    grid_y_coords = grid_y[:, 0]
    initial_bounds = (0, len(grid_x_coords) - 1, 0, len(grid_y_coords) - 1)
    
    final_rectangles = recursive_split(initial_bounds, aspect, grid_x_coords, grid_y_coords, 0, **params)

    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    fig.suptitle('分析步骤 2: 最终矩形区域划分 (递归分割法)', fontsize=16)
    grid_z_plot = np.nan_to_num(grid_z, nan=np.nanmean(depths))
    contour = ax.contourf(grid_x, grid_y, grid_z_plot, levels=40, cmap='viridis')
    ax.contour(grid_x, grid_y, grid_z_plot, levels=40, colors='white', linewidths=0.5, alpha=0.7)
    fig.colorbar(contour, ax=ax, label='深度 (m)')
    ax.set_title(f'最终划分: {len(final_rectangles)} 个无重叠区域')
    ax.set_xlabel('横坐标'), ax.set_ylabel('纵坐标'), ax.set_aspect('equal', 'box')

    for i, (x_min, x_max, y_min, y_max) in enumerate(final_rectangles):
        if x_min >= x_max or y_min >= y_max: continue
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min + (x_max - x_min)/2, y_min + (y_max - y_min)/2, str(i),
                ha='center', va='center', color='white', fontsize=10, weight='bold')

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig('final_division_analysis.png')
    print("最终划分方案图已保存至 'final_division_analysis.png'")
    plt.show(block=True)
    return final_rectangles

# --- 报告生成模块 ---
def generate_markdown_report(rectangles, cluster_params, split_params, report_file='region_division_report.md'):
    """生成包含分析参数和结果的Markdown报告。"""
    print(f"\n--- 步骤 3: 生成分析报告 ---")
    
    print(f"关键参数总结:")
    print(f"- 聚类数量 (n_clusters): {cluster_params['n_clusters']}")
    print(f"- 最大递归深度 (max_depth): {split_params['max_depth']}")
    print(f"- 坡向方差阈值 (aspect_variance_threshold): {split_params['aspect_variance_threshold']}")
    print(f"- 区域最小网格数 (min_cells): {split_params['min_cells']}")
    print(f"- 最终划分区域总数: {len(rectangles)} 个")


# --- 主执行函数 ---
def run_full_analysis(data_path='output.csv'):
    """执行完整的两步区域划分分析并生成报告。"""
    setup_chinese_font()

    # --- 数据加载与预处理 (公共步骤) ---
    print("--- 开始数据加载与预处理 ---")
    df = pd.read_csv(data_path)
    points = df[['横坐标', '纵坐标']].values
    depths = df['深度'].values

    grid_y_coords, grid_x_coords = [np.linspace(df[col].min(), df[col].max(), 200) for col in ['纵坐标', '横坐标']]
    grid_y, grid_x = np.meshgrid(grid_y_coords, grid_x_coords, indexing='ij')
    grid_z = griddata(points, depths, (grid_x, grid_y), method='cubic')
    dy, dx = np.gradient(grid_z)
    slope, aspect = np.sqrt(dx**2 + dy**2), np.arctan2(-dy, -dx)
    print("数据预处理完成。")

    # --- 步骤 1: 聚类分析 ---
    cluster_params = {'n_clusters': 8}
    analyze_and_plot_clustering(grid_x, grid_y, grid_z, slope, aspect, points, depths, **cluster_params)

    # --- 步骤 2: 递归分割 ---
    split_params = {
        'max_depth': 3,
        'aspect_variance_threshold': 0.7,
        'min_cells': 200
    }
    final_rectangles = analyze_and_plot_split(grid_x, grid_y, grid_z, aspect, depths, split_params)

    # --- 步骤 3: 生成报告 ---
    generate_markdown_report(final_rectangles, cluster_params, split_params)

if __name__ == '__main__':
    run_full_analysis() 