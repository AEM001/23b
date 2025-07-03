import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

# --- 配置 ---
# 设置字体为支持中文字符的本地可用字体
# 常用选项: 'SimHei', 'Heiti TC', 'Microsoft YaHei'
mpl.rcParams['font.sans-serif'] = ['Heiti TC'] 
mpl.rcParams['axes.unicode_minus'] = False

INPUT_FILE = "/Users/Mac/Downloads/23b/3/result3.xlsx"
OUTPUT_FILE = "survey_top_down_view.png"

# 海域尺寸
L_ew_nm = 4  # 东西方向长度 (海里)
L_ns_nm = 2  # 南北方向长度 (海里)
NM_TO_M = 1852  # 转换系数
L_ew_m = L_ew_nm * NM_TO_M
L_ns_m = L_ns_nm * NM_TO_M

# --- 可视化函数 ---
def create_top_down_view(data_file):
    """
    创建测线覆盖的顶视图。
    """
    try:
        df = pd.read_excel(data_file)
    except FileNotFoundError:
        print(f"错误：输入文件 '{data_file}' 未找到。请先运行 problem3_solution.py。")
        return

    fig, ax = plt.subplots(figsize=(15, 7.5))

    # 1. 绘制测量区域边界
    boundary = patches.Rectangle(
        (0, 0), L_ew_m, L_ns_m,
        linewidth=2, edgecolor='black', facecolor='none',
        label='待测海域边界'
    )
    ax.add_patch(boundary)

    # 2. 定义测绘带颜色并迭代绘制
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 颜色小循环
    for i, row in df.iterrows():
        line_pos = row['离西侧距离 (m)']
        left_edge = row['左覆盖边界 (m)']
        right_edge = row['右覆盖边界 (m)']
        width = right_edge - left_edge
        
        # 绘制覆盖测绘带（未填充的彩色矩形）
        swath = patches.Rectangle(
            (left_edge, 0), width, L_ns_m,
            linewidth=1.5, edgecolor=colors[i % len(colors)], facecolor='none'
        )
        ax.add_patch(swath)

        # 绘制测线（船只轨迹）
        ax.plot([line_pos, line_pos], [0, L_ns_m], color='black', linestyle='--', linewidth=0.8, 
                label='测线' if i == 0 else "")

    # 3. 配置图表外观
    ax.set_xlabel('东西方向距离 (m)')
    ax.set_ylabel('南北方向距离 (m)')
    ax.set_title('测线布设顶视图 (仅边界)', fontsize=16, fontweight='bold')
    
    # 设置坐标轴限制和宽高比
    ax.set_xlim(-500, L_ew_m + 500)
    ax.set_ylim(-200, L_ns_m + 200)
    ax.set_aspect('equal', adjustable='box')

    # 创建自定义图例
    legend_patches = [
        patches.Patch(color='black', fill=False, label='海域边界'),
        patches.Patch(color='gray', fill=False, linewidth=1.5, label='测线覆盖范围 (边界)'),
        plt.Line2D([0], [0], color='black', linestyle='--', lw=1, label='测量船航迹')
    ]
    ax.legend(handles=legend_patches, loc='upper right')

    ax.grid(True, linestyle=':', alpha=0.5)

    # 4. 保存图表
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"可视化顶视图已保存至: {OUTPUT_FILE}")

# --- 主程序执行 ---
if __name__ == '__main__':
    create_top_down_view(INPUT_FILE) 