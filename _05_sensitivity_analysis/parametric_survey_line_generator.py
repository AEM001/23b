#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数化测线生成器
基于原始的智能边界扩展优化算法，但将关键参数设为可配置
用于灵敏度分析
"""

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

class ParametricSurveyLineGenerator:
    """参数化测线生成器类"""
    
    def __init__(self, 
                 safety_margin_factor=0.2,      # 安全边距系数（默认20%）
                 angle_correction_factor=0.5,   # 角度修正系数（默认0.5）
                 max_angle_factor=3.0,          # 最大角度系数限制（默认3.0）
                 min_direction_component=0.1,   # 最小方向分量（防除零，默认0.1）
                 extension_base_factor=1.0):    # 基础延伸系数（默认1.0为完整覆盖宽度）
        """
        初始化参数化测线生成器
        
        参数说明：
        - safety_margin_factor: 安全边距系数，在基础延伸之上的额外安全缓冲
        - angle_correction_factor: 角度修正系数，控制角度修正的强度
        - max_angle_factor: 角度系数的最大限制值
        - min_direction_component: 防止除零的最小方向分量
        - extension_base_factor: 基础延伸系数，控制基础延伸距离
        """
        self.safety_margin_factor = safety_margin_factor
        self.angle_correction_factor = angle_correction_factor
        self.max_angle_factor = max_angle_factor
        self.min_direction_component = min_direction_component
        self.extension_base_factor = extension_base_factor
        
        # 加载深度插值器
        self.depth_interpolator = self._load_depth_interpolator()
    
    def _load_depth_interpolator(self):
        """加载深度插值器"""
        df = pd.read_csv('../_00_source_data/output.csv')
        x_nm = np.array(df['横坐标'].values, dtype=float)
        y_nm = np.array(df['纵坐标'].values, dtype=float) 
        depth = np.array(df['深度'].values, dtype=float)
        return LinearNDInterpolator(list(zip(x_nm, y_nm)), depth)
    
    def calculate_swath_width(self, depth, beam_angle=120):
        """根据水深计算条带宽度"""
        swath_width_m = 2 * depth * np.tan(np.radians(beam_angle / 2))
        swath_width_nm = swath_width_m / 1852
        return min(swath_width_nm, 0.2)  # 限制最大宽度0.2海里
    
    def find_line_region_intersections_parametric(self, x_start, y_start, x_end, y_end, 
                                                x_min, x_max, y_min, y_max):
        """
        参数化的测线与矩形区域边界交点计算及智能延伸
        使用可配置的参数进行延伸计算
        """
        # 首先找到基本的边界交点
        def get_boundary_intersections(x1, y1, x2, y2):
            """计算直线与矩形边界的所有交点"""
            intersections = []
            
            # 与各边界的交点
            # 左边界 x = x_min
            if x2 != x1:
                t = (x_min - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    y = y1 + t * (y2 - y1)
                    if y_min <= y <= y_max:
                        intersections.append((x_min, y, t, 'left'))
            
            # 右边界 x = x_max  
            if x2 != x1:
                t = (x_max - x1) / (x2 - x1)
                if 0 <= t <= 1:
                    y = y1 + t * (y2 - y1)
                    if y_min <= y <= y_max:
                        intersections.append((x_max, y, t, 'right'))
            
            # 下边界 y = y_min
            if y2 != y1:
                t = (y_min - y1) / (y2 - y1)
                if 0 <= t <= 1:
                    x = x1 + t * (x2 - x1)
                    if x_min <= x <= x_max:
                        intersections.append((x, y_min, t, 'bottom'))
            
            # 上边界 y = y_max
            if y2 != y1:
                t = (y_max - y1) / (y2 - y1)
                if 0 <= t <= 1:
                    x = x1 + t * (x2 - x1)
                    if x_min <= x <= x_max:
                        intersections.append((x, y_max, t, 'top'))
            
            return sorted(intersections, key=lambda x: x[2])  # 按参数t排序
        
        # 获取边界交点
        intersections = get_boundary_intersections(x_start, y_start, x_end, y_end)
        
        if len(intersections) < 2:
            return None
        
        # 取前两个交点作为基本的裁剪结果
        x1, y1 = intersections[0][0], intersections[0][1]
        x2, y2 = intersections[-1][0], intersections[-1][1]
        
        # 计算测线方向向量
        line_length = np.sqrt((x_end - x_start)**2 + (y_end - y_start)**2)
        if line_length == 0:
            return None
            
        dx = (x_end - x_start) / line_length
        dy = (y_end - y_start) / line_length
        
        # 参数化的角度系数计算
        def calculate_angle_factor(boundary_type, dx, dy):
            """根据测线与边界的夹角计算延伸系数（参数化版本）"""
            if boundary_type in ['left', 'right']:
                # 与垂直边界的夹角，dx越小角度越小，需要更大延伸
                angle_factor = 1.0 / max(abs(dx), self.min_direction_component)
            else:  # top, bottom
                # 与水平边界的夹角，dy越小角度越小，需要更大延伸  
                angle_factor = 1.0 / max(abs(dy), self.min_direction_component)
            
            return min(angle_factor, self.max_angle_factor)
        
        # 参数化的延伸距离计算
        def get_extension_distance_parametric(x, y, boundary_type):
            """计算在某个边界点需要的延伸距离（参数化版本）"""
            # 查询该点的深度
            depth = self.depth_interpolator(x, y)
            if np.isnan(depth):
                depth = 110  # 使用平均深度作为备选
                
            # 计算覆盖宽度
            swath_width = self.calculate_swath_width(depth)
            
            # 角度修正系数
            angle_factor = calculate_angle_factor(boundary_type, dx, dy)
            
            # 参数化延伸策略：
            # 1. 基础延伸 = extension_base_factor * 覆盖宽度
            # 2. 角度修正：根据入射角和angle_correction_factor增加延伸
            # 3. 边界安全边距：额外增加safety_margin_factor比例
            base_extension = swath_width * self.extension_base_factor
            angle_extension = base_extension * (angle_factor - 1) * self.angle_correction_factor
            safety_margin = swath_width * self.safety_margin_factor
            
            total_extension = base_extension + angle_extension + safety_margin
            
            return total_extension
        
        # 计算起点延伸距离
        boundary_type_start = intersections[0][3]
        ext_dist_start = get_extension_distance_parametric(x1, y1, boundary_type_start)
        
        # 计算终点延伸距离  
        boundary_type_end = intersections[-1][3]
        ext_dist_end = get_extension_distance_parametric(x2, y2, boundary_type_end)
        
        # 向外延伸测线
        x1_ext = x1 - dx * ext_dist_start
        y1_ext = y1 - dy * ext_dist_start
        x2_ext = x2 + dx * ext_dist_end  
        y2_ext = y2 + dy * ext_dist_end
        
        return x1_ext, y1_ext, x2_ext, y2_ext
    
    def optimize_survey_lines(self, lines_df, region_boundaries):
        """
        对现有的测线进行参数化的智能长度优化
        """
        optimized_lines = []
        
        for _, line in lines_df.iterrows():
            region_id = line['region_id']
            
            # 获取区域边界
            region_bounds = region_boundaries[region_boundaries['区域编号'] == region_id].iloc[0]
            x_min, x_max = region_bounds['X_min'], region_bounds['X_max']
            y_min, y_max = region_bounds['Y_min'], region_bounds['Y_max']
            
            # 对测线进行参数化智能边界延伸
            result = self.find_line_region_intersections_parametric(
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
                
                optimized_lines.append(optimized_line)
        
        return pd.DataFrame(optimized_lines)
    
    def get_parameter_summary(self):
        """返回当前参数设置的摘要"""
        return {
            'safety_margin_factor': self.safety_margin_factor,
            'angle_correction_factor': self.angle_correction_factor,
            'max_angle_factor': self.max_angle_factor,
            'min_direction_component': self.min_direction_component,
            'extension_base_factor': self.extension_base_factor
        }

def generate_survey_lines_with_parameters(parameter_config, output_filename=None):
    """
    使用指定参数配置生成测线
    
    参数:
    - parameter_config: 参数配置字典
    - output_filename: 输出文件名（可选）
    
    返回:
    - 优化后的测线DataFrame
    """
    # 创建参数化生成器
    generator = ParametricSurveyLineGenerator(**parameter_config)
    
    # 加载原始测线数据
    lines_df_original = pd.read_csv("../_03_line_generation/survey_lines_q4.csv")
    
    # 区域边界数据
    region_boundaries_data = {
        '区域编号': [0, 1, 2, 3, 4, 5, 6],
        'X_min': [0.00, 0.98, 0.00, 1.99, 2.99, 1.99, 2.99],
        'X_max': [0.98, 1.99, 1.99, 2.99, 4.00, 2.99, 4.00],
        'Y_min': [0.00, 0.00, 2.49, 0.00, 0.00, 2.49, 2.49],
        'Y_max': [2.49, 2.49, 5.00, 2.49, 2.49, 5.00, 5.00]
    }
    region_boundaries_df = pd.DataFrame(region_boundaries_data)
    
    # 生成优化的测线
    lines_df_optimized = generator.optimize_survey_lines(lines_df_original, region_boundaries_df)
    
    # 保存结果（如果指定了文件名）
    if output_filename:
        lines_df_optimized.to_csv(output_filename, index=False, float_format='%.4f')
    
    return lines_df_optimized, generator.get_parameter_summary()

if __name__ == '__main__':
    # 测试默认参数
    default_params = {
        'safety_margin_factor': 0.2,
        'angle_correction_factor': 0.5,
        'max_angle_factor': 3.0,
        'min_direction_component': 0.1,
        'extension_base_factor': 1.0
    }
    
    lines_df, param_summary = generate_survey_lines_with_parameters(
        default_params, 
        "test_parametric_lines.csv"
    )
    
    print("参数化测线生成器测试完成")
    print(f"生成测线数量: {len(lines_df)}")
    print(f"总长度: {lines_df['length_optimized_nm'].sum():.2f} 海里")
    print("参数设置:", param_summary) 