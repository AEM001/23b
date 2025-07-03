#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标计算适配器
调用第四问的指标计算方法，为灵敏度分析提供统一接口
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加第四问目录到路径，以便导入其指标计算模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '_04_final_evaluation'))

try:
    from final_metrics_calculator_real_coverage import (
        load_original_data,
        calculate_coverage_metrics
    )
except ImportError as e:
    print(f"无法导入指标计算模块: {e}")
    sys.exit(1)

class MetricsCalculatorAdapter:
    """指标计算适配器类"""
    
    def __init__(self):
        """初始化适配器，加载插值器"""
        print("正在初始化指标计算适配器...")
        self.interpolator = load_original_data()
        print("插值器加载完成")
    
    def calculate_comprehensive_metrics(self, lines_df, verbose=True):
        """
        计算综合指标
        
        参数:
        - lines_df: 测线DataFrame，必须包含坐标和长度信息
        - verbose: 是否输出详细信息
        
        返回:
        - 包含所有指标的字典
        """
        if verbose:
            print("正在计算综合指标...")
        
        # 1. 计算测线总长度
        total_length = lines_df['length_optimized_nm'].sum()
        if verbose:
            print(f"   - 测线总长度: {total_length:.2f} 海里")
        
        # 2. 计算覆盖指标（覆盖率、漏测率、超额重叠长度）
        coverage_metrics = calculate_coverage_metrics(lines_df, self.interpolator)
        
        # 3. 整合所有指标
        comprehensive_metrics = {
            # 基础指标
            'total_length_nm': total_length,
            'total_length_km': total_length * 1.852,
            'total_lines': len(lines_df),
            
            # 覆盖指标
            'coverage_rate': coverage_metrics['coverage_rate'],
            'miss_rate': coverage_metrics['miss_rate'],
            'miss_rate_percent': coverage_metrics['miss_rate'] * 100,
            'excess_overlap_length_nm': coverage_metrics['excess_overlap_length_nm'],
            'excess_overlap_length_km': coverage_metrics['excess_overlap_length_nm'] * 1.852,
            
            # 效率指标
            'excess_overlap_ratio': coverage_metrics['excess_overlap_length_nm'] / total_length,
            'excess_overlap_ratio_percent': (coverage_metrics['excess_overlap_length_nm'] / total_length) * 100,
            
            # 综合评分（可以根据需要调整权重）
            'efficiency_score': self._calculate_efficiency_score(
                coverage_metrics['coverage_rate'],
                coverage_metrics['miss_rate'],
                coverage_metrics['excess_overlap_length_nm'] / total_length,
                total_length
            )
        }
        
        if verbose:
            print(f"   - 覆盖率: {comprehensive_metrics['coverage_rate']:.4f} ({(1-comprehensive_metrics['miss_rate'])*100:.2f}%)")
            print(f"   - 漏测率: {comprehensive_metrics['miss_rate_percent']:.2f}%")
            print(f"   - 超额重叠长度: {comprehensive_metrics['excess_overlap_length_nm']:.2f} 海里")
            print(f"   - 超额重叠比例: {comprehensive_metrics['excess_overlap_ratio_percent']:.2f}%")
            print(f"   - 效率评分: {comprehensive_metrics['efficiency_score']:.4f}")
        
        return comprehensive_metrics
    
    def _calculate_efficiency_score(self, coverage_rate, miss_rate, excess_ratio, total_length):
        """
        计算效率评分（0-1之间，越高越好）
        
        评分考虑因素：
        1. 覆盖率（权重40%）
        2. 漏测惩罚（权重30%）
        3. 超额重叠惩罚（权重20%）
        4. 长度效率（权重10%）
        """
        # 覆盖率得分（0-1）
        coverage_score = coverage_rate
        
        # 漏测惩罚（漏测率越高惩罚越大）
        miss_penalty = miss_rate * 2  # 漏测惩罚加倍
        
        # 超额重叠惩罚（比例越高惩罚越大）
        excess_penalty = min(excess_ratio * 5, 0.5)  # 最大惩罚0.5
        
        # 长度效率（相对于基准长度的惩罚）
        baseline_length = 280  # 海里，作为基准
        length_penalty = max(0, (total_length - baseline_length) / baseline_length * 0.1)
        
        # 综合评分
        efficiency_score = max(0, 
            0.4 * coverage_score - 
            0.3 * miss_penalty - 
            0.2 * excess_penalty - 
            0.1 * length_penalty
        )
        
        return min(efficiency_score, 1.0)  # 限制在0-1之间
    
    def quick_metrics(self, lines_df):
        """
        快速计算主要指标（仅总长度和测线数量）
        用于快速比较
        """
        return {
            'total_length_nm': lines_df['length_optimized_nm'].sum(),
            'total_lines': len(lines_df)
        }

def calculate_metrics_for_lines(lines_df, verbose=True):
    """
    便捷函数：为给定的测线计算指标
    
    参数:
    - lines_df: 测线DataFrame
    - verbose: 是否输出详细信息
    
    返回:
    - 指标字典
    """
    adapter = MetricsCalculatorAdapter()
    return adapter.calculate_comprehensive_metrics(lines_df, verbose)

if __name__ == '__main__':
    # 测试适配器
    print("测试指标计算适配器...")
    
    # 尝试加载现有的测线数据进行测试
    try:
        test_lines = pd.read_csv("../_03_line_generation/survey_lines_q4_optimized.csv")
        print(f"加载测试数据: {len(test_lines)} 条测线")
        
        # 计算指标
        metrics = calculate_metrics_for_lines(test_lines)
        
        print("\n=== 测试结果 ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
                
    except FileNotFoundError:
        print("未找到测试数据文件，适配器初始化测试完成")
    except Exception as e:
        print(f"测试过程中出现错误: {e}") 