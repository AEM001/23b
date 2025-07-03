#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试灵敏度分析系统的基本功能
"""

import sys
import os
import pandas as pd

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

def test_config_loading():
    """测试配置加载"""
    print("=== 测试配置加载 ===")
    try:
        from sensitivity_analysis_config import (
            BASELINE_CONFIG,
            get_all_test_configs,
            get_config_description
        )
        
        print(f"基准配置: {BASELINE_CONFIG}")
        
        configs = get_all_test_configs()
        print(f"总配置数量: {len(configs)}")
        
        # 显示前几个配置
        print("\n前5个配置:")
        for i, (name, config) in enumerate(configs[:5]):
            desc = get_config_description(name, config)
            print(f"{i+1}. {name}: {desc}")
        
        print("✓ 配置加载测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 配置加载测试失败: {e}")
        return False

def test_parametric_generator():
    """测试参数化生成器"""
    print("\n=== 测试参数化生成器 ===")
    try:
        from parametric_survey_line_generator import generate_survey_lines_with_parameters
        from sensitivity_analysis_config import BASELINE_CONFIG
        
        # 测试基准配置
        print("测试基准配置...")
        lines_df, param_summary = generate_survey_lines_with_parameters(
            BASELINE_CONFIG,
            output_filename=None  # 不保存文件
        )
        
        print(f"生成测线数量: {len(lines_df)}")
        print(f"测线总长度: {lines_df['length_optimized_nm'].sum():.2f} 海里")
        print(f"参数摘要: {param_summary}")
        
        # 测试修改配置
        print("\n测试修改配置...")
        modified_config = BASELINE_CONFIG.copy()
        modified_config['safety_margin_factor'] = 0.1  # 修改安全边距
        
        lines_df2, param_summary2 = generate_survey_lines_with_parameters(
            modified_config,
            output_filename=None
        )
        
        print(f"修改后测线数量: {len(lines_df2)}")
        print(f"修改后测线总长度: {lines_df2['length_optimized_nm'].sum():.2f} 海里")
        
        length_diff = lines_df2['length_optimized_nm'].sum() - lines_df['length_optimized_nm'].sum()
        print(f"长度差异: {length_diff:.2f} 海里")
        
        print("✓ 参数化生成器测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 参数化生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_sensitivity():
    """测试简单的灵敏度分析"""
    print("\n=== 测试简单灵敏度分析 ===")
    try:
        from parametric_survey_line_generator import generate_survey_lines_with_parameters
        from sensitivity_analysis_config import BASELINE_CONFIG
        
        # 测试不同的安全边距系数
        safety_margins = [0.1, 0.2, 0.3]
        results = []
        
        for margin in safety_margins:
            config = BASELINE_CONFIG.copy()
            config['safety_margin_factor'] = margin
            
            lines_df, _ = generate_survey_lines_with_parameters(config)
            total_length = lines_df['length_optimized_nm'].sum()
            
            results.append({
                'safety_margin': margin,
                'total_length': total_length
            })
            
            print(f"安全边距 {margin:.1f}: {total_length:.2f} 海里")
        
        # 计算敏感度
        length_range = max(r['total_length'] for r in results) - min(r['total_length'] for r in results)
        margin_range = max(r['safety_margin'] for r in results) - min(r['safety_margin'] for r in results)
        
        print(f"\n敏感度分析结果:")
        print(f"长度变化范围: {length_range:.2f} 海里")
        print(f"参数变化范围: {margin_range:.1f}")
        print(f"相对敏感度: {length_range/min(r['total_length'] for r in results):.4f}")
        
        print("✓ 简单灵敏度分析测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 简单灵敏度分析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("测线延伸参数灵敏度分析系统 - 功能测试")
    print("=" * 60)
    
    tests = [
        test_config_loading,
        test_parametric_generator,
        test_simple_sensitivity
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"测试总结: {passed}/{len(tests)} 项测试通过")
    
    if passed == len(tests):
        print("✓ 所有测试通过，系统运行正常!")
        print("\n可以运行完整的灵敏度分析:")
        print("python sensitivity_analysis_main.py")
    else:
        print("✗ 部分测试失败，请检查系统配置")
    
    print("=" * 60)

if __name__ == '__main__':
    main() 