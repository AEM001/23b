#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灵敏度分析参数配置
定义要测试的不同参数组合
"""

# 基准参数配置（原始优化方案）
BASELINE_CONFIG = {
    'safety_margin_factor': 0.2,      # 安全边距系数 20%
    'angle_correction_factor': 0.5,   # 角度修正系数 0.5
    'max_angle_factor': 3.0,          # 最大角度系数限制 3.0
    'min_direction_component': 0.1,   # 最小方向分量 0.1
    'extension_base_factor': 1.0       # 基础延伸系数 1.0（完整覆盖宽度）
}

# 安全边距系数灵敏度分析
SAFETY_MARGIN_VARIATIONS = [
    {'safety_margin_factor': 0.0},   # 无安全边距
    {'safety_margin_factor': 0.1},   # 10%安全边距
    {'safety_margin_factor': 0.15},  # 15%安全边距
    {'safety_margin_factor': 0.2},   # 20%安全边距（基准）
    {'safety_margin_factor': 0.25},  # 25%安全边距
    {'safety_margin_factor': 0.3},   # 30%安全边距
    {'safety_margin_factor': 0.4},   # 40%安全边距
]

# 角度修正系数灵敏度分析
ANGLE_CORRECTION_VARIATIONS = [
    {'angle_correction_factor': 0.0},   # 无角度修正
    {'angle_correction_factor': 0.2},   # 弱角度修正
    {'angle_correction_factor': 0.3},   # 较弱角度修正
    {'angle_correction_factor': 0.5},   # 中等角度修正（基准）
    {'angle_correction_factor': 0.7},   # 较强角度修正
    {'angle_correction_factor': 1.0},   # 强角度修正
    {'angle_correction_factor': 1.5},   # 很强角度修正
]

# 最大角度系数限制灵敏度分析
MAX_ANGLE_FACTOR_VARIATIONS = [
    {'max_angle_factor': 1.5},  # 较小限制
    {'max_angle_factor': 2.0},  # 中小限制
    {'max_angle_factor': 2.5},  # 中等限制
    {'max_angle_factor': 3.0},  # 基准限制
    {'max_angle_factor': 4.0},  # 较大限制
    {'max_angle_factor': 5.0},  # 大限制
    {'max_angle_factor': 10.0}, # 很大限制（几乎无限制）
]

# 基础延伸系数灵敏度分析
EXTENSION_BASE_VARIATIONS = [
    {'extension_base_factor': 0.5},   # 半覆盖宽度延伸
    {'extension_base_factor': 0.7},   # 0.7倍覆盖宽度延伸
    {'extension_base_factor': 0.8},   # 0.8倍覆盖宽度延伸
    {'extension_base_factor': 1.0},   # 完整覆盖宽度延伸（基准）
    {'extension_base_factor': 1.2},   # 1.2倍覆盖宽度延伸
    {'extension_base_factor': 1.5},   # 1.5倍覆盖宽度延伸
    {'extension_base_factor': 2.0},   # 双倍覆盖宽度延伸
]

# 最小方向分量灵敏度分析
MIN_DIRECTION_VARIATIONS = [
    {'min_direction_component': 0.05},  # 更小阈值，更大角度修正
    {'min_direction_component': 0.08},  # 较小阈值
    {'min_direction_component': 0.1},   # 基准阈值
    {'min_direction_component': 0.12},  # 较大阈值
    {'min_direction_component': 0.15},  # 更大阈值，更小角度修正
    {'min_direction_component': 0.2},   # 大阈值
]

# 组合测试：极端保守配置
CONSERVATIVE_CONFIG = {
    'safety_margin_factor': 0.4,      # 高安全边距
    'angle_correction_factor': 1.0,   # 强角度修正
    'max_angle_factor': 5.0,          # 大角度系数限制
    'min_direction_component': 0.05,  # 小方向分量阈值
    'extension_base_factor': 1.5       # 大基础延伸
}

# 组合测试：极端激进配置
AGGRESSIVE_CONFIG = {
    'safety_margin_factor': 0.0,      # 无安全边距
    'angle_correction_factor': 0.0,   # 无角度修正
    'max_angle_factor': 1.0,          # 小角度系数限制
    'min_direction_component': 0.2,   # 大方向分量阈值
    'extension_base_factor': 0.5       # 小基础延伸
}

# 组合测试：平衡配置
BALANCED_CONFIG = {
    'safety_margin_factor': 0.15,     # 中等安全边距
    'angle_correction_factor': 0.3,   # 中等角度修正
    'max_angle_factor': 2.5,          # 中等角度系数限制
    'min_direction_component': 0.12,  # 中等方向分量阈值
    'extension_base_factor': 0.8       # 中等基础延伸
}

def get_all_test_configs():
    """
    获取所有测试配置的列表
    每个配置都是基于基准配置的变体
    """
    all_configs = []
    
    # 添加基准配置
    all_configs.append(('baseline', BASELINE_CONFIG.copy()))
    
    # 添加单参数变化的配置
    for i, variation in enumerate(SAFETY_MARGIN_VARIATIONS):
        config = BASELINE_CONFIG.copy()
        config.update(variation)
        all_configs.append((f'safety_margin_{i}', config))
    
    for i, variation in enumerate(ANGLE_CORRECTION_VARIATIONS):
        config = BASELINE_CONFIG.copy()
        config.update(variation)
        all_configs.append((f'angle_correction_{i}', config))
    
    for i, variation in enumerate(MAX_ANGLE_FACTOR_VARIATIONS):
        config = BASELINE_CONFIG.copy()
        config.update(variation)
        all_configs.append((f'max_angle_{i}', config))
    
    for i, variation in enumerate(EXTENSION_BASE_VARIATIONS):
        config = BASELINE_CONFIG.copy()
        config.update(variation)
        all_configs.append((f'extension_base_{i}', config))
    
    for i, variation in enumerate(MIN_DIRECTION_VARIATIONS):
        config = BASELINE_CONFIG.copy()
        config.update(variation)
        all_configs.append((f'min_direction_{i}', config))
    
    # 添加组合配置
    all_configs.append(('conservative', CONSERVATIVE_CONFIG.copy()))
    all_configs.append(('aggressive', AGGRESSIVE_CONFIG.copy()))
    all_configs.append(('balanced', BALANCED_CONFIG.copy()))
    
    return all_configs

def get_parameter_analysis_groups():
    """
    获取按参数分组的分析配置
    返回字典，键为参数名，值为变化列表
    """
    return {
        'safety_margin_factor': [(f'safety_{i}', BASELINE_CONFIG.copy()) 
                                for i in range(len(SAFETY_MARGIN_VARIATIONS))],
        'angle_correction_factor': [(f'angle_{i}', BASELINE_CONFIG.copy()) 
                                   for i in range(len(ANGLE_CORRECTION_VARIATIONS))],
        'max_angle_factor': [(f'max_angle_{i}', BASELINE_CONFIG.copy()) 
                            for i in range(len(MAX_ANGLE_FACTOR_VARIATIONS))],
        'extension_base_factor': [(f'extension_{i}', BASELINE_CONFIG.copy()) 
                                 for i in range(len(EXTENSION_BASE_VARIATIONS))],
        'min_direction_component': [(f'min_dir_{i}', BASELINE_CONFIG.copy()) 
                                   for i in range(len(MIN_DIRECTION_VARIATIONS))]
    }

def get_config_description(config_name, config):
    """
    获取配置的描述性文字
    """
    descriptions = {
        'baseline': '基准配置（原始优化方案）',
        'conservative': '保守配置（高延伸、强修正）',
        'aggressive': '激进配置（低延伸、弱修正）',
        'balanced': '平衡配置（中等延伸和修正）'
    }
    
    if config_name in descriptions:
        return descriptions[config_name]
    
    # 对于参数变化配置，生成描述
    if 'safety_margin' in config_name:
        return f"安全边距系数: {config['safety_margin_factor']:.2f}"
    elif 'angle_correction' in config_name:
        return f"角度修正系数: {config['angle_correction_factor']:.2f}"
    elif 'max_angle' in config_name:
        return f"最大角度系数: {config['max_angle_factor']:.1f}"
    elif 'extension_base' in config_name:
        return f"基础延伸系数: {config['extension_base_factor']:.1f}"
    elif 'min_direction' in config_name:
        return f"最小方向分量: {config['min_direction_component']:.2f}"
    else:
        return config_name

if __name__ == '__main__':
    # 测试配置
    configs = get_all_test_configs()
    print(f"总共生成 {len(configs)} 个测试配置")
    
    print("\n前5个配置示例:")
    for i, (name, config) in enumerate(configs[:5]):
        print(f"{i+1}. {name}: {get_config_description(name, config)}")
        print(f"   参数: {config}")
        print()
    
    print(f"...（省略中间配置）")
    print(f"总计: {len(configs)} 个配置待测试") 