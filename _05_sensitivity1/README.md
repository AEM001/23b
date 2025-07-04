# 测线延伸参数灵敏度分析系统

## 概述

本系统对测线边界智能扩展优化算法中的关键参数进行系统性的灵敏度分析，旨在：

1. **量化参数影响**：分析各参数对测线总长度的影响程度
2. **优化参数配置**：找到最优的参数组合
3. **指导实际应用**：为不同海域条件提供参数调优建议

## 系统架构

### 核心模块

1. **`parametric_survey_line_generator.py`** - 参数化测线生成器
   - 将原始硬编码参数变为可配置
   - 提供统一的测线生成接口

2. **`sensitivity_analysis_config.py`** - 参数配置定义
   - 定义基准配置和各种参数变化
   - 提供配置管理功能

3. **`metrics_calculator_adapter.py`** - 指标计算适配器
   - 调用第四问的指标计算方法
   - 提供统一的性能评估接口

4. **`sensitivity_analysis_main.py`** - 主分析程序
   - 运行全面的参数测试
   - 生成分析报告和可视化

## 分析的关键参数

### 1. 安全边距系数 (safety_margin_factor)
- **含义**：在基础延伸距离之上的额外安全缓冲
- **默认值**：0.2 (20%)
- **测试范围**：0.0 ~ 0.4
- **影响**：值越大，测线延伸越长，覆盖越安全但效率越低

### 2. 角度修正系数 (angle_correction_factor)
- **含义**：控制测线与边界夹角修正的强度
- **默认值**：0.5
- **测试范围**：0.0 ~ 1.5
- **影响**：值越大，小夹角时延伸距离增加越多

### 3. 最大角度系数限制 (max_angle_factor)
- **含义**：角度修正系数的上限
- **默认值**：3.0
- **测试范围**：1.5 ~ 10.0
- **影响**：防止极端情况下的过度延伸

### 4. 基础延伸系数 (extension_base_factor)
- **含义**：基础延伸距离相对于覆盖宽度的比例
- **默认值**：1.0 (完整覆盖宽度)
- **测试范围**：0.5 ~ 2.0
- **影响**：核心参数，直接决定延伸的基础长度

### 5. 最小方向分量 (min_direction_component)
- **含义**：防止除零的最小方向分量阈值
- **默认值**：0.1
- **测试范围**：0.05 ~ 0.2
- **影响**：间接影响角度修正的计算

## 使用方法

### 1. 快速开始

```bash
cd _05_sensitivity_analysis
python sensitivity_analysis_main.py
```

### 2. 自定义分析

```python
from parametric_survey_line_generator import generate_survey_lines_with_parameters

# 定义自定义参数
custom_config = {
    'safety_margin_factor': 0.15,
    'angle_correction_factor': 0.3,
    'max_angle_factor': 2.5,
    'min_direction_component': 0.12,
    'extension_base_factor': 0.8
}

# 生成测线
lines_df, param_summary = generate_survey_lines_with_parameters(
    custom_config, 
    output_filename="custom_lines.csv"
)

print(f"总长度: {lines_df['length_optimized_nm'].sum():.2f} 海里")
```

### 3. 单参数测试

```python
from sensitivity_analysis_config import BASELINE_CONFIG

# 测试不同的安全边距系数
safety_margins = [0.0, 0.1, 0.2, 0.3, 0.4]
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
```

## 输出结果

### 1. 数据文件
- `sensitivity_analysis_results.csv` - 所有配置的详细结果
- `lines_<config_name>.csv` - 各配置的测线数据（可选）

### 2. 可视化图表
- `parameter_sensitivity_comparison.png` - 参数敏感度对比
- `parameter_curves_detailed.png` - 参数变化曲线
- `key_configs_comparison.png` - 关键配置对比

### 3. 分析报告
- `sensitivity_analysis_report.md` - 综合分析报告

## 分析指标

### 1. 基础指标
- **测线总长度**：所有测线长度之和
- **测线数量**：生成的测线条数
- **生成成功率**：成功生成测线的配置比例

### 2. 敏感度指标
- **敏感度评分**：参数变化对结果影响的相对大小
- **最大变化百分比**：参数变化引起的最大结果变化
- **参数影响范围**：参数变化导致的结果变化范围

### 3. 效率指标（扩展功能）
- **覆盖率**：海域覆盖的完整性
- **漏测率**：未覆盖区域的比例
- **超额重叠长度**：重叠率超过20%的测线长度

## 实际应用建议

### 1. 不同海域类型的参数建议

**平坦海域**：
- 可以使用较小的安全边距系数（0.1-0.15）
- 角度修正可以适中（0.3-0.5）
- 基础延伸系数可以保守（0.8-1.0）

**复杂地形海域**：
- 建议使用较大的安全边距系数（0.2-0.3）
- 强化角度修正（0.5-1.0）
- 增大基础延伸系数（1.0-1.5）

**高精度要求**：
- 优先保证覆盖完整性，使用保守参数
- 安全边距系数 ≥ 0.2
- 基础延伸系数 ≥ 1.0

**效率优先**：
- 在保证基本覆盖的前提下减少冗余
- 安全边距系数 ≤ 0.15
- 基础延伸系数 ≤ 0.8

### 2. 参数调优策略

1. **先调核心参数**：基础延伸系数和安全边距系数对结果影响最大
2. **再调修正参数**：角度修正系数和最大角度系数用于精细调节
3. **最后调技术参数**：最小方向分量等技术参数一般保持默认值

### 3. 验证方法

1. **测线长度验证**：比较不同参数配置的测线总长度
2. **覆盖率验证**：使用第四问的指标计算方法验证覆盖效果
3. **实际测试验证**：在小范围海域进行实际测试验证

## 注意事项

1. **计算时间**：全参数分析可能需要较长时间，建议先进行快速测试
2. **内存使用**：大量参数组合可能占用较多内存
3. **结果解释**：需要结合实际应用场景解释分析结果
4. **参数边界**：极端参数值可能导致计算失败或无意义结果

## 扩展功能

### 1. 添加新参数
在 `sensitivity_analysis_config.py` 中定义新的参数变化列表，并在生成器中实现相应逻辑。

### 2. 自定义指标
在 `metrics_calculator_adapter.py` 中添加新的性能指标计算方法。

### 3. 可视化扩展
在主分析程序中添加新的图表类型和分析维度。

## 技术架构

```
灵敏度分析系统
├── 参数配置层 (Config)
├── 测线生成层 (Generator)  
├── 指标计算层 (Metrics)
├── 分析执行层 (Analyzer)
└── 结果输出层 (Output)
```

每一层都提供清晰的接口，便于维护和扩展。 