# 快速开始指南

本指南帮助您快速上手使用本项目复现论文结果。

## 安装

### 1. 克隆项目（如果尚未克隆）

```bash
git clone <repository-url>
cd sbl_paper_reproduction
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

或者使用开发模式安装：

```bash
pip install -e .
```

## 基础使用

### 示例1：基础频率估计

```python
import numpy as np
from sbl import SBLDictParam, generate_signal

# 参数设置
N = 64  # 信号长度
theta_true = np.array([0.1, 0.2, 0.3])  # 真实频率
alpha_true = np.array([1.0, 0.8, 1.2])  # 真实幅度
snr_db = 20  # 信噪比

# 生成信号
y, x = generate_signal(N, theta_true, alpha_true, snr_db, seed=42)

# 运行SBL算法
sbl = SBLDictParam(N=N, max_iter=50, verbose=True)
theta_est, alpha_est, K_est = sbl.fit(y)

print(f"估计的频率: {theta_est}")
print(f"估计的幅度: {np.abs(alpha_est)}")
```

### 示例2：运行预定义示例

```bash
# 基础使用示例
python examples/basic_usage.py

# 算法对比
python experiments/comparison.py

# 复现论文结果（需要较长时间）
python examples/reproduce_paper_results.py
```

## 算法对比

本项目实现了多种频率估计算法：

### 1. SBL-DictParam（本文提出）

```python
from sbl import SBLDictParam

sbl = SBLDictParam(N=64, max_iter=50)
theta, alpha, K = sbl.fit(y)
```

**优势**：
- 无需网格化，避免字典失配
- 自动估计模型阶数
- 对密集频率分量鲁棒

### 2. ESPRIT（子空间方法）

```python
from sbl.algorithms.esprit import ESPRIT

esprit = ESPRIT(K=3, method='ls')
theta, alpha, K = esprit.fit(y)
```

**特点**：
- 经典子空间方法
- 需要预先指定K
- 对噪声敏感

### 3. MUSIC（子空间方法）

```python
from sbl.algorithms.music import MUSIC

music = MUSIC(K=3, grid_size=512)
theta, alpha, K = music.fit(y)
```

**特点**：
- 伪谱估计
- 需要网格搜索
- 分辨率受限于网格

### 4. OMP（贪婪方法）

```python
from sbl.algorithms.omp import OMP

omp = OMP(K=3, grid_size=256)
theta, alpha, K = omp.fit(y)
```

**特点**：
- 快速贪婪算法
- 需要固定字典
- 存在基失配误差

## 性能评估

### 评估指标

```python
from sbl.utils.metrics import (
    frequency_rmse,
    amplitude_rmse,
    detection_rate,
    model_order_accuracy
)

# 频率RMSE
freq_error = frequency_rmse(theta_true, theta_est)

# 幅度RMSE
amp_error = amplitude_rmse(alpha_true, alpha_est, theta_true, theta_est)

# 检测率
det_rate = detection_rate(theta_true, theta_est)

# 模型阶数准确率
order_acc = model_order_accuracy(K_true, K_est)
```

### 可视化

```python
from sbl.utils.visualization import (
    plot_estimation_results,
    plot_comparison,
    plot_spectrum
)

# 绘制估计结果
plot_estimation_results(y, theta_true, alpha_true, theta_est, alpha_est)

# 算法对比
plot_comparison(y, theta_true, alpha_true, results_dict)

# 绘制频谱
freq_grid = np.linspace(0, 1, 512)
spectrum = sbl.get_spectrum(freq_grid)
plot_spectrum(freq_grid, {'SBL': spectrum}, theta_true, theta_est)
```

## 测试场景

### 场景1：良好分离的频率

```python
from sbl.models.signal_model import generate_test_scenario

scenario = generate_test_scenario(
    N=64,
    K=3,
    snr_db=20,
    freq_separation=3.0/64,  # 约3个频率分辨单元
    seed=42
)

y = scenario['y']
theta_true = scenario['theta']
```

### 场景2：密集频率（挑战性）

```python
from sbl.models.signal_model import generate_closely_spaced_scenario

scenario = generate_closely_spaced_scenario(
    N=64,
    K=3,
    snr_db=20,
    center_freq=0.25,
    freq_span=2.0/64,  # 低于瑞利限
    seed=42
)
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_sbl.py -v
pytest tests/test_signal_model.py -v
```

## 常见问题

### Q1: 算法收敛很慢怎么办？

**A**: 尝试调整以下参数：
- 减少 `max_iter`
- 增大 `tol`
- 调整 `theta_learning_rate`

```python
sbl = SBLDictParam(
    N=64,
    max_iter=30,  # 减少迭代次数
    tol=1e-4,     # 放宽收敛容差
    theta_learning_rate=0.05  # 增大学习率
)
```

### Q2: 估计的模型阶数不准确？

**A**: 可能原因：
- SNR太低
- 频率过于密集
- 需要调整 `prune_threshold`

```python
sbl = SBLDictParam(
    N=64,
    prune_threshold=1e-8,  # 调整剪枝阈值
    add_threshold=0.1      # 调整添加阈值
)
```

### Q3: 如何加速蒙特卡洛实验？

**A**:
- 减少试验次数 `num_trials`
- 使用并行计算（需要额外实现）
- 减少SNR测试点

```python
# 在 reproduce_paper_results.py 中
results = run_monte_carlo_experiment(
    N=64,
    K=3,
    snr_range_db=np.arange(10, 31, 10),  # 减少SNR点
    num_trials=50  # 减少试验次数
)
```

## 进阶使用

### 自定义原子函数

```python
def custom_atom(theta, N):
    """自定义原子函数"""
    # 实现您的原子函数
    return atom_vector

def custom_derivative(theta, N):
    """原子函数的导数"""
    # 实现导数
    return derivative_vector

sbl = SBLDictParam(
    N=64,
    atom_func=custom_atom,
    atom_derivative=custom_derivative
)
```

### 批量实验

```python
import pandas as pd

results_list = []

for snr in [10, 15, 20, 25, 30]:
    for trial in range(100):
        scenario = generate_test_scenario(N=64, K=3, snr_db=snr, seed=trial)
        theta_est, alpha_est, K_est = sbl.fit(scenario['y'])

        results_list.append({
            'snr': snr,
            'trial': trial,
            'freq_rmse': frequency_rmse(scenario['theta'], theta_est),
            'K_est': K_est
        })

df = pd.DataFrame(results_list)
df.to_csv('results/batch_experiment.csv', index=False)
```

## 引用

如果您使用本代码，请引用原论文：

```bibtex
@article{hansen2014sparse,
  title={A sparse Bayesian learning algorithm with dictionary parameter estimation},
  author={Hansen, Thomas L and Badiu, Mihai A and Fleury, Bernard H and Rao, Bhaskar D},
  journal={IEEE Signal Processing Letters},
  year={2014}
}
```

## 获取帮助

- 查看 [README.md](README.md) 了解详细文档
- 查看 `examples/` 目录获取更多示例
- 提交 Issue 报告问题或建议

## 下一步

1. 尝试运行 `examples/basic_usage.py`
2. 修改参数，观察算法行为
3. 在自己的数据上测试算法
4. 复现论文中的图表结果

祝您使用愉快！
