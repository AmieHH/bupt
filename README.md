# Sparse Bayesian Learning with Dictionary Parameter Estimation

论文复现项目：**A Sparse Bayesian Learning Algorithm With Dictionary Parameter Estimation**

作者：Thomas L. Hansen, Mihai A. Badiu, Bernard H. Fleury, Bhaskar D. Rao

## 📋 项目简介

本项目完整复现了论文中提出的稀疏贝叶斯学习（SBL）算法，该算法能够在连续参数空间中进行稀疏信号分解，无需将参数空间离散化到固定网格。

### 核心特性

- ✅ **无网格化**：直接在连续参数空间工作，避免字典失配问题
- ✅ **自动模型选择**：同时估计模型阶数 K、原子参数 θ 和权重系数 α
- ✅ **频谱估计**：特别适用于密集频率分量的频谱估计问题
- ✅ **贝叶斯框架**：基于稀疏先验的概率建模

## 🔬 论文摘要

论文解决的问题：将含噪信号稀疏分解为由未知连续参数指定的原子。例如，估计复正弦波叠加的模型阶数、频率和幅度。

**关键创新**：
- 避免参数空间离散化，直接使用含参数化原子的信号模型
- 基于Tipping和Faul的"快速推断方案"开发新的SBL算法
- 能够同时估计原子参数、模型阶数和权重系数

**性能**：在密集频率分量的频谱估计实验中，本算法优于最先进的子空间方法（如ESPRIT）和压缩感知方法。

## 🚀 快速开始

### 安装依赖

```bash
cd sbl_paper_reproduction
pip install -r requirements.txt
```

### 基础使用示例

```python
from sbl.algorithms.sbl_dict_param import SBLDictParam
from sbl.models.signal_model import generate_signal
import numpy as np

# 生成测试信号
N = 64  # 信号长度
K = 3   # 真实频率数量
true_freqs = np.array([0.1, 0.15, 0.3])
true_amps = np.array([1.0, 0.8, 1.2])
snr_db = 20

y, x_true = generate_signal(N, true_freqs, true_amps, snr_db)

# 运行SBL算法
sbl = SBLDictParam(N=N, max_iter=100, tol=1e-6)
theta_est, alpha_est, K_est = sbl.fit(y)

print(f"估计的频率数量: {K_est}")
print(f"估计的频率: {theta_est}")
print(f"估计的幅度: {np.abs(alpha_est)}")
```

### 复现论文结果

```bash
python examples/reproduce_paper_results.py
```

## 📁 项目结构

```
sbl_paper_reproduction/
├── README.md                          # 项目说明
├── requirements.txt                   # 依赖包
├── setup.py                          # 安装脚本
├── sbl/                              # 核心算法包
│   ├── __init__.py
│   ├── algorithms/                   # 算法实现
│   │   ├── __init__.py
│   │   ├── sbl_dict_param.py        # 本文提出的SBL算法
│   │   ├── esprit.py                # ESPRIT对比算法
│   │   ├── music.py                 # MUSIC对比算法
│   │   └── omp.py                   # OMP对比算法
│   ├── models/                      # 信号模型
│   │   ├── __init__.py
│   │   ├── signal_model.py          # 信号生成和模型
│   │   └── atoms.py                 # 原子函数（傅里叶等）
│   └── utils/                       # 工具函数
│       ├── __init__.py
│       ├── metrics.py               # 评估指标（RMSE等）
│       └── visualization.py         # 可视化工具
├── experiments/                      # 实验脚本
│   ├── __init__.py
│   ├── spectral_estimation.py       # 频谱估计实验
│   └── comparison.py                # 算法对比实验
├── tests/                           # 单元测试
│   ├── __init__.py
│   ├── test_sbl.py
│   └── test_signal_model.py
├── examples/                        # 使用示例
│   ├── basic_usage.py               # 基础使用
│   └── reproduce_paper_results.py   # 复现论文图表
├── data/                            # 数据目录
└── results/                         # 结果输出目录
```

## 🧮 算法原理

### 信号模型

观测信号模型：
```
y = Σ ψ(θᵢ)αᵢ + w,  i = 1,...,K
```

其中：
- `y ∈ ℂᴺ`：观测信号
- `ψ(θᵢ) ∈ ℂᴺ`：原子函数（如傅里叶原子）
- `θᵢ ∈ [0,1)`：连续参数（如归一化频率）
- `αᵢ ∈ ℂ`：复数权重系数
- `w ∈ ℂᴺ`：高斯白噪声
- `K`：模型阶数（信号分量数量）

### 傅里叶原子

对于频谱估计问题，原子函数为：
```
ψ(θᵢ) = [e^(j2πθᵢ·0), e^(j2πθᵢ·1), ..., e^(j2πθᵢ·(N-1))]ᵀ
```

### SBL框架

1. **概率建模**：
   - 观测likelihood: p(y|θ,α,λ) = 𝒩(y|Ψ(θ)α, λ⁻¹I)
   - 稀疏先验: p(α|γ,λ) = ∏ᵢ 𝒩(αᵢ|0, λ⁻¹γᵢ)
   - 超参数先验: p(γᵢ) = Ga(γᵢ|c,d)

2. **推断**：
   - 迭代优化超参数 γ 和字典参数 θ
   - 原子的添加、删除和更新
   - 自动确定模型阶数 K

3. **优势**：
   - 无需预先指定模型阶数
   - 避免网格化导致的失配误差
   - 对密集频率分量鲁棒

## 📊 实验结果

### 实验1：密集频率估计

- 信号长度：N = 64
- 频率间隔：Δf = 1/N（瑞利限附近）
- SNR范围：0-30 dB
- 蒙特卡洛次数：500次

**性能对比**（RMSE）：
- SBL-DictParam（本文）：最优性能
- ESPRIT：在低SNR下性能下降
- MUSIC：需要准确的模型阶数
- OMP（网格化）：存在字典失配误差

### 实验2：不同频率间隔

测试算法在不同频率间隔下的性能：
- 超瑞利分辨率（Δf < 1/N）
- 瑞利限附近（Δf ≈ 1/N）
- 宽间隔（Δf > 1/N）

## 🔧 主要功能模块

### 1. SBL算法 (`sbl/algorithms/sbl_dict_param.py`)

核心功能：
- `fit(y)`: 估计参数θ、α和K
- `add_atom()`: 添加新原子
- `delete_atom()`: 删除不显著原子
- `update_atom()`: 更新原子参数
- `compute_marginal_likelihood()`: 计算边际似然

### 2. 对比算法

- **ESPRIT** (`esprit.py`): 子空间方法，估计正弦信号频率
- **MUSIC** (`music.py`): 多信号分类，频谱估计
- **OMP** (`omp.py`): 正交匹配追踪，需要固定字典

### 3. 信号模型 (`sbl/models/signal_model.py`)

- `generate_signal()`: 生成多分量正弦信号
- `add_noise()`: 添加高斯白噪声
- `compute_snr()`: 计算信噪比

### 4. 评估指标 (`sbl/utils/metrics.py`)

- `frequency_rmse()`: 频率估计均方根误差
- `amplitude_rmse()`: 幅度估计均方根误差
- `model_order_accuracy()`: 模型阶数估计准确率
- `resolution_probability()`: 分辨概率

## 📚 理论背景

### 为什么避免网格化？

传统方法将连续参数空间离散化到固定网格，存在以下问题：

1. **字典失配**：真实参数可能不在网格点上
2. **字典相干性**：细网格导致列高度相关，估计不稳定
3. **计算复杂度**：细网格增加字典尺寸，计算代价高

### SBL的优势

1. **自动稀疏性**：通过ARD先验自动实现稀疏
2. **参数连续优化**：直接优化连续参数，无离散化误差
3. **鲁棒性**：对模型失配和噪声鲁棒

## 🎯 应用场景

1. **频谱估计**：密集频率分量的频谱分析
2. **到达方向估计（DOA）**：阵列信号处理
3. **压缩感知**：连续参数的稀疏重构
4. **雷达信号处理**：目标检测和参数估计
5. **通信系统**：信道估计和同步

## 🧪 测试

运行所有测试：
```bash
pytest tests/ -v
```

运行特定测试：
```bash
pytest tests/test_sbl.py -v
```

## 📈 性能提示

1. **初始化**：好的初始化可以加快收敛
2. **停止准则**：根据应用调整容差参数
3. **最大迭代次数**：复杂场景可能需要更多迭代
4. **数值稳定性**：注意矩阵求逆的条件数

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📖 引用

如果使用本代码，请引用原论文：

```bibtex
@article{hansen2014sparse,
  title={A sparse Bayesian learning algorithm with dictionary parameter estimation},
  author={Hansen, Thomas L and Badiu, Mihai A and Fleury, Bernard H and Rao, Bhaskar D},
  journal={IEEE Signal Processing Letters},
  year={2014}
}
```

## 📧 联系方式

如有问题，请提交GitHub Issue。

## 🔗 相关资源

- 原论文链接：[待添加]
- SBL综述：Wipf & Rao (2007)
- 子空间方法：ESPRIT, MUSIC
- 压缩感知：Candes, Donoho

---

**最后更新**: 2025-11-05
