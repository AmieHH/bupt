# 基于信道预训练的泛化定位技术

本项目实现了一个基于信道状态信息(CSI)预训练的泛化定位技术pipeline，包含预训练阶段（掩码重建自监督学习）和微调阶段（米级弱标签和轨迹一致性约束）。

## 项目结构

```
.
├── config.py              # 配置文件
├── data_utils.py          # 数据预处理和加载工具
├── models.py              # 模型定义（预训练和微调模型）
├── trainer.py             # 训练器（预训练和微调）
├── evaluation.py          # 评估和可视化工具
├── main.py               # 主程序入口
├── requirements.txt       # 依赖包
└── README.md             # 说明文档
```

## 核心特性

### 预训练阶段
- **掩码重建自监督学习**: 通过随机掩码CSI信号的部分子载波，让模型学习重建原始信号
- **Transformer编码器**: 使用基于注意力机制的编码器学习CSI信号的本质特征和内在结构
- **位置编码**: 为时间序列数据添加位置信息，增强模型对时序特征的理解

### 微调阶段
- **位置回归头**: 基于预训练编码器特征进行2D位置预测
- **轨迹一致性约束**: 通过平滑性损失和时间窗口一致性损失优化轨迹预测
- **米级精度**: 利用弱标签数据实现高精度终端位置预测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 完整训练pipeline

```bash
python main.py --mode full --output_dir outputs
```

这将运行完整的预训练+微调pipeline，包括：
- 生成合成数据（如果没有真实数据）
- 预训练阶段：掩码重建自监督学习
- 微调阶段：位置预测和轨迹一致性约束

### 2. 分阶段训练

#### 预训练阶段
```bash
python main.py --mode pretrain --output_dir outputs
```

#### 微调阶段
```bash
python main.py --mode finetune --pretrained_path checkpoints/pretrain/best.pth --output_dir outputs
```

### 3. 模型评估

```bash
python main.py --mode evaluate --finetuned_path checkpoints/finetune/best.pth --output_dir outputs
```

### 4. 使用自定义数据

```bash
python main.py --mode full --data_path /path/to/your/data --output_dir outputs
```

数据格式要求：
- 训练数据：`data_path/train/` 目录下的 `.mat` 或 `.pkl` 文件
- 验证数据：`data_path/val/` 目录下的 `.mat` 或 `.pkl` 文件
- 测试数据：`data_path/test/` 目录下的 `.mat` 或 `.pkl` 文件

每个数据文件应包含：
- `csi_data`: CSI数据，形状为 `[samples, antennas, subcarriers]`
- `positions`: 位置标签，形状为 `[samples, 2]` (x, y坐标)

## 配置参数

主要配置参数在 `config.py` 中：

### 数据配置
- `num_antennas`: 天线数量 (默认: 64)
- `num_subcarriers`: 子载波数量 (默认: 64)
- `sequence_length`: 时间序列长度 (默认: 100)
- `batch_size`: 批次大小 (默认: 32)

### 模型配置
- `pretrain_hidden_dim`: 预训练模型隐藏维度 (默认: 256)
- `pretrain_num_layers`: 预训练模型层数 (默认: 6)
- `pretrain_num_heads`: 注意力头数 (默认: 8)
- `finetune_hidden_dim`: 微调模型隐藏维度 (默认: 256)

### 训练配置
- `pretrain_epochs`: 预训练轮数 (默认: 100)
- `finetune_epochs`: 微调轮数 (默认: 50)
- `learning_rate`: 学习率 (默认: 1e-4)
- `trajectory_loss_weight`: 轨迹一致性损失权重 (默认: 0.1)

## 模型架构

### 预训练模型
```
CSI输入 [B, T, A, S, 2] 
    ↓
输入投影 [B, T×A×S, hidden_dim]
    ↓
位置编码 + 掩码token
    ↓
Transformer编码器 (6层, 8头)
    ↓
输出投影 [B, T×A×S, 2]
    ↓
重建损失 (MSE)
```

### 微调模型
```
CSI输入 [B, T, A, S, 2]
    ↓
预训练编码器 (冻结或微调)
    ↓
全局平均池化 [B, T, hidden_dim]
    ↓
位置回归头 (MLP)
    ↓
位置预测 [B, T, 2] + 轨迹一致性损失
```

## 评估指标

### 位置预测精度
- **MSE**: 均方误差
- **MAE**: 平均绝对误差
- **RMSE**: 均方根误差
- **R²**: 决定系数

### 轨迹一致性
- **轨迹误差**: 预测轨迹与真实轨迹的总长度差异
- **速度误差**: 速度预测的均方误差
- **加速度误差**: 加速度预测的均方误差

## 可视化功能

评估完成后会生成以下可视化图表：
- 轨迹对比图：显示预测轨迹与真实轨迹的对比
- 误差分布图：位置误差的统计分布
- 特征可视化：使用t-SNE降维显示特征空间
- 3D轨迹图：交互式3D轨迹对比
- 训练曲线：损失函数随训练轮数的变化

## 实验设置

### 硬件要求
- GPU: 推荐使用CUDA兼容的GPU
- 内存: 至少8GB RAM
- 存储: 至少10GB可用空间

### 软件环境
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (如果使用GPU)

## 自定义配置

可以通过JSON文件自定义配置：

```json
{
  "data": {
    "num_antennas": 32,
    "num_subcarriers": 32,
    "sequence_length": 50,
    "batch_size": 64
  },
  "model": {
    "pretrain_hidden_dim": 512,
    "pretrain_num_layers": 8,
    "pretrain_num_heads": 16
  },
  "pretrain": {
    "epochs": 200,
    "learning_rate": 5e-5
  }
}
```

然后使用：
```bash
python main.py --mode full --config custom_config.json
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 减小 `sequence_length`
   - 使用 `--device cpu` 强制使用CPU

2. **数据加载错误**
   - 检查数据路径是否正确
   - 确认数据格式是否符合要求
   - 查看错误日志中的具体信息

3. **模型收敛慢**
   - 调整学习率
   - 增加预训练轮数
   - 检查数据质量

### 日志和调试

- 训练日志保存在 `runs/` 目录下，可使用TensorBoard查看
- 如果启用wandb，可在wandb界面查看训练进度
- 检查点保存在 `checkpoints/` 目录下

## 引用

如果您使用了本项目，请引用：

```bibtex
@misc{channel_pretrain_localization,
  title={基于信道预训练的泛化定位技术},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/channel-pretrain-localization}
}
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进本项目。

## 联系方式

如有问题，请通过以下方式联系：
- Email: your.email@example.com
- GitHub Issues: https://github.com/your-repo/issues