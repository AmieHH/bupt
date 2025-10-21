"""
配置文件 - 基于信道预训练的泛化定位技术
"""
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径
    data_root: str = "/workspace/data"
    train_data_path: str = "/workspace/data/train"
    val_data_path: str = "/workspace/data/val"
    test_data_path: str = "/workspace/data/test"
    
    # CSI数据参数
    num_antennas: int = 64  # 天线数量
    num_subcarriers: int = 64  # 子载波数量
    sequence_length: int = 100  # 时间序列长度
    
    # 数据增强
    use_augmentation: bool = True
    noise_std: float = 0.1
    mask_ratio: float = 0.15  # 掩码比例
    
    # 数据加载
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ModelConfig:
    """模型配置"""
    # 预训练模型参数
    pretrain_hidden_dim: int = 256
    pretrain_num_layers: int = 6
    pretrain_num_heads: int = 8
    pretrain_dropout: float = 0.1
    
    # 微调模型参数
    finetune_hidden_dim: int = 256
    finetune_num_layers: int = 4
    finetune_dropout: float = 0.1
    
    # 位置回归头
    position_dim: int = 2  # 2D位置 (x, y)
    position_mlp_layers: List[int] = None
    
    def __post_init__(self):
        if self.position_mlp_layers is None:
            self.position_mlp_layers = [128, 64, 32]

@dataclass
class PretrainConfig:
    """预训练配置"""
    # 训练参数
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    
    # 掩码重建参数
    mask_ratio: float = 0.15
    mask_token_id: int = -1
    
    # 优化器
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    
    # 保存和日志
    save_interval: int = 10
    log_interval: int = 100
    checkpoint_dir: str = "/workspace/checkpoints/pretrain"

@dataclass
class FinetuneConfig:
    """微调配置"""
    # 训练参数
    epochs: int = 50
    learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    
    # 轨迹一致性约束
    trajectory_loss_weight: float = 0.1
    smoothness_weight: float = 0.05
    temporal_window: int = 5
    
    # 优化器
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    
    # 保存和日志
    save_interval: int = 5
    log_interval: int = 50
    checkpoint_dir: str = "/workspace/checkpoints/finetune"

@dataclass
class Config:
    """总配置"""
    data: DataConfig = None
    model: ModelConfig = None
    pretrain: PretrainConfig = None
    finetune: FinetuneConfig = None
    
    # 设备配置
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    seed: int = 42
    
    # 实验配置
    experiment_name: str = "channel_pretrain_localization"
    use_wandb: bool = True
    wandb_project: str = "channel-localization"
    
    def __post_init__(self):
        # 初始化默认配置
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.pretrain is None:
            self.pretrain = PretrainConfig()
        if self.finetune is None:
            self.finetune = FinetuneConfig()
        
        # 创建必要的目录
        os.makedirs(self.pretrain.checkpoint_dir, exist_ok=True)
        os.makedirs(self.finetune.checkpoint_dir, exist_ok=True)
        os.makedirs(self.data.data_root, exist_ok=True)