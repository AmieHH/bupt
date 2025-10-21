"""
基于信道预训练的泛化定位模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class CSIEncoder(nn.Module):
    """CSI编码器 - 基于Transformer"""
    
    def __init__(self, 
                 input_dim: int = 2,  # 幅度和相位
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 sequence_length: int = 100,
                 num_antennas: int = 64,
                 num_subcarriers: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=sequence_length * num_antennas * num_subcarriers)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 掩码token嵌入
        self.mask_token = nn.Parameter(torch.randn(hidden_dim))
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                csi: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            csi: [batch_size, seq_len, num_ant, num_sub, input_dim]
            mask: [batch_size, seq_len, num_ant, num_sub] 掩码，True表示保留，False表示掩码
        
        Returns:
            reconstructed: 重建的CSI
            features: 编码后的特征
        """
        batch_size, seq_len, num_ant, num_sub, input_dim = csi.shape
        
        # 展平为序列
        csi_flat = csi.view(batch_size, seq_len * num_ant * num_sub, input_dim)
        
        # 应用掩码
        if mask is not None:
            mask_flat = mask.view(batch_size, seq_len * num_ant * num_sub)
            # 将掩码位置替换为mask token
            masked_csi = csi_flat.clone()
            mask_positions = ~mask_flat  # False表示需要掩码的位置
            
            # 创建掩码后的输入，将掩码位置置零
            masked_csi = csi_flat.clone()
            for i in range(batch_size):
                batch_mask_positions = mask_positions[i]
                if batch_mask_positions.any():
                    masked_csi[i, batch_mask_positions] = 0
        else:
            masked_csi = csi_flat
        
        # 输入投影
        x = self.input_projection(masked_csi)  # [batch_size, seq_len*num_ant*num_sub, hidden_dim]
        
        # 位置编码
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        # Dropout
        x = self.dropout(x)
        
        # Transformer编码
        features = self.transformer(x)  # [batch_size, seq_len*num_ant*num_sub, hidden_dim]
        
        # 输出投影
        reconstructed = self.output_projection(features)  # [batch_size, seq_len*num_ant*num_sub, input_dim]
        
        # 重塑回原始形状
        reconstructed = reconstructed.view(batch_size, seq_len, num_ant, num_sub, input_dim)
        features = features.view(batch_size, seq_len, num_ant, num_sub, self.hidden_dim)
        
        return reconstructed, features

class PretrainModel(nn.Module):
    """预训练模型 - 掩码重建自监督学习"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.encoder = CSIEncoder(
            input_dim=2,  # 幅度和相位
            hidden_dim=config.model.pretrain_hidden_dim,
            num_layers=config.model.pretrain_num_layers,
            num_heads=config.model.pretrain_num_heads,
            dropout=config.model.pretrain_dropout,
            sequence_length=config.data.sequence_length,
            num_antennas=config.data.num_antennas,
            num_subcarriers=config.data.num_subcarriers
        )
        
    def forward(self, masked_csi: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            masked_csi: [batch_size, seq_len, num_ant, num_sub, 2]
            mask: [batch_size, seq_len, num_ant, num_sub]
        
        Returns:
            Dict containing reconstructed CSI and loss
        """
        # 获取原始CSI（用于计算损失）
        original_csi = masked_csi.clone()
        original_csi[~mask.unsqueeze(-1).expand_as(masked_csi)] = 0
        
        # 编码和重建
        reconstructed, features = self.encoder(masked_csi, mask)
        
        # 计算重建损失（只在掩码位置）
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)
        masked_positions = ~mask_expanded
        
        # 只计算掩码位置的损失
        if masked_positions.any():
            loss = F.mse_loss(
                reconstructed[masked_positions], 
                original_csi[masked_positions]
            )
        else:
            loss = torch.tensor(0.0, device=reconstructed.device)
        
        return {
            'reconstructed': reconstructed,
            'features': features,
            'loss': loss
        }

class PositionRegressionHead(nn.Module):
    """位置回归头"""
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dims: list = [128, 64, 32],
                 output_dim: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, seq_len, num_ant, num_sub, input_dim]
        
        Returns:
            positions: [batch_size, seq_len, output_dim]
        """
        batch_size, seq_len, num_ant, num_sub, input_dim = features.shape
        
        # 全局平均池化
        global_features = features.mean(dim=(2, 3))  # [batch_size, seq_len, input_dim]
        
        # 位置回归
        positions = self.mlp(global_features)  # [batch_size, seq_len, output_dim]
        
        return positions

class TrajectoryConsistencyLoss(nn.Module):
    """轨迹一致性损失"""
    
    def __init__(self, 
                 smoothness_weight: float = 0.05,
                 temporal_window: int = 5):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.temporal_window = temporal_window
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [batch_size, seq_len, 2]
        
        Returns:
            trajectory_loss: 轨迹一致性损失
        """
        batch_size, seq_len, _ = positions.shape
        
        # 计算速度（一阶差分）
        velocity = positions[:, 1:] - positions[:, :-1]  # [batch_size, seq_len-1, 2]
        
        # 计算加速度（二阶差分）
        acceleration = velocity[:, 1:] - velocity[:, :-1]  # [batch_size, seq_len-2, 2]
        
        # 平滑性损失：加速度的L2范数
        smoothness_loss = torch.mean(torch.norm(acceleration, dim=-1))
        
        # 时间窗口内的位置一致性
        consistency_loss = 0.0
        for i in range(seq_len - self.temporal_window + 1):
            window_positions = positions[:, i:i+self.temporal_window]  # [batch_size, window, 2]
            
            # 计算窗口内位置的方差
            window_var = torch.var(window_positions, dim=1)  # [batch_size, 2]
            consistency_loss += torch.mean(window_var)
        
        consistency_loss = consistency_loss / (seq_len - self.temporal_window + 1)
        
        # 总轨迹损失
        trajectory_loss = self.smoothness_weight * smoothness_loss + consistency_loss
        
        return trajectory_loss

class FinetuneModel(nn.Module):
    """微调模型 - 位置预测"""
    
    def __init__(self, config, pretrained_encoder: Optional[CSIEncoder] = None):
        super().__init__()
        
        self.config = config
        
        # 使用预训练的编码器或创建新的
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = CSIEncoder(
                input_dim=2,
                hidden_dim=config.model.finetune_hidden_dim,
                num_layers=config.model.finetune_num_layers,
                num_heads=config.model.pretrain_num_heads,  # 保持头数一致
                dropout=config.model.finetune_dropout,
                sequence_length=config.data.sequence_length,
                num_antennas=config.data.num_antennas,
                num_subcarriers=config.data.num_subcarriers
            )
        
        # 位置回归头
        self.position_head = PositionRegressionHead(
            input_dim=config.model.finetune_hidden_dim,
            hidden_dims=config.model.position_mlp_layers,
            output_dim=config.model.position_dim,
            dropout=config.model.finetune_dropout
        )
        
        # 轨迹一致性损失
        self.trajectory_loss_fn = TrajectoryConsistencyLoss(
            smoothness_weight=config.finetune.smoothness_weight,
            temporal_window=config.finetune.temporal_window
        )
        
    def forward(self, csi: torch.Tensor, positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            csi: [batch_size, seq_len, num_ant, num_sub, 2]
            positions: [batch_size, seq_len, 2] 用于训练时的标签
        
        Returns:
            Dict containing predicted positions and losses
        """
        # 编码CSI特征
        _, features = self.encoder(csi)  # [batch_size, seq_len, num_ant, num_sub, hidden_dim]
        
        # 预测位置
        predicted_positions = self.position_head(features)  # [batch_size, seq_len, 2]
        
        result = {'predicted_positions': predicted_positions}
        
        if positions is not None:
            # 计算位置预测损失
            position_loss = F.mse_loss(predicted_positions, positions)
            result['position_loss'] = position_loss
            
            # 计算轨迹一致性损失
            trajectory_loss = self.trajectory_loss_fn(predicted_positions)
            result['trajectory_loss'] = trajectory_loss
            
            # 总损失
            total_loss = position_loss + self.config.finetune.trajectory_loss_weight * trajectory_loss
            result['total_loss'] = total_loss
        
        return result

def load_pretrained_encoder(checkpoint_path: str, config) -> CSIEncoder:
    """加载预训练的编码器"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建编码器
    encoder = CSIEncoder(
        input_dim=2,
        hidden_dim=config.model.pretrain_hidden_dim,
        num_layers=config.model.pretrain_num_layers,
        num_heads=config.model.pretrain_num_heads,
        dropout=config.model.pretrain_dropout,
        sequence_length=config.data.sequence_length,
        num_antennas=config.data.num_antennas,
        num_subcarriers=config.data.num_subcarriers
    )
    
    # 加载权重
    encoder.load_state_dict(checkpoint['model_state_dict'])
    
    return encoder