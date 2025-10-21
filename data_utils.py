"""
CSI数据预处理和加载工具
"""
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict
import os
import pickle
from scipy.io import loadmat
import random

class CSIDataset(Dataset):
    """CSI数据集类"""
    
    def __init__(self, 
                 data_path: str,
                 sequence_length: int = 100,
                 num_antennas: int = 64,
                 num_subcarriers: int = 64,
                 use_augmentation: bool = True,
                 noise_std: float = 0.1,
                 is_pretrain: bool = True):
        """
        Args:
            data_path: 数据路径
            sequence_length: 时间序列长度
            num_antennas: 天线数量
            num_subcarriers: 子载波数量
            use_augmentation: 是否使用数据增强
            noise_std: 噪声标准差
            is_pretrain: 是否为预训练模式
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.use_augmentation = use_augmentation
        self.noise_std = noise_std
        self.is_pretrain = is_pretrain
        
        # 加载数据
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """加载CSI数据"""
        data = []
        
        if os.path.isdir(self.data_path):
            # 从目录加载多个文件
            for filename in os.listdir(self.data_path):
                if filename.endswith('.mat') or filename.endswith('.pkl'):
                    file_path = os.path.join(self.data_path, filename)
                    file_data = self._load_single_file(file_path)
                    data.extend(file_data)
        else:
            # 加载单个文件
            data = self._load_single_file(self.data_path)
            
        return data
    
    def _load_single_file(self, file_path: str) -> List[Dict]:
        """加载单个数据文件"""
        data = []
        
        if file_path.endswith('.mat'):
            # 加载MATLAB文件
            mat_data = loadmat(file_path)
            csi_data = mat_data.get('csi_data', mat_data.get('CSI', None))
            positions = mat_data.get('positions', mat_data.get('pos', None))
            
            if csi_data is not None:
                # 处理CSI数据格式
                if len(csi_data.shape) == 3:  # [samples, antennas, subcarriers]
                    for i in range(csi_data.shape[0]):
                        sample = {
                            'csi': csi_data[i],
                            'position': positions[i] if positions is not None else None
                        }
                        data.append(sample)
                        
        elif file_path.endswith('.pkl'):
            # 加载pickle文件
            with open(file_path, 'rb') as f:
                file_data = pickle.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
        
        return data
    
    def _preprocess_csi(self, csi: np.ndarray) -> torch.Tensor:
        """预处理CSI数据"""
        # 转换为复数
        if csi.dtype != np.complex128:
            csi = csi.astype(np.complex128)
        
        # 提取幅度和相位
        amplitude = np.abs(csi)
        phase = np.angle(csi)
        
        # 归一化
        amplitude = (amplitude - np.mean(amplitude)) / (np.std(amplitude) + 1e-8)
        phase = (phase - np.mean(phase)) / (np.std(phase) + 1e-8)
        
        # 堆叠幅度和相位
        csi_processed = np.stack([amplitude, phase], axis=-1)
        
        return torch.FloatTensor(csi_processed)
    
    def _apply_augmentation(self, csi: torch.Tensor) -> torch.Tensor:
        """应用数据增强"""
        if not self.use_augmentation:
            return csi
        
        # 添加高斯噪声
        if self.noise_std > 0:
            noise = torch.randn_like(csi) * self.noise_std
            csi = csi + noise
        
        # 随机缩放
        scale_factor = 1.0 + random.uniform(-0.1, 0.1)
        csi = csi * scale_factor
        
        return csi
    
    def _create_mask(self, csi: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建掩码用于预训练"""
        batch_size, seq_len, num_ant, num_sub, channels = csi.shape
        
        # 创建掩码
        mask = torch.ones(batch_size, seq_len, num_ant, num_sub, dtype=torch.bool)
        num_mask = int(seq_len * num_ant * num_sub * mask_ratio)
        
        for i in range(batch_size):
            # 随机选择要掩码的位置
            flat_indices = torch.randperm(seq_len * num_ant * num_sub)[:num_mask]
            mask_indices = torch.unravel_index(flat_indices, (seq_len, num_ant, num_sub))
            mask[i, mask_indices[0], mask_indices[1], mask_indices[2]] = False
        
        # 创建掩码后的输入
        masked_csi = csi.clone()
        masked_csi[~mask.unsqueeze(-1).expand_as(csi)] = 0  # 将掩码位置置零
        
        return masked_csi, mask
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx]
        csi = sample['csi']
        position = sample.get('position', None)
        
        # 预处理CSI数据
        csi_processed = self._preprocess_csi(csi)
        
        # 确保序列长度
        if csi_processed.shape[0] < self.sequence_length:
            # 填充
            pad_length = self.sequence_length - csi_processed.shape[0]
            csi_processed = F.pad(csi_processed, (0, 0, 0, 0, 0, 0, 0, pad_length), mode='constant', value=0)
        elif csi_processed.shape[0] > self.sequence_length:
            # 截断
            csi_processed = csi_processed[:self.sequence_length]
        
        # 添加批次维度
        csi_processed = csi_processed.unsqueeze(0)  # [1, seq_len, num_ant, num_sub, channels]
        
        # 应用数据增强
        csi_processed = self._apply_augmentation(csi_processed)
        
        result = {'csi': csi_processed.squeeze(0)}  # 移除批次维度
        
        if self.is_pretrain:
            # 预训练模式：创建掩码
            masked_csi, mask = self._create_mask(csi_processed, mask_ratio=0.15)
            result.update({
                'masked_csi': masked_csi.squeeze(0),
                'mask': mask.squeeze(0)
            })
        else:
            # 微调模式：包含位置信息
            if position is not None:
                # 确保位置数据有正确的时间维度
                if len(position.shape) == 1:
                    # 如果位置是一维的，扩展为时间序列
                    position = position.reshape(1, -1).repeat(csi_processed.shape[0], 1)
                result['position'] = torch.FloatTensor(position)
            else:
                result['position'] = torch.zeros(csi_processed.shape[0], 2)  # 默认位置
        
        return result

def create_data_loaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 预训练数据加载器
    pretrain_dataset = CSIDataset(
        data_path=config.data.train_data_path,
        sequence_length=config.data.sequence_length,
        num_antennas=config.data.num_antennas,
        num_subcarriers=config.data.num_subcarriers,
        use_augmentation=config.data.use_augmentation,
        noise_std=config.data.noise_std,
        is_pretrain=True
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # 微调数据加载器
    finetune_train_dataset = CSIDataset(
        data_path=config.data.train_data_path,
        sequence_length=config.data.sequence_length,
        num_antennas=config.data.num_antennas,
        num_subcarriers=config.data.num_subcarriers,
        use_augmentation=config.data.use_augmentation,
        noise_std=config.data.noise_std,
        is_pretrain=False
    )
    
    finetune_train_loader = DataLoader(
        finetune_train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    # 验证数据加载器
    val_dataset = CSIDataset(
        data_path=config.data.val_data_path,
        sequence_length=config.data.sequence_length,
        num_antennas=config.data.num_antennas,
        num_subcarriers=config.data.num_subcarriers,
        use_augmentation=False,  # 验证时不使用数据增强
        noise_std=0.0,
        is_pretrain=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return pretrain_loader, finetune_train_loader, val_loader

def generate_synthetic_data(config, num_samples: int = 1000, save_path: str = None):
    """生成合成CSI数据用于测试"""
    print(f"生成 {num_samples} 个合成CSI样本...")
    
    data = []
    for i in range(num_samples):
        # 生成随机CSI数据
        csi_real = np.random.randn(config.data.sequence_length, 
                                  config.data.num_antennas, 
                                  config.data.num_subcarriers)
        csi_imag = np.random.randn(config.data.sequence_length, 
                                  config.data.num_antennas, 
                                  config.data.num_subcarriers)
        csi = csi_real + 1j * csi_imag
        
        # 生成随机位置
        position = np.random.rand(2) * 100  # 0-100米范围内
        
        sample = {
            'csi': csi,
            'position': position
        }
        data.append(sample)
    
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"合成数据已保存到: {save_path}")
    
    return data