"""
训练器 - 预训练和微调pipeline
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
import os
import time
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from config import Config
from models import PretrainModel, FinetuneModel, load_pretrained_encoder
from data_utils import create_data_loaders, generate_synthetic_data

class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, config: Config, model: nn.Module, device: str = 'cuda'):
        self.config = config
        self.model = model
        self.device = device
        
        # 移动模型到设备
        self.model.to(device)
        
        # 初始化优化器和调度器
        self.optimizer = None
        self.scheduler = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # 日志
        self.writer = None
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志记录"""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        elif self.config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb not available, skipping wandb initialization")
        
        # TensorBoard
        log_dir = f"runs/{self.config.experiment_name}"
        self.writer = SummaryWriter(log_dir)
        
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.pretrain.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.pretrain.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            self.best_loss = loss
            
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        
        print(f"加载检查点: epoch {self.current_epoch}, loss {self.best_loss:.4f}")

class PretrainTrainer(BaseTrainer):
    """预训练训练器"""
    
    def __init__(self, config: Config, device: str = 'cuda'):
        model = PretrainModel(config)
        super().__init__(config, model, device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.pretrain.learning_rate,
            weight_decay=config.pretrain.weight_decay
        )
        
        # 设置学习率调度器
        if config.pretrain.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=config.pretrain.epochs,
                eta_min=config.pretrain.learning_rate * 0.01
            )
        elif config.pretrain.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.pretrain.epochs // 3,
                gamma=0.1
            )
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            masked_csi = batch['masked_csi'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(masked_csi, mask)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 记录到TensorBoard
            if batch_idx % self.config.pretrain.log_interval == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss', loss.item(), global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                masked_csi = batch['masked_csi'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                outputs = self.model(masked_csi, mask)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader=None):
        """训练模型"""
        print("开始预训练...")
        
        for epoch in range(self.current_epoch, self.config.pretrain.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
            else:
                val_loss = train_loss
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 记录日志
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 记录到wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if epoch % self.config.pretrain.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        print("预训练完成!")

class FinetuneTrainer(BaseTrainer):
    """微调训练器"""
    
    def __init__(self, config: Config, pretrained_encoder_path: str, device: str = 'cuda'):
        # 加载预训练编码器
        if pretrained_encoder_path is not None:
            pretrained_encoder = load_pretrained_encoder(pretrained_encoder_path, config)
        else:
            pretrained_encoder = None
        
        # 创建微调模型
        model = FinetuneModel(config, pretrained_encoder)
        super().__init__(config, model, device)
        
        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.finetune.learning_rate,
            weight_decay=config.finetune.weight_decay
        )
        
        # 设置学习率调度器
        if config.finetune.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.finetune.epochs,
                eta_min=config.finetune.learning_rate * 0.01
            )
        elif config.finetune.scheduler == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.finetune.epochs // 3,
                gamma=0.1
            )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_losses = {'total': 0.0, 'position': 0.0, 'trajectory': 0.0}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            csi = batch['csi'].to(self.device)
            positions = batch['position'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(csi, positions)
            
            # 计算损失
            total_loss = outputs['total_loss']
            position_loss = outputs['position_loss']
            trajectory_loss = outputs['trajectory_loss']
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_losses['total'] += total_loss.item()
            total_losses['position'] += position_loss.item()
            total_losses['trajectory'] += trajectory_loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'pos': f"{position_loss.item():.4f}",
                'traj': f"{trajectory_loss.item():.4f}"
            })
            
            # 记录到TensorBoard
            if batch_idx % self.config.finetune.log_interval == 0:
                global_step = self.current_epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('Train/TotalLoss', total_loss.item(), global_step)
                self.writer.add_scalar('Train/PositionLoss', position_loss.item(), global_step)
                self.writer.add_scalar('Train/TrajectoryLoss', trajectory_loss.item(), global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], global_step)
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def validate(self, val_loader) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_losses = {'total': 0.0, 'position': 0.0, 'trajectory': 0.0}
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                csi = batch['csi'].to(self.device)
                positions = batch['position'].to(self.device)
                
                outputs = self.model(csi, positions)
                
                # 记录损失
                total_losses['total'] += outputs['total_loss'].item()
                total_losses['position'] += outputs['position_loss'].item()
                total_losses['trajectory'] += outputs['trajectory_loss'].item()
                num_batches += 1
                
                # 收集预测和目标
                predictions = outputs['predicted_positions'].cpu().numpy()
                targets = positions.cpu().numpy()
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # 计算平均损失
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        # 计算位置预测指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 计算MSE和MAE
        mse = mean_squared_error(all_targets.reshape(-1, 2), all_predictions.reshape(-1, 2))
        mae = mean_absolute_error(all_targets.reshape(-1, 2), all_predictions.reshape(-1, 2))
        
        avg_losses['mse'] = mse
        avg_losses['mae'] = mae
        
        return avg_losses
    
    def train(self, train_loader, val_loader):
        """训练模型"""
        print("开始微调...")
        
        for epoch in range(self.current_epoch, self.config.finetune.epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses = self.validate(val_loader)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 记录日志
            print(f"Epoch {epoch}:")
            print(f"  Train - Total: {train_losses['total']:.4f}, Position: {train_losses['position']:.4f}, Trajectory: {train_losses['trajectory']:.4f}")
            print(f"  Val   - Total: {val_losses['total']:.4f}, Position: {val_losses['position']:.4f}, Trajectory: {val_losses['trajectory']:.4f}")
            print(f"  Val   - MSE: {val_losses['mse']:.4f}, MAE: {val_losses['mae']:.4f}")
            
            # 记录到wandb
            if self.config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch,
                    'train_total_loss': train_losses['total'],
                    'train_position_loss': train_losses['position'],
                    'train_trajectory_loss': train_losses['trajectory'],
                    'val_total_loss': val_losses['total'],
                    'val_position_loss': val_losses['position'],
                    'val_trajectory_loss': val_losses['trajectory'],
                    'val_mse': val_losses['mse'],
                    'val_mae': val_losses['mae'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # 保存检查点
            is_best = val_losses['total'] < self.best_loss
            if epoch % self.config.finetune.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_losses['total'], is_best)
        
        print("微调完成!")

def run_pretrain(config: Config):
    """运行预训练"""
    print("=" * 50)
    print("开始预训练阶段")
    print("=" * 50)
    
    # 创建数据加载器
    pretrain_loader, _, _ = create_data_loaders(config)
    
    # 创建训练器
    trainer = PretrainTrainer(config)
    
    # 训练
    trainer.train(pretrain_loader)
    
    print("预训练完成!")

def run_finetune(config: Config, pretrained_encoder_path: str):
    """运行微调"""
    print("=" * 50)
    print("开始微调阶段")
    print("=" * 50)
    
    # 创建数据加载器
    _, train_loader, val_loader = create_data_loaders(config)
    
    # 创建训练器
    trainer = FinetuneTrainer(config, pretrained_encoder_path)
    
    # 训练
    trainer.train(train_loader, val_loader)
    
    print("微调完成!")

def run_full_pipeline(config: Config):
    """运行完整pipeline"""
    print("=" * 50)
    print("开始完整训练pipeline")
    print("=" * 50)
    
    # 生成合成数据（如果没有真实数据）
    if not os.path.exists(config.data.train_data_path):
        print("生成合成数据...")
        os.makedirs(config.data.train_data_path, exist_ok=True)
        os.makedirs(config.data.val_data_path, exist_ok=True)
        
        # 生成训练数据
        train_data = generate_synthetic_data(config, num_samples=1000, 
                                           save_path=os.path.join(config.data.train_data_path, 'train.pkl'))
        
        # 生成验证数据
        val_data = generate_synthetic_data(config, num_samples=200, 
                                         save_path=os.path.join(config.data.val_data_path, 'val.pkl'))
    
    # 1. 预训练阶段
    run_pretrain(config)
    
    # 2. 微调阶段
    pretrained_encoder_path = os.path.join(config.pretrain.checkpoint_dir, 'best.pth')
    run_finetune(config, pretrained_encoder_path)
    
    print("=" * 50)
    print("完整pipeline完成!")
    print("=" * 50)