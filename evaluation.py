"""
评估和可视化工具
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.manifold import TSNE
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_position_accuracy(self, data_loader) -> Dict[str, float]:
        """评估位置预测精度"""
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                csi = batch['csi'].to(self.device)
                positions = batch['position'].to(self.device)
                
                outputs = self.model(csi, positions)
                predictions = outputs['predicted_positions'].cpu().numpy()
                targets = positions.cpu().numpy()
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # 合并所有预测和目标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 确保形状一致
        if all_predictions.shape != all_targets.shape:
            # 如果形状不一致，调整到相同形状
            min_samples = min(all_predictions.shape[0], all_targets.shape[0])
            all_predictions = all_predictions[:min_samples]
            all_targets = all_targets[:min_samples]
        
        # 计算指标
        mse = mean_squared_error(all_targets.reshape(-1, 2), all_predictions.reshape(-1, 2))
        mae = mean_absolute_error(all_targets.reshape(-1, 2), all_predictions.reshape(-1, 2))
        rmse = np.sqrt(mse)
        r2 = r2_score(all_targets.reshape(-1, 2), all_predictions.reshape(-1, 2))
        
        # 计算每个维度的指标
        mse_x = mean_squared_error(all_targets[:, :, 0].flatten(), all_predictions[:, :, 0].flatten())
        mse_y = mean_squared_error(all_targets[:, :, 1].flatten(), all_predictions[:, :, 1].flatten())
        mae_x = mean_absolute_error(all_targets[:, :, 0].flatten(), all_predictions[:, :, 0].flatten())
        mae_y = mean_absolute_error(all_targets[:, :, 1].flatten(), all_predictions[:, :, 1].flatten())
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mse_x': mse_x,
            'mse_y': mse_y,
            'mae_x': mae_x,
            'mae_y': mae_y
        }
    
    def evaluate_trajectory_consistency(self, data_loader) -> Dict[str, float]:
        """评估轨迹一致性"""
        trajectory_errors = []
        velocity_errors = []
        acceleration_errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                csi = batch['csi'].to(self.device)
                positions = batch['position'].to(self.device)
                
                outputs = self.model(csi, positions)
                predictions = outputs['predicted_positions'].cpu().numpy()
                targets = positions.cpu().numpy()
                
                # 计算轨迹误差
                for i in range(len(predictions)):
                    pred_traj = predictions[i]  # [seq_len, 2]
                    target_traj = targets[i]    # [seq_len, 2]
                    
                    # 轨迹总长度误差
                    pred_length = np.sum(np.linalg.norm(np.diff(pred_traj, axis=0), axis=1))
                    target_length = np.sum(np.linalg.norm(np.diff(target_traj, axis=0), axis=1))
                    trajectory_errors.append(abs(pred_length - target_length))
                    
                    # 速度误差
                    pred_velocity = np.diff(pred_traj, axis=0)
                    target_velocity = np.diff(target_traj, axis=0)
                    velocity_error = np.mean(np.linalg.norm(pred_velocity - target_velocity, axis=1))
                    velocity_errors.append(velocity_error)
                    
                    # 加速度误差
                    pred_acceleration = np.diff(pred_velocity, axis=0)
                    target_acceleration = np.diff(target_velocity, axis=0)
                    if len(pred_acceleration) > 0 and len(target_acceleration) > 0:
                        acc_error = np.mean(np.linalg.norm(pred_acceleration - target_acceleration, axis=1))
                        acceleration_errors.append(acc_error)
        
        return {
            'trajectory_error': np.mean(trajectory_errors),
            'velocity_error': np.mean(velocity_errors),
            'acceleration_error': np.mean(acceleration_errors) if acceleration_errors else 0.0
        }
    
    def extract_features(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """提取CSI特征用于可视化"""
        all_features = []
        all_positions = []
        
        with torch.no_grad():
            for batch in data_loader:
                csi = batch['csi'].to(self.device)
                positions = batch['position'].to(self.device)
                
                # 获取编码器特征
                _, features = self.model.encoder(csi)
                
                # 全局平均池化
                global_features = features.mean(dim=(2, 3)).cpu().numpy()  # [batch_size, seq_len, hidden_dim]
                
                all_features.append(global_features)
                all_positions.append(positions.cpu().numpy())
        
        all_features = np.concatenate(all_features, axis=0)
        all_positions = np.concatenate(all_positions, axis=0)
        
        return all_features, all_positions

class Visualizer:
    """可视化工具"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_trajectory_comparison(self, 
                                 predictions: np.ndarray, 
                                 targets: np.ndarray, 
                                 save_path: str = None):
        """绘制轨迹对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 选择几个样本进行可视化
        num_samples = min(4, len(predictions))
        
        for i in range(num_samples):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            pred_traj = predictions[i]
            target_traj = targets[i]
            
            # 绘制轨迹
            ax.plot(target_traj[:, 0], target_traj[:, 1], 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Prediction', alpha=0.7)
            
            # 标记起点和终点
            ax.scatter(target_traj[0, 0], target_traj[0, 1], c='green', s=100, marker='o', label='Start')
            ax.scatter(target_traj[-1, 0], target_traj[-1, 1], c='red', s=100, marker='s', label='End')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title(f'Trajectory {i+1}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_error_distribution(self, 
                              predictions: np.ndarray, 
                              targets: np.ndarray, 
                              save_path: str = None):
        """绘制误差分布图"""
        # 计算位置误差
        position_errors = np.linalg.norm(predictions - targets, axis=-1)  # [batch_size, seq_len]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 误差分布直方图
        axes[0, 0].hist(position_errors.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('Position Error (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Position Error Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # X坐标误差
        x_errors = np.abs(predictions[:, :, 0] - targets[:, :, 0])
        axes[0, 1].hist(x_errors.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_xlabel('X Error (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('X Coordinate Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Y坐标误差
        y_errors = np.abs(predictions[:, :, 1] - targets[:, :, 1])
        axes[1, 0].hist(y_errors.flatten(), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Y Error (m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Y Coordinate Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差随时间变化
        mean_errors = np.mean(position_errors, axis=0)
        std_errors = np.std(position_errors, axis=0)
        time_steps = np.arange(len(mean_errors))
        
        axes[1, 1].plot(time_steps, mean_errors, 'b-', linewidth=2, label='Mean Error')
        axes[1, 1].fill_between(time_steps, 
                               mean_errors - std_errors, 
                               mean_errors + std_errors, 
                               alpha=0.3, color='blue', label='±1 Std')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Position Error (m)')
        axes[1, 1].set_title('Error vs Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_visualization(self, 
                                 features: np.ndarray, 
                                 positions: np.ndarray, 
                                 save_path: str = None):
        """绘制特征可视化"""
        # 使用t-SNE降维
        print("进行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # 展平特征
        features_flat = features.reshape(-1, features.shape[-1])
        positions_flat = positions.reshape(-1, 2)
        
        # 随机采样以加速t-SNE
        n_samples = min(5000, len(features_flat))
        indices = np.random.choice(len(features_flat), n_samples, replace=False)
        features_sample = features_flat[indices]
        positions_sample = positions_flat[indices]
        
        features_2d = tsne.fit_transform(features_sample)
        
        # 创建散点图
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按位置着色
        scatter1 = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                 c=positions_sample[:, 0], cmap='viridis', alpha=0.6)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].set_title('Feature Space (colored by X position)')
        plt.colorbar(scatter1, ax=axes[0], label='X Position (m)')
        
        scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                 c=positions_sample[:, 1], cmap='plasma', alpha=0.6)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('Feature Space (colored by Y position)')
        plt.colorbar(scatter2, ax=axes[1], label='Y Position (m)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_trajectory(self, 
                          predictions: np.ndarray, 
                          targets: np.ndarray, 
                          save_path: str = None):
        """绘制3D轨迹图"""
        fig = go.Figure()
        
        # 选择几个样本
        num_samples = min(3, len(predictions))
        colors = ['red', 'blue', 'green']
        
        for i in range(num_samples):
            pred_traj = predictions[i]
            target_traj = targets[i]
            
            # 添加时间维度
            time_steps = np.arange(len(pred_traj))
            
            # 预测轨迹
            fig.add_trace(go.Scatter3d(
                x=pred_traj[:, 0],
                y=pred_traj[:, 1],
                z=time_steps,
                mode='lines+markers',
                name=f'Prediction {i+1}',
                line=dict(color=colors[i], width=4),
                marker=dict(size=3)
            ))
            
            # 真实轨迹
            fig.add_trace(go.Scatter3d(
                x=target_traj[:, 0],
                y=target_traj[:, 1],
                z=time_steps,
                mode='lines+markers',
                name=f'Ground Truth {i+1}',
                line=dict(color=colors[i], width=2, dash='dash'),
                marker=dict(size=3)
            ))
        
        fig.update_layout(
            title='3D Trajectory Comparison',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Time Step'
            ),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
        fig.show()
    
    def plot_training_curves(self, 
                           train_losses: List[float], 
                           val_losses: List[float] = None,
                           save_path: str = None):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失对数曲线
        plt.subplot(1, 2, 2)
        plt.semilogy(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            plt.semilogy(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training and Validation Loss (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def comprehensive_evaluation(model, data_loader, config, save_dir: str = "evaluation_results"):
    """综合评估"""
    print("开始综合评估...")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建评估器和可视化器
    evaluator = Evaluator(model)
    visualizer = Visualizer(save_dir)
    
    # 1. 位置精度评估
    print("评估位置预测精度...")
    position_metrics = evaluator.evaluate_position_accuracy(data_loader)
    print("位置预测指标:")
    for metric, value in position_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 2. 轨迹一致性评估
    print("评估轨迹一致性...")
    trajectory_metrics = evaluator.evaluate_trajectory_consistency(data_loader)
    print("轨迹一致性指标:")
    for metric, value in trajectory_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 3. 提取特征和位置用于可视化
    print("提取特征用于可视化...")
    features, positions = evaluator.extract_features(data_loader)
    
    # 4. 获取预测结果用于可视化
    print("获取预测结果...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            csi = batch['csi'].to(evaluator.device)
            pos = batch['position'].to(evaluator.device)
            
            outputs = model(csi, pos)
            predictions = outputs['predicted_positions'].cpu().numpy()
            targets = pos.cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 5. 生成可视化
    print("生成可视化图表...")
    
    # 轨迹对比
    visualizer.plot_trajectory_comparison(
        all_predictions, all_targets,
        save_path=os.path.join(save_dir, 'trajectory_comparison.png')
    )
    
    # 误差分布
    visualizer.plot_error_distribution(
        all_predictions, all_targets,
        save_path=os.path.join(save_dir, 'error_distribution.png')
    )
    
    # 特征可视化
    visualizer.plot_feature_visualization(
        features, positions,
        save_path=os.path.join(save_dir, 'feature_visualization.png')
    )
    
    # 3D轨迹
    visualizer.plot_3d_trajectory(
        all_predictions, all_targets,
        save_path=os.path.join(save_dir, '3d_trajectory.html')
    )
    
    # 6. 保存评估结果
    results = {
        'position_metrics': position_metrics,
        'trajectory_metrics': trajectory_metrics,
        'config': config.__dict__
    }
    
    import json
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"评估完成! 结果保存在: {save_dir}")
    
    return results