"""
使用示例 - 演示如何使用基于信道预训练的泛化定位技术
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from data_utils import generate_synthetic_data, create_data_loaders
from models import PretrainModel, FinetuneModel
from trainer import PretrainTrainer, FinetuneTrainer
from evaluation import comprehensive_evaluation

def example_pretrain():
    """预训练示例"""
    print("=" * 50)
    print("预训练示例")
    print("=" * 50)
    
    # 创建配置
    config = Config()
    config.pretrain.epochs = 5  # 示例中减少轮数
    config.data.batch_size = 8
    
    # 生成合成数据
    print("生成合成数据...")
    train_data = generate_synthetic_data(config, num_samples=100, 
                                       save_path="example_data/train.pkl")
    
    # 创建数据加载器
    pretrain_loader, _, _ = create_data_loaders(config)
    
    # 创建预训练模型
    model = PretrainModel(config)
    print(f"预训练模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = PretrainTrainer(config)
    
    # 训练一个batch来演示
    print("演示预训练过程...")
    for i, batch in enumerate(pretrain_loader):
        if i >= 3:  # 只演示前3个batch
            break
            
        masked_csi = batch['masked_csi']
        mask = batch['mask']
        
        # 前向传播
        outputs = model(masked_csi, mask)
        
        print(f"Batch {i+1}:")
        print(f"  输入形状: {masked_csi.shape}")
        print(f"  掩码形状: {mask.shape}")
        print(f"  重建形状: {outputs['reconstructed'].shape}")
        print(f"  特征形状: {outputs['features'].shape}")
        print(f"  损失: {outputs['loss'].item():.4f}")
        print()

def example_finetune():
    """微调示例"""
    print("=" * 50)
    print("微调示例")
    print("=" * 50)
    
    # 创建配置
    config = Config()
    config.finetune.epochs = 3  # 示例中减少轮数
    config.data.batch_size = 8
    
    # 生成合成数据
    print("生成合成数据...")
    train_data = generate_synthetic_data(config, num_samples=100, 
                                       save_path="example_data/train.pkl")
    val_data = generate_synthetic_data(config, num_samples=50, 
                                     save_path="example_data/val.pkl")
    
    # 创建数据加载器
    _, train_loader, val_loader = create_data_loaders(config)
    
    # 创建微调模型
    model = FinetuneModel(config)
    print(f"微调模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    trainer = FinetuneTrainer(config, "dummy_pretrained.pth")  # 使用虚拟路径
    
    # 训练一个batch来演示
    print("演示微调过程...")
    for i, batch in enumerate(train_loader):
        if i >= 3:  # 只演示前3个batch
            break
            
        csi = batch['csi']
        positions = batch['position']
        
        # 前向传播
        outputs = model(csi, positions)
        
        print(f"Batch {i+1}:")
        print(f"  CSI形状: {csi.shape}")
        print(f"  位置形状: {positions.shape}")
        print(f"  预测位置形状: {outputs['predicted_positions'].shape}")
        print(f"  位置损失: {outputs['position_loss'].item():.4f}")
        print(f"  轨迹损失: {outputs['trajectory_loss'].item():.4f}")
        print(f"  总损失: {outputs['total_loss'].item():.4f}")
        print()

def example_evaluation():
    """评估示例"""
    print("=" * 50)
    print("评估示例")
    print("=" * 50)
    
    # 创建配置
    config = Config()
    config.data.batch_size = 8
    
    # 生成合成数据
    print("生成合成数据...")
    test_data = generate_synthetic_data(config, num_samples=50, 
                                      save_path="example_data/test.pkl")
    
    # 创建数据加载器
    _, _, test_loader = create_data_loaders(config)
    
    # 创建微调模型
    model = FinetuneModel(config)
    
    # 演示评估过程
    print("演示评估过程...")
    
    # 获取一个batch的数据
    batch = next(iter(test_loader))
    csi = batch['csi']
    positions = batch['position']
    
    # 前向传播
    with torch.no_grad():
        outputs = model(csi, positions)
        predictions = outputs['predicted_positions'].numpy()
        targets = positions.numpy()
    
    # 计算基本指标
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    print(f"位置预测指标:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # 绘制轨迹对比图
    plt.figure(figsize=(12, 8))
    
    for i in range(min(4, len(predictions))):
        plt.subplot(2, 2, i+1)
        
        pred_traj = predictions[i]
        target_traj = targets[i]
        
        plt.plot(target_traj[:, 0], target_traj[:, 1], 'b-', linewidth=2, label='Ground Truth')
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=2, label='Prediction')
        
        plt.scatter(target_traj[0, 0], target_traj[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(target_traj[-1, 0], target_traj[-1, 1], c='red', s=100, marker='s', label='End')
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Trajectory {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("轨迹对比图已保存为: example_trajectory_comparison.png")

def example_custom_config():
    """自定义配置示例"""
    print("=" * 50)
    print("自定义配置示例")
    print("=" * 50)
    
    # 创建自定义配置
    config = Config()
    
    # 修改数据配置
    config.data.num_antennas = 32
    config.data.num_subcarriers = 32
    config.data.sequence_length = 50
    config.data.batch_size = 16
    
    # 修改模型配置
    config.model.pretrain_hidden_dim = 128
    config.model.pretrain_num_layers = 4
    config.model.pretrain_num_heads = 4
    
    # 修改训练配置
    config.pretrain.epochs = 10
    config.pretrain.learning_rate = 5e-5
    config.finetune.epochs = 5
    config.finetune.learning_rate = 1e-5
    
    print("自定义配置:")
    print(f"  天线数量: {config.data.num_antennas}")
    print(f"  子载波数量: {config.data.num_subcarriers}")
    print(f"  序列长度: {config.data.sequence_length}")
    print(f"  批次大小: {config.data.batch_size}")
    print(f"  预训练隐藏维度: {config.model.pretrain_hidden_dim}")
    print(f"  预训练层数: {config.model.pretrain_num_layers}")
    print(f"  预训练轮数: {config.pretrain.epochs}")
    print(f"  微调轮数: {config.finetune.epochs}")

def main():
    """主函数 - 运行所有示例"""
    print("基于信道预训练的泛化定位技术 - 使用示例")
    print("=" * 60)
    
    # 创建示例数据目录
    import os
    os.makedirs("example_data", exist_ok=True)
    
    try:
        # 1. 预训练示例
        example_pretrain()
        
        # 2. 微调示例
        example_finetune()
        
        # 3. 评估示例
        example_evaluation()
        
        # 4. 自定义配置示例
        example_custom_config()
        
        print("=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()