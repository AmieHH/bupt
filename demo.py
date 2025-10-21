"""
演示脚本 - 快速体验基于信道预训练的泛化定位技术
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from config import Config
from data_utils import generate_synthetic_data, create_data_loaders
from models import PretrainModel, FinetuneModel
from trainer import PretrainTrainer, FinetuneTrainer
from evaluation import Evaluator, Visualizer

def quick_demo():
    """快速演示"""
    print("🚀 基于信道预训练的泛化定位技术 - 快速演示")
    print("=" * 60)
    
    # 创建配置
    config = Config()
    config.pretrain.epochs = 2  # 快速演示
    config.finetune.epochs = 2
    config.data.batch_size = 4
    config.data.sequence_length = 20
    config.data.num_antennas = 16
    config.data.num_subcarriers = 16
    
    print("📊 配置信息:")
    print(f"  设备: {config.device}")
    print(f"  天线数量: {config.data.num_antennas}")
    print(f"  子载波数量: {config.data.num_subcarriers}")
    print(f"  序列长度: {config.data.sequence_length}")
    print(f"  批次大小: {config.data.batch_size}")
    print()
    
    # 生成合成数据
    print("📁 生成合成数据...")
    os.makedirs("demo_data/train", exist_ok=True)
    os.makedirs("demo_data/val", exist_ok=True)
    
    train_data = generate_synthetic_data(
        config, num_samples=50, 
        save_path="demo_data/train/train.pkl"
    )
    val_data = generate_synthetic_data(
        config, num_samples=20, 
        save_path="demo_data/val/val.pkl"
    )
    
    print(f"  训练样本: {len(train_data)}")
    print(f"  验证样本: {len(val_data)}")
    print()
    
    # 更新数据路径
    config.data.train_data_path = "demo_data/train"
    config.data.val_data_path = "demo_data/val"
    
    # 创建数据加载器
    pretrain_loader, finetune_train_loader, val_loader = create_data_loaders(config)
    
    # 1. 预训练演示
    print("🔧 预训练阶段演示...")
    pretrain_model = PretrainModel(config)
    print(f"  预训练模型参数: {sum(p.numel() for p in pretrain_model.parameters()):,}")
    
    # 演示一个batch的预训练
    batch = next(iter(pretrain_loader))
    masked_csi = batch['masked_csi']
    mask = batch['mask']
    
    with torch.no_grad():
        outputs = pretrain_model(masked_csi, mask)
        print(f"  输入形状: {masked_csi.shape}")
        print(f"  掩码形状: {mask.shape}")
        print(f"  重建形状: {outputs['reconstructed'].shape}")
        print(f"  重建损失: {outputs['loss'].item():.4f}")
    print()
    
    # 2. 微调演示
    print("🎯 微调阶段演示...")
    finetune_model = FinetuneModel(config)
    print(f"  微调模型参数: {sum(p.numel() for p in finetune_model.parameters()):,}")
    
    # 演示一个batch的微调
    batch = next(iter(finetune_train_loader))
    csi = batch['csi']
    positions = batch['position']
    
    with torch.no_grad():
        outputs = finetune_model(csi, positions)
        print(f"  CSI形状: {csi.shape}")
        print(f"  位置形状: {positions.shape}")
        print(f"  预测位置形状: {outputs['predicted_positions'].shape}")
        print(f"  位置损失: {outputs['position_loss'].item():.4f}")
        print(f"  轨迹损失: {outputs['trajectory_loss'].item():.4f}")
    print()
    
    # 3. 评估演示
    print("📈 评估演示...")
    evaluator = Evaluator(finetune_model, config.device)
    
    # 简化评估：只测试一个batch
    batch = next(iter(val_loader))
    csi = batch['csi'].to(config.device)
    positions = batch['position'].to(config.device)
    
    with torch.no_grad():
        outputs = finetune_model(csi, positions)
        predictions = outputs['predicted_positions'].cpu().numpy()
        targets = positions.cpu().numpy()
        
        # 计算基本指标
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        print("  位置预测指标:")
        print(f"    MSE: {mse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print()
    
    # 4. 可视化演示
    print("🎨 可视化演示...")
    visualizer = Visualizer("demo_visualizations")
    
    # 获取预测结果
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            csi = batch['csi'].to(config.device)
            positions = batch['position'].to(config.device)
            
            outputs = finetune_model(csi, positions)
            predictions = outputs['predicted_positions'].cpu().numpy()
            targets = positions.cpu().numpy()
            
            all_predictions.append(predictions)
            all_targets.append(targets)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 绘制轨迹对比图
    print("  生成轨迹对比图...")
    visualizer.plot_trajectory_comparison(
        all_predictions, all_targets,
        save_path="demo_visualizations/trajectory_comparison.png"
    )
    
    # 绘制误差分布图
    print("  生成误差分布图...")
    visualizer.plot_error_distribution(
        all_predictions, all_targets,
        save_path="demo_visualizations/error_distribution.png"
    )
    
    print("  可视化图表已保存到: demo_visualizations/")
    print()
    
    # 5. 总结
    print("✅ 演示完成!")
    print("=" * 60)
    print("📋 演示总结:")
    print("  1. ✅ 预训练模型: 掩码重建自监督学习")
    print("  2. ✅ 微调模型: 位置预测 + 轨迹一致性约束")
    print("  3. ✅ 评估指标: 位置精度 + 轨迹一致性")
    print("  4. ✅ 可视化: 轨迹对比 + 误差分布")
    print()
    print("🔗 下一步:")
    print("  - 运行完整训练: python main.py --mode full")
    print("  - 查看详细文档: README.md")
    print("  - 运行使用示例: python example_usage.py")
    print("=" * 60)

if __name__ == '__main__':
    quick_demo()