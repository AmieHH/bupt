"""
主程序 - 基于信道预训练的泛化定位技术
"""
import torch
import numpy as np
import argparse
import os
import random
from config import Config
from trainer import run_pretrain, run_finetune, run_full_pipeline
from evaluation import comprehensive_evaluation
from models import FinetuneModel, load_pretrained_encoder
from data_utils import create_data_loaders

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='基于信道预训练的泛化定位技术')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['pretrain', 'finetune', 'full', 'evaluate'],
                       help='运行模式: pretrain(预训练), finetune(微调), full(完整pipeline), evaluate(评估)')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='预训练模型路径（微调模式需要）')
    parser.add_argument('--finetuned_path', type=str, default=None,
                       help='微调模型路径（评估模式需要）')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备: auto, cpu, cuda')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    if args.config:
        # 从文件加载配置
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
    else:
        # 使用默认配置
        config = Config()
    
    # 更新配置
    if args.data_path:
        config.data.train_data_path = os.path.join(args.data_path, 'train')
        config.data.val_data_path = os.path.join(args.data_path, 'val')
        config.data.test_data_path = os.path.join(args.data_path, 'test')
    
    if args.device != 'auto':
        config.device = args.device
    else:
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config.seed = args.seed
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("基于信道预训练的泛化定位技术")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"设备: {config.device}")
    print(f"随机种子: {config.seed}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    if args.mode == 'pretrain':
        # 预训练模式
        print("开始预训练...")
        run_pretrain(config)
        
    elif args.mode == 'finetune':
        # 微调模式
        if not args.pretrained_path:
            print("错误: 微调模式需要指定预训练模型路径 --pretrained_path")
            return
        
        print("开始微调...")
        run_finetune(config, args.pretrained_path)
        
    elif args.mode == 'full':
        # 完整pipeline
        print("开始完整训练pipeline...")
        run_full_pipeline(config)
        
    elif args.mode == 'evaluate':
        # 评估模式
        if not args.finetuned_path:
            print("错误: 评估模式需要指定微调模型路径 --finetuned_path")
            return
        
        print("开始评估...")
        
        # 加载微调模型
        checkpoint = torch.load(args.finetuned_path, map_location=config.device)
        pretrained_encoder = load_pretrained_encoder(
            os.path.join(config.pretrain.checkpoint_dir, 'best.pth'), 
            config
        )
        model = FinetuneModel(config, pretrained_encoder)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.device)
        
        # 创建测试数据加载器
        _, _, test_loader = create_data_loaders(config)
        
        # 综合评估
        results = comprehensive_evaluation(
            model, test_loader, config, 
            save_dir=os.path.join(args.output_dir, 'evaluation')
        )
        
        print("评估完成!")
        print(f"位置预测RMSE: {results['position_metrics']['rmse']:.4f}m")
        print(f"位置预测MAE: {results['position_metrics']['mae']:.4f}m")
        print(f"轨迹一致性误差: {results['trajectory_metrics']['trajectory_error']:.4f}m")
    
    print("=" * 60)
    print("程序执行完成!")
    print("=" * 60)

if __name__ == '__main__':
    main()