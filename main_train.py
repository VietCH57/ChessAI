#!/usr/bin/env python
import torch
import argparse
import os
from trainer import AlphaZeroTrainer

def main():
    parser = argparse.ArgumentParser(description='Huấn luyện AlphaZero Chess')
    parser.add_argument('--iterations', type=int, default=100, help='Số iterations huấn luyện')
    parser.add_argument('--episodes', type=int, default=500, help='Số episodes mỗi iteration')
    parser.add_argument('--mcts_sims', type=int, default=800, help='Số MCTS simulations')
    parser.add_argument('--epochs', type=int, default=100, help='Số epochs huấn luyện NN')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint để tiếp tục huấn luyện')
    parser.add_argument('--model_dir', type=str, default='./alphazero_models/', help='Thư mục lưu models')
    parser.add_argument('--data_dir', type=str, default='./alphazero_data/', help='Thư mục lưu data')
    
    args = parser.parse_args()
    
    # Cấu hình
    config = {
        'num_iterations': args.iterations,
        'num_episodes': args.episodes,
        'num_mcts_sims': args.mcts_sims,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'model_dir': args.model_dir,
        'data_dir': args.data_dir,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=== ALPHAZERO CHESS TRAINING ===")
    print(f"Device: {config['device']}")
    print(f"Iterations: {config['num_iterations']}")
    print(f"Episodes per iteration: {config['num_episodes']}")
    print(f"MCTS simulations: {config['num_mcts_sims']}")
    print("=" * 35)
    
    # Tạo trainer
    trainer = AlphaZeroTrainer(config)
    
    # Tải checkpoint nếu có
    start_iteration = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        start_iteration = trainer.load_checkpoint(args.checkpoint)
        print(f"Tiếp tục từ iteration {start_iteration}")
    
    # Bắt đầu huấn luyện
    trainer.train()

if __name__ == "__main__":
    main()