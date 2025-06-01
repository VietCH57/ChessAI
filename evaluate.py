#!/usr/bin/env python
import torch
import argparse
from neural_network import AlphaZeroNet
from mcts import MCTS
from trainer import AlphaZeroAI
from headless import HeadlessChessGame
from ai_template import MyChessAI

def load_model(model_path: str, device: str = 'cuda'):
    """Tải model AlphaZero"""
    checkpoint = torch.load(model_path, map_location=device)
    
    neural_network = AlphaZeroNet().to(device)
    neural_network.load_state_dict(checkpoint['model_state_dict'])
    neural_network.eval()
    
    mcts = MCTS(neural_network, simulations=800)
    ai = AlphaZeroAI(neural_network, mcts, temperature=0.1)
    
    return ai

def evaluate_vs_random(alphazero_ai, num_games=100):
    """Đánh giá AlphaZero vs Random AI"""
    random_ai = MyChessAI()
    game_engine = HeadlessChessGame()
    
    print(f"Đánh giá AlphaZero vs Random AI ({num_games} games)...")
    
    stats = game_engine.run_many_games(
        white_ai=alphazero_ai,
        black_ai=random_ai,
        num_games=num_games,
        swap_sides=True
    )
    
    print(f"Kết quả:")
    print(f"  AlphaZero win rate: {stats['white_win_percentage']:.1f}%")
    print(f"  Random AI win rate: {stats['black_win_percentage']:.1f}%")
    print(f"  Draw rate: {stats['draw_percentage']:.1f}%")
    print(f"  Average game length: {stats['avg_game_length']:.1f} moves")
    
    return stats

def evaluate_vs_alphazero(alphazero_ai1, alphazero_ai2, num_games=50):
    """Đánh giá 2 AlphaZero models với nhau"""
    game_engine = HeadlessChessGame()
    
    print(f"Đánh giá AlphaZero vs AlphaZero ({num_games} games)...")
    
    stats = game_engine.run_many_games(
        white_ai=alphazero_ai1,
        black_ai=alphazero_ai2,
        num_games=num_games,
        swap_sides=True
    )
    
    print(f"Kết quả:")
    print(f"  Model 1 win rate: {stats['white_win_percentage']:.1f}%")
    print(f"  Model 2 win rate: {stats['black_win_percentage']:.1f}%")
    print(f"  Draw rate: {stats['draw_percentage']:.1f}%")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Đánh giá AlphaZero Chess')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn model AlphaZero')
    parser.add_argument('--opponent', type=str, default='random', help='Đối thủ: random hoặc đường dẫn model khác')
    parser.add_argument('--games', type=int, default=100, help='Số games đánh giá')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda hoặc cpu')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Tải model chính
    print(f"Đang tải model: {args.model}")
    alphazero_ai = load_model(args.model, device)
    
    if args.opponent == 'random':
        # Đánh giá vs Random AI
        evaluate_vs_random(alphazero_ai, args.games)
    else:
        # Đánh giá vs AlphaZero khác
        print(f"Đang tải model đối thủ: {args.opponent}")
        opponent_ai = load_model(args.opponent, device)
        evaluate_vs_alphazero(alphazero_ai, opponent_ai, args.games)

if __name__ == "__main__":
    main()