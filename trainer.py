#!/usr/bin/env python
import torch
import numpy as np
import os
import pickle
import random
from typing import List, Tuple, Dict, Any
from collections import deque
import datetime

from neural_network import AlphaZeroNet, train_network
from mcts import MCTS
from chess_board import ChessBoard, PieceColor
from interface import ChessAI
from headless import HeadlessChessGame

class AlphaZeroTrainer:
    """Main trainer cho AlphaZero Chess"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Default configuration
        self.config = {
            'num_iterations': 100,
            'num_episodes': 500,
            'num_mcts_sims': 800,
            'num_epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'temperature': 1.0,
            'temperature_threshold': 15,
            'max_game_length': 200,
            'memory_size': 100000,
            'checkpoint_interval': 10,
            'evaluation_games': 100,
            'model_dir': './alphazero_models/',
            'data_dir': './alphazero_data/',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if config:
            self.config.update(config)
        
        # Tạo directories
        os.makedirs(self.config['model_dir'], exist_ok=True)
        os.makedirs(self.config['data_dir'], exist_ok=True)
        
        # Initialize components
        self.device = torch.device(self.config['device'])
        self.neural_network = AlphaZeroNet().to(self.device)
        self.mcts = MCTS(self.neural_network, self.config['num_mcts_sims'])
        
        # Training data memory
        self.memory = deque(maxlen=self.config['memory_size'])
        
        # Training statistics
        self.training_stats = {
            'iterations': [],
            'episodes': [],
            'losses': [],
            'evaluation_scores': []
        }
    
    def train(self):
        """Main training loop"""
        print("Bắt đầu huấn luyện AlphaZero Chess...")
        print(f"Device: {self.device}")
        print(f"Cấu hình: {self.config}")
        
        for iteration in range(self.config['num_iterations']):
            print(f"\n=== ITERATION {iteration + 1}/{self.config['num_iterations']} ===")
            
            # 1. Self-play để tạo training data
            print("Bước 1: Self-play...")
            self._self_play(iteration)
            
            # 2. Huấn luyện neural network
            print("Bước 2: Huấn luyện neural network...")
            if len(self.memory) > self.config['batch_size']:
                self._train_network(iteration)
            
            # 3. Đánh giá model
            if (iteration + 1) % self.config['checkpoint_interval'] == 0:
                print("Bước 3: Đánh giá model...")
                self._evaluate_model(iteration)
                
                # Lưu checkpoint
                self._save_checkpoint(iteration)
        
        print("\nHoàn thành huấn luyện!")
        self._save_final_model()
    
    def _self_play(self, iteration: int):
        """Thực hiện self-play để tạo training data"""
        ai = AlphaZeroAI(self.neural_network, self.mcts, 
                        temperature=self.config['temperature'],
                        temperature_threshold=self.config['temperature_threshold'])
        
        game_engine = HeadlessChessGame()
        episode_data = []
        
        for episode in range(self.config['num_episodes']):
            print(f"  Episode {episode + 1}/{self.config['num_episodes']}")
            
            # Chạy một game
            game_data = self._play_single_game(ai, game_engine)
            episode_data.extend(game_data)
            
            if (episode + 1) % 50 == 0:
                print(f"    Đã hoàn thành {episode + 1} games, thu thập {len(episode_data)} positions")
        
        # Thêm data vào memory
        self.memory.extend(episode_data)
        print(f"  Tổng data trong memory: {len(self.memory)} positions")
        
        # Lưu training data
        data_file = os.path.join(self.config['data_dir'], f'selfplay_data_iter_{iteration}.pkl')
        with open(data_file, 'wb') as f:
            pickle.dump(episode_data, f)
    
    def _play_single_game(self, ai: 'AlphaZeroAI', game_engine: HeadlessChessGame) -> List[Tuple]:
        """Chơi một game và thu thập training data"""
        game_engine.reset_game()
        game_data = []
        move_count = 0
        
        while not game_engine.game_over and move_count < self.config['max_game_length']:
            current_board = game_engine.board.copy_board()
            current_color = current_board.turn
            
            # Lấy action probabilities từ MCTS
            try:
                move, action_probs = ai.get_move_with_probs(current_board, current_color)
                
                if move is None:
                    break
                
                # Lưu (state, policy, None) - value sẽ được điền sau
                board_tensor = ai._board_to_tensor(current_board)
                game_data.append((board_tensor.numpy(), action_probs, None))
                
                # Thực hiện move
                game_engine.board.move_piece(move[0], move[1])
                game_engine.check_game_state()
                move_count += 1
                
            except Exception as e:
                print(f"Error in game: {e}")
                break
        
        # Xác định kết quả game và gán values
        if game_engine.result['winner'] == PieceColor.WHITE:
            result = 1.0
        elif game_engine.result['winner'] == PieceColor.BLACK:
            result = -1.0
        else:
            result = 0.0
        
        # Gán values cho các positions
        final_game_data = []
        for i, (state, policy, _) in enumerate(game_data):
            # Value từ perspective của player tại thời điểm đó
            player_result = result if i % 2 == 0 else -result
            final_game_data.append((state, policy, player_result))
        
        return final_game_data
    
    def _train_network(self, iteration: int):
        """Huấn luyện neural network với data từ memory"""
        # Lấy training data từ memory
        training_data = list(self.memory)
        random.shuffle(training_data)
        
        # Chuyển đổi thành numpy arrays
        states = np.array([data[0] for data in training_data])
        policies = np.array([data[1] for data in training_data])
        values = np.array([data[2] for data in training_data])
        
        dataset = np.column_stack([states, policies, values])
        
        # Huấn luyện
        losses = train_network(
            self.neural_network,
            dataset,
            epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            save_dir=self.config['model_dir'],
            device=self.config['device']
        )
        
        self.training_stats['losses'].extend(losses)
    
    def _evaluate_model(self, iteration: int):
        """Đánh giá model hiện tại"""
        # Tạo AI với model hiện tại
        current_ai = AlphaZeroAI(self.neural_network, self.mcts, temperature=0.1)
        
        # Tạo AI random để so sánh
        from ai_template import MyChessAI
        random_ai = MyChessAI()
        
        # Chạy evaluation games
        game_engine = HeadlessChessGame()
        stats = game_engine.run_many_games(
            white_ai=current_ai,
            black_ai=random_ai,
            num_games=self.config['evaluation_games'],
            max_moves=self.config['max_game_length'],
            swap_sides=True
        )
        
        win_rate = stats['white_win_percentage']
        print(f"  Evaluation: Win rate vs Random AI: {win_rate:.1f}%")
        
        self.training_stats['evaluation_scores'].append(win_rate)
    
    def _save_checkpoint(self, iteration: int):
        """Lưu checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.neural_network.state_dict(),
            'memory': list(self.memory),
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config['model_dir'], 
            f'checkpoint_iter_{iteration}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"  Đã lưu checkpoint: {checkpoint_path}")
    
    def _save_final_model(self):
        """Lưu model cuối cùng"""
        final_model = {
            'model_state_dict': self.neural_network.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }
        
        final_path = os.path.join(self.config['model_dir'], 'final_alphazero_model.pth')
        torch.save(final_model, final_path)
        print(f"Đã lưu model cuối cùng: {final_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Tải checkpoint để tiếp tục huấn luyện"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.neural_network.load_state_dict(checkpoint['model_state_dict'])
        self.memory = deque(checkpoint['memory'], maxlen=self.config['memory_size'])
        self.training_stats = checkpoint['training_stats']
        
        print(f"Đã tải checkpoint từ iteration {checkpoint['iteration']}")
        return checkpoint['iteration']

class AlphaZeroAI(ChessAI):
    """AI sử dụng AlphaZero cho ChessGame engine"""
    
    def __init__(self, neural_network, mcts, temperature=0.1, temperature_threshold=15):
        self.neural_network = neural_network
        self.mcts = mcts
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.move_count = 0
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple:
        """Lấy nước đi từ AlphaZero AI"""
        move, _ = self.get_move_with_probs(board, color)
        return move
    
    def get_move_with_probs(self, board: ChessBoard, color: PieceColor) -> Tuple:
        """Lấy nước đi và action probabilities"""
        self.move_count += 1
        
        # Điều chỉnh temperature theo game progress
        current_temp = self.temperature if self.move_count > self.temperature_threshold else 1.0
        
        # Thực hiện MCTS search
        move, action_probs = self.mcts.search(board, temperature=current_temp)
        
        return move, action_probs
    
    def _board_to_tensor(self, board_state: ChessBoard) -> torch.Tensor:
        """Chuyển board state thành tensor"""
        return self.mcts._board_to_tensor(board_state)
    
    def reset_game(self):
        """Reset cho game mới"""
        self.move_count = 0