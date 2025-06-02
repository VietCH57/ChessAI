import numpy as np
import torch
import time
import os
import random
import traceback  # Import for better error reporting
from interface import ChessAI
from train_ai import TrainableChessAI
from typing import Tuple  
from chess_board import ChessBoard, Position, PieceColor  
from board_encoder import ChessEncoder
from alpha_zero_mcts import AlphaZeroMCTS
from alphazero_model import AlphaZeroNetwork

class AlphaZeroPlayer(TrainableChessAI):
    """AlphaZero chess AI"""
    def __init__(self, config=None):
        """Initialize AlphaZero player"""
        # Default configuration
        default_config = {
            'num_simulations': 800,     # Number of MCTS simulations per move
            'c_puct': 1.0,              # Exploration constant in PUCT formula
            'temperature': 1.0,         # Temperature for move selection
            'num_res_blocks': 20,       # Number of residual blocks
            'num_filters': 256,         # Number of filters
            'exploration_rate': 0.0,    # Random move probability
            'model_file': None,         # Path to model file
            'use_cuda': True,           # Whether to use CUDA
        }
        
        # Update default config with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize base class
        super().__init__(exploration_rate=self.config['exploration_rate'])
        
        # Đảm bảo rõ ràng GPU được sử dụng nếu có thể
        if torch.cuda.is_available():
            # Thiết lập CUDA device một cách rõ ràng
            torch.cuda.set_device(0)
            self.device = torch.device("cuda")
            print(f"AlphaZero player using CUDA: {torch.cuda.get_device_name(0)}")
            # Hiển thị thông tin GPU
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            # Khởi tạo GPU với một tensor nhỏ để đảm bảo các tính toán diễn ra trên GPU
            warm_up = torch.ones(1, device=self.device)
            warm_up = warm_up * 2
            torch.cuda.synchronize()  # Đảm bảo xử lý xong
            del warm_up  # Giải phóng bộ nhớ
        else:
            self.device = torch.device("cpu")
            print("AlphaZero player using CPU")
        
        # Initialize neural network
        try:
            self.network = AlphaZeroNetwork(
                num_res_blocks=self.config['num_res_blocks'], 
                num_filters=self.config['num_filters'],
                device=self.device
            )
            
            # Load model if provided
            if self.config['model_file'] and os.path.exists(self.config['model_file']):
                self.load_model(self.config['model_file'])
            
        except Exception as e:
            print(f"Error initializing network: {e}")
            traceback.print_exc()
            raise
        
        # Initialize encoder
        self.encoder = ChessEncoder()
        
        # Initialize MCTS with the same device as network
        self.mcts = AlphaZeroMCTS(
            network=self.network,
            encoder=self.encoder,
            num_simulations=self.config['num_simulations'],
            c_puct=self.config['c_puct'],
            device=self.device  # Đảm bảo MCTS sử dụng cùng device với network
        )
        
        # Storage for self-play data
        self.self_play_data = []
        self.last_state = None
        self.board_history = []
        
        # Test model với input đơn giản và hiển thị thông tin về device
        try:
            test_input = np.zeros((119, 8, 8), dtype=np.float32)
            start_time = time.time()
            policy, value = self.network.predict(test_input)
            inference_time = time.time() - start_time
            print(f"Neural network initialized: policy shape={policy.shape}, value={value}")
            print(f"Test inference time: {inference_time*1000:.2f}ms")
            
            # Kiểm tra GPU usage sau khi dự đoán
            if torch.cuda.is_available():
                print(f"GPU memory used: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        except Exception as e:
            print(f"Warning: Neural network test failed: {e}")
            traceback.print_exc()
    
    def _random_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Get a random valid move (fallback)"""
        valid_moves = []
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece.color == color:
                    moves = board.get_valid_moves(pos)
                    valid_moves.extend([(pos, move.end_pos) for move in moves])
        
        if not valid_moves:
            print(f"ERROR: No valid moves found for {color}")
            # This should never happen unless the game is over
            raise ValueError(f"No valid moves for {color}")
        
        return random.choice(valid_moves)
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Get the next move from the AI"""
        try:
            # Check if it's our turn
            if board.turn != color:
                print(f"Warning: Not {color}'s turn to move")
                return self._random_move(board, color)
                
            # Save the board state for training data
            self.last_state = board.copy_board()
            
            # Initialize last_mcts_policy if needed
            if not hasattr(self, 'last_mcts_policy'):
                self.last_mcts_policy = {}
            
            # Reset MCTS tree for each move
            self.mcts.root = None
            
            # Monitor GPU memory trước khi bắt đầu
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Đảm bảo mọi hoạt động GPU đã hoàn tất
                mem_before = torch.cuda.memory_allocated(0)
                print(f"GPU memory before MCTS: {mem_before/1024**2:.2f} MB")
            
            # Run MCTS simulations to get move probabilities
            start_time = time.time()
            
            # Check if board has any valid moves
            has_valid_moves = False
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    if piece.color == color:
                        moves = board.get_valid_moves(pos)
                        if moves:
                            has_valid_moves = True
                            break
                if has_valid_moves:
                    break
            
            if not has_valid_moves:
                print(f"WARNING: No valid moves for {color}")
                # Game should be over if there are no valid moves
                raise ValueError(f"No valid moves for {color}")
            
            # Standard MCTS move selection - show statistics
            moves, probabilities = self.mcts.get_move_probabilities(
                board, temperature=self.config['temperature'], show_stats=True
            )
            search_time = time.time() - start_time
            print(f"MCTS search time: {search_time:.2f}s for {self.config['num_simulations']} simulations")
            
            # Monitor GPU memory sau MCTS
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_after = torch.cuda.memory_allocated(0)
                print(f"GPU memory after MCTS: {mem_after/1024**2:.2f} MB")
                print(f"GPU memory change: {(mem_after-mem_before)/1024**2:.2f} MB")
            
            # Handle case with no valid moves
            if not moves or len(moves) == 0:
                print("No valid moves returned by MCTS, using random move")
                return self._random_move(board, color)
            
            # Store the MCTS policy for training
            self.last_mcts_policy = {move: prob for move, prob in zip(moves, probabilities)}
            
            # Check probabilities for validity
            if np.isnan(probabilities).any() or np.sum(probabilities) == 0:
                print("Invalid probabilities, using uniform distribution")
                probabilities = np.ones(len(moves)) / len(moves)
            
            # Choose a move based on the probabilities
            chosen_idx = np.random.choice(len(moves), p=probabilities)
            chosen_move = moves[chosen_idx]
            
            # Làm sạch bộ nhớ GPU sau mỗi move
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return chosen_move
            
        except Exception as e:
            print(f"Error in get_move: {e}")
            traceback.print_exc()
            # Fallback to random move in case of error
            print("Falling back to random move due to error")
            return self._random_move(board, color)
    
    def load_model(self, filepath):
        """Load model from file"""
        try:
            if not os.path.exists(filepath):
                print(f"Model file not found: {filepath}")
                return False

            iteration, _ = self.network.load_checkpoint(filepath)
            print(f"Model loaded from {filepath} (iteration {iteration})")
            
            # Đảm bảo model ở đúng device
            self.network.to(self.device)
            
            # Kiểm tra model device
            print(f"Model device after loading: {next(self.network.parameters()).device}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False