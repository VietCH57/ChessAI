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
        
        # Set device
        if self.config['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"AlphaZero player using CUDA: {torch.cuda.get_device_name(0)}")
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
        
        # Initialize MCTS
        self.mcts = AlphaZeroMCTS(
            network=self.network,
            encoder=self.encoder,
            num_simulations=self.config['num_simulations'],
            c_puct=self.config['c_puct']
        )
        
        # Storage for self-play data
        self.self_play_data = []
        self.last_state = None
        self.board_history = []
        
        # Test model with a simple input
        try:
            test_input = np.zeros((119, 8, 8), dtype=np.float32)
            policy, value = self.network.predict(test_input)
            print(f"Neural network initialized: policy shape={policy.shape}, value={value}")
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
            
            # Standard MCTS move selection
            moves, probabilities = self.mcts.get_move_probabilities(
                board, temperature=self.config['temperature']
            )
            search_time = time.time() - start_time
            
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
            
            
            return chosen_move
            
        except Exception as e:
            print(f"Error in get_move: {e}")
            traceback.print_exc()
            # Fallback to random move in case of error
            print("Falling back to random move due to error")
            return self._random_move(board, color)
        
        