import numpy as np
import torch
import time
import os
from interface import ChessAI
from train_ai import TrainableChessAI
from board_encoder import ChessEncoder
from alpha_zero_mcts import AlphaZeroMCTS
from alphazero_model import AlphaZeroNetwork

class AlphaZeroPlayer(TrainableChessAI):
    """
    AlphaZero chess AI 
    """
    def __init__(self, config=None):
        """
        Initialize AlphaZero player
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        default_config = {
            'num_simulations': 800,  # Number of MCTS simulations per move
            'c_puct': 1.0,          # Exploration constant in PUCT formula
            'temperature': 1.0,      # Temperature for move selection (1=explore, 0=best)
            'num_res_blocks': 20,    # Number of residual blocks in the network
            'num_filters': 256,      # Number of filters in convolutional layers
            'exploration_rate': 0.0, # Exploration rate for random moves during play
            'model_file': None,      # Path to saved model file
        }
        
        # Update default config with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialize base class
        super().__init__(exploration_rate=self.config['exploration_rate'])
        
        # Initialize neural network
        self.network = AlphaZeroNetwork(
            num_res_blocks=self.config['num_res_blocks'], 
            num_filters=self.config['num_filters']
        )
        
        # Load model if provided
        if self.config['model_file'] and os.path.exists(self.config['model_file']):
            self.load_model(self.config['model_file'])
        
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
    
    def _best_move(self, board, color):
        """
        Use MCTS to find the best move
        
        Args:
            board: Current board state
            color: Color to play (PieceColor.WHITE or PieceColor.BLACK)
            
        Returns:
            from_pos, to_pos: Selected move
        """
        # Check if it's our turn
        if board.turn != color:
            raise ValueError(f"Not {color}'s turn to move")
        
        # Save the board state for training data
        self.last_state = board.copy_board()
        
        # Run MCTS simulations to get move probabilities
        moves, probabilities = self.mcts.get_move_probabilities(
            board, temperature=self.config['temperature']
        )
        
        # Store the MCTS policy for training
        self.last_mcts_policy = {move: prob for move, prob in zip(moves, probabilities)}
        
        # Choose a move based on the probabilities
        chosen_idx = np.random.choice(len(moves), p=probabilities)
        chosen_move = moves[chosen_idx]
        
        # Update MCTS tree with the chosen move
        self.mcts.update_with_move(chosen_move)
        
        return chosen_move
    
    def record_game_result(self, result):
        """
        Record the result of a game for training data
        
        Args:
            result: Game result (1.0 for win, 0.0 for draw, -1.0 for loss)
        """
        if self.last_state and self.last_mcts_policy:
            # Convert last_mcts_policy to the format expected by the training function
            policy = np.zeros(1968)  # Policy vector size
            for (from_pos, to_pos), prob in self.last_mcts_policy.items():
                move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
                if move_key in self.encoder.move_to_index:
                    idx = self.encoder.move_to_index[move_key]
                    policy[idx] = prob
            
            # Encode the board
            encoded_state = self.encoder.encode_board(self.last_state, self.board_history)
            
            # Add the training example
            self.self_play_data.append({
                'state': encoded_state,
                'policy': policy,
                'value': result
            })
            
            # Clear last state and policy
            self.last_state = None
            self.last_mcts_policy = None
    
    def save_model(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath):
        """
        Load a model from a file
        
        Args:
            filepath: Path to the model file
        """
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        if 'config' in checkpoint:
            # Update only the network-related config parameters
            for key in ['num_res_blocks', 'num_filters']:
                if key in checkpoint['config']:
                    self.config[key] = checkpoint['config'][key]
        
        # Reset MCTS after loading new model
        self.mcts = AlphaZeroMCTS(
            network=self.network,
            encoder=self.encoder,
            num_simulations=self.config['num_simulations'],
            c_puct=self.config['c_puct']
        )