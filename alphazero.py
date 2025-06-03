import torch
from typing import Tuple, List
from chess_board import ChessBoard, Position, PieceColor
from network import AlphaZeroNetwork
from mcts import AlphaZeroMCTS
from config import AlphaZeroConfig
from interface import ChessAI
from selfplay import AlphaZeroAI

class AlphaZeroChessAI(AlphaZeroAI):
    """AlphaZero trained model for chess play using a pre-trained checkpoint"""
    
    @classmethod
    def from_checkpoint(cls, model_path: str, num_simulations: int = 200):
        """
        Create an AlphaZero player from a trained model checkpoint.
        """
        # Create configuration
        config = AlphaZeroConfig(
            num_simulations=num_simulations,
            num_residual_blocks=12,
            num_filters=128,
            batch_size=32,
            max_moves_per_game=200,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize network
        network = AlphaZeroNetwork(config)
        
        # Load model weights
        device = torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            network.load_state_dict(checkpoint)
            network = network.to(device)  # Ensure network is on the right device
            print(f"Loaded AlphaZero model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        network.eval()  # Set to evaluation mode
        
        # Return the AI instance
        return cls(network, config, training_mode=False)
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Get the best move from the trained AlphaZero model"""
        # Always use evaluation mode (no exploration)
        num_sims = self.config.num_simulations
        temperature = 0.0  # Deterministic selection
        
        # Run MCTS to get action probabilities (no Dirichlet noise in evaluation)
        action_probs, _ = self.mcts.search(
            board, num_sims, temperature=temperature, add_noise=False
        )
        
        # Get legal moves
        legal_moves = self._get_legal_moves_cached(board, color)
        if not legal_moves:
            raise ValueError(f"No legal moves for {color}")
        
        # Get probabilities for legal moves
        move_probs = []
        for from_pos, to_pos in legal_moves:
            move_idx = self.mcts.move_encoder.encode_move(from_pos, to_pos)
            prob = action_probs[move_idx] if move_idx >= 0 else 0.0
            move_probs.append((from_pos, to_pos, prob))
        
        # Select best move (highest probability)
        best_move = max(move_probs, key=lambda x: x[2])
        from_pos, to_pos, _ = best_move
        
        self.move_count += 1
        return from_pos, to_pos