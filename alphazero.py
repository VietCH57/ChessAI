import torch
from typing import Tuple, List
from chess_board import ChessBoard, Position, PieceColor
from network import AlphaZeroNetwork
from mcts import AlphaZeroMCTS
from config import AlphaZeroConfig
from interface import ChessAI
from selfplay import AlphaZeroAI

import torch
from typing import Tuple
from chess_board import ChessBoard, Position, PieceColor
from config import AlphaZeroConfig
from interface import ChessAI
from selfplay import AlphaZeroAI

class AlphaZeroChessAI(AlphaZeroAI):
    """AlphaZero trained model for chess play using a pre-trained TorchScript model"""

    @classmethod
    def from_checkpoint(cls, model_path: str, num_simulations: int = 400):
        config = AlphaZeroConfig(
            num_simulations=num_simulations,
            num_residual_blocks=12,
            num_filters=128,
            batch_size=64,
            max_moves_per_game=400,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(config.device)
        
        # Determine if the path points to a TorchScript model or a state dict
        try:
            if model_path.endswith("_weights.pt"):
                # Load regular state dict
                network = AlphaZeroNetwork(config).to(device)
                network.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
                network.eval()
                print(f"Loaded weights from {model_path}")
            else:
                try:
                    # Try loading as TorchScript model first
                    network = torch.jit.load(model_path, map_location=device)
                    network.eval()
                    print(f"Loaded TorchScript model from {model_path}")
                except Exception:
                    # If that fails, try loading as a state dict with weights_only=False for PyTorch 2.6+
                    try:
                        # Create new network
                        network = AlphaZeroNetwork(config).to(device)
                        # Load with weights_only=False
                        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                        
                        # Handle both raw state dict and wrapped checkpoint
                        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                            network.load_state_dict(checkpoint['state_dict'])
                        else:
                            network.load_state_dict(checkpoint)
                        network.eval()
                        print(f"Loaded model state dict from {model_path} with weights_only=False")
                    except Exception as e:
                        print(f"Error loading with weights_only=False: {e}")
                        # Last attempt - create empty network and warn
                        network = AlphaZeroNetwork(config).to(device)
                        network.eval()
                        print(f"Warning: Could not load checkpoint, using randomly initialized network")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create empty network as fallback
            network = AlphaZeroNetwork(config).to(device)
            network.eval()
            print("Using randomly initialized network due to loading error")

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