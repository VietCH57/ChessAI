import os
import json
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
from chess_board import ChessBoard, Position, PieceColor
from mcts import AlphaZeroMCTS
from network import AlphaZeroNetwork, BoardEncoder
from config import AlphaZeroConfig
from interface import ChessAI

class AlphaZeroAI(ChessAI):
    """ AlphaZero AI with reduced simulation overhead"""
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig, 
                 training_mode: bool = False):
        self.network = network
        self.config = config
        self.mcts = AlphaZeroMCTS(network, config)
        self.training_mode = training_mode
        self.move_count = 0
        
        # Cache for move encoding
        self._move_cache = {}
        
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Optimized move selection"""
        # Adaptive simulation count based on game phase
        base_sims = self.config.num_simulations
        if self.training_mode:
            # Reduce simulations in opening (first 10 moves)
            if self.move_count < 10:
                num_sims = max(50, base_sims // 4)
            # Reduce simulations in very late game (after move 80)
            elif self.move_count > 80:
                num_sims = max(100, base_sims // 2)
            else:
                num_sims = base_sims
        else:
            num_sims = base_sims
        
        # Temperature schedule
        if self.training_mode:
            temperature = 1.0 if self.move_count < self.config.temperature_threshold else 0.0
            add_noise = True
        else:
            temperature = 0.0
            add_noise = False
        
        # Run MCTS
        action_probs, _ = self.mcts.search(
            board, num_sims, temperature=temperature, add_noise=add_noise
        )
        
        # Fast move selection
        legal_moves = self._get_legal_moves_cached(board, color)
        if not legal_moves:
            raise ValueError(f"No legal moves for {color}")
        
        # Vectorized probability computation
        move_indices = np.array([self.mcts.move_encoder.encode_move(from_pos, to_pos) 
                                for from_pos, to_pos in legal_moves])
        valid_indices = move_indices >= 0
        
        if not np.any(valid_indices):
            # Fallback to uniform distribution
            move_idx = np.random.randint(len(legal_moves))
            selected_move = legal_moves[move_idx]
        else:
            legal_probs = action_probs[move_indices[valid_indices]]
            if legal_probs.sum() > 0:
                legal_probs = legal_probs / legal_probs.sum()
                # Sample from valid moves only
                valid_moves = [legal_moves[i] for i in range(len(legal_moves)) if valid_indices[i]]
                if temperature == 0.0:
                    move_idx = np.argmax(legal_probs)
                    selected_move = valid_moves[move_idx]
                else:
                    move_idx = np.random.choice(len(valid_moves), p=legal_probs)
                    selected_move = valid_moves[move_idx]
            else:
                selected_move = legal_moves[0]
        
        self.move_count += 1
        return selected_move
    
    def _get_legal_moves_cached(self, board: ChessBoard, color: PieceColor) -> List[Tuple[Position, Position]]:
        """Cached legal move generation"""
        legal_moves = []
        for row in range(8):
            for col in range(8):
                from_pos = Position(row, col)
                piece = board.get_piece(from_pos)
                if piece and piece.color == color:
                    try:
                        valid_moves = board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            legal_moves.append((from_pos, move.end_pos))
                    except:
                        continue
        return legal_moves
    
    def reset_move_count(self):
        self.move_count = 0

class SelfPlayGenerator:
    """Self-play generator"""
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig):
        self.network = network
        self.config = config
        self.board_encoder = BoardEncoder()
        
    def generate_games(self, num_games: int) -> List[Dict[str, Any]]:
        """Generate multiple self-play games"""
        all_training_data = []
        
        for game_num in range(num_games):
            print(f"Generating self-play game {game_num + 1}/{num_games}")
            
            try:
                game_data = self.generate_game()
                all_training_data.extend(game_data)
                
                print(f"Generated {game_num + 1} games, {len(all_training_data)} training examples")
            except Exception as e:
                print(f"Error generating game {game_num + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_training_data
        
    def generate_games_parallel(self, num_games: int, num_processes: int = None) -> List[Dict[str, Any]]:
        """Generate games in parallel"""
        if num_processes is None:
            num_processes = min(cpu_count(), 4)  # Limit to 4 processes
        
        games_per_process = max(1, num_games // num_processes)
        remaining_games = num_games % num_processes
        
        # Create arguments for each process
        process_args = []
        for i in range(num_processes):
            games_for_this_process = games_per_process + (1 if i < remaining_games else 0)
            if games_for_this_process > 0:
                process_args.append((games_for_this_process, self.config))
        
        # Run processes
        with Pool(processes=len(process_args)) as pool:
            results = pool.map(generate_games_worker, process_args)
        
        # Combine results
        all_training_data = []
        for result in results:
            all_training_data.extend(result)
        
        return all_training_data
    
    def generate_game(self) -> List[Dict[str, Any]]:
        """Generate a single self-play game"""
        print("Starting self-play game generation")
        
        board = ChessBoard()
        ai = AlphaZeroAI(self.network, self.config, training_mode=True)
        ai.reset_move_count()
        
        game_data = []
        board_history = []
        move_count = 0
        max_moves = self.config.max_moves_per_game
        
        while (not self._is_game_over(board) and move_count < max_moves):
            print(f"Move {move_count + 1}/{max_moves}")
            
            # Store board state
            board_history.append(board.copy_board())
            
            try:
                # Get MCTS policy
                temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
                
                # Use adaptive simulation count
                if move_count < 10:
                    num_sims = max(50, ai.config.num_simulations // 4)
                else:
                    num_sims = ai.config.num_simulations // 2
                
                mcts_policy, _ = ai.mcts.search(
                    board, 
                    num_sims,
                    temperature=temperature,
                    add_noise=True
                )
                
                # Fast board encoding
                board_state = self.board_encoder.encode_board(board, board_history[-8:])
                
                # Store training example
                game_data.append({
                    'state': board_state,
                    'policy': mcts_policy,
                    'player': board.turn,
                    'move_count': move_count
                })
                
                # Make move
                from_pos, to_pos = ai.get_move(board, board.turn)
                move = board.move_piece(from_pos, to_pos)
                if not move:
                    print("Invalid move returned by AI")
                    break
                    
                move_count += 1
                print(f"Move completed: {from_pos.row},{from_pos.col} -> {to_pos.row},{to_pos.col}")
                
            except Exception as e:
                print(f"Error in self-play move {move_count}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Set outcomes
        outcome = self._get_game_outcome(board)
        print(f"Game ended with outcome: {outcome}")
        
        for example in game_data:
            if example['player'] == PieceColor.WHITE:
                example['outcome'] = outcome
            else:
                example['outcome'] = -outcome
        
        print(f"Generated game with {len(game_data)} training examples")
        return game_data
    
    def _is_game_over(self, board: ChessBoard) -> bool:
        try:
            return (board.is_checkmate(board.turn) or 
                    board.is_stalemate(board.turn) or
                    board.is_fifty_move_rule_draw() or
                    board.is_threefold_repetition())
        except:
            return False
    
    def _get_game_outcome(self, board: ChessBoard) -> float:
        try:
            if board.is_checkmate(board.turn):
                return -1.0 if board.turn == PieceColor.WHITE else 1.0
            else:
                return 0.0
        except:
            return 0.0

def generate_games_worker(args):
    """Worker function for parallel game generation"""
    num_games, config = args
    
    # Create network for this process
    network = AlphaZeroNetwork(config)
    if torch.cuda.is_available() and config.device == "cuda":
        network = network.cuda()
    
    generator = SelfPlayGenerator(network, config)
    
    training_data = []
    for game_num in range(num_games):
        try:
            game_data = generator.generate_game()
            training_data.extend(game_data)
        except Exception as e:
            print(f"Error in worker game {game_num}: {e}")
            continue
    
    return training_data