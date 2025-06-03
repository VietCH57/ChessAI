import os
import json
import numpy as np
import torch
import time
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
from chess_board import ChessBoard, Position, PieceColor
from mcts import AlphaZeroMCTS
from network import AlphaZeroNetwork, BoardEncoder
from config import AlphaZeroConfig
from interface import ChessAI

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class AlphaZeroAI(ChessAI):
    """AlphaZero AI with MCTS optimization"""
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig, 
                 training_mode: bool = False):
        self.network = network
        self.config = config
        
        # Ensure network is on correct device
        if torch.cuda.is_available() and config.device == "cuda":
            self.network = self.network.cuda()
            
        self.mcts = AlphaZeroMCTS(network, config)
        self.training_mode = training_mode
        self.move_count = 0
        
        # Cache legal moves to avoid recomputation
        self._legal_moves_cache = {}
        
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Get move using MCTS - with adaptive simulations"""
        # Adaptive simulation count based on game phase
        base_sims = self.config.num_simulations
        
        # In training mode, use adaptive simulation count for efficiency
        if self.training_mode:
            if self.move_count < 10:
                # Opening phase - less accuracy needed
                num_sims = max(base_sims // 4, 50)
            elif self.move_count > 80:
                # Endgame - reduce slightly for performance
                num_sims = max(base_sims // 2, 100)
            else:
                # Middlegame - full simulation count
                num_sims = base_sims
        else:
            # In evaluation/play mode, always use full count
            num_sims = base_sims
        
        # Temperature schedule
        if self.training_mode:
            temperature = 1.0 if self.move_count < self.config.temperature_threshold else 0.0
            add_noise = True  # Use Dirichlet noise in training
        else:
            temperature = 0.0  # Deterministic selection in evaluation
            add_noise = False
        
        # Run MCTS
        action_probs, _ = self.mcts.search(
            board, num_sims, temperature=temperature, add_noise=add_noise
        )
        
        # Get legal moves
        legal_moves = self._get_legal_moves_cached(board, color)
        if not legal_moves:
            raise ValueError(f"No legal moves for {color}")
        
        # Efficiently select move from probabilities
        selected_move = self._select_move_efficiently(legal_moves, action_probs, temperature)
        
        self.move_count += 1
        return selected_move
    
    def _select_move_efficiently(self, legal_moves, action_probs, temperature):
        """Efficiently select move from action probabilities"""
        # Calculate probabilities for legal moves using vectorized operations
        move_indices = np.array([self.mcts.move_encoder.encode_move(from_pos, to_pos) 
                                for from_pos, to_pos in legal_moves])
        valid_indices = move_indices >= 0
        
        if not np.any(valid_indices):
            # No valid moves in policy - use uniform random
            move_idx = np.random.randint(len(legal_moves))
            return legal_moves[move_idx]
        
        # Extract probabilities for valid moves
        valid_moves = [legal_moves[i] for i in range(len(legal_moves)) if valid_indices[i]]
        valid_probs = action_probs[move_indices[valid_indices]]
        
        # Normalize probabilities
        if valid_probs.sum() > 0:
            valid_probs = valid_probs / valid_probs.sum()
        else:
            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)
        
        # Select move based on temperature
        if temperature == 0.0 or len(valid_moves) == 1:
            # Deterministic - select highest probability
            move_idx = np.argmax(valid_probs)
            return valid_moves[move_idx]
        else:
            # Sample from distribution
            move_idx = np.random.choice(len(valid_moves), p=valid_probs)
            return valid_moves[move_idx]
    
    def _get_legal_moves_cached(self, board: ChessBoard, color: PieceColor) -> List[Tuple[Position, Position]]:
        """Get all legal moves with caching"""
        # Create a unique key for this board state
        board_hash = board.get_board_hash() if hasattr(board, 'get_board_hash') else None
        
        if board_hash and board_hash in self._legal_moves_cache:
            return self._legal_moves_cache[board_hash]
        
        # Calculate legal moves
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
        
        # Cache result
        if board_hash:
            self._legal_moves_cache[board_hash] = legal_moves
            
            # Limit cache size
            if len(self._legal_moves_cache) > 1000:
                # Remove oldest entries
                for key in list(self._legal_moves_cache.keys())[:100]:
                    self._legal_moves_cache.pop(key)
                    
        return legal_moves
    
    def reset_move_count(self):
        self.move_count = 0
        self._legal_moves_cache.clear()
        
        # Also reset MCTS tree
        if hasattr(self.mcts, 'root'):
            self.mcts.root = None

class SelfPlayGenerator:
    """Self-play generator with optimization"""
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig):
        self.network = network
        self.config = config
        self.board_encoder = BoardEncoder()
        
        # Pre-calculate some common board positions for curriculum learning
        self._starting_positions = None
        if hasattr(self, '_create_curriculum_positions'):
            self._starting_positions = self._create_curriculum_positions()
        
    def generate_games(self, num_games: int) -> List[Dict[str, Any]]:
        """Generate multiple self-play games - no batching for compatibility"""
        all_training_data = []
        
        # Simple progress tracking
        start_time = time.time()
        examples_per_game = []
        
        for game_num in range(num_games):
            print(f"Generating self-play game {game_num + 1}/{num_games}")
            
            try:
                # Generate game data
                game_data = self.generate_game()
                all_training_data.extend(game_data)
                examples_per_game.append(len(game_data))
                
                # Calculate stats
                elapsed = time.time() - start_time
                examples_per_sec = len(all_training_data) / max(1, elapsed)
                avg_examples = sum(examples_per_game) / len(examples_per_game)
                
                # Display progress
                eta = (num_games - (game_num + 1)) * elapsed / (game_num + 1)
                print(f"Generated {game_num + 1} games, {len(all_training_data)} examples "
                      f"({examples_per_sec:.1f} ex/s, avg {avg_examples:.1f} ex/game, ETA: {eta/60:.1f}m)")
                
            except Exception as e:
                print(f"Error generating game {game_num + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return all_training_data
    
    def generate_games_parallel(self, num_games: int, num_processes: int = None) -> List[Dict[str, Any]]:
        """Generate games in parallel - improves throughput on multi-core systems"""
        if num_processes is None:
            num_processes = min(cpu_count(), 4)  # Limit to 4 processes by default
        
        # Distribute games across processes
        games_per_process = max(1, num_games // num_processes)
        remaining_games = num_games % num_processes
        
        # Create arguments for each process
        process_args = []
        for i in range(num_processes):
            games_for_this_process = games_per_process + (1 if i < remaining_games else 0)
            if games_for_this_process > 0:
                # Create a CPU-only config for workers to avoid CUDA issues
                worker_config = AlphaZeroConfig(
                    num_iterations=self.config.num_iterations,
                    episodes_per_iteration=self.config.episodes_per_iteration,
                    num_simulations=self.config.num_simulations,
                    batch_size=self.config.batch_size,
                    learning_rate=self.config.learning_rate,
                    num_residual_blocks=self.config.num_residual_blocks,
                    num_filters=self.config.num_filters,
                    device="cpu",  # Force CPU for workers
                    num_workers=1
                )
                process_args.append((games_for_this_process, worker_config))
        
        # Run processes
        print(f"Generating {num_games} games using {len(process_args)} processes (CPU mode)")
        with Pool(processes=len(process_args)) as pool:
            results = pool.map(SelfPlayGenerator.generate_games_worker, process_args)
        
        # Combine results
        all_training_data = []
        for result in results:
            if result:  # Check if result is not None
                all_training_data.extend(result)
        
        return all_training_data
    
    def generate_game(self) -> List[Dict[str, Any]]:
        """Generate a single self-play game with optimizations"""
        # Initialize game
        board = ChessBoard()
        ai = AlphaZeroAI(self.network, self.config, training_mode=True)
        ai.reset_move_count()
        
        # Track data for training examples
        game_data = []
        board_history = []  # For encoding historical positions
        move_count = 0
        max_moves = self.config.max_moves_per_game
        
        # Main game loop
        while (not self._is_game_over(board) and move_count < max_moves):
            # Store current board state in history
            board_history.append(board.copy_board())
            if len(board_history) > 8:  # Only keep last 8 positions
                board_history.pop(0)
            
            try:
                # Determine temperature - higher in early game for exploration
                temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
                
                # Run MCTS to get policy and value estimate
                # Adaptive simulation count for performance
                num_sims = self.config.num_simulations
                if move_count < 10:
                    num_sims = max(50, num_sims // 4)  # Fewer simulations in opening
                elif move_count < 30:
                    num_sims = max(100, num_sims // 2)  # Increase in early middlegame
                
                # Get MCTS policy
                mcts_policy, _ = ai.mcts.search(
                    board, num_sims, temperature=temperature, add_noise=True
                )
                
                # Encode current board state for training
                board_state = self.board_encoder.encode_board(board, board_history)
                
                # Store training example
                game_data.append({
                    'state': board_state,
                    'policy': mcts_policy,
                    'player': board.turn,
                    'move_count': move_count
                })
                
                # Make move selected by AI
                from_pos, to_pos = ai.get_move(board, board.turn)
                move = board.move_piece(from_pos, to_pos)
                if not move:
                    print("Invalid move returned by AI")
                    break
                    
                move_count += 1
                
            except Exception as e:
                print(f"Error in self-play move {move_count}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Determine game outcome
        outcome = self._get_game_outcome(board)
        
        # Update outcomes in all training examples
        for example in game_data:
            # Outcome is from the perspective of the player who made the move
            if example['player'] == PieceColor.WHITE:
                example['outcome'] = outcome
            else:
                example['outcome'] = -outcome  # Flip for black
        
        return game_data
    
    def _is_game_over(self, board: ChessBoard) -> bool:
        """Check if game is over"""
        try:
            return (board.is_checkmate(board.turn) or 
                    board.is_stalemate(board.turn) or
                    board.is_fifty_move_rule_draw() or
                    board.is_threefold_repetition())
        except:
            return False
    
    def _get_game_outcome(self, board: ChessBoard) -> float:
        """Get game outcome from white's perspective"""
        try:
            if board.is_checkmate(board.turn):
                return -1.0 if board.turn == PieceColor.WHITE else 1.0
            else:
                return 0.0  # Draw
        except:
            return 0.0

    @staticmethod
    def generate_games_worker(args):
        """Worker function for parallel game generation"""
        # Disable pygame display for worker processes
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        
        # Suppress pygame initialization message
        import sys
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        # Now it's safe to import pygame modules if needed
        try:
            import pygame
            pygame.init()
        except ImportError:
            pass
        
        # Restore stdout
        sys.stdout = old_stdout
        
        try:
            num_games, config = args
            
            print(f"Worker process starting - will generate {num_games} games")
            
            # Create network for this process - IMPORTANT: Force CPU mode!
            config.device = "cpu"  # Override to ensure CPU mode
            network = AlphaZeroNetwork(config)
            
            # Create generator
            generator = SelfPlayGenerator(network, config)
            
            # Generate games
            training_data = []
            for game_num in range(num_games):
                try:
                    print(f"Worker generating game {game_num+1}/{num_games}")
                    game_data = generator.generate_game()
                    print(f"Completed game {game_num+1} with {len(game_data)} examples")
                    training_data.extend(game_data)
                except Exception as e:
                    print(f"Error in worker game {game_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"Worker completed all {num_games} games")
            return training_data
            
        except Exception as e:
            print(f"Critical worker error: {e}")
            import traceback
            traceback.print_exc()
            return []