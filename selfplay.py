import os
import json
import numpy as np
import torch
import time
import sys
import traceback
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict, Any
from chess_board import ChessBoard, Position, PieceColor
from mcts import AlphaZeroMCTS, BatchedMCTS
from network import AlphaZeroNetwork, BoardEncoder, MoveEncoder  
from config import AlphaZeroConfig
from interface import ChessAI

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

class AlphaZeroAI(ChessAI):
    """AlphaZero AI with MCTS optimization"""
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig, 
                 training_mode: bool = False, device=None, batch_size=1):
        self.network = network
        self.config = config
        
        # Set device correctly
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        
        # Ensure network is on correct device
        self.network = self.network.to(self.device)
            
        # CUDA optimized MCTS
        self.mcts = AlphaZeroMCTS(network, config, device=self.device, batch_size=batch_size)
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
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig, 
                 device=None, use_cuda_mcts=False):
        self.network = network
        self.config = config
        self.board_encoder = BoardEncoder()
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() and config.device == "cuda" else "cpu")
        
        # Ensure network is on correct device
        self.network = self.network.to(self.device)
        
        # CUDA optimizations for MCTS
        self.use_cuda_mcts = use_cuda_mcts
        self.mcts_batch_size = 64 if use_cuda_mcts else 1
        
        # For batched inference in MCTS
        if self.use_cuda_mcts and self.device.type == 'cuda':
            self.inference_queue = []
            self.inference_results = {}
            
            # Pre-allocate tensors for batched inference
            self.batch_states = torch.zeros((self.mcts_batch_size, 
                                           config.input_planes, 
                                           8, 8), 
                                          dtype=torch.float32, 
                                          device=self.device)
        
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
        ai = AlphaZeroAI(
            self.network, 
            self.config, 
            training_mode=True,
            device=self.device,
            batch_size=self.mcts_batch_size
        )
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
        
class GPUParallelSelfPlayGenerator:
    """Self-play generator that runs multiple games in parallel on a single GPU"""
    
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig, device=None, num_parallel_games=8):
        self.network = network
        self.config = config
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        
        # Set device and ensure it's GPU
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != 'cuda':
            print("Warning: Using CPU for GPUParallelSelfPlayGenerator. Performance will be suboptimal.")
            
        # Ensure network is on correct device
        self.network = self.network.to(self.device)
        
        # Set evaluation mode for consistent inference
        self.network.eval()
        
        # Optimize number of parallel games based on available memory
        if self.device.type == 'cuda':
            try:
                free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                # Adaptive parallel games - more conservative to avoid OOM errors
                mem_per_game = 250 * 1024 * 1024  # ~250MB per game
                max_by_memory = int(free_memory * 0.7 / mem_per_game)  # Use 70% of free memory
                self.num_parallel_games = min(num_parallel_games, max(2, max_by_memory))
                print(f"Adjusted to {self.num_parallel_games} parallel games based on GPU memory")
            except Exception as e:
                print(f"Error calculating GPU memory: {e}")
                self.num_parallel_games = min(num_parallel_games, 4)  # Conservative default
        else:
            self.num_parallel_games = 1  # Just one game on CPU
            
        print(f"Running {self.num_parallel_games} parallel games on GPU")
        
        # For tracking all parallel games
        self.active_games = []
        self.game_data = []
        
    def generate_games(self, num_games: int) -> List[Dict[str, Any]]:
        """Generate multiple games in parallel on GPU"""
        all_training_data = []
        games_completed = 0
        
        # Track performance metrics
        start_time = time.time()
        
        # Continue until we've generated the requested number of games
        while games_completed < num_games:
            # Determine batch size for this iteration
            batch_size = min(self.num_parallel_games, num_games - games_completed)
            
            # Initialize batch of games
            self._initialize_game_batch(batch_size)
            
            # Run batch of games to completion
            batch_data = self._run_game_batch()
            
            # Add completed games to training data
            for game_examples in batch_data:
                all_training_data.extend(game_examples)
            
            # Update completion count
            games_completed += batch_size
            
            # Show progress
            elapsed = time.time() - start_time
            examples_per_sec = len(all_training_data) / max(1, elapsed)
            print(f"Completed {games_completed}/{num_games} games, "
                f"{len(all_training_data)} examples ({examples_per_sec:.1f} ex/sec)")
        
        return all_training_data
        
    def _initialize_game_batch(self, batch_size: int):
        """Initialize a batch of games for parallel processing"""
        # Reset tracking data
        self.active_games = []
        self.game_data = [[] for _ in range(batch_size)]
        
        # Create fresh chess boards and MCTS trees
        for i in range(batch_size):
            board = ChessBoard()
            self.active_games.append({
                'board': board,
                'move_count': 0,
                'history': [],
                'active': True,
                'index': i
            })
    
    def _run_game_batch(self) -> List[List[Dict[str, Any]]]:
        """Run a batch of games to completion, using GPU for MCTS"""
        max_moves = self.config.max_moves_per_game
        
        # Set network to eval mode to ensure consistent inference
        self.network.eval()
        
        game_step = 0
        # Track active games
        active_game_count = len(self.active_games)
        
        # Create a shared MCTS for all games
        mcts = BatchedMCTS(
            self.network, 
            self.config, 
            self.device,
            batch_size=min(64, active_game_count * 4)
        )
        
        # Continue until all games are complete or max steps reached
        max_steps = 500  # Safeguard against infinite loops
        
        while active_game_count > 0 and game_step < max_steps:
            # Print progress every 10 steps
            if game_step % 10 == 0:
                moves_completed = sum(game['move_count'] for game in self.active_games)
                print(f"Step {game_step}, active games: {active_game_count}, total moves: {moves_completed}")
            
            # Collect active boards
            active_boards = []
            active_indices = []
            for i, game in enumerate(self.active_games):
                if game['active'] and game['move_count'] < max_moves:
                    active_boards.append(game['board'])
                    active_indices.append(i)
            
            # If no active boards, break
            if not active_boards:
                break
            
            # Run batch MCTS to get policies
            try:
                with torch.amp.autocast(device_type='cuda', enabled=self.device.type=='cuda'):
                    policies_and_values = mcts.batch_search(active_boards)
                    
                # Verify the result is valid
                if policies_and_values is None:
                    print("Error: batch_search returned None")
                    # Create fallback policies with uniform distribution
                    policies_and_values = [(np.ones(4096)/4096, 0.0) for _ in active_boards]
            except Exception as e:
                print(f"Error in batch search: {e}")
                # Create fallback policies with uniform distribution
                policies_and_values = [(np.ones(4096)/4096, 0.0) for _ in active_boards]
            
            # Apply moves
            for i, game_idx in enumerate(active_indices):
                if i >= len(policies_and_values):
                    print(f"Warning: index {i} out of range for policies_and_values with length {len(policies_and_values)}")
                    continue
                    
                game = self.active_games[game_idx]
                policy, value = policies_and_values[i]
                
                # Save board state before move
                board_state = self.board_encoder.encode_board(game['board'])
                
                # Select move from policy
                try:
                    # Get legal moves
                    legal_moves = []
                    board = game['board']
                    for row in range(8):
                        for col in range(8):
                            from_pos = Position(row, col)
                            piece = board.get_piece(from_pos)
                            if piece and piece.color == board.turn:
                                try:
                                    valid_moves = board.get_valid_moves(from_pos)
                                    for move in valid_moves:
                                        legal_moves.append((from_pos, move.end_pos))
                                except Exception as e:
                                    print(f"Error getting valid moves: {e}")
                    
                    if not legal_moves:
                        print(f"No legal moves for game {game_idx}, turn {board.turn}")
                        game['active'] = False
                        continue
                    
                    # Get move probabilities
                    move_indices = np.array([self.move_encoder.encode_move(from_pos, to_pos) 
                                        for from_pos, to_pos in legal_moves])
                    valid_indices = move_indices >= 0
                    
                    if not np.any(valid_indices):
                        # No valid moves found
                        move_idx = np.random.randint(len(legal_moves))
                        selected_move = legal_moves[move_idx]
                    else:
                        # Select move based on policy
                        valid_moves = [legal_moves[i] for i in range(len(legal_moves)) if valid_indices[i]]
                        valid_probs = policy[move_indices[valid_indices]]
                        
                        # Normalize probabilities
                        if valid_probs.sum() > 0:
                            valid_probs = valid_probs / valid_probs.sum()
                        else:
                            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)
                        
                        # Sample move
                        move_idx = np.random.choice(len(valid_moves), p=valid_probs)
                        selected_move = valid_moves[move_idx]
                    
                    # Apply move
                    from_pos, to_pos = selected_move
                    move_result = board.move_piece(from_pos, to_pos)
                    
                    if move_result:
                        # Add to training data
                        self.game_data[game_idx].append({
                            'state': board_state,
                            'policy': policy,
                            'player': PieceColor.WHITE if game['move_count'] % 2 == 0 else PieceColor.BLACK,
                            'move_count': game['move_count']
                        })
                        
                        # Update move count
                        game['move_count'] += 1
                        print(f"Game {game_idx}: Move {game['move_count']} applied") if game['move_count'] % 10 == 0 else None
                    else:
                        # Invalid move
                        print(f"Game {game_idx}: Invalid move {from_pos} to {to_pos}")
                        game['active'] = False
                except Exception as e:
                    print(f"Error processing game {game_idx}: {e}")
                    traceback.print_exc()
                    game['active'] = False
                
                # Check for game end
                if game['active'] and (self._is_game_over(game['board']) or game['move_count'] >= max_moves):
                    game['active'] = False
                    print(f"Game {game_idx} complete after {game['move_count']} moves")
                    
                    # Get outcome
                    outcome = self._get_game_outcome(game['board'])
                    
                    # Update all examples with outcome
                    for example in self.game_data[game_idx]:
                        if example['player'] == PieceColor.WHITE:
                            example['outcome'] = outcome
                        else:
                            example['outcome'] = -outcome
            
            # Update active count
            active_game_count = sum(1 for game in self.active_games if game['active'])
            game_step += 1
            
            # Clear cache occasionally
            if game_step % 20 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # If we reached max steps, terminate remaining games
        if game_step >= max_steps:
            print(f"Warning: Reached maximum steps ({max_steps}), terminating remaining games")
            for game_idx, game in enumerate(self.active_games):
                if game['active']:
                    game['active'] = False
                    outcome = 0.0  # Draw for timed out games
                    
                    # Update examples with draw outcome
                    for example in self.game_data[game_idx]:
                        if example['player'] == PieceColor.WHITE:
                            example['outcome'] = outcome
                        else:
                            example['outcome'] = -outcome
        
        return self.game_data
    
    def _evaluate_batch(self, mcts_evaluator) -> List[Tuple[np.ndarray, Tuple[Position, Position]]]:
        """Evaluate all active games in batch and return policies and selected moves"""
        active_boards = []
        active_indices = []
        
        # Collect active boards
        for i, game in enumerate(self.active_games):
            if game['active']:
                active_boards.append(game['board'])
                active_indices.append(i)
        
        # If no active boards, return empty results
        if not active_boards:
            return [(None, None)] * len(self.active_games)
        
        # Batch evaluate with MCTS - this is where GPU acceleration happens
        try:
            # FIX: Update deprecated autocast syntax
            with torch.amp.autocast(device_type='cuda', enabled=self.device.type=='cuda'):
                policies_and_values = mcts_evaluator.batch_search(active_boards)
        except Exception as e:
            print(f"Error in batch MCTS: {e}")
            policies_and_values = [(np.ones(4096)/4096, 0.0) for _ in active_boards]  # Fallback
        
        # Process results and select moves
        results = []
        for i in range(len(self.active_games)):
            if i in active_indices:
                idx = active_indices.index(i)
                policy, value = policies_and_values[idx]
                
                # Select move from policy
                board = self.active_games[i]['board']
                game_idx = i  # Pass the game index
                move = self._select_move_from_policy(board, policy, game_idx)
                results.append((policy, move))
            else:
                results.append((None, None))
        
        return results

    def _select_move_from_policy(self, board: ChessBoard, policy: np.ndarray, game_idx: int) -> Tuple[Position, Position]:
        """Select a move based on the policy distribution"""
        # Get legal moves
        legal_moves = []
        for row in range(8):
            for col in range(8):
                from_pos = Position(row, col)
                piece = board.get_piece(from_pos)
                if piece and piece.color == board.turn:
                    try:
                        valid_moves = board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            legal_moves.append((from_pos, move.end_pos))
                    except Exception as e:
                        continue
        
        if not legal_moves:
            return None
        
        # Get move indices
        move_indices = np.array([self.move_encoder.encode_move(from_pos, to_pos) 
                            for from_pos, to_pos in legal_moves])
        valid_indices = move_indices >= 0
        
        if not np.any(valid_indices):
            # No valid moves found in policy
            return legal_moves[np.random.randint(len(legal_moves))]
        
        # Extract probabilities for valid moves
        valid_moves = [legal_moves[i] for i in range(len(legal_moves)) if valid_indices[i]]
        valid_probs = policy[move_indices[valid_indices]]
        
        # Normalize probabilities
        if valid_probs.sum() > 0:
            valid_probs = valid_probs / valid_probs.sum()
        else:
            valid_probs = np.ones(len(valid_moves)) / len(valid_moves)
        
        # Determine temperature based on move count
        move_count = self.active_games[game_idx]['move_count']
        temperature = 1.0 if move_count < self.config.temperature_threshold else 0.0
        
        # Select move based on temperature
        if temperature == 0.0 or len(valid_moves) == 1:
            # Deterministic - select highest probability
            move_idx = np.argmax(valid_probs)
        else:
            # Sample from distribution
            move_idx = np.random.choice(len(valid_moves), p=valid_probs)
            
        return valid_moves[move_idx]
    
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
        except Exception as e:
            print(f"Error getting game outcome: {e}")
            return 0.0  # Default to draw on error