import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from headless import HeadlessChessGame
from chess_board import ChessBoard, Position, PieceType, PieceColor
from alpha_zero_player import AlphaZeroPlayer
from alphazero_model import AlphaZeroNetwork
from board_encoder import ChessEncoder
from tqdm import tqdm
import concurrent.futures

class SelfPlayDataset(Dataset):
    """Dataset for AlphaZero self-play data"""
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return (example['state'], example['policy'], example['value'])

class AlphaZeroTrainer:
    """
    AlphaZero training pipeline with self-play, neural network training, and evaluation
    """
    def __init__(self, config=None):
        """
        Initialize the trainer with configuration
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        default_config = {
            # Self-play parameters
            'num_self_play_games': 100,        # Number of self-play games per iteration
            'num_parallel_games': 8,           # Number of parallel self-play games (adjust based on GPU memory)
            'num_simulations': 800,            # MCTS simulations per move during self-play
            'max_moves_per_game': 512,         # Maximum moves per self-play game
            
            # Training parameters
            'batch_size': 2048,                # Batch size for training
            'epochs': 20,                      # Epochs per training iteration
            'learning_rate': 0.001,            # Learning rate
            'weight_decay': 1e-4,              # L2 regularization coefficient
            'num_iterations': 100,             # Total training iterations
            'scheduler': 'cosine',             # Learning rate scheduler ('cosine', 'step', or 'none')
            
            # Network parameters
            'num_res_blocks': 20,              # Residual blocks in the network
            'num_filters': 256,                # Filters in convolutional layers
            
            # Evaluation parameters
            'evaluation_games': 40,            # Number of games for evaluation
            'evaluation_threshold': 0.55,      # Win rate threshold to update best model
            
            # MCTS parameters
            'c_puct': 1.0,                     # Exploration constant in PUCT formula
            'temperature_init': 1.0,           # Initial temperature for move selection
            'temperature_final': 0.25,         # Final temperature after temperature_drop_move
            'temperature_drop_move': 30,       # Move number to drop temperature
            
            # File paths
            'output_dir': 'alphazero_models',  # Directory to save models
            'replay_buffer_size': 500000,      # Maximum number of examples in replay buffer
            
            # CUDA parameters
            'use_cuda': True,                  # Whether to use CUDA for training
            'mixed_precision': True,           # Whether to use mixed precision (FP16) training
            'num_workers': 4,                  # Number of dataloader workers
            'pin_memory': True,                # Pin memory for faster GPU transfer
        }
        
        # Update default config with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Set device
        if self.config['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Training on CUDA: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            self.device = torch.device("cpu")
            print("Training on CPU")
            self.config['mixed_precision'] = False  # Disable mixed precision on CPU
        
        # Initialize neural network
        self.network = AlphaZeroNetwork(
            num_res_blocks=self.config['num_res_blocks'], 
            num_filters=self.config['num_filters'],
            device=self.device
        )
        
        # Initialize best network (copy of the current network)
        self.best_network = AlphaZeroNetwork(
            num_res_blocks=self.config['num_res_blocks'], 
            num_filters=self.config['num_filters'],
            device=self.device
        )
        self.best_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup mixed precision training if enabled
        self.scaler = torch.cuda.amp.GradScaler() if self.config['mixed_precision'] else None
        
        # Initialize replay buffer
        self.replay_buffer = []
        
        # Initialize game engine
        self.game_engine = HeadlessChessGame()
        
        # Initialize encoder
        self.encoder = ChessEncoder()
        
        # Save configuration
        config_file = os.path.join(self.config['output_dir'], 'config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def run_self_play_game(self, game_idx, player_config):
        """Run a single self-play game and return the examples"""
        try:
            # Create AlphaZero player with current network
            player = AlphaZeroPlayer(player_config)
            
            # Set the player's network to the current network state
            if hasattr(self, 'shared_state_dict'):
                player.network.load_state_dict(self.shared_state_dict)
            
            # Run a self-play game
            start_time = time.time()
            result = self.game_engine.run_game(
                white_ai=player,
                black_ai=player,
                max_moves=self.config['max_moves_per_game'],
                collect_data=True
            )
            
            # Get examples from the player
            examples = player.self_play_data
            
            # Print game result
            game_time = time.time() - start_time
            moves = result['moves']
            winner = result['winner'].value if result['winner'] is not None else 'draw'
            
            return {
                'game_idx': game_idx,
                'examples': examples,
                'moves': moves,
                'winner': winner,
                'time': game_time
            }
        except Exception as e:
            print(f"Error in self-play game {game_idx}: {e}")
            return {'game_idx': game_idx, 'examples': [], 'error': str(e)}
    
    # Update the self_play method in AlphaZeroTrainer
    def self_play(self, iteration):
        """
        Conduct self-play games to generate training data
        
        Args:
            iteration: Current training iteration
            
        Returns:
            List of self-play examples
        """
        print(f"\nStarting self-play for iteration {iteration}...")
        
        # Create player configuration
        player_config = {
            'num_simulations': self.config['num_simulations'],
            'c_puct': self.config['c_puct'],
            'temperature': self.config['temperature_init'],
            'num_res_blocks': self.config['num_res_blocks'],
            'num_filters': self.config['num_filters'],
            'exploration_rate': 0.0,  # No random exploration during training
            'use_cuda': self.config['use_cuda']
        }
        
        # Save current network state for sharing with workers
        self.shared_state_dict = self.network.state_dict()
        
        # Try a single self-play game first to validate the setup
        print("Running test self-play game...")
        test_player = AlphaZeroPlayer(player_config)
        test_player.network.load_state_dict(self.shared_state_dict)
        
        test_result = self.game_engine.run_game(
            white_ai=test_player,
            black_ai=test_player,
            max_moves=self.config['max_moves_per_game'],
            collect_data=True
        )
        
        if test_result["moves"] == 0:
            print("WARNING: Test game had 0 moves. There may be an issue with the game logic.")
            print("Trying again with debug enabled...")
            
            # Run a debug game with additional logging
            debug_result = self.game_engine.run_game(
                white_ai=test_player,
                black_ai=test_player,
                max_moves=self.config['max_moves_per_game'],
                collect_data=True
            )
            
            if debug_result["moves"] == 0:
                print("CRITICAL: Self-play is failing with 0 moves consistently.")
                print("Reducing simulations and trying simplified self-play...")
                
                # Try with drastically reduced simulations as a last resort
                test_player.config['num_simulations'] = 100
                final_attempt = self.game_engine.run_game(
                    white_ai=test_player,
                    black_ai=test_player,
                    max_moves=self.config['max_moves_per_game'],
                    collect_data=True
                )
                
                if final_attempt["moves"] == 0:
                    print("CRITICAL FAILURE: Unable to generate valid self-play games.")
                    print("Returning empty example list - check game logic or AI implementation.")
                    return []
        
        # If we made it here, self-play is working (or at least the test game worked)
        all_examples = test_player.self_play_data
        
        # Continue with parallel self-play if the test game was successful
        num_games = max(0, self.config['num_self_play_games'] - 1)  # Subtract the test game
        parallel_games = min(self.config['num_parallel_games'], num_games)
        
        if num_games > 0:
            print(f"Running remaining {num_games} self-play games in parallel...")
            
            # Use ThreadPoolExecutor for parallel self-play games
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_games) as executor:
                # Submit all games
                futures = [
                    executor.submit(self.run_self_play_game, game_idx, player_config)
                    for game_idx in range(num_games)
                ]
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if 'error' in result:
                        print(f"Game {result['game_idx']} error: {result['error']}")
                        continue
                        
                    # Add examples from this game
                    all_examples.extend(result['examples'])
                    
                    # Log game summary
                    print(f"Game {result['game_idx']+1}: {result['moves']} moves, "
                        f"Winner: {result['winner']}, Time: {result['time']:.1f}s")
        
        print(f"Self-play completed with {len(all_examples)} training examples")
        return all_examples
    
    def train_network(self, examples):
        """
        Train the neural network on examples
        
        Args:
            examples: List of self-play examples
            
        Returns:
            Training loss
        """
        print("\nTraining neural network...")
        
        # Prepare dataset and dataloader
        dataset = SelfPlayDataset(examples)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        # Set network to training mode
        self.network.train()
        
        # Create learning rate scheduler
        if self.config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['epochs']
            )
        elif self.config['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        else:
            scheduler = None
        
        # Train the network for multiple epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        batch_count = 0
        
        # Create progress bar for epochs
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_batches = 0
            
            # Create progress bar for batches
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for states, policies, values in pbar:
                # Move data to device and convert types
                states = states.float().to(self.device)
                policies = policies.float().to(self.device)
                values = values.float().unsqueeze(1).to(self.device)  # Add batch dimension
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Mixed precision forward pass
                if self.config['mixed_precision']:
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        policy_logits, value_pred = self.network(states)
                        
                        # Calculate loss components
                        policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
                        value_loss = F.mse_loss(value_pred, values)
                        
                        # Total loss (weighted sum of policy and value losses)
                        loss = policy_loss + value_loss
                        
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    # Forward pass
                    policy_logits, value_pred = self.network(states)
                    
                    # Calculate loss components
                    policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
                    value_loss = F.mse_loss(value_pred, values)
                    
                    # Total loss (weighted sum of policy and value losses)
                    loss = policy_loss + value_loss
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': epoch_loss / epoch_batches,
                    'policy': epoch_policy_loss / epoch_batches,
                    'value': epoch_value_loss / epoch_batches
                })
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Update total statistics
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
            avg_epoch_policy_loss = epoch_policy_loss / epoch_batches if epoch_batches > 0 else 0
            avg_epoch_value_loss = epoch_value_loss / epoch_batches if epoch_batches > 0 else 0
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}, "
                  f"Loss: {avg_epoch_loss:.4f}, "
                  f"Policy: {avg_epoch_policy_loss:.4f}, "
                  f"Value: {avg_epoch_value_loss:.4f}")
            
            total_loss += avg_epoch_loss
            total_policy_loss += avg_epoch_policy_loss
            total_value_loss += avg_epoch_value_loss
            batch_count += 1
        
        # Return average losses across all epochs
        return {
            'loss': total_loss / self.config['epochs'],
            'policy_loss': total_policy_loss / self.config['epochs'],
            'value_loss': total_value_loss / self.config['epochs']
        }
    
    def evaluate(self):
        """
        Evaluate the current network against the best network
        
        Returns:
            float: Win rate of current network against best network
        """
        print("\nEvaluating current network against best network...")
        
        # Create players with current and best networks
        current_player_config = {
            'num_simulations': self.config['num_simulations'],
            'c_puct': self.config['c_puct'],
            'temperature': 0.0,  # Use best move during evaluation
            'num_res_blocks': self.config['num_res_blocks'],
            'num_filters': self.config['num_filters'],
            'use_cuda': self.config['use_cuda']
        }
        
        best_player_config = current_player_config.copy()
        
        current_player = AlphaZeroPlayer(current_player_config)
        current_player.network = self.network
        
        best_player = AlphaZeroPlayer(best_player_config)
        best_player.network = self.best_network
        
        # Play evaluation games
        win_count = 0
        draw_count = 0
        loss_count = 0
        
        # Create a progress bar
        pbar = tqdm(range(self.config['evaluation_games']), desc="Evaluation games")
        for game_idx in pbar:
            # Alternate colors to ensure fairness
            if game_idx % 2 == 0:
                white_ai = current_player
                black_ai = best_player
                current_is_white = True
            else:
                white_ai = best_player
                black_ai = current_player
                current_is_white = False
            
            # Run the game
            result = self.game_engine.run_game(
                white_ai=white_ai,
                black_ai=black_ai,
                max_moves=self.config['max_moves_per_game']
            )
            
            # Determine result from current player's perspective
            if result['winner'] is None:  # Draw
                draw_count += 1
                result_text = "Draw"
            elif (result['winner'] == PieceColor.WHITE and current_is_white) or \
                 (result['winner'] == PieceColor.BLACK and not current_is_white):
                win_count += 1
                result_text = "Win"
            else:
                loss_count += 1
                result_text = "Loss"
            
            # Update progress bar
            pbar.set_postfix({
                'W': win_count,
                'D': draw_count,
                'L': loss_count,
                'last': result_text
            })
        
        # Calculate win rate (counting draws as 0.5 wins)
        win_rate = (win_count + 0.5 * draw_count) / self.config['evaluation_games']
        
        print(f"Evaluation results: {win_count} wins, {draw_count} draws, {loss_count} losses")
        print(f"Win rate: {win_rate:.2f}")
        
        return win_rate
    
    def update_best_network(self):
        """Update the best network with the current network"""
        self.best_network.load_state_dict(self.network.state_dict())
        print("Best network updated with current network weights")
    
    def save_checkpoint(self, iteration, is_best=False):
        """Save training checkpoint"""
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        current_model_path = os.path.join(checkpoint_dir, f'model_iter_{iteration}.pt')
        best_model_path = os.path.join(self.config['output_dir'], 'best_model.pt')
        latest_model_path = os.path.join(self.config['output_dir'], 'latest_model.pt')
        
        # Save current model
        self.network.save_checkpoint(
            current_model_path,
            optimizer=self.optimizer,
            iteration=iteration,
            config=self.config
        )
        
        # Save as latest model for convenience
        self.network.save_checkpoint(
            latest_model_path,
            optimizer=self.optimizer,
            iteration=iteration,
            config=self.config
        )
        
        # Save best model if indicated
        if is_best:
            self.best_network.save_checkpoint(
                best_model_path,
                iteration=iteration,
                config=self.config
            )
            print(f"New best model saved at iteration {iteration}")
        
        print(f"Checkpoint saved at iteration {iteration}")
    
    def load_checkpoint(self, model_path):
        """Load training checkpoint"""
        try:
            # Load model
            iteration, loaded_config = self.network.load_checkpoint(model_path, optimizer=self.optimizer)
            
            # Update config from loaded checkpoint
            if loaded_config:
                for key, value in loaded_config.items():
                    if key in self.config:
                        self.config[key] = value
            
            # Load best model if available (same location with 'best_' prefix)
            best_model_path = os.path.join(os.path.dirname(model_path), 'best_model.pt')
            if os.path.exists(best_model_path):
                self.best_network.load_checkpoint(best_model_path)
            else:
                # If no best model, use the loaded model as best
                self.best_network.load_state_dict(self.network.state_dict())
            
            print(f"Checkpoint loaded from {model_path} (iteration {iteration})")
            return iteration
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return 0
    
    def train(self, start_iteration=0):
        """
        Run the full AlphaZero training pipeline
        
        Args:
            start_iteration: Starting iteration number (for resuming training)
            
        Returns:
            The trained best network
        """
        print("Starting AlphaZero training pipeline...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Main training loop
        for iteration in range(start_iteration, self.config['num_iterations']):
            print(f"\n{'='*50}")
            print(f"Starting iteration {iteration+1}/{self.config['num_iterations']}")
            print(f"{'='*50}")
            
            # Self-play phase
            examples = self.self_play(iteration)
            
            # Add examples to replay buffer
            self.replay_buffer.extend(examples)
            
            # Limit replay buffer size
            if len(self.replay_buffer) > self.config['replay_buffer_size']:
                # Remove oldest examples
                excess = len(self.replay_buffer) - self.config['replay_buffer_size']
                self.replay_buffer = self.replay_buffer[excess:]
            
            # Sample from replay buffer for training
            if len(self.replay_buffer) > self.config['batch_size']:
                sample_size = min(len(self.replay_buffer), self.config['batch_size'] * 30)
                train_examples = random.sample(self.replay_buffer, sample_size)
                
                # Train network
                loss_dict = self.train_network(train_examples)
                print(f"Training completed with average loss: {loss_dict['loss']:.4f} "
                      f"(Policy: {loss_dict['policy_loss']:.4f}, Value: {loss_dict['value_loss']:.4f})")
            
            # Evaluate and update best network
            if (iteration + 1) % 1 == 0:  # Evaluate every iteration
                win_rate = self.evaluate()
                
                is_best = False
                if win_rate >= self.config['evaluation_threshold']:
                    print(f"New best network with win rate {win_rate:.2f}")
                    self.update_best_network()
                    is_best = True
                else:
                    print(f"Current network not better than best network (win rate: {win_rate:.2f})")
            
            # Save checkpoint
            self.save_checkpoint(iteration + 1, is_best)
        
        print("\nTraining completed!")
        return self.best_network