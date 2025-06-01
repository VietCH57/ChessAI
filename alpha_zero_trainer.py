import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from headless import HeadlessChessGame
from chess_board import ChessBoard, Position, PieceType, PieceColor
from alpha_zero_player import AlphaZeroPlayer
from alphazero_model import AlphaZeroNetwork
from board_encoder import ChessEncoder

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
            'num_self_play_games': 100,         # Number of self-play games per iteration
            'num_simulations': 800,             # MCTS simulations per move during self-play
            'max_moves_per_game': 512,          # Maximum moves per self-play game
            
            # Training parameters
            'batch_size': 1024,                # Batch size for training
            'epochs': 20,                      # Epochs per training iteration
            'learning_rate': 0.001,            # Learning rate
            'weight_decay': 1e-4,              # L2 regularization coefficient
            'num_iterations': 100,             # Total training iterations
            
            # Network parameters
            'num_res_blocks': 20,              # Residual blocks in the network
            'num_filters': 256,                # Filters in convolutional layers
            
            # Evaluation parameters
            'evaluation_games': 40,             # Number of games for evaluation
            'evaluation_threshold': 0.55,       # Win rate threshold to update best model
            
            # MCTS parameters
            'c_puct': 1.0,                     # Exploration constant in PUCT formula
            'temperature_init': 1.0,            # Initial temperature for move selection
            'temperature_final': 0.25,          # Final temperature after temperature_drop_move
            'temperature_drop_move': 30,        # Move number to drop temperature
            
            # File paths
            'output_dir': 'alphazero_models',   # Directory to save models
            'replay_buffer_size': 200000,       # Maximum number of examples in replay buffer
        }
        
        # Update default config with provided config
        self.config = default_config.copy()
        if config:
            self.config.update(config)
            
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize neural network
        self.network = AlphaZeroNetwork(
            num_res_blocks=self.config['num_res_blocks'], 
            num_filters=self.config['num_filters']
        )
        
        # Initialize best network (copy of the current network)
        self.best_network = AlphaZeroNetwork(
            num_res_blocks=self.config['num_res_blocks'], 
            num_filters=self.config['num_filters']
        )
        self.best_network.load_state_dict(self.network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
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
    
    def self_play(self, iteration):
        """
        Conduct self-play games to generate training data
        
        Args:
            iteration: Current training iteration
            
        Returns:
            List of self-play examples
        """
        print(f"\nStarting self-play for iteration {iteration}...")
        
        # Create AlphaZero player with current network
        player_config = {
            'num_simulations': self.config['num_simulations'],
            'c_puct': self.config['c_puct'],
            'temperature': self.config['temperature_init'],
            'num_res_blocks': self.config['num_res_blocks'],
            'num_filters': self.config['num_filters'],
            'exploration_rate': 0.0  # No random exploration during training
        }
        
        player = AlphaZeroPlayer(player_config)
        player.network = self.network  # Use current network
        
        # Initialize storage for self-play examples
        examples = []
        
        # Conduct self-play games
        for game_idx in range(self.config['num_self_play_games']):
            print(f"Self-play game {game_idx+1}/{self.config['num_self_play_games']}")
            start_time = time.time()
            
            # Run a self-play game
            result = self.game_engine.run_game(
                white_ai=player,
                black_ai=player,
                max_moves=self.config['max_moves_per_game'],
                collect_data=True
            )
            
            # Get examples from the player
            examples.extend(player.self_play_data)
            player.self_play_data = []  # Clear the player's data
            
            # Print game result
            game_time = time.time() - start_time
            moves = result['moves']
            winner = result['winner'].value if result['winner'] is not None else 'draw'
            print(f"Game completed in {moves} moves ({game_time:.1f}s). Result: {winner}")
        
        print(f"Self-play completed with {len(examples)} training examples")
        return examples
    
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
            num_workers=4
        )
        
        # Set network to training mode
        self.network.train()
        
        # Train the network for multiple epochs
        total_loss = 0
        for epoch in range(self.config['epochs']):
            epoch_loss = 0
            batch_count = 0
            
            for states, policies, values in dataloader:
                # Convert to appropriate format
                states = states.float()
                policies = policies.float()
                values = values.float().unsqueeze(1)  # Add batch dimension
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_logits, value_pred = self.network(states)
                
                # Calculate loss components
                policy_loss = -torch.mean(torch.sum(policies * F.log_softmax(policy_logits, dim=1), dim=1))
                value_loss = F.mse_loss(value_pred, values)
                
                # Total loss (weighted sum of policy and value losses, plus L2 regularization from optimizer)
                loss = policy_loss + value_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                batch_count += 1
            
            # Epoch statistics
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            print(f"Epoch {epoch+1}/{self.config['epochs']}, Loss: {avg_epoch_loss:.4f}")
            
            total_loss += avg_epoch_loss
        
        # Return average loss across all epochs
        return total_loss / self.config['epochs']
    
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
            'num_filters': self.config['num_filters']
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
        
        for game_idx in range(self.config['evaluation_games']):
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
                print(f"Game {game_idx+1}: Draw")
            elif (result['winner'] == PieceColor.WHITE and current_is_white) or \
                 (result['winner'] == PieceColor.BLACK and not current_is_white):
                win_count += 1
                print(f"Game {game_idx+1}: Current network won")
            else:
                loss_count += 1
                print(f"Game {game_idx+1}: Best network won")
        
        # Calculate win rate (counting draws as 0.5 wins)
        win_rate = (win_count + 0.5 * draw_count) / self.config['evaluation_games']
        
        print(f"Evaluation results: {win_count} wins, {draw_count} draws, {loss_count} losses")
        print(f"Win rate: {win_rate:.2f}")
        
        return win_rate
    
    def update_best_network(self):
        """Update the best network with the current network"""
        self.best_network.load_state_dict(self.network.state_dict())
        print("Best network updated with current network weights")
    
    def save_checkpoint(self, iteration):
        """Save training checkpoint"""
        current_model_path = os.path.join(self.config['output_dir'], f'model_iter_{iteration}.pt')
        best_model_path = os.path.join(self.config['output_dir'], 'best_model.pt')
        
        # Save current model
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, current_model_path)
        
        # Save best model
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.best_network.state_dict(),
            'config': self.config
        }, best_model_path)
        
        print(f"Checkpoint saved at iteration {iteration}")
    
    def load_checkpoint(self, model_path):
        """Load training checkpoint"""
        checkpoint = torch.load(model_path)
        
        # Load model state
        self.network.load_state_dict(checkpoint['model_state_dict'])
        
        # Load best model if available
        if 'best_model_state_dict' in checkpoint:
            self.best_network.load_state_dict(checkpoint['best_model_state_dict'])
        else:
            self.best_network.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Update config from checkpoint if available
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        # Return the iteration number
        return checkpoint.get('iteration', 0)
    
    def train(self):
        """
        Run the full AlphaZero training pipeline
        """
        print("Starting AlphaZero training pipeline...")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Main training loop
        for iteration in range(self.config['num_iterations']):
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
                loss = self.train_network(train_examples)
                print(f"Training completed with average loss: {loss:.4f}")
            
            # Evaluate and update best network
            if (iteration + 1) % 1 == 0:  # Evaluate every iteration
                win_rate = self.evaluate()
                
                if win_rate >= self.config['evaluation_threshold']:
                    print(f"New best network with win rate {win_rate:.2f}")
                    self.update_best_network()
                else:
                    print(f"Current network not better than best network (win rate: {win_rate:.2f})")
            
            # Save checkpoint
            self.save_checkpoint(iteration + 1)
        
        print("\nTraining completed!")
        return self.best_network