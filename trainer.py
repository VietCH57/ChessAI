import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from collections import deque
import random
from datetime import datetime

from network import AlphaZeroNetwork
from config import AlphaZeroConfig
from selfplay import SelfPlayGenerator, AlphaZeroAI
from headless import HeadlessChessGame

class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training"""
    
    def __init__(self, training_examples: List[Dict[str, Any]]):
        self.examples = training_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return (
            torch.FloatTensor(example['state']),
            torch.FloatTensor(example['policy']),
            torch.FloatTensor([example['outcome']])
        )

class AlphaZeroTrainer:
    """Main AlphaZero training pipeline"""
    
    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.setup_directories()
        
        # Initialize network
        self.network = AlphaZeroNetwork(config)
        if torch.cuda.is_available() and config.device == "cuda":
            self.network = self.network.cuda()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training components
        self.selfplay_generator = SelfPlayGenerator(self.network, config)
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Training stats
        self.iteration = 0
        self.training_history = []
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
    
    def train(self):
        """Main training loop"""
        print("Starting AlphaZero training...")
        print(f"Configuration: {self.config}")
        
        for iteration in range(self.config.num_iterations):
            self.iteration = iteration
            print(f"\n=== Iteration {iteration + 1}/{self.config.num_iterations} ===")
            
            # Generate self-play games
            print("Generating self-play games...")
            training_examples = self.selfplay_generator.generate_games(
                self.config.episodes_per_iteration
            )
            
            # Add to replay buffer
            self.replay_buffer.extend(training_examples)
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            
            # Train network
            print("Training network...")
            train_loss = self.train_network()
            
            # Evaluate new model
            if iteration % self.config.checkpoint_interval == 0:
                print("Evaluating model...")
                win_rate = self.evaluate_model()
                
                # Save model if it's better
                if win_rate > self.config.win_rate_threshold:
                    print(f"New model wins {win_rate:.2%} - saving as best model")
                    self.save_model("best_model.pt")
                else:
                    print(f"New model wins {win_rate:.2%} - keeping old model")
                    self.load_model("best_model.pt")
            
            # Save checkpoint
            self.save_model(f"checkpoint_iter_{iteration}.pt")
            
            # Log progress
            self.training_history.append({
                'iteration': iteration,
                'train_loss': train_loss,
                'buffer_size': len(self.replay_buffer)
            })
            
            self.save_training_history()
    
    def train_network(self) -> float:
        """Train the neural network on replay buffer data"""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        # Sample training batch
        batch_examples = random.sample(list(self.replay_buffer), 
                                     min(len(self.replay_buffer), 
                                         self.config.batch_size * 10))
        
        dataset = AlphaZeroDataset(batch_examples)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, 
                              shuffle=True, num_workers=0)
        
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        for states, target_policies, target_values in dataloader:
            if torch.cuda.is_available() and self.config.device == "cuda":
                states = states.cuda()
                target_policies = target_policies.cuda()
                target_values = target_values.cuda()
            
            # Forward pass
            policy_logits, predicted_values = self.network(states)
            
            # Calculate losses
            value_loss = nn.MSELoss()(predicted_values.squeeze(), target_values.squeeze())
            policy_loss = -torch.sum(target_policies * torch.log_softmax(policy_logits, dim=1)) / states.size(0)
            
            total_loss_batch = value_loss + policy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Average training loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate_model(self) -> float:
        """Evaluate current model against previous best model"""
        if not os.path.exists(os.path.join(self.config.model_dir, "best_model.pt")):
            # First iteration - no previous model to compare against
            return 1.0
        
        # Load previous best model
        prev_network = AlphaZeroNetwork(self.config)
        if torch.cuda.is_available() and self.config.device == "cuda":
            prev_network = prev_network.cuda()
        
        prev_model_path = os.path.join(self.config.model_dir, "best_model.pt")
        prev_network.load_state_dict(torch.load(prev_model_path))
        
        # Create AIs
        current_ai = AlphaZeroAI(self.network, self.config, training_mode=False)
        previous_ai = AlphaZeroAI(prev_network, self.config, training_mode=False)
        
        # Run evaluation games
        game_engine = HeadlessChessGame()
        wins = 0
        total_games = self.config.evaluation_games
        
        for game_num in range(total_games):
            # Alternate colors
            if game_num % 2 == 0:
                white_ai, black_ai = current_ai, previous_ai
            else:
                white_ai, black_ai = previous_ai, current_ai
            
            # Reset move counters
            white_ai.reset_move_count()
            black_ai.reset_move_count()
            
            # Run game
            result = game_engine.run_game(
                white_ai, black_ai, 
                max_moves=self.config.max_moves_per_game,
                collect_data=False
            )
            
            # Check if current model won
            if game_num % 2 == 0:  # Current model was white
                if result['winner'] and result['winner'].value == 'white':
                    wins += 1
                elif result['winner'] is None:  # Draw
                    wins += 0.5
            else:  # Current model was black
                if result['winner'] and result['winner'].value == 'black':
                    wins += 1
                elif result['winner'] is None:  # Draw
                    wins += 0.5
        
        win_rate = wins / total_games
        return win_rate
    
    def save_model(self, filename: str):
        """Save model state"""
        model_path = os.path.join(self.config.model_dir, filename)
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename: str):
        """Load model state"""
        model_path = os.path.join(self.config.model_dir, filename)
        if os.path.exists(model_path):
            self.network.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file not found: {model_path}")
    
    def save_training_history(self):
        """Save training history"""
        history_path = os.path.join(self.config.data_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_training_history(self):
        """Load training history"""
        history_path = os.path.join(self.config.data_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)