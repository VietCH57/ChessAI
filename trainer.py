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
import time
from datetime import datetime, timedelta

# TPU imports - kiểm tra và import nếu có thể
USE_TPU = os.environ.get('COLAB_TPU_ADDR') is not None
if USE_TPU:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        print("TPU available, configuring PyTorch XLA")
    except ImportError:
        print("PyTorch XLA not available. Install with: pip install torch_xla")
        USE_TPU = False

from network import AlphaZeroNetwork
from config import AlphaZeroConfig
from selfplay import SelfPlayGenerator, AlphaZeroAI
from headless import HeadlessChessGame

class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training with batching optimizations"""
    
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
    """Main AlphaZero training pipeline with optimizations"""
    
    def __init__(self, config: AlphaZeroConfig):
        self.config = config
        self.setup_directories()
        
        # Initialize network
        self.network = AlphaZeroNetwork(config)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Move network to correct device - with TPU support
        if USE_TPU:
            self.device = xm.xla_device()
            print(f"Using TPU device: {self.device}")
        else:
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        self.network = self.network.to(self.device)
        
        # Enable cuDNN benchmarking for faster training (only for GPU)
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        # Initialize scheduler for adaptive learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)
        
        # Training components
        self.selfplay_generator = SelfPlayGenerator(self.network, config)
        self.replay_buffer = deque(maxlen=config.replay_buffer_size)
        
        # Training stats
        self.iteration = 0
        self.training_history = []
        self.start_time = time.time()
        
        # Try to load previous model
        if os.path.exists(os.path.join(config.model_dir, "best_model.pt")):
            print("Loading previous best model...")
            self.load_model("best_model.pt")
            
        # Data augmentation
        self.use_augmentation = True
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
    
    def train(self):
        """Main training loop with optimizations"""
        print("Starting AlphaZero training...")
        print(f"Device: {self.device}")
        print(f"Configuration: {self.config}")
        
        for iteration in range(self.iteration, self.config.num_iterations):
            self.iteration = iteration
            iter_start_time = time.time()
            print(f"\n=== Iteration {iteration + 1}/{self.config.num_iterations} ===")
            
            # Generate self-play games
            print("Generating self-play games...")
            try:
                if self.config.num_workers > 1 and hasattr(self.selfplay_generator, 'generate_games_parallel'):
                    # Use parallel generation if available
                    training_examples = self.selfplay_generator.generate_games_parallel(
                        self.config.episodes_per_iteration,
                        num_processes=self.config.num_workers
                    )
                else:
                    # Fallback to sequential generation
                    training_examples = self.selfplay_generator.generate_games(
                        self.config.episodes_per_iteration
                    )
            except Exception as e:
                print(f"Error in parallel generation: {e}. Falling back to sequential mode.")
                import traceback
                traceback.print_exc()
                # Emergency fallback
                training_examples = self.selfplay_generator.generate_games(
                    self.config.episodes_per_iteration
                )
            
            # Skip training if no examples generated
            if not training_examples:
                print("No training examples generated. Skipping iteration.")
                continue
            
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
            self.save_model("last_checkpoint.pt")
            
            # Calculate iteration time
            iter_time = time.time() - iter_start_time
            
            # Log progress
            self.training_history.append({
                'iteration': iteration,
                'train_loss': train_loss,
                'buffer_size': len(self.replay_buffer),
                'iter_time': iter_time,
                'total_time': time.time() - self.start_time
            })
            
            # Calculate and show ETA
            avg_iter_time = iter_time
            remaining_iters = self.config.num_iterations - (iteration + 1)
            eta_seconds = avg_iter_time * remaining_iters
            eta = timedelta(seconds=int(eta_seconds))
            
            print(f"Iteration {iteration + 1} completed in {iter_time/60:.1f}m")
            print(f"Estimated time remaining: {eta}")
            
            self.save_training_history()
            
            # Update learning rate based on loss
            self.scheduler.step(train_loss)
    
    def train_network(self) -> float:
        """Train the neural network with optimized data handling and TPU support"""
        if len(self.replay_buffer) < self.config.batch_size:
            return 0.0
        
        # Sample training data
        num_samples = min(len(self.replay_buffer), 2048)  # Limit to reasonable amount
        batch_examples = random.sample(list(self.replay_buffer), num_samples)
        
        # Data augmentation
        if self.use_augmentation:
            augmented_examples = self._augment_training_data(batch_examples)
            batch_examples.extend(augmented_examples)
        
        # Create dataset and dataloader
        dataset = AlphaZeroDataset(batch_examples)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=0,  # TPU requires num_workers=0
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Wrap dataloader with ParallelLoader for TPU
        if USE_TPU:
            dataloader = pl.ParallelLoader(dataloader, [self.device]).per_device_loader(self.device)
        
        # Training loop
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        # Track loss components
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        # Number of epochs over the data
        num_epochs = 1
        
        for epoch in range(num_epochs):
            for states, target_policies, target_values in dataloader:
                # Move tensors to device - non_blocking for better performance
                states = states.to(self.device, non_blocking=True)
                target_policies = target_policies.to(self.device, non_blocking=True)
                target_values = target_values.to(self.device, non_blocking=True)
                
                # Forward pass
                policy_logits, predicted_values = self.network(states)
                
                # Calculate losses
                value_loss = nn.MSELoss()(predicted_values.squeeze(), target_values.squeeze())
                policy_loss = -torch.sum(target_policies * torch.log_softmax(policy_logits, dim=1)) / states.size(0)
                
                # Combined loss
                total_loss_batch = value_loss + policy_loss
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                
                # Clip gradients to prevent explosions
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                
                # Optimizer step with TPU sync
                if USE_TPU:
                    xm.optimizer_step(self.optimizer)
                    xm.mark_step()  # Important for TPU performance
                else:
                    self.optimizer.step()
                
                # Track losses
                total_loss += total_loss_batch.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1
        
        # Calculate average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_policy_loss = policy_loss_sum / max(num_batches, 1)
        avg_value_loss = value_loss_sum / max(num_batches, 1)
        
        print(f"Training: Loss={avg_loss:.4f} (Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f})")
        return avg_loss
    
    def _augment_training_data(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply simple horizontal flip data augmentation"""
        augmented = []
        
        # Only augment a portion of the data
        num_to_augment = len(examples) // 4
        for i in range(num_to_augment):
            example = examples[i]
            
            # Create a copy for augmentation
            new_example = {
                'player': example['player'],
                'move_count': example['move_count'],
                'outcome': example['outcome']
            }
            
            # Horizontal flip of board state
            state = example['state'].copy()
            # Flip the board planes
            for plane_idx in range(state.shape[0]):
                state[plane_idx] = np.fliplr(state[plane_idx])
            new_example['state'] = state
            
            # Horizontal flip of policy
            policy = example['policy'].copy()
            flipped_policy = np.zeros_like(policy)
            
            # Remap each move
            for from_row in range(8):
                for from_col in range(8):
                    for to_row in range(8):
                        for to_col in range(8):
                            # Original move index
                            orig_idx = from_row * 8 * 8 * 8 + from_col * 8 * 8 + to_row * 8 + to_col
                            
                            # Flipped move index (flip columns)
                            flipped_from_col = 7 - from_col
                            flipped_to_col = 7 - to_col
                            flipped_idx = from_row * 8 * 8 * 8 + flipped_from_col * 8 * 8 + to_row * 8 + flipped_to_col
                            
                            flipped_policy[flipped_idx] = policy[orig_idx]
            
            new_example['policy'] = flipped_policy
            augmented.append(new_example)
            
        return augmented
    
    def evaluate_model(self) -> float:
        """Evaluate current model against previous best model"""
        # Skip if no previous model
        if not os.path.exists(os.path.join(self.config.model_dir, "best_model.pt")):
            return 1.0
        
        # Load previous best model
        prev_network = AlphaZeroNetwork(self.config)
        prev_network = prev_network.to(self.device)
        
        prev_model_path = os.path.join(self.config.model_dir, "best_model.pt")
        prev_network.load_state_dict(torch.load(prev_model_path, map_location=self.device))
        
        # Create AIs
        current_ai = AlphaZeroAI(self.network, self.config, training_mode=False)
        previous_ai = AlphaZeroAI(prev_network, self.config, training_mode=False)
        
        # Run evaluation games
        game_engine = HeadlessChessGame()
        wins = 0
        total_games = self.config.evaluation_games
        
        # Track game results
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        
        for game_num in range(total_games):
            # Alternate colors
            if game_num % 2 == 0:
                white_ai, black_ai = current_ai, previous_ai
            else:
                white_ai, black_ai = previous_ai, current_ai
            
            # Reset move counters
            white_ai.reset_move_count()
            black_ai.reset_move_count()
            
            # Run game with reduced simulation count for speed
            config_backup = self.config.num_simulations
            self.config.num_simulations = max(200, self.config.num_simulations // 2)
            
            result = game_engine.run_game(
                white_ai, black_ai, 
                max_moves=self.config.max_moves_per_game,
                collect_data=False
            )
            
            # Restore simulation count
            self.config.num_simulations = config_backup
            
            # Track results
            if result['winner'] is None:
                # Draw
                results['draws'] += 1
                wins += 0.5
            elif (game_num % 2 == 0 and result['winner'].value == 'white') or \
                 (game_num % 2 == 1 and result['winner'].value == 'black'):
                # Current model won
                results['wins'] += 1
                wins += 1
            else:
                # Current model lost
                results['losses'] += 1
            
            # Display progress
            win_rate = wins / (game_num + 1)
            print(f"Evaluation: Game {game_num+1}/{total_games}, current win rate: {win_rate:.2%}")
        
        # Calculate final win rate
        win_rate = wins / total_games
        print(f"Evaluation complete: Win rate: {win_rate:.2%} (W:{results['wins']}, L:{results['losses']}, D:{results['draws']})")
        return win_rate
    
    def save_model(self, filename: str):
        """Save model state"""
        # Make sure the directory exists
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Fix potential path issues by normalizing the filename
        if os.path.sep in filename:
            # Extract just the basename if a path was provided
            filename = os.path.basename(filename)
        
        model_path = os.path.join(self.config.model_dir, filename)
        torch.save(self.network.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename: str):
        """Load model state"""
        model_path = os.path.join(self.config.model_dir, filename)
        if os.path.exists(model_path):
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
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
                
                # Update iteration if history exists
                if self.training_history:
                    self.iteration = self.training_history[-1]['iteration'] + 1