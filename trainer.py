import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import time
import json
import traceback
from collections import deque
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple
from collections import deque
import random
import time
from datetime import datetime, timedelta
from mcts import AlphaZeroMCTS, BatchedMCTS
from selfplay import SelfPlayGenerator, AlphaZeroAI, GPUParallelSelfPlayGenerator

# Enhanced CUDA configuration
USE_TPU = os.environ.get('COLAB_TPU_ADDR') is not None
USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True
    
    # Set memory allocation to avoid fragmentation
    if hasattr(torch.cuda, 'memory_stats'):
        print(f"CUDA available: {torch.cuda.device_count()} devices")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    
    # Try to ensure CUDA operations are asynchronous when possible
    torch.cuda.set_device(0)  # Set default device

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
    """Dataset for AlphaZero training with GPU optimizations"""
    
    def __init__(self, training_examples: List[Dict[str, Any]], device=None):
        self.examples = training_examples
        self.device = device
        
        # Pre-process data in batches for faster GPU loading
        if len(training_examples) > 0 and device is not None and device.type == 'cuda':
            self.pre_process()
    
    def pre_process(self):
        """Pre-process data in batches to GPU for faster loading"""
        # Maximum batch size for pre-processing
        batch_size = 1024
        
        # Process data in batches to avoid memory issues
        for start_idx in range(0, len(self.examples), batch_size):
            end_idx = min(start_idx + batch_size, len(self.examples))
            batch = self.examples[start_idx:end_idx]
            
            # Convert to tensors and move to device
            states = torch.stack([torch.FloatTensor(ex['state']) for ex in batch])
            policies = torch.stack([torch.FloatTensor(ex['policy']) for ex in batch])
            outcomes = torch.FloatTensor([ex['outcome'] for ex in batch]).unsqueeze(1)
            
            # Move to device
            if self.device.type == 'cuda':
                states = states.pin_memory()
                policies = policies.pin_memory()
                outcomes = outcomes.pin_memory()
            
            # Update examples with pre-processed tensors
            for i, idx in enumerate(range(start_idx, end_idx)):
                self.examples[idx]['state_tensor'] = states[i]
                self.examples[idx]['policy_tensor'] = policies[i]
                self.examples[idx]['outcome_tensor'] = outcomes[i]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Use pre-processed tensors if available
        if 'state_tensor' in example:
            return (
                example['state_tensor'],
                example['policy_tensor'],
                example['outcome_tensor']
            )
        
        # Otherwise convert on-the-fly
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
        
        # CUDA configuration and optimization
        if USE_TPU:
            self.device = xm.xla_device()
            print(f"Using TPU device: {self.device}")
        else:
            if USE_CUDA:
                # Set up CUDA device with optimizations
                torch.cuda.empty_cache()
                
                # Get GPU device properties
                device_id = 0
                device_props = torch.cuda.get_device_properties(device_id)
                print(f"Using GPU: {device_props.name} with {device_props.total_memory/1024**2:.0f}MB memory")
                
                # Set memory allocation strategy for MCTS
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    # Reserve memory for MCTS batch inference
                    torch.cuda.set_per_process_memory_fraction(0.8, device_id)
                
                self.device = torch.device(f"cuda:{device_id}")
            else:
                self.device = torch.device("cpu")
        
        self.network = self.network.to(self.device)
        
        # Enable cuDNN benchmarking for faster training
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            
        # Initialize scheduler for adaptive learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)
        
        # Configure selfplay with CUDA optimizations
        self.selfplay_generator = SelfPlayGenerator(
            self.network, 
            config,
            device=self.device,
            use_cuda_mcts=self.device.type == 'cuda'  # Enable CUDA MCTS
        )
        
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
        
        # For CUDA optimization, create a pool of inference batches
        self.mcts_batch_size = 64 if self.device.type == 'cuda' else 1
        print(f"MCTS batch size: {self.mcts_batch_size}")
        
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        os.makedirs(self.config.data_dir, exist_ok=True)
    
    def train(self):
        """Main training loop with improved error handling"""
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
                if hasattr(self.config, 'use_gpu_parallel') and self.config.use_gpu_parallel:
                    # Use GPU parallel self-play generator
                    gpu_generator = GPUParallelSelfPlayGenerator(
                        self.network, 
                        self.config, 
                        device=self.device,
                        num_parallel_games=self.config.parallel_games
                    )
                    training_examples = gpu_generator.generate_games(self.config.episodes_per_iteration)
                else:
                    # Use standard self-play generator
                    training_examples = self.selfplay_generator.generate_games(self.config.episodes_per_iteration)
                    
            except Exception as e:
                print(f"Error in self-play generation: {e}")
                traceback.print_exc()
                training_examples = []
            
            # Skip training if no examples generated
            if not training_examples:
                print("No training examples generated, skipping iteration")
                continue
            
            # Add to replay buffer
            self.replay_buffer.extend(training_examples)
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            
            # Train neural network
            avg_loss = self.train_network()
            
            # SAVE CHECKPOINT FIRST - before evaluation
            if (iteration + 1) % self.config.checkpoint_interval == 0:
                try:
                    self.save_checkpoint(iteration + 1)
                    print(f"Checkpoint {iteration + 1} saved successfully")
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                    traceback.print_exc()
                
                # THEN evaluate model
                try:
                    win_rate = self.evaluate_model()
                    
                    if win_rate >= self.config.win_rate_threshold:
                        self.save_best_model()
                        print(f"New best model saved (win rate: {win_rate:.3f})")
                        
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    traceback.print_exc()
                    print("Continuing training without evaluation...")
            
            # Record training stats
            elapsed_time = time.time() - iter_start_time
            stats = {
                'iteration': iteration,
                'examples_generated': len(training_examples),
                'replay_buffer_size': len(self.replay_buffer),
                'avg_loss': avg_loss,
                'time_seconds': elapsed_time
            }
            self.training_history.append(stats)
            
            # Save training history
            self.save_training_history()
            
            # Display progress
            total_time = time.time() - self.start_time
            examples_per_second = sum(stat['examples_generated'] for stat in self.training_history) / total_time
            print(f"Iteration {iteration + 1} completed in {elapsed_time:.1f}s")
            print(f"Total training time: {total_time/3600:.2f} hours")
            print(f"Examples/second: {examples_per_second:.1f}")
            print(f"Estimated completion: {(self.config.num_iterations - iteration - 1) * elapsed_time / 3600:.2f} hours")
    
    def train_network(self) -> float:
        """Train the neural network with GPU-optimized data handling"""
        print(f"Starting network training with {len(self.replay_buffer)} examples in replay buffer")
        
        if len(self.replay_buffer) < self.config.batch_size:
            print("Not enough examples for training")
            return 0.0
        
        # Sample training data
        num_samples = min(len(self.replay_buffer), 8192)  # Increased batch size for GPU
        batch_examples = random.sample(list(self.replay_buffer), num_samples)
        print(f"Sampled {len(batch_examples)} examples for training")
        
        # Log example structure for debugging
        print(f"Example structure: {list(batch_examples[0].keys())}")
        print(f"State shape: {batch_examples[0]['state'].shape}")
        print(f"Policy shape: {batch_examples[0]['policy'].shape}")
        
        # Data augmentation on GPU if possible
        if self.use_augmentation:
            if self.device.type == 'cuda':
                print("Performing GPU-accelerated data augmentation...")
                augmented_examples = self._augment_training_data_gpu(batch_examples)
            else:
                print("Performing CPU data augmentation...")
                augmented_examples = self._augment_training_data(batch_examples)
            batch_examples.extend(augmented_examples)
            print(f"Data augmentation added {len(augmented_examples)} examples, total: {len(batch_examples)}")
        
        # Create dataset and dataloader with device info
        print("Creating dataset and dataloader...")
        dataset = AlphaZeroDataset(batch_examples, device=self.device)
        
        # Configure dataloader for maximum GPU utilization
        num_workers = 0 if USE_TPU else (0 if self.device.type == 'cuda' else 4)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=False if USE_TPU else (num_workers > 0)
        )
        print(f"Created dataloader with {len(dataloader)} batches, batch size: {self.config.batch_size}")
        
        # Training loop
        self.network.train()
        total_loss = 0.0
        num_batches = 0
        
        # Track loss components
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        
        # Number of epochs over the data
        num_epochs = 1
        
        # Enable mixed precision training for faster computation on compatible GPUs
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        if scaler:
            print("Using mixed precision training with gradient scaling")
        
        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")
        
        # Track time for performance monitoring
        start_time = time.time()
        batch_start_time = start_time
        
        print("===== STARTING NETWORK TRAINING =====")
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            epoch_start_time = time.time()
            
            for batch_idx, (states, target_policies, target_values) in enumerate(dataloader):
                # Log batch shapes for debugging
                if batch_idx == 0:
                    print(f"Batch tensor shapes - States: {states.shape}, Policies: {target_policies.shape}, Values: {target_values.shape}")
                
                # Move tensors to device - non_blocking for better performance
                states = states.to(self.device, non_blocking=True)
                target_policies = target_policies.to(self.device, non_blocking=True)
                target_values = target_values.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                self.optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # Forward pass
                        policy_logits, value = self.network(states)
                        
                        # Calculate loss
                        policy_loss = F.cross_entropy(policy_logits, target_policies)
                        value_loss = F.mse_loss(value, target_values)
                        total_loss_batch = policy_loss + value_loss
                    
                    # Backward pass with gradient scaling
                    scaler.scale(total_loss_batch).backward()
                    
                    # Clip gradients to prevent explosions (with scaling)
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                    
                    # Update weights with scaling
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    # Regular forward pass (no mixed precision)
                    policy_logits, value = self.network(states)
                    
                    # Calculate loss
                    policy_loss = F.cross_entropy(policy_logits, target_policies)
                    value_loss = F.mse_loss(value, target_values)
                    total_loss_batch = policy_loss + value_loss
                    
                    # Backward pass
                    total_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                    self.optimizer.step()
                
                # Track losses
                total_loss += total_loss_batch.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1
                
                # Log every batch (was every 10)
                batch_time = time.time() - batch_start_time
                total_time = time.time() - start_time
                examples_per_sec = (batch_idx + 1) * self.config.batch_size / total_time
                
                print(f"Batch {batch_idx+1}/{len(dataloader)} - Loss: {total_loss_batch.item():.4f} (P: {policy_loss.item():.4f}, V: {value_loss.item():.4f}) - {batch_time:.2f}s/batch, {examples_per_sec:.1f} ex/s")
                
                # GPU memory tracking
                if self.device.type == 'cuda' and batch_idx % 5 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"GPU Memory: {mem_allocated:.1f}MB allocated, {mem_reserved:.1f}MB reserved")
                
                # Reset batch timer
                batch_start_time = time.time()
                
                # GPU memory management - clear cache periodically
                if self.device.type == 'cuda' and num_batches % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Epoch statistics
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Calculate average losses
        avg_loss = total_loss / max(num_batches, 1)
        avg_policy_loss = policy_loss_sum / max(num_batches, 1)
        avg_value_loss = value_loss_sum / max(num_batches, 1)
        
        # Total training statistics
        total_training_time = time.time() - start_time
        print(f"===== TRAINING COMPLETE =====")
        print(f"Total training time: {total_training_time:.2f}s")
        print(f"Training: Loss={avg_loss:.4f} (Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f})")
        print(f"Processed {num_batches} batches, {num_batches * self.config.batch_size} examples")
        print(f"Speed: {num_batches * self.config.batch_size / total_training_time:.1f} examples/second")
        
        return avg_loss
    
    def _augment_training_data_gpu(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply data augmentation directly on GPU for better performance"""
        augmented = []
        
        # Only augment a portion of the data
        num_to_augment = len(examples) // 4
        if num_to_augment == 0:
            return augmented
            
        # Convert states and policies to tensors for batch processing
        states = torch.tensor(np.array([ex['state'] for ex in examples[:num_to_augment]]), 
                            dtype=torch.float32, device=self.device)
        policies = torch.tensor(np.array([ex['policy'] for ex in examples[:num_to_augment]]), 
                                dtype=torch.float32, device=self.device)
        
        # Horizontal flip of board state (process in batch)
        flipped_states = states.clone()
        for plane_idx in range(states.shape[1]):
            flipped_states[:, plane_idx] = torch.flip(states[:, plane_idx], dims=[2])
        
        # Horizontal flip of policy (process in batch)
        flipped_policies = torch.zeros_like(policies)
        batch_size = policies.shape[0]
        
        # Create mapping tensor for move indices
        move_indices = torch.arange(0, 8*8*8*8, device=self.device).reshape(8, 8, 8, 8)
        flipped_move_indices = torch.flip(move_indices, dims=[1, 3])
        
        # Map the policies using the flipped indices
        for b in range(batch_size):
            policy_flat = policies[b].reshape(8, 8, 8, 8)
            flipped_policy_flat = torch.zeros_like(policy_flat)
            
            for from_row in range(8):
                for from_col in range(8):
                    flipped_from_col = 7 - from_col
                    for to_row in range(8):
                        for to_col in range(8):
                            flipped_to_col = 7 - to_col
                            flipped_policy_flat[from_row, flipped_from_col, to_row, flipped_to_col] = policy_flat[from_row, from_col, to_row, to_col]
            
            flipped_policies[b] = flipped_policy_flat.reshape(-1)
        
        # Move back to CPU and create new examples
        flipped_states_cpu = flipped_states.cpu().numpy()
        flipped_policies_cpu = flipped_policies.cpu().numpy()
        
        for i in range(num_to_augment):
            new_example = {
                'player': examples[i]['player'],
                'move_count': examples[i]['move_count'],
                'outcome': examples[i]['outcome'],
                'state': flipped_states_cpu[i],
                'policy': flipped_policies_cpu[i]
            }
            augmented.append(new_example)
        
        return augmented
        
    def find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        checkpoint_files = []
        
        try:
            for file in os.listdir(self.config.model_dir):
                if file.startswith("checkpoint_") and file.endswith(".pt"):
                    try:
                        # Extract iteration number from filename
                        iteration_str = file.replace("checkpoint_", "").replace(".pt", "")
                        iteration_num = int(iteration_str)
                        checkpoint_files.append((iteration_num, file))
                    except ValueError:
                        continue
            
            if checkpoint_files:
                # Sort by iteration number and return the latest
                latest = max(checkpoint_files, key=lambda x: x[0])
                return latest[1]  # Return filename
            
        except Exception as e:
            print(f"Error finding latest checkpoint: {e}")
        
        return None

    def evaluate_model(self) -> float:
        """Evaluate current model against previous best model with CUDA optimizations"""
        print("Starting model evaluation against previous best")
        
        # Find previous best model with improved logic
        prev_model_path = None
        
        # Priority 1: Look for best_model.pt
        if os.path.exists(os.path.join(self.config.model_dir, "best_model.pt")):
            prev_model_path = os.path.join(self.config.model_dir, "best_model.pt")
            print("Using best_model.pt for comparison")
        else:
            # Priority 2: Look for latest checkpoint
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                prev_model_path = os.path.join(self.config.model_dir, latest_checkpoint)
                print(f"Using latest checkpoint {latest_checkpoint} for comparison")
        
        if not prev_model_path:
            print("No previous model found, accepting current model as best")
            return 1.0
        
        try:
            # Load previous best model
            prev_network = AlphaZeroNetwork(self.config)
            prev_network = prev_network.to(self.device)
            
            # Try different loading methods for PyTorch 2.6+ compatibility
            try:
                # First try: weights_only=False for PyTorch 2.6+
                prev_network.load_state_dict(torch.load(prev_model_path, map_location=self.device, weights_only=False))
                print("Loaded previous model with weights_only=False")
            except Exception as e1:
                try:
                    # Second try: TorchScript load
                    prev_network = torch.jit.load(prev_model_path, map_location=self.device)
                    print("Loaded previous model as TorchScript")
                except Exception as e2:
                    print(f"Could not load previous model: {e1}, {e2}")
                    print("Accepting current model as best due to loading error")
                    return 1.0
            
            # Enable eval mode
            self.network.eval()
            prev_network.eval()
            
            # Create AIs with CUDA optimization flags
            current_ai = AlphaZeroAI(
                self.network, 
                self.config, 
                training_mode=False,
                device=self.device,
                batch_size=self.mcts_batch_size if self.device.type == 'cuda' else 1
            )
            
            previous_ai = AlphaZeroAI(
                prev_network, 
                self.config, 
                training_mode=False,
                device=self.device,
                batch_size=self.mcts_batch_size if self.device.type == 'cuda' else 1
            )
            
            # Run evaluation games
            game_engine = HeadlessChessGame()
            wins = 0
            total_games = self.config.evaluation_games
            
            # Track game results
            results = {'wins': 0, 'losses': 0, 'draws': 0}
            
            # Clear GPU cache before evaluation
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
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
                    results['draws'] += 1
                elif (game_num % 2 == 0 and result['winner'].value == 'white') or \
                    (game_num % 2 == 1 and result['winner'].value == 'black'):
                    results['wins'] += 1
                    wins += 1
                else:
                    results['losses'] += 1
                
                # Display progress
                win_rate = wins / (game_num + 1)
                print(f"Evaluation: Game {game_num+1}/{total_games}, current win rate: {win_rate:.2%}")
                
                # Periodically clear GPU cache during evaluation
                if self.device.type == 'cuda' and game_num % 5 == 0:
                    torch.cuda.empty_cache()
            
            # Calculate final win rate
            win_rate = wins / total_games
            print(f"Evaluation complete: Win rate: {win_rate:.2%} (W:{results['wins']}, L:{results['losses']}, D:{results['draws']})")
            
            # Switch back to training mode
            self.network.train()
            
            return win_rate
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("Accepting current model as best due to evaluation error")
            return 1.0
        
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint"""
        try:
            checkpoint_path = os.path.join(self.config.model_dir, f"checkpoint_{iteration}.pt")
            
            # Save state dict instead of TorchScript to avoid loading issues
            torch.save(self.network.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Also save as TorchScript for deployment (separate file)
            try:
                script_path = os.path.join(self.config.model_dir, f"checkpoint_{iteration}_script.pt")
                scripted_model = torch.jit.script(self.network)
                torch.jit.save(scripted_model, script_path)
                print(f"TorchScript model saved: {script_path}")
            except Exception as e:
                print(f"Warning: Could not save TorchScript version: {e}")
                
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    def save_best_model(self):
        """Save the best model"""
        try:
            best_path = os.path.join(self.config.model_dir, "best_model.pt")
            
            # Save state dict instead of TorchScript
            torch.save(self.network.state_dict(), best_path)
            print(f"Best model saved: {best_path}")
            
            # Also save as TorchScript for deployment (separate file)
            try:
                script_path = os.path.join(self.config.model_dir, "best_model_script.pt")
                scripted_model = torch.jit.script(self.network)
                torch.jit.save(scripted_model, script_path)
                print(f"Best TorchScript model saved: {script_path}")
            except Exception as e:
                print(f"Warning: Could not save TorchScript version: {e}")
                
        except Exception as e:
            print(f"Error saving best model: {e}")
        
    def save_model(self, filename: str):
        """Save model state with dual format for both training and inference"""
        # Make sure the directory exists
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Fix potential path issues by normalizing the filename
        if os.path.sep in filename:
            # Extract just the basename if a path was provided
            filename = os.path.basename(filename)
        
        model_path = os.path.join(self.config.model_dir, filename)
        
        # ALWAYS save a state_dict version for training compatibility
        try:
            torch.save(self.network.state_dict(), model_path)
            print(f"Model weights saved to {model_path}")
        except Exception as e:
            print(f"Error saving model weights: {e}")
            # Try saving with explicit CPU transfer
            try:
                state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}
                torch.save(state_dict, model_path)
                print(f"Model weights saved to {model_path} (CPU transfer)")
            except Exception as e2:
                print(f"Failed to save model: {e2}")
                raise e2
        
        # Additionally save TorchScript version if possible
        try:
            script_path = os.path.splitext(model_path)[0] + "_script.pt"
            if self.device.type == 'cuda':
                # Move to CPU for TorchScript saving to avoid device issues
                cpu_network = self.network.cpu()
                scripted_model = torch.jit.script(cpu_network)
                torch.jit.save(scripted_model, script_path)
                # Move back to original device
                self.network = self.network.to(self.device)
            else:
                scripted_model = torch.jit.script(self.network)
                torch.jit.save(scripted_model, script_path)
            print(f"TorchScript model saved to {script_path}")
        except Exception as e:
            print(f"Warning: Could not save TorchScript version: {e}")

    def load_model(self, filename: str):
        """Load model with compatibility for both training and PyTorch 2.6+"""
        model_path = os.path.join(self.config.model_dir, filename)
        weights_path = os.path.splitext(model_path)[0] + "_weights.pt"
        
        # First try to load from weights file (for training)
        if os.path.exists(weights_path):
            try:
                self.network.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=False))
                print(f"Model loaded from weights: {weights_path}")
                return
            except Exception as e:
                print(f"Failed to load weights file {weights_path}: {e}")
        
        # If weights file doesn't exist or failed to load, try main file
        if os.path.exists(model_path):
            try:
                # Try PyTorch 2.6+ compatible loading first
                self.network.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
                print(f"Model loaded from: {model_path}")
                return
            except Exception as e1:
                try:
                    # Try TorchScript loading
                    self.network = torch.jit.load(model_path, map_location=self.device)
                    print(f"TorchScript model loaded from: {model_path}")
                    return
                except Exception as e2:
                    print(f"Failed to load model: {e1}, {e2}")
                    raise e1
        else:
            print(f"No model file found at {model_path} or {weights_path}")
            raise FileNotFoundError(f"Model not found: {model_path}")
    
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