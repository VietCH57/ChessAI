"""
AlphaZero Configuration
This file contains default configuration settings for the AlphaZero implementation.
"""

# Default configuration for AlphaZero training
DEFAULT_CONFIG = {
    # Self-play parameters
    'num_self_play_games': 200,         # Number of self-play games per iteration
    'num_parallel_games': 8,            # Number of parallel self-play games
    'num_simulations': 800,             # MCTS simulations per move during self-play
    'max_moves_per_game': 512,          # Maximum moves per self-play game
    
    # Training parameters
    'batch_size': 2048,                 # Batch size for training
    'epochs': 40,                       # Epochs per training iteration
    'learning_rate': 0.001,             # Learning rate
    'weight_decay': 1e-4,               # L2 regularization coefficient
    'num_iterations': 100,              # Total training iterations
    'scheduler': 'cosine',              # Learning rate scheduler ('cosine', 'step', or 'none')
    
    # Network parameters
    'num_res_blocks': 20,               # Residual blocks in the network
    'num_filters': 256,                 # Filters in convolutional layers
    
    # Evaluation parameters
    'evaluation_games': 80,             # Number of games for evaluation
    'evaluation_threshold': 0.55,       # Win rate threshold to update best model
    
    # MCTS parameters
    'c_puct': 1.0,                      # Exploration constant in PUCT formula
    'temperature_init': 1.0,            # Initial temperature for move selection
    'temperature_final': 0.25,          # Final temperature after temperature_drop_move
    'temperature_drop_move': 30,        # Move number to drop temperature
    
    # File paths
    'output_dir': 'alphazero_models',   # Directory to save models
    'replay_buffer_size': 500000,       # Maximum number of examples in replay buffer
    
    # CUDA parameters
    'use_cuda': True,                   # Whether to use CUDA for training
    'mixed_precision': True,            # Whether to use mixed precision training
    'num_workers': 4,                   # Number of dataloader workers
    'pin_memory': True,                 # Pin memory for faster GPU transfer
}

def load_config(config_path=None):
    """
    Load configuration from file if provided, otherwise use default config
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    import os
    
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    return config