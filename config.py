import dataclasses
from typing import Optional

@dataclasses.dataclass
class AlphaZeroConfig:
    """Configuration for AlphaZero training with optimization flags"""
    
    # Network Architecture
    num_residual_blocks: int = 20
    num_filters: int = 256
    input_planes: int = 119
    
    # MCTS Parameters
    num_simulations: int = 400
    c_puct: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # Training Parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    episodes_per_iteration: int = 100
    num_iterations: int = 1000
    
    # Self-play
    max_moves_per_game: int = 200
    temperature_threshold: int = 30  # moves after which temperature = 0
    
    # Evaluation
    evaluation_games: int = 100
    win_rate_threshold: float = 0.55
    
    # Storage
    replay_buffer_size: int = 100000
    checkpoint_interval: int = 10
    model_dir: str = "models"
    data_dir: str = "training_data"
    
    # Hardware
    device: str = "cuda"  # "cuda", "cpu", or "tpu"
    num_workers: int = 4
    tpu_cores: int = 8  # Number of TPU cores to use
    
    # Optimization flags
    tree_reuse: bool = True 
    data_augmentation: bool = True  
    use_jit: bool = True  
    half_precision: bool = True  
    batch_mcts: bool = True 
    use_cache: bool = True  
    memory_efficient: bool = True