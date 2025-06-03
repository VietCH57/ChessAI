import argparse
import torch
import os
import psutil
import time
import math
from config import AlphaZeroConfig
from trainer import AlphaZeroTrainer

# Check for TPU availability
USE_TPU = os.environ.get('COLAB_TPU_ADDR') is not None

if USE_TPU:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print("TPU available, configuring PyTorch XLA")
        TPU_DEVICE = xm.xla_device()
    except ImportError:
        print("PyTorch XLA not available. Install with: pip install torch_xla")
        USE_TPU = False

def get_optimal_gpu_settings():
    """Determine optimal settings based on GPU capabilities"""
    try:
        if not torch.cuda.is_available():
            return {}
            
        # Get GPU information
        device_props = torch.cuda.get_device_properties(0)
        gpu_name = device_props.name
        gpu_mem_gb = device_props.total_memory / 1e9  # GB
        compute_capability = f"{device_props.major}.{device_props.minor}"
        cuda_cores = 0
        
        # Estimate CUDA cores based on compute capability
        if device_props.major == 8:  # Ampere (RTX 30xx)
            cuda_cores = device_props.multi_processor_count * 128
        elif device_props.major == 7:  # Volta/Turing (RTX 20xx, GTX 16xx)
            cuda_cores = device_props.multi_processor_count * 64
        elif device_props.major == 6:  # Pascal (GTX 10xx)
            cuda_cores = device_props.multi_processor_count * 128
        else:  # Older architectures
            cuda_cores = device_props.multi_processor_count * 64
            
        print(f"GPU: {gpu_name}, Memory: {gpu_mem_gb:.1f}GB, Compute: {compute_capability}, CUDA Cores: {cuda_cores}")
        
        # Calculate optimal parallel games based on GPU memory and cores
        # Each game roughly needs ~200MB for network + MCTS
        max_parallel_by_mem = math.floor(gpu_mem_gb * 0.7 / 0.2)  # Use 70% of memory
        max_parallel_by_cores = math.ceil(cuda_cores / 128)  # Heuristic based on cores
        
        optimal_parallel_games = min(max(2, min(max_parallel_by_mem, max_parallel_by_cores)), 16)
        
        # Set network size based on memory
        if gpu_mem_gb < 4:  # Low memory GPU
            network_size = "small"  # 6 blocks, 64 filters
            optimal_batch_size = 32
            mcts_batch_size = 16
        elif gpu_mem_gb < 8:  # Mid-range GPU 
            network_size = "medium"  # 12 blocks, 128 filters
            optimal_batch_size = 64
            mcts_batch_size = 32
        else:  # High-end GPU
            network_size = "large"  # 19+ blocks, 256 filters
            optimal_batch_size = 128
            mcts_batch_size = 64
            
        # Tensor cores available on Volta (7.0), Turing (7.5), Ampere (8.0+)
        has_tensor_cores = device_props.major >= 7
        
        return {
            "parallel_games": optimal_parallel_games,
            "network_size": network_size,
            "batch_size": optimal_batch_size,
            "mcts_batch_size": mcts_batch_size,
            "use_mixed_precision": has_tensor_cores,
            "gpu_mem_gb": gpu_mem_gb
        }
    except Exception as e:
        print(f"Error determining GPU settings: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Chess Training')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=50)       
    parser.add_argument('--episodes-per-iter', type=int, default=50)
    parser.add_argument('--simulations', type=int, default=200)      
    parser.add_argument('--batch-size', type=int, default=64)         
    parser.add_argument('--learning-rate', type=float, default=0.002) 
    
    # Network size
    parser.add_argument('--residual-blocks', type=int, default=12)   
    parser.add_argument('--filters', type=int, default=128)          
    
    # Hardware & optimization
    parser.add_argument('--device', type=str, default='auto', 
                        help='Device to use: auto, cpu, cuda, or tpu')
    parser.add_argument('--workers', type=int, default=4)           
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use-tpu', action='store_true', help='Force TPU usage')
    parser.add_argument('--tpu-cores', type=int, default=8, help='Number of TPU cores to use')
    parser.add_argument('--disable-tree-reuse', action='store_true', help='Disable MCTS tree reuse')
    parser.add_argument('--disable-data-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--jit', action='store_true', help='Use PyTorch JIT compilation')
    parser.add_argument('--half-precision', action='store_true', help='Use half precision (FP16)')
    parser.add_argument('--batch-mcts', action='store_true', help='Use batched MCTS evaluation')
    parser.add_argument('--use-cache', action='store_true', help='Use board evaluation cache')
    parser.add_argument('--parallel-games', type=int, default=0, 
                       help='Number of parallel games to run on GPU (0=auto)')
    parser.add_argument('--mcts-batch-size', type=int, default=0,
                       help='Batch size for MCTS evaluations (0=auto)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if args.use_tpu or USE_TPU:
            device = 'tpu'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    # Get GPU-specific optimizations if using CUDA
    gpu_settings = {}
    if device == 'cuda' and torch.cuda.is_available():
        gpu_settings = get_optimal_gpu_settings()
        print(f"GPU optimization: {gpu_settings}")
        
        # Enable CUDA optimizations
        if not args.disable_tree_reuse:
            print("Enabling MCTS tree reuse for GPU")
        
        # Activate tensor cores if available
        if gpu_settings.get("use_mixed_precision", False) and not args.half_precision:
            print("GPU supports tensor cores - enabling mixed precision by default")
            args.half_precision = True
            
        # Force batched MCTS for GPU
        if not args.batch_mcts:
            print("Enabling batched MCTS for GPU")
            args.batch_mcts = True
    
    # Configure optimal batch size
    if device == 'tpu':
        # TPU typically works well with larger batch sizes
        optimal_batch_size = max(128, args.batch_size)
        # TPU-optimized parameters
        optimal_filters = 256  # TPUs handle larger models efficiently
        optimal_blocks = 20    # More blocks for better model capacity
        optimal_workers = 0    # TPU requires num_workers=0
        print(f"TPU mode: Using {args.tpu_cores} cores with batch size {optimal_batch_size}")
        use_gpu_parallel = False
        parallel_games = 0
        mcts_batch_size = 1
    elif device == 'cuda' and torch.cuda.is_available():
        # GPU settings
        network_size = gpu_settings.get("network_size", "medium")
        
        if network_size == "small":
            optimal_filters = min(64, args.filters)
            optimal_blocks = min(6, args.residual_blocks)
        elif network_size == "medium":
            optimal_filters = min(128, args.filters)
            optimal_blocks = min(12, args.residual_blocks)
        else:  # large
            optimal_filters = min(256, args.filters) 
            optimal_blocks = min(19, args.residual_blocks)
            
        optimal_batch_size = gpu_settings.get("batch_size", args.batch_size)
        
        # Set up GPU parallel parameters
        use_gpu_parallel = True
        parallel_games = args.parallel_games if args.parallel_games > 0 else gpu_settings.get("parallel_games", 4)
        mcts_batch_size = args.mcts_batch_size if args.mcts_batch_size > 0 else gpu_settings.get("mcts_batch_size", 32)
        
        print(f"GPU mode: Using {parallel_games} parallel games with MCTS batch size {mcts_batch_size}")
        
        # Optimize workers for GPU
        logical_cores = psutil.cpu_count(logical=True) or 4
        optimal_workers = min(args.workers, max(2, logical_cores // 2))
    else:
        # CPU mode
        print("Running on CPU. Consider using GPU for faster training.")
        optimal_batch_size = min(args.batch_size, 16)  # Smaller batches on CPU
        optimal_filters = min(args.filters, 64)        # Smaller network on CPU
        optimal_blocks = min(args.residual_blocks, 6)  # Fewer blocks on CPU
        use_gpu_parallel = False
        parallel_games = 0
        mcts_batch_size = 1
        
        # Optimize workers for CPU
        physical_cores = psutil.cpu_count(logical=False) or 2
        logical_cores = psutil.cpu_count(logical=True) or physical_cores
        optimal_workers = min(args.workers, max(1, physical_cores - 1))
    
    # Memory optimization settings
    memory_efficient = True  # Always use memory optimization
    
    # Create configuration
    config = AlphaZeroConfig(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iter,
        num_simulations=args.simulations,
        batch_size=optimal_batch_size,
        learning_rate=args.learning_rate,
        num_residual_blocks=optimal_blocks,
        num_filters=optimal_filters,
        device=device,
        num_workers=optimal_workers,
        
        # Optimization flags
        tree_reuse=not args.disable_tree_reuse,
        data_augmentation=not args.disable_data_augmentation,
        use_jit=args.jit,
        half_precision=args.half_precision,
        batch_mcts=args.batch_mcts,
        use_cache=args.use_cache,
        memory_efficient=memory_efficient,
        
        # GPU parallel settings
        use_gpu_parallel=use_gpu_parallel,
        parallel_games=parallel_games,
        mcts_batch_size=mcts_batch_size
    )
    
    print("\nAlphaZero Chess Training")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iteration}")
    print(f"MCTS simulations: {config.num_simulations}")
    print(f"Network: {config.num_residual_blocks} blocks, {config.num_filters} filters")
    print(f"Optimizations: JIT={config.use_jit}, FP16={config.half_precision}, Tree reuse={config.tree_reuse}")
    print(f"GPU acceleration: {config.use_gpu_parallel}, Parallel games: {config.parallel_games}")
    print(f"Memory efficient: {config.memory_efficient}, Batch MCTS: {config.batch_mcts}")
    print("=" * 50)
    
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Initialize trainer
    trainer = AlphaZeroTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        try:
            trainer.load_model(args.resume)
            trainer.load_training_history()
            print(f"Resumed training from {args.resume}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Start training
    try:
        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model("models/interrupted_checkpoint.pt")
        trainer.save_training_history()
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_model("models/error_checkpoint.pt")
        trainer.save_training_history()
        raise

if __name__ == "__main__":
    main()