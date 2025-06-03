import argparse
import torch
import os
import psutil
import time
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
    
    # Configure optimal batch size
    if device == 'tpu':
        # TPU typically works well with larger batch sizes
        optimal_batch_size = max(128, args.batch_size)
        # TPU-optimized parameters
        optimal_filters = 256  # TPUs handle larger models efficiently
        optimal_blocks = 20    # More blocks for better model capacity
        optimal_workers = 0    # TPU requires num_workers=0
        print(f"TPU mode: Using {args.tpu_cores} cores with batch size {optimal_batch_size}")
    elif device == 'cuda' and torch.cuda.is_available():
        # GPU settings
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if gpu_mem < 6:  # Low memory GPU (< 6GB)
                optimal_batch_size = min(32, args.batch_size)
                optimal_filters = min(64, args.filters)
                optimal_blocks = min(6, args.residual_blocks)
            elif gpu_mem < 11:  # Mid-range GPU (6-11GB)
                optimal_batch_size = min(64, args.batch_size)
                optimal_filters = min(128, args.filters)
                optimal_blocks = min(12, args.residual_blocks)
            else:  # High-end GPU (>11GB)
                optimal_batch_size = min(128, args.batch_size)
                optimal_filters = args.filters
                optimal_blocks = args.residual_blocks
                
            print(f"GPU Memory: {gpu_mem:.1f}GB, Optimized batch size: {optimal_batch_size}")
            
            # Optimize workers for GPU
            logical_cores = psutil.cpu_count(logical=True) or 4
            optimal_workers = min(args.workers, max(2, logical_cores // 2))
        except Exception as e:
            print(f"Error detecting GPU specs: {e}, using default parameters")
            optimal_batch_size = args.batch_size
            optimal_filters = args.filters
            optimal_blocks = args.residual_blocks
            optimal_workers = args.workers
    else:
        # CPU mode
        print("Running on CPU. Consider using GPU for faster training.")
        optimal_batch_size = min(args.batch_size, 16)  # Smaller batches on CPU
        optimal_filters = min(args.filters, 64)        # Smaller network on CPU
        optimal_blocks = min(args.residual_blocks, 6)  # Fewer blocks on CPU
        
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
        memory_efficient=memory_efficient
    )
    
    print("\nAlphaZero Chess Training")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iteration}")
    print(f"MCTS simulations: {config.num_simulations}")
    print(f"Network: {config.num_residual_blocks} blocks, {config.num_filters} filters")
    print(f"Optimizations: JIT={config.use_jit}, FP16={config.half_precision}, Tree reuse={config.tree_reuse}")
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