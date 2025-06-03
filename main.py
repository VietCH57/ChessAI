import argparse
import torch
import os
import psutil
import time
from config import AlphaZeroConfig
from trainer import AlphaZeroTrainer

import torch.multiprocessing as mp
if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=4)           
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--disable-tree-reuse', action='store_true', help='Disable MCTS tree reuse')
    parser.add_argument('--disable-data-augmentation', action='store_true', help='Disable data augmentation')
    parser.add_argument('--jit', action='store_true', help='Use PyTorch JIT compilation')
    parser.add_argument('--half-precision', action='store_true', help='Use half precision (FP16)')
    parser.add_argument('--batch-mcts', action='store_true', help='Use batched MCTS evaluation')
    parser.add_argument('--use-cache', action='store_true', help='Use board evaluation cache')
    
    args = parser.parse_args()
    
    # Configure CUDA for optimal performance
    if args.device == 'cuda' and torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for faster convolutions
        torch.backends.cudnn.benchmark = True
        
        # Optimize batch size based on available GPU memory
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
        except Exception as e:
            print(f"Error detecting GPU specs: {e}, using default parameters")
            optimal_batch_size = args.batch_size
            optimal_filters = args.filters
            optimal_blocks = args.residual_blocks
    else:
        # CPU mode
        print("Running on CPU. Consider using GPU for faster training.")
        optimal_batch_size = min(args.batch_size, 16)  # Smaller batches on CPU
        optimal_filters = min(args.filters, 64)        # Smaller network on CPU
        optimal_blocks = min(args.residual_blocks, 6)  # Fewer blocks on CPU
    
    # Optimize number of workers based on CPU cores
    physical_cores = psutil.cpu_count(logical=False) or 2
    optimal_workers = min(args.workers, max(1, physical_cores - 1))
    print(f"CPU Cores: {physical_cores}, Using {optimal_workers} worker processes")
    
    # Memory optimization settings
    memory_efficient = True  # Always use memory optimization
    
    # Overide (because i'm too lazy to type the full command line arguments)
    tree_reuse = True
    data_augmentation = True
    use_jit = True
    half_precision = True
    batch_mcts = True
    use_cache = True
    
    # Create configuration
    config = AlphaZeroConfig(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iter,
        num_simulations=args.simulations,
        batch_size=optimal_batch_size,
        learning_rate=args.learning_rate,
        num_residual_blocks=optimal_blocks,
        num_filters=optimal_filters,
        device=args.device if torch.cuda.is_available() else "cpu",
        num_workers=optimal_workers,
        
        # Sử dụng biến override thay vì args
        tree_reuse=tree_reuse,
        data_augmentation=data_augmentation,
        use_jit=use_jit,
        half_precision=half_precision,
        batch_mcts=batch_mcts,
        use_cache=use_cache,
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