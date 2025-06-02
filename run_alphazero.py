import os
import argparse
import torch
from alpha_zero_trainer import AlphaZeroTrainer
from alphazero_config import load_config

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training with CUDA Support")
    parser.add_argument("--config", type=str, help="Path to config file", default=None)
    parser.add_argument("--output_dir", type=str, help="Directory for output files", default="alphazero_models")
    parser.add_argument("--iterations", type=int, help="Number of training iterations", default=100)
    parser.add_argument("--self_play_games", type=int, help="Number of self-play games per iteration", default=100)
    parser.add_argument("--parallel_games", type=int, help="Number of parallel self-play games", default=8)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=2048)
    parser.add_argument("--epochs", type=int, help="Epochs per training iteration", default=20)
    parser.add_argument("--simulations", type=int, help="MCTS simulations per move", default=800)
    parser.add_argument("--load_model", type=str, help="Path to model to continue training", default=None)
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (alias for --no_cuda)")
    parser.add_argument("--no_mixed_precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--num_filters", type=int, help="Number of filters in convolutional layers", default=256)
    parser.add_argument("--num_res_blocks", type=int, help="Number of residual blocks", default=20)
    args = parser.parse_args()

    # Load the base configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    config.update({
        'output_dir': args.output_dir,
        'num_iterations': args.iterations,
        'num_self_play_games': args.self_play_games,
        'num_parallel_games': args.parallel_games,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'num_simulations': args.simulations,
        'num_filters': args.num_filters,
        'num_res_blocks': args.num_res_blocks,
        'use_cuda': not (args.no_cuda or args.cpu) and torch.cuda.is_available(),
        'mixed_precision': not args.no_mixed_precision and torch.cuda.is_available(),
    })
    
    # Log CUDA status
    if config['use_cuda']:
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Running on CPU only")

    # Initialize trainer
    trainer = AlphaZeroTrainer(config)

    # Load model if provided
    start_iter = 0
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        start_iter = trainer.load_checkpoint(args.load_model)
        print(f"Continuing training from iteration {start_iter}")
    
    # Run training
    best_network = trainer.train(start_iteration=start_iter)

    print("Training completed successfully!")
    
    return 0

if __name__ == "__main__":
    main()