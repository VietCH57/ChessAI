import os
import argparse
from alpha_zero_trainer import AlphaZeroTrainer

def main():
    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument("--config", type=str, help="Path to config file", default=None)
    parser.add_argument("--output_dir", type=str, help="Directory for output files", default="alphazero_models")
    parser.add_argument("--iterations", type=int, help="Number of training iterations", default=100)
    parser.add_argument("--self_play_games", type=int, help="Number of self-play games per iteration", default=100)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=1024)
    parser.add_argument("--epochs", type=int, help="Epochs per training iteration", default=20)
    parser.add_argument("--simulations", type=int, help="MCTS simulations per move", default=800)
    parser.add_argument("--load_model", type=str, help="Path to model to continue training", default=None)
    args = parser.parse_args()

    # Create configuration
    config = {
        'output_dir': args.output_dir,
        'num_iterations': args.iterations,
        'num_self_play_games': args.self_play_games,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'num_simulations': args.simulations
    }

    # Initialize trainer
    trainer = AlphaZeroTrainer(config)

    # Load model if provided
    start_iter = 0
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        start_iter = trainer.load_checkpoint(args.load_model)
        print(f"Continuing training from iteration {start_iter}")
    
    # Run training
    best_network = trainer.train()

    print("Training completed successfully!")
    
    return 0

if __name__ == "__main__":
    main()