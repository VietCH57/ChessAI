import argparse
import torch
from config import AlphaZeroConfig
from trainer import AlphaZeroTrainer

def main():
    parser = argparse.ArgumentParser(description='AlphaZero Chess Training for Tesla P100')
    
    # Tesla P100 optimized parameters
    parser.add_argument('--iterations', type=int, default=50)       
    parser.add_argument('--episodes-per-iter', type=int, default=25)  
    parser.add_argument('--simulations', type=int, default=200)      
    parser.add_argument('--batch-size', type=int, default=64)         
    parser.add_argument('--learning-rate', type=float, default=0.002) 
    
    # Network size optimized for P100
    parser.add_argument('--residual-blocks', type=int, default=12)   
    parser.add_argument('--filters', type=int, default=128)          
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--workers', type=int, default=4)           
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # Optimized configuration for speed
    config = AlphaZeroConfig(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iter,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_residual_blocks=args.residual_blocks,
        num_filters=args.filters,
        device=args.device if torch.cuda.is_available() else 'cpu',
        num_workers=args.workers,
        max_moves_per_game=100,  # Reduced for faster games
        checkpoint_interval=3,   # More frequent saves
        evaluation_games=20      # Fewer evaluation games
    )
    
    print("Fast AlphaZero Chess Training")
    print("=" * 50)
    print(f"Device: {config.device}")
    print(f"Iterations: {config.num_iterations}")
    print(f"Episodes per iteration: {config.episodes_per_iteration}")
    print(f"MCTS simulations: {config.num_simulations}")
    print(f"Network: {config.num_residual_blocks} blocks, {config.num_filters} filters")
    print("=" * 50)
    
    trainer = AlphaZeroTrainer(config)
    
    if args.resume:
        trainer.load_model(args.resume)
        trainer.load_training_history()
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        trainer.save_model("interrupted_checkpoint.pt")
    except Exception as e:
        print(f"\nTraining error: {e}")
        trainer.save_model("error_checkpoint.pt")
        raise

if __name__ == "__main__":
    main()