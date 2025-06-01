import argparse
from chess_game import ChessGame
from alpha_zero_player import AlphaZeroPlayer

def main():
    parser = argparse.ArgumentParser(description="Play chess with AlphaZero")
    parser.add_argument("--model", type=str, help="Path to AlphaZero model", required=True)
    parser.add_argument("--simulations", type=int, help="MCTS simulations per move", default=800)
    parser.add_argument("--temperature", type=float, help="Temperature for move selection", default=0.1)
    parser.add_argument("--as_black", action="store_true", help="Play as black pieces")
    args = parser.parse_args()

    # Create AlphaZero player with model
    config = {
        'model_file': args.model,
        'num_simulations': args.simulations,
        'temperature': args.temperature,
    }
    alphazero_player = AlphaZeroPlayer(config)

    # Create chess game
    game = ChessGame()
    
    # Set AI player
    if args.as_black:
        game.toggle_ai(white_ai=alphazero_player)
        print("You are playing as BLACK")
    else:
        game.toggle_ai(black_ai=alphazero_player)
        print("You are playing as WHITE")
    
    # Run the game
    game.run()
    
    return 0

if __name__ == "__main__":
    main()