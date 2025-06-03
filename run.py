from chess_game import ChessGame
from chess_board import PieceColor
from minimax import MinimaxChessAI
from alphabeta import AlphaBetaChessAI
from alphazero import AlphaZeroChessAI

def main():
    """
    Script chạy game cờ vua với AI tùy chỉnh.
    """
    # Khởi tạo game
    game = ChessGame()
    
    
    # Cấu hình game để sử dụng AI của bạn (quân trắng)
    # game.toggle_ai(white_ai=my_ai)
    alphazero_ai = AlphaZeroChessAI.from_checkpoint("models\\last_checkpoint.pt", num_simulations=400)
    alphabeta_ai = AlphaBetaChessAI(depth=3)
    
    # HOẶC, để AI của bạn chơi với AI kháckhác
    game.toggle_ai(white_ai=alphazero_ai, black_ai=alphabeta_ai)
    
    # HOẶC, để chơi lại chính AI của bạn
    # game.toggle_ai(black_ai=AlphaBetaChessAI(depth=3))
    
    """
    Tóm lại là game.toggle_ai cho phép bạn chọn phe nào là AI, phe nào được bỏ trống thì sẽ là người chơi
    """
    # Chạy game
    game.run()

if __name__ == "__main__":
    main()