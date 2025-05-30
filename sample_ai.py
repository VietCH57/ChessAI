from chess_board import ChessBoard, Position, Piece, PieceType, PieceColor, Move
from interface import ChessAI

class RandomAI(ChessAI):
    """
    A simple implementation of a random move AI for demonstration purposes.
    """
    
    def __init__(self):
        import random
        self.random = random
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> tuple[Position, Position]:
        """Get a random valid move for the given color"""
        # Find all pieces of the given color
        valid_positions = []
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece.color == color:
                    valid_moves = board.get_valid_moves(pos)
                    if valid_moves:
                        for move in valid_moves:
                            valid_positions.append((pos, move.end_pos))
        
        if not valid_positions:
            raise ValueError(f"No valid moves for {color}")
        
        return self.random.choice(valid_positions)
    