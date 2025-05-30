from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from chess_board import ChessBoard, Position, Piece, Move, PieceType, PieceColor

class ChessAI(ABC):
    """
    Abstract base class defining the interface for AI programs to interact with the chess game.
    AI developers should implement this interface for their chess AI.
    """
    
    @abstractmethod
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """
        Get the next move from the AI.
        
        Args:
            board: The current chess board state
            color: The color (PieceColor.WHITE or PieceColor.BLACK) the AI is playing as
            
        Returns:
            Tuple of (from_position, to_position) representing the move
        """
        pass


class ChessAIManager:
    """
    Manager to facilitate interaction between the chess game and AI players.
    """
    
    def __init__(self, chess_game):
        """
        Initialize the AI manager with a game instance.
        
        Args:
            chess_game: The ChessGame instance to manage
        """
        self.chess_game = chess_game
        self.board = chess_game.board
        self.white_ai = None
        self.black_ai = None
        self.game_active = False
        self.move_delay = 500  # Delay between AI moves in milliseconds
        self.last_move_time = 0
    
    def register_ai(self, ai: ChessAI, color: PieceColor) -> None:
        """
        Register an AI for a specific color.
        
        Args:
            ai: The AI instance implementing ChessAI
            color: PieceColor.WHITE or PieceColor.BLACK
        """
        if color == PieceColor.WHITE:
            self.white_ai = ai
        elif color == PieceColor.BLACK:
            self.black_ai = ai
        else:
            raise ValueError("Color must be PieceColor.WHITE or PieceColor.BLACK")
    
    def start_ai_game(self):
        """
        Start a game where AI(s) play automatically.
        This can be AI vs AI or AI vs human, depending on which AIs are registered.
        """
        self.game_active = True
    
    def stop_ai_game(self):
        """Stop the automatic AI gameplay"""
        self.game_active = False
    
    def update(self, current_time):
        """
        Update method to be called in the game loop.
        Handles AI moves when appropriate.
        
        Args:
            current_time: Current game time in milliseconds (from pygame.time.get_ticks())
        """
        if not self.game_active or self.chess_game.game_over:
            return
            
        # Check if it's time for AI to move
        if current_time - self.last_move_time < self.move_delay:
            return
            
        current_color = self.board.turn
        ai_player = self.white_ai if current_color == PieceColor.WHITE else self.black_ai
        
        # If it's AI's turn and we have an AI for this color
        if ai_player:
            # Get AI's move
            try:
                from_pos, to_pos = ai_player.get_move(self.board, current_color)
                # Apply the move
                self.make_move(from_pos, to_pos)
                self.last_move_time = current_time
            except Exception as e:
                print(f"AI error: {e}")
                self.game_active = False
    
    def make_move(self, from_pos: Position, to_pos: Position) -> bool:
        """
        Apply an AI move to the game.
        
        Args:
            from_pos: Starting position
            to_pos: Target position
            
        Returns:
            True if move was successful, False otherwise
        """
        # Use the board's move_piece method
        move = self.board.move_piece(from_pos, to_pos)
        
        if move:
            # Update game state after move
            self.chess_game.check_game_state()
            return True
        return False
    
    def get_valid_moves_from_position(self, position: Position) -> List[Position]:
        """
        Get all valid destination positions for a piece at the given position.
        
        Args:
            position: Position of the piece
            
        Returns:
            List of valid destination positions
        """
        moves = self.board.get_valid_moves(position)
        return [move.end_pos for move in moves]

    def get_all_valid_moves(self, color: PieceColor) -> Dict[Position, List[Position]]:
        """
        Get all valid moves for all pieces of the given color.
        
        Args:
            color: The color to get moves for
            
        Returns:
            Dictionary mapping piece positions to lists of valid destination positions
        """
        valid_moves = {}
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = self.board.get_piece(pos)
                if piece.color == color:
                    moves = self.get_valid_moves_from_position(pos)
                    if moves:
                        valid_moves[pos] = moves
        return valid_moves


