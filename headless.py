from chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move
from interface import ChessAI
from typing import Dict, List, Tuple, Optional, Any
import time
import random
import json
import os

class HeadlessChessGame:
    """
    Optimized headless chess game for training AI
    """
    
    def __init__(self):
        """Initialize headless chess game"""
        self.board = ChessBoard()
        self.game_over = False
        self.result_message = ""
        self.result = {
            "winner": None,
            "reason": None,
            "moves": 0,
            "history": []
        }
    
    def reset_game(self):
        """Reset the game to initial state"""
        self.board = ChessBoard()
        self.game_over = False
        self.result_message = ""
        self.result = {
            "winner": None,
            "reason": None,
            "moves": 0,
            "history": []
        }
    
    def check_game_state(self):
        """Check for game end conditions"""
        opponent_color = PieceColor.BLACK if self.board.turn == PieceColor.WHITE else PieceColor.WHITE
        
        if self.board.is_checkmate(self.board.turn):
            self.game_over = True
            self.result_message = f"Checkmate! {opponent_color.value.capitalize()} wins!"
            self.result["winner"] = opponent_color
            self.result["reason"] = "checkmate"
        elif self.board.is_stalemate(self.board.turn):
            self.game_over = True
            self.result_message = "Stalemate! Draw."
            self.result["reason"] = "stalemate"
        elif self.board.is_fifty_move_rule_draw():
            self.game_over = True
            self.result_message = "Draw by fifty-move rule."
            self.result["reason"] = "fifty_move_rule"
        elif self.board.is_threefold_repetition():
            self.game_over = True
            self.result_message = "Draw by threefold repetition."
            self.result["reason"] = "threefold_repetition"
    
    def run_game(self, white_ai: ChessAI, black_ai: ChessAI, max_moves=200, collect_data=False):
        """
        Run a complete game between two AIs at maximum speed
        
        Args:
            white_ai: AI for white
            black_ai: AI for black
            max_moves: Maximum moves before forced draw
            collect_data: Whether to collect detailed data for training
            
        Returns:
            dict: Game result and info
        """
        self.reset_game()
        move_count = 0
        game_history = []
        
        # Start time for FPS calculation
        start_time = time.time()
        
        # Debug info
        print(f"Starting game: White={type(white_ai).__name__}, Black={type(black_ai).__name__}")
        
        # Run game until end or max moves reached
        while not self.game_over and move_count < max_moves:
            # Determine current AI
            current_ai = white_ai if self.board.turn == PieceColor.WHITE else black_ai
            current_color = self.board.turn
            
            # Save current state for training if needed
            if collect_data:
                current_state = self._board_to_state()
            else:
                current_state = None
            
            try:
                # Get move from AI
                print(f"Getting move for {current_color.value}...")
                from_pos, to_pos = current_ai.get_move(self.board, current_color)
                print(f"AI returned move: {from_pos.row},{from_pos.col} -> {to_pos.row},{to_pos.col}")
                
                # Execute move
                move = self.board.move_piece(from_pos, to_pos)
                
                if move:
                    # If collecting data, save move info
                    if collect_data:
                        move_data = {
                            'move_number': move_count + 1,
                            'from': (from_pos.row, from_pos.col),
                            'to': (to_pos.row, to_pos.col),
                            'piece_type': move.piece.type.value,
                            'piece_color': move.piece.color.value,
                            'captured': None if not move.captured_piece or move.captured_piece.type == PieceType.EMPTY 
                                    else move.captured_piece.type.value,
                            'is_castling': move.is_castling,
                            'is_en_passant': move.is_en_passant,
                            'state_before': current_state
                        }
                        game_history.append(move_data)
                    
                    # Check for game end
                    self.check_game_state()
                    move_count += 1
                    print(f"Move {move_count}: {current_color.value} played {from_pos.row},{from_pos.col} -> {to_pos.row},{to_pos.col}")
                else:
                    # Invalid move (shouldn't happen with proper AI)
                    print(f"ERROR: Invalid move from {current_color.value} AI: {from_pos.row},{from_pos.col} -> {to_pos.row},{to_pos.col}")
                    self.result["reason"] = "invalid_move"
                    self.game_over = True
                    
            except Exception as e:
                # Handle AI errors
                print(f"ERROR in AI move: {str(e)}")
                self.result["reason"] = f"ai_error: {str(e)}"
                self.game_over = True
        
        # Check if move limit reached
        if move_count >= max_moves and not self.game_over:
            self.game_over = True
            self.result["reason"] = "move_limit"
            
        # Calculate FPS (moves/second)
        elapsed_time = time.time() - start_time
        fps = move_count / elapsed_time if elapsed_time > 0 else 0
        
        # Update result
        self.result["moves"] = move_count
        self.result["time_seconds"] = elapsed_time  
        self.result["moves_per_second"] = fps
        
        if collect_data:
            self.result["history"] = game_history
        
        print(f"Game completed: {move_count} moves, Winner: {self.result['winner'].value if self.result['winner'] else 'draw'}, Reason: {self.result['reason']}")
        
        return self.result
    
    def _board_to_state(self):
        """
        Convert board to state representation for AI
        """
        state = {}
        
        # Pieces
        pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row][col]
                if piece.type != PieceType.EMPTY:
                    pieces.append({
                        'position': (row, col),
                        'type': piece.type.value,
                        'color': piece.color.value,
                        'has_moved': piece.has_moved
                    })
                    
        state['pieces'] = pieces
        state['turn'] = self.board.turn.value
        
        # Current turn
        state['half_move_clock'] = self.board.half_move_clock
        state['move_number'] = len(self.board.move_history) // 2 + 1
        
        # Check info
        state['white_in_check'] = self.board.is_check(PieceColor.WHITE)
        state['black_in_check'] = self.board.is_check(PieceColor.BLACK)
        
        return state