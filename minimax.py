from chess_board import ChessBoard, Position, PieceType, PieceColor
from interface import ChessAI
import random
import time

class MinimaxChessAI(ChessAI):
    """
    Chess AI that uses the Minimax algorithm to evaluate and choose the best move.
    """
    
    def __init__(self, depth=3):
        """
        Initialize the Minimax AI.
        
        Args:
            depth: How many moves ahead to look
        """
        self.depth = depth
        self.max_time = 10  # Maximum time in seconds for a move decision
        self.piece_values = {
            PieceType.PAWN: 100,
            PieceType.KNIGHT: 320,
            PieceType.BISHOP: 330,
            PieceType.ROOK: 500,
            PieceType.QUEEN: 900,
            PieceType.KING: 20000,
            PieceType.EMPTY: 0
        }
        
        # Position evaluation tables to encourage good piece placement
        self.pawn_table = [
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [ 5,  5, 10, 25, 25, 10,  5,  5],
            [ 0,  0,  0, 20, 20,  0,  0,  0],
            [ 5, -5,-10,  0,  0,-10, -5,  5],
            [ 5, 10, 10,-20,-20, 10, 10,  5],
            [ 0,  0,  0,  0,  0,  0,  0,  0]
        ]
        
        self.knight_table = [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ]
        
        self.bishop_table = [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5,  5,  5,  5,  5,-10],
            [-10,  0,  5,  0,  0,  5,  0,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ]
        
        self.rook_table = [
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [ 0,  0,  0,  5,  5,  0,  0,  0]
        ]
        
        self.queen_table = [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [ -5,  0,  5,  5,  5,  5,  0, -5],
            [  0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]
        
        self.king_mid_table = [
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [ 20, 20,  0,  0,  0,  0, 20, 20],
            [ 20, 30, 10,  0,  0, 10, 30, 20]
        ]
        
        self.king_end_table = [
            [-50,-40,-30,-20,-20,-30,-40,-50],
            [-30,-20,-10,  0,  0,-10,-20,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 30, 40, 40, 30,-10,-30],
            [-30,-10, 20, 30, 30, 20,-10,-30],
            [-30,-30,  0,  0,  0,  0,-30,-30],
            [-50,-30,-30,-30,-30,-30,-30,-50]
        ]
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> tuple[Position, Position]:
        """
        Get the best move using the Minimax algorithm.
        
        Args:
            board: The current chess board state
            color: The color (PieceColor.WHITE or PieceColor.BLACK) the AI is playing as
            
        Returns:
            Tuple of (from_position, to_position) representing the move
        """
        # Set a time limit for move calculation
        self.start_time = time.time()
        self.time_limit_exceeded = False
        
        # Track best move and score
        best_move = None
        best_score = float('-inf')
        
        # Dynamic depth adjustment based on position complexity
        if self.count_pieces(board) <= 10:  # Endgame with few pieces
            current_depth = min(self.depth + 1, 4)  # Go deeper in endgame, but cap at depth 4
        else:
            current_depth = self.depth
        
        try:
            # Get all valid moves for all pieces of this color
            all_moves = []
            for row in range(8):
                for col in range(8):
                    from_pos = Position(row, col)
                    piece = board.get_piece(from_pos)
                    
                    if piece.color == color:
                        valid_moves = board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            all_moves.append((from_pos, move.end_pos))
            
            # Safety check - if no moves, return None (game should be over)
            if not all_moves:
                print("No valid moves found - game should be over")
                # Return a dummy move (this should never be executed in a real game)
                for row in range(8):
                    for col in range(8):
                        if board.get_piece(Position(row, col)).color == color:
                            return (Position(row, col), Position(row, col))
            
            # Randomize move order for less predictable play
            random.shuffle(all_moves)
            
            # Find the best move
            for from_pos, to_pos in all_moves:
                # Check if we're out of time
                if time.time() - self.start_time > self.max_time:
                    print("Time limit exceeded, using best move found so far")
                    break
                
                # Make the move on a copy of the board
                temp_board = board.copy_board()
                try:
                    temp_board.move_piece(from_pos, to_pos)
                except Exception as e:
                    print(f"Error making move: {e}")
                    continue
                
                # Evaluate the position using minimax
                try:
                    score = self.minimax(temp_board, current_depth - 1, False, color)
                except Exception as e:
                    print(f"Error in minimax: {e}")
                    continue
                
                # Update best move if this move is better
                if score > best_score:
                    best_score = score
                    best_move = (from_pos, to_pos)
            
            # If we found a valid move, return it
            if best_move:
                return best_move
                
            # If we didn't find a good move (shouldn't happen), return the first valid move
            return all_moves[0]
            
        except Exception as e:
            print(f"Error in get_move: {e}")
            # Fallback to a simple move selection in case of error
            return self.get_fallback_move(board, color)
    
    def get_fallback_move(self, board, color):
        """Fallback move selection in case of errors"""
        for row in range(8):
            for col in range(8):
                from_pos = Position(row, col)
                piece = board.get_piece(from_pos)
                
                if piece.color == color:
                    valid_moves = board.get_valid_moves(from_pos)
                    if valid_moves:
                        return (from_pos, valid_moves[0].end_pos)
        
        # If no moves found (shouldn't happen), raise an error
        raise ValueError(f"No valid moves found for {color}")
    
    def count_pieces(self, board):
        """Count the total number of pieces on the board"""
        count = 0
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    count += 1
        return count
    
    def minimax(self, board, depth, is_maximizing, ai_color):
        """
        Implementation of the Minimax algorithm.
        
        Args:
            board: The current board state
            depth: How many moves to look ahead from this position
            is_maximizing: True if the current player is trying to maximize the score
            ai_color: Color of the AI player (for evaluation)
            
        Returns:
            The best score for this position
        """
        # Check if we've exceeded the time limit
        if time.time() - self.start_time > self.max_time:
            self.time_limit_exceeded = True
            # Return the current evaluation if maximizing, or inverse if minimizing
            current_eval = self.evaluate_board(board, ai_color)
            return current_eval if is_maximizing else -current_eval
        
        # Base case: reached leaf node or game is over
        if depth == 0 or self.time_limit_exceeded:
            return self.evaluate_board(board, ai_color)
        
        # Check for checkmate and stalemate
        opponent_color = PieceColor.BLACK if ai_color == PieceColor.WHITE else PieceColor.WHITE
        if board.is_checkmate(opponent_color):
            return 10000  # AI wins
        elif board.is_checkmate(ai_color):
            return -10000  # AI loses
        elif board.is_stalemate(board.turn):
            return 0  # Draw
        
        current_color = board.turn
        
        if is_maximizing:
            max_score = float('-inf')
            
            # Get all pieces of current color
            for row in range(8):
                for col in range(8):
                    if self.time_limit_exceeded:
                        break
                        
                    from_pos = Position(row, col)
                    piece = board.get_piece(from_pos)
                    
                    if piece.color == current_color:
                        valid_moves = board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            # Make the move on a copy
                            temp_board = board.copy_board()
                            temp_board.move_piece(from_pos, move.end_pos)
                            
                            # Recurse and find the best move
                            score = self.minimax(temp_board, depth - 1, False, ai_color)
                            max_score = max(max_score, score)
            
            return max_score
        else:
            min_score = float('inf')
            
            # Try all possible moves
            for row in range(8):
                for col in range(8):
                    if self.time_limit_exceeded:
                        break
                        
                    from_pos = Position(row, col)
                    piece = board.get_piece(from_pos)
                    
                    if piece.color == current_color:
                        valid_moves = board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            # Make the move on a copy
                            temp_board = board.copy_board()
                            temp_board.move_piece(from_pos, move.end_pos)
                            
                            # Recurse and find the best move
                            score = self.minimax(temp_board, depth - 1, True, ai_color)
                            min_score = min(min_score, score)
            
            return min_score
    
    def evaluate_board(self, board, ai_color):
        """
        Evaluates the board position from the perspective of color.
        A positive score is good for the AI, negative is bad.
        
        Args:
            board: The chess board to evaluate
            ai_color: The color of the AI player
            
        Returns:
            A numerical score representing the quality of the position
        """
        score = 0
        opponent_color = PieceColor.BLACK if ai_color == PieceColor.WHITE else PieceColor.WHITE
        
        # Material count
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                
                if piece.type == PieceType.EMPTY:
                    continue
                
                # Get base material value
                value = self.piece_values[piece.type]
                
                # Add position value based on piece type
                position_value = 0
                if piece.type == PieceType.PAWN:
                    position_value = self.pawn_table[row][col]
                elif piece.type == PieceType.KNIGHT:
                    position_value = self.knight_table[row][col]
                elif piece.type == PieceType.BISHOP:
                    position_value = self.bishop_table[row][col]
                elif piece.type == PieceType.ROOK:
                    position_value = self.rook_table[row][col]
                elif piece.type == PieceType.QUEEN:
                    position_value = self.queen_table[row][col]
                elif piece.type == PieceType.KING:
                    # Use different tables for the king depending on game phase
                    if self.is_endgame(board):
                        position_value = self.king_end_table[row][col]
                    else:
                        position_value = self.king_mid_table[row][col]
                
                # Invert position tables for black
                if piece.color == PieceColor.BLACK:
                    position_value = position_value * -1
                
                # Add to the total score
                if piece.color == ai_color:
                    score += value + position_value
                else:
                    score -= value + position_value
        
        return score
    
    def is_endgame(self, board):
        """Simple check for endgame - less than 10 pieces total"""
        piece_count = 0
        for row in range(8):
            for col in range(8):
                if board.get_piece(Position(row, col)).type != PieceType.EMPTY:
                    piece_count += 1
        return piece_count < 10