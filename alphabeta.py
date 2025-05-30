from chess_board import ChessBoard, Position, PieceType, PieceColor
from interface import ChessAI
import random
import time

class AlphaBetaChessAI(ChessAI):
    """
    Chess AI that uses the Alpha-Beta Pruning algorithm to evaluate and choose the best move.
    This is an optimization of the Minimax algorithm that prunes branches that won't affect
    the final decision, allowing for a deeper search in the same amount of time.
    """
    
    def __init__(self, depth=4):
        """
        Initialize the Alpha-Beta Pruning AI.
        
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
        Get the best move using the Alpha-Beta Pruning algorithm.
        
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
        alpha = float('-inf')
        beta = float('inf')
        
        # Adjust depth based on game phase
        if self.count_pieces(board) <= 10:  # Endgame with few pieces
            current_depth = min(self.depth, 4)  # Cap at depth 4 to prevent timeouts
        else:
            current_depth = min(self.depth - 1, 3)  # Reduce depth in middlegame
        
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
            
            # Apply move ordering heuristics to improve alpha-beta pruning efficiency
            all_moves = self.order_moves(board, all_moves)
            
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
                
                # Evaluate the position using alpha-beta
                try:
                    score = self.alpha_beta(temp_board, current_depth - 1, alpha, beta, False, color)
                except Exception as e:
                    print(f"Error in alpha_beta: {e}")
                    continue
                
                # Update best move if this move is better
                if score > best_score:
                    best_score = score
                    best_move = (from_pos, to_pos)
                
                # Update alpha
                alpha = max(alpha, best_score)
            
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
    
    def order_moves(self, board, moves):
        """
        Order moves to improve alpha-beta pruning efficiency.
        Captures and checks are examined first.
        
        Args:
            board: Current board state
            moves: List of moves to order
            
        Returns:
            Ordered list of moves
        """
        # Score each move to determine order
        scored_moves = []
        for from_pos, to_pos in moves:
            score = 0
            piece = board.get_piece(from_pos)
            target = board.get_piece(to_pos)
            
            # Prioritize captures based on MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if target.type != PieceType.EMPTY:
                score += 10 * self.piece_values[target.type] - self.piece_values[piece.type]
            
            # Prioritize pawn promotions
            if piece.type == PieceType.PAWN and (to_pos.row == 0 or to_pos.row == 7):
                score += 900  # Value of a queen
            
            scored_moves.append((score, (from_pos, to_pos)))
        
        # Sort moves by score in descending order
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        
        # Return only the moves, without scores
        return [move for _, move in scored_moves]
    
    def alpha_beta(self, board, depth, alpha, beta, is_maximizing, ai_color):
        """
        Implementation of the Alpha-Beta Pruning algorithm.
        
        Args:
            board: The current board state
            depth: How many moves to look ahead from this position
            alpha: The best score that the maximizing player can guarantee
            beta: The best score that the minimizing player can guarantee
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
        
        # Check for game-ending conditions
        opponent_color = PieceColor.BLACK if ai_color == PieceColor.WHITE else PieceColor.WHITE
        if board.is_checkmate(opponent_color):
            return 10000  # AI wins
        elif board.is_checkmate(ai_color):
            return -10000  # AI loses
        elif board.is_stalemate(board.turn):
            return 0  # Draw
        
        current_color = board.turn
        
        # Get all possible moves
        all_moves = []
        for row in range(8):
            for col in range(8):
                if self.time_limit_exceeded:
                    break
                    
                from_pos = Position(row, col)
                piece = board.get_piece(from_pos)
                
                if piece.color == current_color:
                    valid_moves = board.get_valid_moves(from_pos)
                    for move in valid_moves:
                        all_moves.append((from_pos, move.end_pos))
        
        # Apply move ordering for better pruning
        all_moves = self.order_moves(board, all_moves)
        
        if is_maximizing:
            max_score = float('-inf')
            
            for from_pos, to_pos in all_moves:
                if self.time_limit_exceeded:
                    break
                    
                # Make the move on a copy
                temp_board = board.copy_board()
                temp_board.move_piece(from_pos, to_pos)
                
                # Recurse and find the best score
                score = self.alpha_beta(temp_board, depth - 1, alpha, beta, False, ai_color)
                max_score = max(max_score, score)
                
                # Update alpha
                alpha = max(alpha, max_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
            return max_score
        else:
            min_score = float('inf')
            
            for from_pos, to_pos in all_moves:
                if self.time_limit_exceeded:
                    break
                    
                # Make the move on a copy
                temp_board = board.copy_board()
                temp_board.move_piece(from_pos, to_pos)
                
                # Recurse and find the best score
                score = self.alpha_beta(temp_board, depth - 1, alpha, beta, True, ai_color)
                min_score = min(min_score, score)
                
                # Update beta
                beta = min(beta, min_score)
                
                # Alpha-beta pruning
                if beta <= alpha:
                    break
            
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
        
        # Count material and piece position value
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                
                if piece.type == PieceType.EMPTY:
                    continue
                
                # Get base material value
                value = self.piece_values[piece.type]
                
                # Add position value based on piece type (simplified)
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
                    if self.count_pieces(board) < 10:  # Simple endgame check
                        position_value = self.king_end_table[row][col]
                    else:
                        position_value = self.king_mid_table[row][col]
                
                # Invert position tables for black pieces
                if piece.color == PieceColor.BLACK:
                    position_value = position_value * -1
                
                # Add to the total score
                if piece.color == ai_color:
                    score += value + position_value
                else:
                    score -= value + position_value
        
        return score