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
        self.position_history = {}  # Track position frequency
        self.previous_moves = []    # Track last few moves
        
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
    
    def hash_board(self, board):
        """Create a simple hash representation of the board for repetition detection"""
        board_str = ""
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    board_str += f"{piece.type.value}{piece.color.value}{row}{col}"
        return board_str
    
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
        
        # Track board position before making a move
        board_hash = self.hash_board(board)
        self.position_history[board_hash] = self.position_history.get(board_hash, 0) + 1
        
        # Track best moves (not just the single best)
        best_moves = []
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
                
                # Check if this is a repeated move
                if (from_pos, to_pos) in self.previous_moves:
                    continue  # Skip repeated moves unless we have no alternatives
                
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
                
                # Update best moves list
                if score > best_score + 10:  # Significantly better move
                    best_score = score
                    best_moves = [(from_pos, to_pos)]
                elif abs(score - best_score) <= 10:  # Similar score, keep as alternative
                    best_moves.append((from_pos, to_pos))
            
            # If we found valid moves, select one based on our strategy
            if best_moves:
                # Check if any move was already played recently
                fresh_moves = [move for move in best_moves if move not in self.previous_moves]
                if fresh_moves:
                    selected_move = random.choice(fresh_moves)
                else:
                    selected_move = random.choice(best_moves)
                
                # Update move history
                self.previous_moves.append(selected_move)
                if len(self.previous_moves) > 6:  # Only keep track of last 6 moves
                    self.previous_moves.pop(0)
                
                return selected_move
                
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
        
        # Count material and piece position value
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                
                if piece.type == PieceType.EMPTY:
                    continue
                
                # Get base material value
                value = self.piece_values[piece.type]
                
                # Get position value based on piece type
                position_value = 0
                
                # For Black pieces, we need to flip the row coordinate to get the correct table position
                # since all tables are designed from White's perspective
                table_row = row if piece.color == PieceColor.WHITE else 7 - row
                
                if piece.type == PieceType.PAWN:
                    position_value = self.pawn_table[table_row][col]
                    
                    # Add bonus for pawn advancement
                    if piece.color == PieceColor.WHITE:
                        advancement = 7 - row  # How far the pawn has advanced
                    else:
                        advancement = row  # For black pawns
                    
                    position_value += advancement * 5  # Small bonus for advancement
                    
                elif piece.type == PieceType.KNIGHT:
                    position_value = self.knight_table[table_row][col]
                elif piece.type == PieceType.BISHOP:
                    position_value = self.bishop_table[table_row][col]
                elif piece.type == PieceType.ROOK:
                    position_value = self.rook_table[table_row][col]
                elif piece.type == PieceType.QUEEN:
                    position_value = self.queen_table[table_row][col]
                elif piece.type == PieceType.KING:
                    # Use different tables for the king depending on game phase
                    if self.count_pieces(board) < 10:
                        position_value = self.king_end_table[table_row][col]
                    else:
                        position_value = self.king_mid_table[table_row][col]
                
                # Add to the total score
                if piece.color == ai_color:
                    score += value + position_value
                else:
                    score -= value + position_value
        
        # Center control bonus
        for row in range(2, 6):
            for col in range(2, 6):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    center_bonus = 10
                    if piece.color == ai_color:
                        score += center_bonus
                    else:
                        score -= center_bonus
        
        # Add penalty for repetitive positions
        board_hash = self.hash_board(board)
        repetition_count = self.position_history.get(board_hash, 0)
        if repetition_count > 0:
            repetition_penalty = repetition_count * 50
            if board.turn == ai_color:
                score -= repetition_penalty
            else:
                score += repetition_penalty
        
        # Bonus for having both bishops (bishop pair)
        white_bishops = 0
        black_bishops = 0
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type == PieceType.BISHOP:
                    if piece.color == PieceColor.WHITE:
                        white_bishops += 1
                    else:
                        black_bishops += 1
        
        if white_bishops >= 2:
            if ai_color == PieceColor.WHITE:
                score += 50  # Bonus for having bishop pair
            else:
                score -= 50
        
        if black_bishops >= 2:
            if ai_color == PieceColor.BLACK:
                score += 50
            else:
                score -= 50
        
        return score