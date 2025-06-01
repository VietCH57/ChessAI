import numpy as np
from chess_board import ChessBoard, Position, PieceType, PieceColor

class ChessEncoder:
    """
    Encodes a chess board into the AlphaZero 119-plane representation
    
    Encoding follows AlphaZero paper:
    - 6 planes for each piece type for white (6 types)
    - 6 planes for each piece type for black (6 types)
    - 5 planes for number of repetitions (0, 1, 2, 3, >=4)
    - 1 plane for side to move (1=white, 0=black) 
    - 1 plane for move number
    - 4 planes for castling rights (white kingside, white queenside, black kingside, black queenside)
    - 1 plane for no-progress count (half-move clock for 50-move rule)
    - 8 planes for history of last moves
    Total: 6 + 6 + 5 + 1 + 1 + 4 + 1 + 8*12 = 119 planes
    """
    
    def __init__(self):
        # Map piece types to indices
        self.piece_to_index = {
            (PieceType.PAWN, PieceColor.WHITE): 0,
            (PieceType.KNIGHT, PieceColor.WHITE): 1,
            (PieceType.BISHOP, PieceColor.WHITE): 2,
            (PieceType.ROOK, PieceColor.WHITE): 3,
            (PieceType.QUEEN, PieceColor.WHITE): 4,
            (PieceType.KING, PieceColor.WHITE): 5,
            (PieceType.PAWN, PieceColor.BLACK): 6,
            (PieceType.KNIGHT, PieceColor.BLACK): 7,
            (PieceType.BISHOP, PieceColor.BLACK): 8,
            (PieceType.ROOK, PieceColor.BLACK): 9,
            (PieceType.QUEEN, PieceColor.BLACK): 10,
            (PieceType.KING, PieceColor.BLACK): 11,
        }
        
        # Mapping from move coordinates to policy indices
        # This maps 8x8x8x8 possible from-to moves to indices in the policy vector
        self.move_to_index = {}
        idx = 0
        for from_row in range(8):
            for from_col in range(8):
                for to_row in range(8):
                    for to_col in range(8):
                        self.move_to_index[(from_row, from_col, to_row, to_col)] = idx
                        idx += 1
        
        # Add underpromotion moves (knight, bishop, rook)
        for from_col in range(8):
            for to_col in range(8):
                # White pawn promotion (from row 6 to row 7)
                if abs(from_col - to_col) <= 1:  # Capture or straight
                    for piece_idx in range(1, 4):  # knight, bishop, rook
                        self.move_to_index[(6, from_col, 7, to_col, piece_idx)] = idx
                        idx += 1
                
                # Black pawn promotion (from row 1 to row 0)
                if abs(from_col - to_col) <= 1:  # Capture or straight
                    for piece_idx in range(1, 4):  # knight, bishop, rook
                        self.move_to_index[(1, from_col, 0, to_col, piece_idx)] = idx
                        idx += 1
    
    def encode_board(self, board, move_history=None):
        """
        Encode a chess board into a 119-plane representation
        
        Args:
            board: ChessBoard instance
            move_history: List of previous board states for repetition planes
            
        Returns:
            np.array with shape (119, 8, 8)
        """
        # Initialize planes
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        # 1-12: Piece type planes (white and black)
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    idx = self.piece_to_index.get((piece.type, piece.color))
                    if idx is not None:
                        planes[idx, row, col] = 1.0
        
        # 13-17: Repetition count (0, 1, 2, 3, >=4)
        repetition_count = 0
        if board.position_history:
            # Count repetitions of current position
            current_position = board.position_history[-1]
            repetition_count = sum(1 for pos in board.position_history if pos == current_position) - 1
        
        planes[12 + min(repetition_count, 4)] = 1.0  # Use min() to cap at 4+ repetitions
        
        # 18: Side to move (1 for white, 0 for black)
        if board.turn == PieceColor.WHITE:
            planes[17] = 1.0
        
        # 19: Move number (normalized to [0, 1])
        planes[18] = min(1.0, len(board.move_history) / 100.0)
        
        # 20-23: Castling rights
        # We need to extract castling rights from your board implementation
        # This is an approximation - adjust according to your implementation
        white_king = board.get_piece(Position(7, 4))
        white_rook_kingside = board.get_piece(Position(7, 7))
        white_rook_queenside = board.get_piece(Position(7, 0))
        black_king = board.get_piece(Position(0, 4))
        black_rook_kingside = board.get_piece(Position(0, 7))
        black_rook_queenside = board.get_piece(Position(0, 0))
        
        # White kingside castling
        if (white_king.type == PieceType.KING and 
            white_king.color == PieceColor.WHITE and 
            not white_king.has_moved and
            white_rook_kingside.type == PieceType.ROOK and
            white_rook_kingside.color == PieceColor.WHITE and
            not white_rook_kingside.has_moved):
            planes[19] = 1.0
            
        # White queenside castling
        if (white_king.type == PieceType.KING and 
            white_king.color == PieceColor.WHITE and 
            not white_king.has_moved and
            white_rook_queenside.type == PieceType.ROOK and
            white_rook_queenside.color == PieceColor.WHITE and
            not white_rook_queenside.has_moved):
            planes[20] = 1.0
            
        # Black kingside castling
        if (black_king.type == PieceType.KING and 
            black_king.color == PieceColor.BLACK and 
            not black_king.has_moved and
            black_rook_kingside.type == PieceType.ROOK and
            black_rook_kingside.color == PieceColor.BLACK and
            not black_rook_kingside.has_moved):
            planes[21] = 1.0
            
        # Black queenside castling
        if (black_king.type == PieceType.KING and 
            black_king.color == PieceColor.BLACK and 
            not black_king.has_moved and
            black_rook_queenside.type == PieceType.ROOK and
            black_rook_queenside.color == PieceColor.BLACK and
            not black_rook_queenside.has_moved):
            planes[22] = 1.0
        
        # 24: No-progress count (half-move clock for 50-move rule)
        planes[23] = min(1.0, board.half_move_clock / 100.0)
        
        # 25-119: History of last 8 moves (12 planes per move = 8 * 12 = 96 planes)
        if move_history:
            # Take up to 8 most recent board states
            recent_history = move_history[-8:] if len(move_history) >= 8 else move_history
            
            # For each historical board state
            for i, hist_board in enumerate(recent_history):
                start_plane = 24 + i * 12  # 12 planes per historical board
                
                # Encode pieces
                for row in range(8):
                    for col in range(8):
                        piece = hist_board.get_piece(Position(row, col))
                        if piece.type != PieceType.EMPTY:
                            idx = self.piece_to_index.get((piece.type, piece.color))
                            if idx is not None:
                                planes[start_plane + idx, row, col] = 1.0
        
        return planes
    
    def decode_move(self, policy_output, board):
        """
        Convert a policy output vector to a valid move
        
        Args:
            policy_output: Array of move probabilities (1968,)
            board: Current ChessBoard to validate moves against
            
        Returns:
            (from_pos, to_pos): Position tuple representing the selected move
        """
        # Get all valid moves for the current board
        valid_moves = []
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece.color == board.turn:
                    moves = board.get_valid_moves(pos)
                    for move in moves:
                        valid_moves.append((pos, move.end_pos, move))
        
        # Mask the policy output for only valid moves
        masked_policy = np.zeros(len(policy_output))
        valid_indices = []
        
        for from_pos, to_pos, move_obj in valid_moves:
            # Regular move
            move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
            if move_key in self.move_to_index:
                move_idx = self.move_to_index[move_key]
                masked_policy[move_idx] = policy_output[move_idx]
                valid_indices.append(move_idx)
            
            # Handle promotions
            if move_obj.promotion_piece:
                piece_map = {
                    PieceType.KNIGHT: 1,
                    PieceType.BISHOP: 2,
                    PieceType.ROOK: 3,
                    PieceType.QUEEN: 0  # Queen is default, covered by regular move
                }
                
                # Only handle underpromotions explicitly (knight, bishop, rook)
                if move_obj.promotion_piece != PieceType.QUEEN:
                    promotion_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col, 
                                   piece_map[move_obj.promotion_piece])
                    if promotion_key in self.move_to_index:
                        move_idx = self.move_to_index[promotion_key]
                        masked_policy[move_idx] = policy_output[move_idx]
                        valid_indices.append(move_idx)
        
        # If no valid moves have probability, return a random valid move
        if len(valid_indices) == 0 or np.sum(masked_policy) == 0:
            random_idx = np.random.randint(len(valid_moves))
            return valid_moves[random_idx][0], valid_moves[random_idx][1]
        
        # Select move with highest probability among valid moves
        selected_idx = valid_indices[np.argmax(masked_policy[valid_indices])]
        
        # Find the corresponding move
        for from_pos, to_pos, _ in valid_moves:
            move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
            if move_key in self.move_to_index and self.move_to_index[move_key] == selected_idx:
                return from_pos, to_pos
        
        # Fallback to random valid move (should not reach here)
        random_idx = np.random.randint(len(valid_moves))
        return valid_moves[random_idx][0], valid_moves[random_idx][1]