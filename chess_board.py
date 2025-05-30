import pygame
from enum import Enum

# Constants
BOARD_SIZE = 8
SQUARE_SIZE = 100
WINDOW_SIZE = BOARD_SIZE * SQUARE_SIZE

class PieceType(Enum):
    PAWN = "pawn"
    ROOK = "rook"
    KNIGHT = "knight"
    BISHOP = "bishop"
    QUEEN = "queen" 
    KING = "king"
    EMPTY = "empty"


class PieceColor(Enum):
    WHITE = "white"
    BLACK = "black"
    NONE = "none"


class Piece:
    def __init__(self, piece_type, color, has_moved=False):
        self.type = piece_type
        self.color = color
        self.has_moved = has_moved  # FFor castling and pawn first move

    def __str__(self):
        return f"{self.color.value}_{self.type.value}"

    def __eq__(self, other):
        if not isinstance(other, Piece):
            return False
        return self.type == other.type and self.color == other.color


class Position:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return self.row == other.row and self.col == other.col

    def __hash__(self):
        return hash((self.row, self.col))


class Move:
    def __init__(self, start_pos, end_pos, piece, captured_piece=None, 
                 is_castling=False, is_en_passant=False, promotion_piece=None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.piece = piece
        self.captured_piece = captured_piece
        self.is_castling = is_castling
        self.is_en_passant = is_en_passant
        self.promotion_piece = promotion_piece


class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.turn = PieceColor.WHITE
        self.move_history = []
        self.last_move = None
        self.half_move_clock = 0  # For fifty-move rule
        self.position_history = []  # For threefold repetition
        self.init_board()

    def init_board(self):
        # Set up pawns
        for col in range(BOARD_SIZE):
            self.board[1][col] = Piece(PieceType.PAWN, PieceColor.BLACK)
            self.board[6][col] = Piece(PieceType.PAWN, PieceColor.WHITE)

        # Set up other pieces
        back_row_pieces = [
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN,
            PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK
        ]

        for col in range(BOARD_SIZE):
            self.board[0][col] = Piece(back_row_pieces[col], PieceColor.BLACK)
            self.board[7][col] = Piece(back_row_pieces[col], PieceColor.WHITE)

        # Empty squares
        for row in range(2, 6):
            for col in range(BOARD_SIZE):
                self.board[row][col] = Piece(PieceType.EMPTY, PieceColor.NONE)

        # Record initial position for threefold repetition detection
        self.add_position_to_history()

    def get_piece(self, position):
        return self.board[position.row][position.col]

    def set_piece(self, position, piece):
        self.board[position.row][position.col] = piece

    def copy_board(self):
        """Create a copy of the current board"""
        new_board = ChessBoard()
        
        # Copy pieces
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.get_piece(Position(row, col))
                new_board.board[row][col] = Piece(piece.type, piece.color, piece.has_moved)
        
        # Copy state variables
        new_board.turn = self.turn
        new_board.last_move = self.last_move
        new_board.half_move_clock = self.half_move_clock
        
        return new_board
        
    def move_piece(self, start_pos, end_pos):
        """
        Move a piece on the board and handle special moves
        Returns the move object if the move was successful, None otherwise
        """
        piece = self.get_piece(start_pos)
        
        # Check if it's the correct player's turn
        if piece.color != self.turn:
            return None
        
        # Get all valid moves for this piece
        valid_moves = self.get_valid_moves(start_pos)
        move_found = None
        
        # Find if the requested move is valid
        for move in valid_moves:
            if move.end_pos.row == end_pos.row and move.end_pos.col == end_pos.col:
                move_found = move
                break
        
        if not move_found:
            return None
        
        # Execute the move
        self.apply_move(move_found)
        
        return move_found
    
    def apply_move(self, move):
        """Apply a valid move and update game state"""
        captured_piece = self.get_piece(move.end_pos)
        is_capture = captured_piece.type != PieceType.EMPTY
        is_pawn_move = move.piece.type == PieceType.PAWN
        
        # Handle castling
        if move.is_castling:
            # Move king
            self.set_piece(move.end_pos, move.piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            move.piece.has_moved = True
            
            # Move rook
            row = move.start_pos.row
            if move.end_pos.col > move.start_pos.col:  # Kingside
                rook_start = Position(row, 7)
                rook_end = Position(row, 5)
            else:  # Queenside
                rook_start = Position(row, 0)
                rook_end = Position(row, 3)
            
            rook = self.get_piece(rook_start)
            self.set_piece(rook_start, Piece(PieceType.EMPTY, PieceColor.NONE))
            self.set_piece(rook_end, rook)
            rook.has_moved = True
            
        # Handle en passant
        elif move.is_en_passant:
            # Move pawn
            self.set_piece(move.end_pos, move.piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            
            # Remove captured pawn
            capture_pos = Position(move.start_pos.row, move.end_pos.col)
            self.set_piece(capture_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            is_capture = True
            
        # Handle promotion
        elif is_pawn_move and (move.end_pos.row == 0 or move.end_pos.row == 7):
            # Default promotion to queen
            promotion_piece = Piece(PieceType.QUEEN, move.piece.color, True)
            self.set_piece(move.end_pos, promotion_piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            
        # Regular move
        else:
            self.set_piece(move.end_pos, move.piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            move.piece.has_moved = True
        
        # Update game state
        self.move_history.append(move)
        self.last_move = move
        
        # Update fifty-move rule counter
        if is_pawn_move or is_capture:
            self.half_move_clock = 0
        else:
            self.half_move_clock += 1
        
        # Switch turn
        self.turn = PieceColor.BLACK if self.turn == PieceColor.WHITE else PieceColor.WHITE
        
        # Record position for threefold repetition detection
        self.add_position_to_history()

    def apply_move_without_validation(self, move):
        """Apply a move without validating it (for internal use)"""
        # Handle special moves
        if move.is_castling:
            # Move king
            self.set_piece(move.end_pos, move.piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            
            # Move rook
            row = move.start_pos.row
            if move.end_pos.col > move.start_pos.col:  # Kingside
                rook_start = Position(row, 7)
                rook_end = Position(row, 5)
            else:  # Queenside
                rook_start = Position(row, 0)
                rook_end = Position(row, 3)
            
            rook = self.get_piece(rook_start)
            self.set_piece(rook_start, Piece(PieceType.EMPTY, PieceColor.NONE))
            self.set_piece(rook_end, rook)
            
        elif move.is_en_passant:
            # Move pawn
            self.set_piece(move.end_pos, move.piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            
            # Remove captured pawn
            capture_pos = Position(move.start_pos.row, move.end_pos.col)
            self.set_piece(capture_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            
        else:
            # Regular move
            if move.promotion_piece:
                self.set_piece(move.end_pos, move.promotion_piece)
            else:
                self.set_piece(move.end_pos, move.piece)
            self.set_piece(move.start_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
    
    def get_valid_moves(self, position):
        """Get all valid moves for the piece at the given position"""
        piece = self.get_piece(position)
        if piece.type == PieceType.EMPTY or piece.color != self.turn:
            return []
            
        # Get raw moves for the piece type
        raw_moves = []
        if piece.type == PieceType.PAWN:
            raw_moves = self.get_pawn_moves(position)
        elif piece.type == PieceType.ROOK:
            raw_moves = self.get_rook_moves(position)
        elif piece.type == PieceType.KNIGHT:
            raw_moves = self.get_knight_moves(position)
        elif piece.type == PieceType.BISHOP:
            raw_moves = self.get_bishop_moves(position)
        elif piece.type == PieceType.QUEEN:
            raw_moves = self.get_queen_moves(position)
        elif piece.type == PieceType.KING:
            raw_moves = self.get_king_moves(position)
        
        # Filter out moves that would leave the king in check
        valid_moves = []
        for move in raw_moves:
            # Make the move on a temporary board
            temp_board = self.copy_board()
            temp_board.apply_move_without_validation(move)
            
            # If the king is not in check after the move, it's valid
            if not temp_board.is_check(piece.color):
                valid_moves.append(move)
        
        return valid_moves
    
    def get_pawn_moves(self, position):
        """Get all valid pawn moves from the given position"""
        moves = []
        piece = self.get_piece(position)
        direction = -1 if piece.color == PieceColor.WHITE else 1  # White pawns move up (decreasing row)
        
        # One square forward
        forward_pos = Position(position.row + direction, position.col)
        if 0 <= forward_pos.row < BOARD_SIZE and self.get_piece(forward_pos).type == PieceType.EMPTY:
            moves.append(Move(position, forward_pos, piece))
            
            # Two squares forward from starting position
            if ((piece.color == PieceColor.WHITE and position.row == 6) or 
               (piece.color == PieceColor.BLACK and position.row == 1)):
                double_forward_pos = Position(position.row + 2 * direction, position.col)
                if self.get_piece(double_forward_pos).type == PieceType.EMPTY:
                    moves.append(Move(position, double_forward_pos, piece))
        
        # Capturing diagonally
        for col_offset in [-1, 1]:
            capture_pos = Position(position.row + direction, position.col + col_offset)
            if 0 <= capture_pos.row < BOARD_SIZE and 0 <= capture_pos.col < BOARD_SIZE:
                capture_piece = self.get_piece(capture_pos)
                if capture_piece.type != PieceType.EMPTY and capture_piece.color != piece.color:
                    moves.append(Move(position, capture_pos, piece, capture_piece))
        
        # En passant
        if self.last_move:
            last_move = self.last_move
            # Check if the last move was a pawn moving two squares
            if (last_move.piece.type == PieceType.PAWN and 
                abs(last_move.start_pos.row - last_move.end_pos.row) == 2):
                # Check if our pawn is in position for en passant
                if (position.row == last_move.end_pos.row and 
                    abs(position.col - last_move.end_pos.col) == 1):
                    en_passant_pos = Position(position.row + direction, last_move.end_pos.col)
                    en_passant_move = Move(position, en_passant_pos, piece)
                    en_passant_move.is_en_passant = True
                    en_passant_move.captured_piece = last_move.piece
                    moves.append(en_passant_move)
        
        return moves
    
    def get_rook_moves(self, position):
        """Get all valid rook moves from the given position"""
        moves = []
        piece = self.get_piece(position)
        
        # Directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        for dr, dc in directions:
            for i in range(1, BOARD_SIZE):
                end_row, end_col = position.row + i * dr, position.col + i * dc
                
                # Check if the position is on the board
                if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
                    break
                    
                end_pos = Position(end_row, end_col)
                end_piece = self.get_piece(end_pos)
                
                # Empty square - can move here
                if end_piece.type == PieceType.EMPTY:
                    moves.append(Move(position, end_pos, piece))
                # Enemy piece - can capture and then stop
                elif end_piece.color != piece.color:
                    moves.append(Move(position, end_pos, piece, end_piece))
                    break
                # Own piece - can't move here
                else:
                    break
        
        return moves
    
    def get_knight_moves(self, position):
        """Get all valid knight moves from the given position"""
        moves = []
        piece = self.get_piece(position)
        
        # All possible knight moves
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dr, dc in knight_moves:
            end_row, end_col = position.row + dr, position.col + dc
            
            # Check if the position is on the board
            if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
                continue
                
            end_pos = Position(end_row, end_col)
            end_piece = self.get_piece(end_pos)
            
            # Can move if square is empty or has an enemy piece
            if end_piece.type == PieceType.EMPTY or end_piece.color != piece.color:
                moves.append(Move(position, end_pos, piece, 
                                 end_piece if end_piece.type != PieceType.EMPTY else None))
        
        return moves
    
    def get_bishop_moves(self, position):
        """Get all valid bishop moves from the given position"""
        moves = []
        piece = self.get_piece(position)
        
        # Directions: diagonal (up-left, up-right, down-right, down-left)
        directions = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            for i in range(1, BOARD_SIZE):
                end_row, end_col = position.row + i * dr, position.col + i * dc
                
                # Check if the position is on the board
                if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
                    break
                    
                end_pos = Position(end_row, end_col)
                end_piece = self.get_piece(end_pos)
                
                # Empty square - can move here
                if end_piece.type == PieceType.EMPTY:
                    moves.append(Move(position, end_pos, piece))
                # Enemy piece - can capture and then stop
                elif end_piece.color != piece.color:
                    moves.append(Move(position, end_pos, piece, end_piece))
                    break
                # Own piece - can't move here
                else:
                    break
        
        return moves
    
    def get_queen_moves(self, position):
        """Get all valid queen moves from the given position"""
        # Queen moves like a rook and bishop combined
        return self.get_rook_moves(position) + self.get_bishop_moves(position)
    
    def get_king_moves(self, position):
        """Get all valid king moves from the given position"""
        moves = []
        piece = self.get_piece(position)
        
        # Regular king moves (one square in any direction)
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dr, dc in directions:
            end_row, end_col = position.row + dr, position.col + dc
            
            # Check if the position is on the board
            if not (0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE):
                continue
                
            end_pos = Position(end_row, end_col)
            end_piece = self.get_piece(end_pos)
            
            # Can move if square is empty or has an enemy piece
            if end_piece.type == PieceType.EMPTY or end_piece.color != piece.color:
                moves.append(Move(position, end_pos, piece, 
                                 end_piece if end_piece.type != PieceType.EMPTY else None))
        
        # Castling
        if not piece.has_moved and not self.is_check(piece.color):
            row = position.row
            
            # Kingside castling
            if self.can_castle_kingside(piece.color):
                moves.append(Move(position, Position(row, 6), piece, is_castling=True))
                
            # Queenside castling
            if self.can_castle_queenside(piece.color):
                moves.append(Move(position, Position(row, 2), piece, is_castling=True))
        
        return moves
    
    def can_castle_kingside(self, color):
        """Check if kingside castling is possible for the given color"""
        row = 7 if color == PieceColor.WHITE else 0
        
        # Check if king is in place and hasn't moved
        king_pos = Position(row, 4)
        king = self.get_piece(king_pos)
        if king.type != PieceType.KING or king.has_moved:
            return False
            
        # Check if rook is in place and hasn't moved
        rook_pos = Position(row, 7)
        rook = self.get_piece(rook_pos)
        if rook.type != PieceType.ROOK or rook.color != color or rook.has_moved:
            return False
        
        # Check if squares between king and rook are empty
        for col in range(5, 7):
            if self.get_piece(Position(row, col)).type != PieceType.EMPTY:
                return False
        
        # Check if king would pass through or end up in check
        for col in range(5, 7):
            # Create a temporary board to test if the king would be in check
            temp_board = self.copy_board()
            temp_board.set_piece(king_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            temp_board.set_piece(Position(row, col), Piece(PieceType.KING, color, True))
            if temp_board.is_check(color):
                return False
        
        return True
    
    def can_castle_queenside(self, color):
        """Check if queenside castling is possible for the given color"""
        row = 7 if color == PieceColor.WHITE else 0
        
        # Check if king is in place and hasn't moved
        king_pos = Position(row, 4)
        king = self.get_piece(king_pos)
        if king.type != PieceType.KING or king.has_moved:
            return False
            
        # Check if rook is in place and hasn't moved
        rook_pos = Position(row, 0)
        rook = self.get_piece(rook_pos)
        if rook.type != PieceType.ROOK or rook.color != color or rook.has_moved:
            return False
        
        # Check if squares between king and rook are empty
        for col in range(1, 4):
            if self.get_piece(Position(row, col)).type != PieceType.EMPTY:
                return False
        
        # Check if king would pass through or end up in check
        for col in range(2, 4):
            # Create a temporary board to test if the king would be in check
            temp_board = self.copy_board()
            temp_board.set_piece(king_pos, Piece(PieceType.EMPTY, PieceColor.NONE))
            temp_board.set_piece(Position(row, col), Piece(PieceType.KING, color, True))
            if temp_board.is_check(color):
                return False
        
        return True
    
    def is_check(self, color):
        """Check if the given color's king is in check"""
        # Find the king
        king_position = None
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece.type == PieceType.KING and piece.color == color:
                    king_position = Position(row, col)
                    break
            if king_position:
                break
        
        # Check if any opponent piece can capture the king
        opponent_color = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece_pos = Position(row, col)
                piece = self.get_piece(piece_pos)
                if piece.color == opponent_color:
                    # For each opponent piece, see if it can move to the king's position
                    for move in self.get_raw_moves(piece_pos):
                        if move.end_pos.row == king_position.row and move.end_pos.col == king_position.col:
                            return True
        return False
    
    def get_raw_moves(self, position):
        """Get moves without considering check"""
        piece = self.get_piece(position)
        if piece.type == PieceType.EMPTY:
            return []
            
        if piece.type == PieceType.PAWN:
            return self.get_pawn_moves(position)
        elif piece.type == PieceType.ROOK:
            return self.get_rook_moves(position)
        elif piece.type == PieceType.KNIGHT:
            return self.get_knight_moves(position)
        elif piece.type == PieceType.BISHOP:
            return self.get_bishop_moves(position)
        elif piece.type == PieceType.QUEEN:
            return self.get_queen_moves(position)
        elif piece.type == PieceType.KING:
            # For raw moves, we only consider normal king movement, not castling
            # Since castling depends on check validation
            moves = []
            
            # Regular king moves (one square in any direction)
            directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
            
            for dr, dc in directions:
                end_row, end_col = position.row + dr, position.col + dc
                
                if 0 <= end_row < BOARD_SIZE and 0 <= end_col < BOARD_SIZE:
                    end_pos = Position(end_row, end_col)
                    end_piece = self.get_piece(end_pos)
                    
                    if end_piece.type == PieceType.EMPTY or end_piece.color != piece.color:
                        moves.append(Move(position, end_pos, piece, 
                                        end_piece if end_piece.type != PieceType.EMPTY else None))
            
            return moves
        
        return []
    
    def is_checkmate(self, color):
        """Check if the given color is in checkmate"""
        if not self.is_check(color):
            return False
            
        # Try all possible moves to see if any can get out of check
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                position = Position(row, col)
                piece = self.get_piece(position)
                if piece.color == color:
                    valid_moves = self.get_valid_moves(position)
                    if valid_moves:  # If there's at least one valid move, it's not checkmate
                        return False
        return True
    
    def is_stalemate(self, color):
        """Check if the given color is in stalemate"""
        if self.is_check(color):
            return False
            
        # Check if the player has any valid moves
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                position = Position(row, col)
                piece = self.get_piece(position)
                if piece.color == color:
                    valid_moves = self.get_valid_moves(position)
                    if valid_moves:  # If there's at least one valid move, it's not stalemate
                        return False
        return True
    
    def is_fifty_move_rule_draw(self):
        """Check if the game is a draw due to the fifty-move rule"""
        return self.half_move_clock >= 100  # 50 moves = 100 half-moves
    
    def add_position_to_history(self):
        """Add current position to history for threefold repetition detection"""
        # Create a hashable representation of the current board state
        position_hash = self.get_board_hash()
        self.position_history.append(position_hash)
    
    def get_board_hash(self):
        """Create a hashable representation of the board"""
        # Convert board to a tuple of tuples for hashing
        board_tuple = tuple(tuple((piece.type.value, piece.color.value) for piece in row) for row in self.board)
        
        # Include castling rights in the hash
        castling_rights = []
        
        # White kingside castling
        king_pos = Position(7, 4)
        rook_pos = Position(7, 7)
        king = self.get_piece(king_pos)
        rook = self.get_piece(rook_pos)
        can_castle = (king.type == PieceType.KING and not king.has_moved and 
                      rook.type == PieceType.ROOK and not rook.has_moved)
        castling_rights.append(can_castle)
        
        # White queenside castling
        rook_pos = Position(7, 0)
        rook = self.get_piece(rook_pos)
        can_castle = (king.type == PieceType.KING and not king.has_moved and 
                      rook.type == PieceType.ROOK and not rook.has_moved)
        castling_rights.append(can_castle)
        
        # Black kingside castling
        king_pos = Position(0, 4)
        rook_pos = Position(0, 7)
        king = self.get_piece(king_pos)
        rook = self.get_piece(rook_pos)
        can_castle = (king.type == PieceType.KING and not king.has_moved and 
                      rook.type == PieceType.ROOK and not rook.has_moved)
        castling_rights.append(can_castle)
        
        # Black queenside castling
        rook_pos = Position(0, 0)
        rook = self.get_piece(rook_pos)
        can_castle = (king.type == PieceType.KING and not king.has_moved and 
                      rook.type == PieceType.ROOK and not rook.has_moved)
        castling_rights.append(can_castle)
        
        castling_tuple = tuple(castling_rights)
        
        # Include en passant target in hash if applicable
        en_passant_target = None
        if self.last_move and self.last_move.piece.type == PieceType.PAWN:
            if abs(self.last_move.start_pos.row - self.last_move.end_pos.row) == 2:
                # A pawn moved two squares, making en passant possible
                en_passant_target = (
                    (self.last_move.start_pos.row + self.last_move.end_pos.row) // 2,
                    self.last_move.start_pos.col
                )
        
        return hash((board_tuple, self.turn.value, castling_tuple, en_passant_target))
    
    def is_threefold_repetition(self):
        """Check if the game is a draw due to threefold repetition"""
        current_hash = self.position_history[-1]
        repetition_count = 0
        
        for position_hash in self.position_history:
            if position_hash == current_hash:
                repetition_count += 1
                if repetition_count >= 3:
                    return True
        
        return False