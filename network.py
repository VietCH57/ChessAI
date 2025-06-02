import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from chess_board import ChessBoard, Position, PieceType, PieceColor
from config import AlphaZeroConfig

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZeroNetwork(nn.Module):
    def __init__(self, config: AlphaZeroConfig):
        super().__init__()
        self.config = config
        
        # Initial convolution
        self.conv_block = nn.Sequential(
            nn.Conv2d(config.input_planes, config.num_filters, 3, padding=1),
            nn.BatchNorm2d(config.num_filters),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.num_filters) 
            for _ in range(config.num_residual_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096)  # 64*64 possible moves
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(config.num_filters, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Main body
        x = self.conv_block(x)
        for block in self.residual_blocks:
            x = block(x)
            
        # Heads
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value

class BoardEncoder:
    """Encodes chess board state into 119-plane representation"""
    
    def __init__(self):
        self.piece_planes = 12
        self.history_planes = 96
        self.meta_planes = 11
        
        # Pre-computed piece type mappings
        self.piece_type_to_idx = {
            PieceType.PAWN: 0,
            PieceType.KNIGHT: 1, 
            PieceType.BISHOP: 2,
            PieceType.ROOK: 3,
            PieceType.QUEEN: 4,
            PieceType.KING: 5
        }
        
    def encode_board(self, board: ChessBoard, history: list = None) -> np.ndarray:
        """
        Encode board state into 119-plane tensor
        
        Args:
            board: Current board state
            history: List of previous board states (max 8)
            
        Returns:
            numpy array of shape (119, 8, 8)
        """
        planes = np.zeros((119, 8, 8), dtype=np.float32)
        
        # Fast piece encoding using numpy arrays
        self._encode_pieces(board, planes)
        
        # Fast history encoding
        if history:
            self._encode_history(history, planes)
        
        # Fast metadata encoding
        self._encode_metadata(board, planes)
        
        return planes
    
    def _encode_pieces(self, board: ChessBoard, planes: np.ndarray):
        """Piece encoding using vectorized operations"""
        # Create numpy array from board state
        board_array = np.zeros((8, 8, 2), dtype=np.int8)  # [piece_type, color]
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    piece_idx = self.piece_type_to_idx[piece.type]
                    color_idx = 0 if piece.color == PieceColor.WHITE else 1
                    board_array[row, col, 0] = piece_idx + 1  # +1 to avoid 0 (empty)
                    board_array[row, col, 1] = color_idx
        
        # Vectorized plane filling
        for piece_type_idx in range(6):  # 6 piece types
            for color_idx in range(2):  # 2 colors
                plane_idx = piece_type_idx * 2 + color_idx
                mask = (board_array[:, :, 0] == piece_type_idx + 1) & (board_array[:, :, 1] == color_idx)
                planes[plane_idx] = mask.astype(np.float32)
                
    def _encode_history(self, history: list, planes: np.ndarray):
        """History encoding"""
        start_idx = 12
        for i, hist_board in enumerate(history[-8:]):
            if i >= 8:
                break
            offset = i * 12
            # Reuse piece encoding for history
            hist_planes = np.zeros((12, 8, 8), dtype=np.float32)
            self._encode_pieces(hist_board, hist_planes)
            planes[start_idx + offset:start_idx + offset + 12] = hist_planes
    
    def _encode_metadata(self, board: ChessBoard, planes: np.ndarray):
        """Fast metadata encoding using numpy broadcasting"""
        idx = 12 + 96  # Start after pieces and history
        
        # Current player (vectorized)
        if board.turn == PieceColor.WHITE:
            planes[idx] = 1.0
        idx += 1
        
        # Castling rights (vectorized)
        castling_rights = [
            self._can_castle_kingside(board, PieceColor.WHITE),
            self._can_castle_queenside(board, PieceColor.WHITE),
            self._can_castle_kingside(board, PieceColor.BLACK),
            self._can_castle_queenside(board, PieceColor.BLACK)
        ]
        
        for can_castle in castling_rights:
            if can_castle:
                planes[idx] = 1.0
            idx += 1
        
        # En passant (optimized check)
        if board.last_move and self._is_en_passant_possible(board):
            en_passant_col = board.last_move.end_pos.col
            en_passant_row = 3 if board.turn == PieceColor.WHITE else 4
            planes[idx, en_passant_row, en_passant_col] = 1.0
        idx += 1
        
        # Remaining metadata (vectorized)
        halfmove_norm = min(board.half_move_clock / 100.0, 1.0)
        planes[idx] = halfmove_norm
        idx += 1
        
        move_num = len(board.move_history) // 2 + 1
        move_norm = min(move_num / 200.0, 1.0)
        planes[idx] = move_norm
        idx += 1
        
        # Check status
        if board.is_check(PieceColor.WHITE):
            planes[idx] = 1.0
        idx += 1
        
        if board.is_check(PieceColor.BLACK):
            planes[idx] = 1.0
        idx += 1
        
        # No-progress count
        planes[idx] = halfmove_norm
    
    def _can_castle_kingside(self, board: ChessBoard, color: PieceColor) -> bool:
        return board.can_castle_kingside(color)
    
    def _can_castle_queenside(self, board: ChessBoard, color: PieceColor) -> bool:
        return board.can_castle_queenside(color)
    
    def _is_en_passant_possible(self, board: ChessBoard) -> bool:
        if not board.last_move or board.last_move.piece.type != PieceType.PAWN:
            return False
        return abs(board.last_move.start_pos.row - board.last_move.end_pos.row) == 2

class MoveEncoder:
    """Encodes moves to/from neural network policy format"""
    
    def __init__(self):
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_move_mapping()
    
    def _build_move_mapping(self):
        """Build mapping between moves and policy indices"""
        idx = 0
        
        # Regular moves (64 * 64 = 4096 possible from-to combinations)
        for from_row in range(8):
            for from_col in range(8):
                for to_row in range(8):
                    for to_col in range(8):
                        move_key = (from_row, from_col, to_row, to_col)
                        self.move_to_index[move_key] = idx
                        self.index_to_move[idx] = move_key
                        idx += 1
    
    def encode_move(self, from_pos: Position, to_pos: Position) -> int:
        """Convert move to policy index"""
        move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
        return self.move_to_index.get(move_key, -1)
    
    def decode_move(self, index: int) -> tuple:
        """Convert policy index to move"""
        if index in self.index_to_move:
            from_row, from_col, to_row, to_col = self.index_to_move[index]
            return Position(from_row, from_col), Position(to_row, to_col)
        return None, None