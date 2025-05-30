from chess_board import ChessBoard, Position, Piece, PieceType, PieceColor, Move #Import các lớp cần thiết từ module chess_board
from interface import ChessAI #Import lớp ChessAI từ module interface

class MyChessAI(ChessAI):
    """
    Template cho AI cờ vua.
    Đây là nơi bạn viết logic cho AI của mình.
    """
    
    def __init__(self):
        # Khởi tạo các biến cần thiết cho AI của bạn
        pass
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> tuple[Position, Position]:
        """
        Phương thức chính để quyết định nước đi kế tiếp.
        
        Args:
            board: Trạng thái hiện tại của bàn cờ
            color: Màu quân mà AI đang chơi (PieceColor.WHITE hoặc PieceColor.BLACK)
            
        Returns:
            Tuple gồm (vị_trí_xuất_phát, vị_trí_đích)
        """
        # ===== VIẾT LOGIC CỦA BẠN Ở ĐÂY =====
        
        # Ví dụ đơn giản: Trả về nước đi hợp lệ đầu tiên tìm được
        for row in range(8):
            for col in range(8):
                from_pos = Position(row, col)
                piece = board.get_piece(from_pos)
                
                # Nếu đây là quân cờ của chúng ta
                if piece.color == color:
                    # Lấy tất cả nước đi hợp lệ cho quân này
                    valid_moves = board.get_valid_moves(from_pos)
                    
                    # Nếu có nước đi hợp lệ, trả về nước đi đầu tiên
                    if valid_moves:
                        return (from_pos, valid_moves[0].end_pos)
        
        # Nếu không tìm thấy nước đi nào (điều này không nên xảy ra)
        raise ValueError(f"Không tìm thấy nước đi hợp lệ nào cho {color}!")
        
    def _evaluate_board(self, board: ChessBoard, color: PieceColor) -> float:
        """
        Phương thức phụ để đánh giá trạng thái bàn cờ.
        Giá trị cao hơn nghĩa là tốt hơn cho 'color'.
        
        Args:
            board: Trạng thái bàn cờ cần đánh giá
            color: Màu của bên cần đánh giá
            
        Returns:
            Điểm số đánh giá của bàn cờ
        """
        # Ví dụ về đánh giá đơn giản dựa trên số lượng và giá trị quân cờ
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0  # Vua không có giá trị tính toán
        }
        
        opponent_color = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE
        score = 0
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    value = piece_values.get(piece.type, 0)
                    if piece.color == color:
                        score += value
                    else:
                        score -= value
        
        return score