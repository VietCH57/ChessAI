from headless import HeadlessChessGame
from chess_board import ChessBoard, Position, PieceType, PieceColor
from interface import ChessAI
from typing import Dict, List, Tuple, Optional, Any, Callable
import random
import json
import os
import time
import numpy as np

class ChessTrainer:
    """
    Framework huấn luyện AI cờ vua, đặc biệt cho neural networks và deep learning
    """
    
    def __init__(self, output_dir='training_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.game_engine = HeadlessChessGame()
    
    def train_self_play(self, 
                       ai: ChessAI, 
                       num_games: int = 1000, 
                       opponent_ai: Optional[ChessAI] = None,
                       save_interval: int = 100,
                       max_moves: int = 200,
                       callback: Optional[Callable] = None):
        """
        Huấn luyện AI thông qua self-play hoặc đấu với AI khác
        
        Args:
            ai: AI cần huấn luyện
            num_games: Số trận đấu cho huấn luyện
            opponent_ai: AI đối thủ (nếu None thì đấu với chính nó)
            save_interval: Lưu dữ liệu sau bao nhiêu trận
            max_moves: Số nước đi tối đa mỗi trận
            callback: Hàm gọi sau mỗi trận để cập nhật/huấn luyện model
            
        Returns:
            dict: Thống kê huấn luyện
        """
        if opponent_ai is None:
            opponent_ai = ai  # Self-play
        
        # Chạy nhiều trận đấu
        stats = self.game_engine.run_many_games(
            white_ai=ai, 
            black_ai=opponent_ai, 
            num_games=num_games, 
            max_moves=max_moves,
            collect_data=True,
            swap_sides=True,
            output_file=f"{self.output_dir}/training_stats.json"
        )
        
        # Gọi callback sau mỗi trận nếu được cung cấp
        if callback:
            for i, game in enumerate(stats["games"]):
                callback(ai, game, i)
                
                # Lưu checkpoint định kỳ nếu AI hỗ trợ
                if (i+1) % save_interval == 0 and hasattr(ai, 'save_model'):
                    ai.save_model(f"{self.output_dir}/model_checkpoint_{i+1}")
        
        return stats
    
    def evaluate_ai(self, ai1: ChessAI, ai2: ChessAI, num_games: int = 100, max_moves: int = 200):
        """
        Đánh giá hiệu suất AI bằng cách so sánh với AI khác
        
        Args:
            ai1: AI thứ nhất cần đánh giá
            ai2: AI thứ hai để so sánh
            num_games: Số trận đấu để đánh giá
            max_moves: Số nước đi tối đa mỗi trận
            
        Returns:
            dict: Thống kê đánh giá
        """
        evaluation = self.game_engine.run_many_games(
            white_ai=ai1,
            black_ai=ai2,
            num_games=num_games,
            max_moves=max_moves,
            collect_data=False,
            swap_sides=True,
            output_file=f"{self.output_dir}/evaluation_results.json"
        )
        
        print("\n=== EVALUATION RESULTS ===")
        print(f"Games played: {evaluation['total_games']}")
        print(f"AI1 win rate: {evaluation['white_win_percentage']:.1f}%")
        print(f"AI2 win rate: {evaluation['black_win_percentage']:.1f}%")
        print(f"Draw rate: {evaluation['draw_percentage']:.1f}%")
        print(f"Average game length: {evaluation['avg_game_length']:.1f} moves")
        print("=======================\n")
        
        return evaluation
    
    def board_to_tensor(self, board: ChessBoard):
        """
        Chuyển đổi trạng thái bàn cờ thành tensor cho deep learning
        Format: 12 kênh (6 loại quân x 2 màu) kích thước 8x8
        
        Args:
            board: Bàn cờ hiện tại
            
        Returns:
            numpy array: Tensor biểu diễn bàn cờ kích thước (12, 8, 8)
        """
        # 12 kênh biểu diễn các loại quân cờ
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_types = [PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP, 
                       PieceType.ROOK, PieceType.QUEEN, PieceType.KING]
        
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    # Xác định kênh trong tensor
                    piece_idx = piece_types.index(piece.type)
                    if piece.color == PieceColor.BLACK:
                        piece_idx += 6  # Kênh 6-11 cho quân đen
                    
                    # Đặt giá trị 1 cho vị trí quân cờ
                    tensor[piece_idx, row, col] = 1.0
        
        return tensor
    
    def generate_data_batch(self, num_positions=1000, files=None):
        """
        Tạo batch dữ liệu từ các file lịch sử trận đấu đã lưu
        
        Args:
            num_positions: Số lượng vị trí cờ cần lấy
            files: Danh sách file lịch sử (nếu None, sử dụng tất cả trong output_dir)
            
        Returns:
            tuple: (X, y) là dữ liệu đầu vào và nhãn
        """
        if files is None:
            files = [f for f in os.listdir(self.output_dir) if f.startswith('game_') and f.endswith('.json')]
        
        if not files:
            raise ValueError("Không tìm thấy file dữ liệu trận đấu")
            
        positions = []
        outcomes = []
        
        # Đọc dữ liệu từ các file
        for file in files[:min(len(files), 100)]:  # Giới hạn số file đọc
            filepath = os.path.join(self.output_dir, file)
            try:
                with open(filepath, 'r') as f:
                    game_data = json.load(f)
                
                if 'history' not in game_data or not game_data['history']:
                    continue
                    
                # Xác định kết quả trận đấu
                if game_data.get('winner') == 'white':
                    result = 1.0
                elif game_data.get('winner') == 'black':
                    result = -1.0
                else:
                    result = 0.0
                    
                # Lấy dữ liệu từ mỗi nước đi
                for move in game_data['history']:
                    if 'state_before' in move:
                        positions.append(move['state_before'])
                        outcomes.append(result)
            except Exception as e:
                print(f"Lỗi đọc file {file}: {e}")
        
        # Lấy ngẫu nhiên số lượng vị trí cần thiết
        if len(positions) > num_positions:
            indices = random.sample(range(len(positions)), num_positions)
            positions = [positions[i] for i in indices]
            outcomes = [outcomes[i] for i in indices]
        
        # Chuyển đổi dữ liệu thành tensor
        X = []
        for pos in positions:
            # Chuyển từ dictionary state sang tensor
            # Cần thêm code chuyển đổi tùy thuộc vào định dạng lưu trữ
            # ...
            
            X.append(np.zeros((12, 8, 8)))  # Placeholder - cần thay thế
        
        return np.array(X), np.array(outcomes)


class TrainableChessAI(ChessAI):
    """
    AI cờ vua cơ bản có thể huấn luyện và lưu/tải model
    Lớp cơ sở cho các AI sử dụng deep learning
    """
    
    def __init__(self, exploration_rate=0.1):
        self.exploration_rate = exploration_rate
        self.exploration_mode = False
        self.game_memory = []
    
    def get_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        # Đôi khi chọn nước đi ngẫu nhiên để khám phá
        if self.exploration_mode or random.random() < self.exploration_rate:
            return self._random_move(board, color)
        
        # Sử dụng chiến lược chính
        return self._best_move(board, color)
    
    def _random_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Chọn một nước đi ngẫu nhiên từ các nước đi hợp lệ"""
        valid_moves = []
        for row in range(8):
            for col in range(8):
                pos = Position(row, col)
                piece = board.get_piece(pos)
                if piece.color == color:
                    moves = board.get_valid_moves(pos)
                    for move in moves:
                        valid_moves.append((pos, move.end_pos))
        
        if not valid_moves:
            raise ValueError(f"Không có nước đi hợp lệ nào cho {color}")
            
        return random.choice(valid_moves)
    
    def _best_move(self, board: ChessBoard, color: PieceColor) -> Tuple[Position, Position]:
        """Tìm nước đi tốt nhất - cần được triển khai bởi các lớp con"""
        # Mặc định là chọn ngẫu nhiên
        return self._random_move(board, color)
    
    def set_exploration_mode(self, mode: bool):
        """Bật/tắt chế độ khám phá"""
        self.exploration_mode = mode
    
    def save_model(self, filepath: str):
        """Lưu model - cần được triển khai bởi lớp con"""
        pass
    
    def load_model(self, filepath: str):
        """Tải model - cần được triển khai bởi lớp con"""
        pass