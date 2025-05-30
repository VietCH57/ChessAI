### ChessGame
Tải xuống: \
1 - Clone repo này: \
git clone https://github.com/VietCH57/ChessGame.git \
2 - Chuyển đến thư mục ChessGame rồi tải thư viện pygame: \
pip install pygame \
3 - Chơi game: \
python src/main.py

### UPDATE: Thêm interface cho AI

Cách viết AI:
# 1. Import các module cần thiết
```python
from chess_board import ChessBoard, Position, PieceType, PieceColor
from interface import ChessAI
```

# 2. Tạo class kế thừa ChessAI
```python
class MyChessAI(ChessAI):
    def __init__(self):
        # Khởi tạo AI của bạn
        pass
        
    def get_move(self, board: ChessBoard, color: PieceColor):
        # Logic của AI để chọn nước đi
        # Trả về: tuple(Position từ, Position đến)
        pass
```

# 3. Chạy AI của bạn
```python
from chess_game import ChessGame

game = ChessGame()
game.toggle_ai(white_ai=MyChessAI())  # Cho AI chơi quân trắng
game.run()
```

### Thành phần cơ bản của bàn cờ:
Vị trí:
- position = Position(row, col)  # row và col từ 0-7

Các loại quân cờ:
- PieceType.PAWN    # Tốt
- PieceType.ROOK    # Xe
- PieceType.KNIGHT  # Mã
- PieceType.BISHOP  # Tượng
- PieceType.QUEEN   # Hậu
- PieceType.KING    # Vua
- PieceType.EMPTY   # Ô trống

Màu quân cờ:
- PieceColor.WHITE  # Trắng
- PieceColor.BLACK  # Đen
- PieceColor.NONE   # Không màu (cho ô trống)

### Đối tượng Move
Mỗi nước đi trong danh sách valid_moves là một đối tượng Move với các thuộc tính:
- start_pos: Vị trí xuất phát
- end_pos: Vị trí đích
- piece: Quân cờ được di chuyển
- captured_piece: Quân bị bắt (nếu có)
- is_castling: True nếu là nước nhập thành
- is_en_passant: True nếu là nước bắt tốt qua đường

### Các methods hữu ích từ class ChessBoard:
- board.get_piece(position): Lấy quân cờ tại vị trí
- board.get_valid_moves(position): Lấy danh sách các nước đi hợp lệ từ vị trí
- board.is_check(color): Kiểm tra xem vua có đang bị chiếu không
- board.copy_board(): Tạo bản sao của bàn cờ (để thử nghiệm nước đi)

### Các files mẫu:
Tôi có viết một số files mẫu để mọi người có thể tham khảo cấu trúc: 
- ai_template.py 
- run.py 

### UPDATE: Framework huấn luyện AI

Để mọi người huấn luyện AI cờ vua hiệu quả, tôi đã thêm "headless mode" để có thể chạy các game intances ở tốc độ cao

Ví dụ, để chạy hàng loạt game ở tốc độ cao:
```python
from headless import HeadlessChessGame
from interface import ChessAI

# Tạo AI của bạn
class MyChessAI(ChessAI):
    def get_move(self, board, color):
        # Logic của bạn...
        return from_pos, to_pos

# Chạy trận đấu không giao diện
engine = HeadlessChessGame()
result = engine.run_game(MyChessAI(), MyChessAI(), max_moves=200)
print(f"Kết quả: {result['moves']} nước, {result['moves_per_second']:.1f} nước/giây")

# Chạy hàng loạt trận đấu
stats = engine.run_many_games(
    white_ai=MyChessAI(), 
    black_ai=MyChessAI(), 
    num_games=1000,
    output_file="results.json"
)
```

Để tạo AI có khả năng train:
```python
from train_ai import TrainableChessAI
import numpy as np

class MyTrainableAI(TrainableChessAI):
    def __init__(self):
        super().__init__(exploration_rate=0.1)  # Tỉ lệ khám phá ngẫu nhiên
        self.model = self._create_model()
    
    def _create_model(self):
        # Tạo model ML/DL của bạn
        return {"weights": np.random.random((12*8*8, 1))} # Placeholder cho model ML/DL thật
    
    def _best_move(self, board, color):
        # Sử dụng model để tìm nước đi tốt nhất
        # Trả về tuple (from_position, to_position)
        # ...
        
    def save_model(self, filepath):
        # Lưu model của bạn
        np.save(f"{filepath}.npy", self.model["weights"])
        
    def load_model(self, filepath):
        # Tải model của bạn
        self.model["weights"] = np.load(f"{filepath}.npy")
```

Để huấn luyện AI đó:
```python
from train_ai import ChessTrainer

# Tạo trainer
trainer = ChessTrainer(output_dir="my_training_data")

# Huấn luyện với self-play
my_ai = MyTrainableAI()
stats = trainer.train_self_play(
    my_ai, 
    num_games=1000,
    save_interval=100  # Lưu model sau mỗi 100 trận
)

# Đánh giá hiệu suất
opponent_ai = AnotherAI()
evaluation = trainer.evaluate_ai(
    my_ai, 
    opponent_ai, 
    num_games=100
)
```


# Cụ thể các phương thức hữu ích trong framework này:
HeadlessChessGame: Chạy game không giao diện với tốc độ cao
- run_game(white_ai, black_ai, max_moves): Chạy một trận đấu
- run_many_games(white_ai, black_ai, num_games, swap_sides, output_file): Chạy nhiều trận

ChessTrainer: Framework huấn luyện AI
- train_self_play(ai, num_games, opponent_ai, save_interval): Huấn luyện bằng self-play
- evaluate_ai(ai1, ai2, num_games): Đánh giá hiệu suất AI
- board_to_tensor(board): Chuyển bàn cờ thành tensor cho deep learning

TrainableChessAI: Lớp cơ sở cho AI có thể huấn luyện
- set_exploration_mode(mode): Bật/tắt chế độ khám phá ngẫu nhiên
- save_model(filepath): Lưu model
- load_model(filepath): Tải model đã huấn luyện

Sau khi huấn luyện xong, do là lớp con của ChessAI nên AI của bạn có thể dùng trong interface ở UPDATE trước và có thể chơi trong game như thườngthường
