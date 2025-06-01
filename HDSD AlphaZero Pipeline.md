# AlphaZero Chess - Hướng Dẫn Sử Dụng

Tài liệu này mô tả cách sử dụng pipeline AlphaZero để huấn luyện, đánh giá và triển khai AI cờ vua dựa trên thuật toán AlphaZero trong framework ChessGame.

## Mục Lục
- [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
- [Cài Đặt](#cài-đặt)
- [Cấu Trúc Mã Nguồn](#cấu-trúc-mã-nguồn)
- [Hướng Dẫn Sử Dụng](#hướng-dẫn-sử-dụng)
  - [Huấn Luyện Mô Hình](#huấn-luyện-mô-hình)
  - [Đánh Giá Mô Hình](#đánh-giá-mô-hình)
  - [Chơi Với Mô Hình Đã Huấn Luyện](#chơi-với-mô-hình-đã-huấn-luyện)
- [Cấu Hình Hệ Thống](#cấu-hình-hệ-thống)
- [Giải Thích Chi Tiết Các Tham Số](#giải-thích-chi-tiết-các-tham-số)
- [Mẹo Và Tối Ưu Hóa](#mẹo-và-tối-ưu-hóa)
- [Xử Lý Sự Cố](#xử-lý-sự-cố)

## Yêu Cầu Hệ Thống

Để chạy được AlphaZero pipeline, bạn cần:

- Python 3.8 hoặc cao hơn
- PyTorch 1.8.0 hoặc cao hơn
- NumPy
- Các thư viện phụ thuộc khác được liệt kê trong `requirements.txt`

Khuyến nghị sử dụng GPU để tăng tốc quá trình huấn luyện. AlphaZero là thuật toán tốn tài nguyên tính toán, đặc biệt là trong quá trình MCTS.

## Cài Đặt

1. Clone repository:
```bash
git clone https://github.com/VietCH57/ChessGame.git
cd ChessGame
```

2. Cài đặt các thư viện cần thiết:

3. (Tùy chọn) Cài đặt PyTorch với hỗ trợ CUDA nếu bạn có GPU:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Cấu Trúc Mã Nguồn

Pipeline AlphaZero bao gồm các module chính sau:

- `alphazero_model.py`: Mạng nơ-ron với kiến trúc theo đúng paper AlphaZero
- `board_encoder.py`: Chuyển đổi trạng thái bàn cờ thành biểu diễn 119 kênh
- `alpha_zero_mcts.py`: Thuật toán Monte Carlo Tree Search
- `alpha_zero_player.py`: Triển khai AlphaZero theo interface ChessAI
- `alpha_zero_trainer.py`: Pipeline huấn luyện đầy đủ
- `alphazero_config.py`: Cấu hình mặc định và utility cho hệ thống
- `run_alphazero.py`: Script để khởi chạy quá trình huấn luyện
- `play_with_alphazero.py`: Script để chơi với AlphaZero đã huấn luyện

## Hướng Dẫn Sử Dụng

### Huấn Luyện Mô Hình

Để bắt đầu huấn luyện mô hình AlphaZero:

```bash
python run_alphazero.py --output_dir models --iterations 100 --self_play_games 100
```

Các tham số có thể tùy chỉnh:

- `--output_dir`: Thư mục để lưu các mô hình đã huấn luyện
- `--iterations`: Số vòng lặp huấn luyện
- `--self_play_games`: Số ván đấu tự chơi trong mỗi vòng
- `--batch_size`: Kích thước batch khi huấn luyện
- `--epochs`: Số epoch huấn luyện sau mỗi vòng tự chơi
- `--simulations`: Số lần mô phỏng MCTS cho mỗi nước đi
- `--load_model`: Đường dẫn đến mô hình để tiếp tục huấn luyện

Trong quá trình huấn luyện, hệ thống sẽ:
1. Sinh dữ liệu thông qua tự chơi
2. Huấn luyện mạng nơ-ron trên dữ liệu đã sinh
3. Đánh giá mạng mới với mạng tốt nhất hiện tại
4. Cập nhật mạng tốt nhất nếu mạng mới đạt ngưỡng win rate

### Đánh Giá Mô Hình

Để đánh giá mô hình với AI khác hoặc phiên bản trước:

```bash
python evaluate_alphazero.py --model models/best_model.pt --opponent random --games 100
```

### Chơi Với Mô Hình Đã Huấn Luyện

Để chơi với mô hình đã huấn luyện:

```bash
python play_with_alphazero.py --model models/best_model.pt --simulations 800
```

Thêm `--as_black` để chơi với quân đen.

## Cấu Hình Hệ Thống

Cấu hình mặc định được định nghĩa trong `alphazero_config.py`. Bạn có thể tạo file cấu hình JSON riêng và cung cấp thông qua tham số `--config`:

```bash
python run_alphazero.py --config my_config.json
```

Ví dụ về file cấu hình:
```json
{
    "num_self_play_games": 200,
    "num_simulations": 400,
    "batch_size": 512,
    "learning_rate": 0.002
}
```

## Giải Thích Chi Tiết Các Tham Số

### Tham số Self-play

- **num_self_play_games (100)**: Số trận đấu tự chơi (self-play) trong mỗi vòng lặp huấn luyện. Mỗi trận đấu tạo ra dữ liệu huấn luyện. Tăng giá trị này sẽ cải thiện chất lượng dữ liệu huấn luyện nhưng tốn thời gian hơn.

- **num_simulations (800)**: Số lần mô phỏng MCTS cho mỗi nước đi trong quá trình tự chơi. Thông số này ảnh hưởng trực tiếp đến chất lượng nước đi. Giá trị cao hơn cho kết quả tốt hơn nhưng chậm hơn.

- **max_moves_per_game (512)**: Số nước đi tối đa cho mỗi ván cờ. Tránh các trận đấu kéo dài vô tận. Trên thực tế, các ván cờ hiếm khi vượt quá 200 nước.

### Tham số Huấn luyện

- **batch_size (1024)**: Kích thước batch khi huấn luyện mạng nơ-ron. Batch lớn hơn giúp huấn luyện ổn định hơn và tận dụng tốt GPU.

- **epochs (20)**: Số epoch huấn luyện mạng nơ-ron sau mỗi vòng lặp tự chơi. Tăng giá trị này giúp học tốt hơn từ dữ liệu nhưng có thể gây overfitting.

- **learning_rate (0.001)**: Tốc độ học của optimizer. Ảnh hưởng đến tốc độ và độ ổn định của quá trình huấn luyện.

- **weight_decay (1e-4)**: Hệ số điều chuẩn L2, giúp tránh overfitting bằng cách phạt các trọng số lớn.

- **num_iterations (100)**: Tổng số vòng lặp huấn luyện. Mỗi vòng bao gồm tự chơi, thu thập dữ liệu, huấn luyện và đánh giá.

### Tham số Mạng

- **num_res_blocks (20)**: Số khối residual trong mạng nơ-ron. Khớp với thiết kế của AlphaZero gốc, giúp mạng học các pattern phức tạp.

- **num_filters (256)**: Số filter trong các lớp tích chập. Quyết định độ phức tạp và khả năng học của mạng.

### Tham số Đánh giá

- **evaluation_games (40)**: Số trận đấu để đánh giá mạng mới so với mạng tốt nhất hiện tại.

- **evaluation_threshold (0.55)**: Tỉ lệ thắng tối thiểu để mạng mới thay thế mạng tốt nhất hiện tại. Giá trị 0.55 có nghĩa là mạng mới phải thắng ít nhất 55% trận đấu.

### Tham số MCTS

- **c_puct (1.0)**: Hằng số khám phá trong công thức PUCT. Điều chỉnh cân bằng giữa khai thác (exploitation) và khám phá (exploration) trong tìm kiếm MCTS.

- **temperature_init (1.0)**: Nhiệt độ ban đầu cho việc lựa chọn nước đi. Giá trị cao hơn tạo ra nhiều đa dạng trong nước đi.

- **temperature_final (0.25)**: Nhiệt độ cuối cùng sau khi đạt đến temperature_drop_move. Giá trị thấp hơn ưu tiên nước đi tối ưu hơn.

- **temperature_drop_move (30)**: Số nước đi trước khi giảm nhiệt độ từ init xuống final. Khuyến khích đa dạng ở đầu ván đấu và play tối ưu ở cuối.

### Đường dẫn File

- **output_dir ('alphazero_models')**: Thư mục để lưu các model đã huấn luyện.

- **replay_buffer_size (200000)**: Kích thước tối đa của bộ đệm lặp lại. Lưu trữ ví dụ huấn luyện từ các vòng lặp trước để sử dụng lại.

## Mẹo Và Tối Ưu Hóa

### Điều chỉnh cho Hiệu suất/Tài nguyên Khác nhau:

- **Máy yếu**: Giảm num_simulations (100-200), num_self_play_games (10-20), và num_res_blocks (5-10)
- **CPU-only**: Giảm batch_size (64-256) và num_filters (64-128)
- **GPU mạnh**: Có thể tăng tất cả các giá trị, đặc biệt là batch_size (2048-4096) và num_simulations (1600+)
- **Huấn luyện nhanh**: Giảm num_iterations (10-20) và evaluation_games (10-20)
- **Chất lượng cao**: Tăng num_self_play_games (500+), replay_buffer_size (500000+), và num_simulations (1600+)

### Mẹo Huấn Luyện

1. **Huấn luyện từng bước**: Bắt đầu với một mạng nhỏ và ít simulations, sau đó tăng dần theo thời gian.

2. **Lưu checkpoint thường xuyên**: Luôn đảm bảo lưu các checkpoint để có thể tiếp tục huấn luyện từ bất kỳ điểm nào.

3. **Theo dõi loss**: Nếu loss giảm quá chậm hoặc tăng đột biến, hãy điều chỉnh learning rate.

4. **Đa dạng dữ liệu**: Cân nhắc sử dụng một số opening ngẫu nhiên để tạo ra đa dạng trong dữ liệu huấn luyện.

5. **Tỷ lệ hòa cao là bình thường**: Cờ vua tối ưu thường dẫn đến hòa, đặc biệt là khi AI đấu với chính nó.

## Xử Lý Sự Cố

### Vấn đề phổ biến và giải pháp:

1. **Thiếu bộ nhớ**: 
   - Giảm batch_size
   - Giảm replay_buffer_size
   - Giảm num_simulations

2. **Huấn luyện quá chậm**:
   - Giảm num_self_play_games
   - Giảm num_simulations
   - Sử dụng GPU nếu có thể

3. **Mô hình không cải thiện**:
   - Tăng learning_rate
   - Tăng num_self_play_games
   - Kiểm tra quá trình đánh giá

4. **Lỗi CUDA out of memory**:
   - Giảm batch_size
   - Giảm num_filters
   - Giảm num_res_blocks

---

Nếu bạn cần hỗ trợ thêm hoặc có câu hỏi, hãy mở issue trên GitHub repository.
