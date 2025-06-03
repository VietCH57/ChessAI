import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Import các AI classes
from chess_game import ChessGame
from chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move
from interface import ChessAI
from minimax import MinimaxChessAI
from alphabeta import AlphaBetaChessAI
from alphazero import AlphaZeroChessAI
from headless import HeadlessChessGame

class ChessBenchmark:
    """
    Benchmark system để chạy 2 rounds 50 trận giữa 2 AI với việc đổi bên
    """
    
    def __init__(self, use_gpu=False):
        self.use_gpu = True
        self.device = self.setup_gpu()
        self.max_workers = min(8, multiprocessing.cpu_count())
        
    def setup_gpu(self):
        """Setup GPU device và force sử dụng discrete GPU"""
        if not self.use_gpu:
            return None
            
        try:
            import torch
            
            # Force sử dụng discrete GPU nếu có nhiều GPU
            if torch.cuda.is_available():
                # Liệt kê tất cả GPU
                gpu_count = torch.cuda.device_count()
                print(f"Found {gpu_count} GPU(s):")
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
                
                # Chọn GPU mạnh nhất (thường là discrete GPU)
                if gpu_count > 1:
                    # Tìm GPU có memory lớn nhất (thường là discrete GPU)
                    best_gpu = 0
                    max_memory = 0
                    for i in range(gpu_count):
                        memory = torch.cuda.get_device_properties(i).total_memory
                        if memory > max_memory:
                            max_memory = memory
                            best_gpu = i
                    
                    device = torch.device(f'cuda:{best_gpu}')
                    print(f"Selected GPU {best_gpu} as primary device")
                else:
                    device = torch.device('cuda:0')
                    print("Using single GPU")
                
                # Set default GPU
                torch.cuda.set_device(device)
                
                # Test GPU
                test_tensor = torch.randn(100, 100).to(device)
                _ = torch.mm(test_tensor, test_tensor.t())
                print(f"✓ GPU {device} is working correctly")
                
                return device
            else:
                print("⚠ CUDA is not available, falling back to CPU")
                return None
                
        except ImportError:
            print("⚠ PyTorch not found, falling back to CPU")
            return None
        except Exception as e:
            print(f"⚠ GPU setup failed: {e}, falling back to CPU")
            return None
    
    def run_single_game(self, white_ai, black_ai, game_id):
        """
        Chạy một trận đấu duy nhất giữa 2 AI
        
        Args:
            white_ai: AI chơi quân trắng
            black_ai: AI chơi quân đen  
            game_id: ID của trận đấu
            
        Returns:
            dict: Kết quả trận đấu
        """
        try:
            # Setup GPU context cho thread này
            if self.device is not None:
                import torch
                torch.cuda.set_device(self.device)
                
                # Move AI models to GPU nếu cần
                if hasattr(white_ai, 'to') and callable(white_ai.to):
                    white_ai.to(self.device)
                if hasattr(black_ai, 'to') and callable(black_ai.to):
                    black_ai.to(self.device)
            
            game = HeadlessChessGame()
            result = game.run_game(white_ai, black_ai, max_moves=200, collect_data=False)
            
            # Xác định kết quả từ góc độ quân trắng
            if result["winner"] == PieceColor.WHITE:
                outcome = "win"
            elif result["winner"] == PieceColor.BLACK:
                outcome = "loss"
            else:
                outcome = "draw"
                
            return {
                "game_id": game_id,
                "outcome": outcome,
                "moves": result["moves"],
                "reason": result["reason"]
            }
            
        except Exception as e:
            print(f"Error in game {game_id}: {str(e)}")
            return {
                "game_id": game_id,
                "outcome": "error",
                "moves": 0,
                "reason": f"error: {str(e)}"
            }
    
    def run_50_games(self, white_ai, black_ai, round_name):
        """
        Chạy 50 trận đấu
        
        Args:
            white_ai: AI chơi quân trắng
            black_ai: AI chơi quân đen
            round_name: Tên round để hiển thị
            
        Returns:
            dict: Thống kê kết quả 50 trận
        """
        print(f"\n{round_name}: {type(white_ai).__name__} (White) vs {type(black_ai).__name__} (Black)")
        print(f"Running 50 games on {'GPU' if self.device else 'CPU'}...")
        
        results = {
            "win": 0,
            "draw": 0, 
            "loss": 0,
            "error": 0
        }
        
        start_time = time.time()
        
        # Move AI models to GPU trước khi bắt đầu
        if self.device is not None:
            try:
                import torch
                if hasattr(white_ai, 'to') and callable(white_ai.to):
                    white_ai.to(self.device)
                if hasattr(black_ai, 'to') and callable(black_ai.to):
                    black_ai.to(self.device)
                    
                # Warm up GPU
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: Could not move models to GPU: {e}")
        
        if self.use_gpu and self.max_workers > 1:
            # Chạy song song với ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tất cả 50 games
                future_to_game = {
                    executor.submit(self.run_single_game, white_ai, black_ai, i): i 
                    for i in range(50)
                }
                
                # Collect results
                completed = 0
                for future in as_completed(future_to_game):
                    try:
                        result = future.result()
                        outcome = result["outcome"]
                        
                        if outcome in results:
                            results[outcome] += 1
                        
                        completed += 1
                        if completed % 10 == 0:
                            print(f"Completed {completed}/50 games...")
                            
                    except Exception as exc:
                        print(f"Game generated an exception: {exc}")
                        results["error"] += 1
        else:
            # Chạy tuần tự
            for i in range(50):
                result = self.run_single_game(white_ai, black_ai, i)
                outcome = result["outcome"]
                
                if outcome in results:
                    results[outcome] += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/50 games...")
        
        round_time = time.time() - start_time
        print(f"Round completed in {round_time:.2f} seconds")
        
        # GPU memory cleanup
        if self.device is not None:
            try:
                import torch
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            except:
                pass
        
        return results
    
    def run_benchmark(self, ai1, ai2):
        """
        Chạy benchmark giữa 2 AI: 50 trận AI1 vs AI2, rồi 50 trận AI2 vs AI1
        
        Args:
            ai1: AI thứ nhất
            ai2: AI thứ hai
        """
        print("="*80)
        print("CHESS AI BENCHMARK")
        print("="*80)
        print(f"AI 1: {type(ai1).__name__}")
        print(f"AI 2: {type(ai2).__name__}")
        print(f"Using GPU: {self.use_gpu}")
        if self.device:
            print(f"GPU Device: {self.device}")
        print(f"Max workers: {self.max_workers}")
        print("="*80)
        
        benchmark_start = time.time()
        
        # Round 1: AI1 (White) vs AI2 (Black)
        round1_results = self.run_50_games(ai1, ai2, "Round 1")
        
        # Round 2: AI2 (White) vs AI1 (Black)
        round2_results = self.run_50_games(ai2, ai1, "Round 2")
        
        benchmark_time = time.time() - benchmark_start
        
        # In kết quả theo format yêu cầu
        self.print_results(ai1, ai2, round1_results, round2_results, benchmark_time)
    
    def print_results(self, ai1, ai2, round1_results, round2_results, total_time):
        """In kết quả theo format yêu cầu"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        
        print(f"{'White':<15} | {'Black':<15} | {'Win':<6} | {'Draw':<6} | {'Loss':<6}")
        print("-" * 65)
        
        # Round 1: AI1 White vs AI2 Black
        r1 = round1_results
        print(f"{type(ai1).__name__:<15} | {type(ai2).__name__:<15} | {r1['win']:<6} | {r1['draw']:<6} | {r1['loss']:<6}")
        
        # Round 2: AI2 White vs AI1 Black (từ góc độ AI2 là White)
        r2 = round2_results
        print(f"{type(ai2).__name__:<15} | {type(ai1).__name__:<15} | {r2['win']:<6} | {r2['draw']:<6} | {r2['loss']:<6}")
        
        print("-" * 65)
        
        # Tổng kết từ góc độ AI1
        total_ai1_wins = r1['win'] + r2['loss']  # AI1 thắng khi là White + AI1 thắng khi là Black
        total_ai2_wins = r1['loss'] + r2['win']  # AI2 thắng khi là Black + AI2 thắng khi là White  
        total_draws = r1['draw'] + r2['draw']
        total_errors = r1['error'] + r2['error']
        total_games = 100 - total_errors
        
        print(f"Total valid games: {total_games}/100")
        if total_games > 0:
            print(f"{type(ai1).__name__} total wins: {total_ai1_wins} ({total_ai1_wins/total_games*100:.1f}%)")
            print(f"{type(ai2).__name__} total wins: {total_ai2_wins} ({total_ai2_wins/total_games*100:.1f}%)")
            print(f"Total draws: {total_draws} ({total_draws/total_games*100:.1f}%)")
        
        print(f"Total benchmark time: {total_time:.2f} seconds")
        
        if total_errors > 0:
            print(f"⚠ Errors occurred: {total_errors} games")
        
        print("="*80)

def get_available_ais():
    """Lấy danh sách các AI có sẵn"""
    ais = {}
    
    try:
        ais["1"] = ("Minimax (depth=3)", lambda: MinimaxChessAI(depth=3))
        ais["2"] = ("Alpha-Beta (depth=3)", lambda: AlphaBetaChessAI(depth=3)) 
        ais["3"] = ("Alpha-Beta (depth=4)", lambda: AlphaBetaChessAI(depth=4))
        
        # Kiểm tra AlphaZero
        checkpoint_path = r"D:\Programming\IntroAI\ChessAI\models\checkpoint_10.pt"
        if os.path.exists(checkpoint_path):
            ais["4"] = ("AlphaZero", lambda: AlphaZeroChessAI.from_checkpoint(checkpoint_path))
        
    except Exception as e:
        print(f"Error loading some AIs: {e}")
    
    return ais

def select_ai(prompt_text):
    """Cho phép user chọn AI"""
    ais = get_available_ais()
    
    print(f"\n{prompt_text}")
    print("Available AIs:")
    for key, (name, _) in ais.items():
        print(f"  {key}: {name}")
    
    while True:
        choice = input("Enter your choice: ").strip()
        if choice in ais:
            name, ai_factory = ais[choice]
            try:
                ai = ai_factory()
                print(f"✓ {name} initialized successfully")
                return ai
            except Exception as e:
                print(f"❌ Error initializing {name}: {e}")
                continue
        else:
            print("Invalid choice. Please try again.")

def main():
    """Hàm main để chạy benchmark"""
    
    print("CHESS AI BENCHMARK TOOL")
    print("This tool runs 50 games with AI1 as White vs AI2 as Black,")
    print("then 50 games with AI2 as White vs AI1 as Black.")
    print()
    
    # Chọn 2 AI
    ai1 = MinimaxChessAI(depth=3)  # Mặc định là Minimax
    ai2 = AlphaBetaChessAI(depth=3)  # Mặc định là Alpha-Beta
    
    # Hỏi về GPU
    use_gpu = y
    
    # Khởi tạo benchmark system
    benchmark = ChessBenchmark(use_gpu=use_gpu)
    
    # Chạy benchmark
    try:
        benchmark.run_benchmark(ai1, ai2)
        print("\n🎉 Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()