from chess_board import ChessBoard, Position, PieceType, PieceColor, Piece, Move
from interface import ChessAI
from typing import Dict, List, Tuple, Optional, Any
import time
import random
import json
import os

class HeadlessChessGame:
    """
    Phiên bản tối ưu hiệu suất của ChessGame, không sử dụng giao diện đồ họa
    để chạy các trận đấu với tốc độ tối đa cho việc huấn luyện AI.
    """
    
    def __init__(self):
        """Khởi tạo game cờ vua không giao diện"""
        self.board = ChessBoard()
        self.game_over = False
        self.result_message = ""
        self.result = {
            "winner": None,
            "reason": None,
            "moves": 0,
            "history": []
        }
    
    def reset_game(self):
        """Reset trò chơi về trạng thái ban đầu"""
        self.board = ChessBoard()
        self.game_over = False
        self.result_message = ""
        self.result = {
            "winner": None,
            "reason": None,
            "moves": 0,
            "history": []
        }
    
    def check_game_state(self):
        """Kiểm tra trạng thái kết thúc game (chiếu hết, hòa cờ, etc.)"""
        opponent_color = PieceColor.BLACK if self.board.turn == PieceColor.WHITE else PieceColor.WHITE
        
        if self.board.is_checkmate(self.board.turn):
            self.game_over = True
            self.result_message = f"Checkmate! {opponent_color.value.capitalize()} wins!"
            self.result["winner"] = opponent_color
            self.result["reason"] = "checkmate"
        elif self.board.is_stalemate(self.board.turn):
            self.game_over = True
            self.result_message = "Stalemate! Draw."
            self.result["reason"] = "stalemate"
        elif self.board.is_fifty_move_rule_draw():
            self.game_over = True
            self.result_message = "Draw by fifty-move rule."
            self.result["reason"] = "fifty_move_rule"
        elif self.board.is_threefold_repetition():
            self.game_over = True
            self.result_message = "Draw by threefold repetition."
            self.result["reason"] = "threefold_repetition"
    
    def run_game(self, white_ai: ChessAI, black_ai: ChessAI, max_moves=200, collect_data=False):
        """
        Chạy một ván cờ hoàn chỉnh giữa 2 AI với tốc độ tối đa
        
        Args:
            white_ai: AI cho quân trắng
            black_ai: AI cho quân đen
            max_moves: Số nước đi tối đa trước khi buộc hòa
            collect_data: Có thu thập dữ liệu chi tiết cho huấn luyện hay không
            
        Returns:
            dict: Kết quả trận đấu và thông tin
        """
        self.reset_game()
        move_count = 0
        game_history = []
        
        # Thời gian bắt đầu để tính FPS
        start_time = time.time()
        
        # Chạy game đến khi kết thúc hoặc đạt giới hạn nước đi
        while not self.game_over and move_count < max_moves:
            # Xác định AI cho lượt hiện tại
            current_ai = white_ai if self.board.turn == PieceColor.WHITE else black_ai
            current_color = self.board.turn
            
            # Lưu trạng thái hiện tại nếu cần cho huấn luyện
            if collect_data:
                current_state = self._board_to_state()
            else:
                current_state = None
            
            try:
                # Lấy nước đi từ AI
                from_pos, to_pos = current_ai.get_move(self.board, current_color)
                
                # Thực hiện nước đi
                move = self.board.move_piece(from_pos, to_pos)
                
                if move:
                    # Nếu đang thu thập dữ liệu, lưu thông tin nước đi
                    if collect_data:
                        move_data = {
                            'move_number': move_count + 1,
                            'from': (from_pos.row, from_pos.col),
                            'to': (to_pos.row, to_pos.col),
                            'piece_type': move.piece.type.value,
                            'piece_color': move.piece.color.value,
                            'captured': None if not move.captured_piece or move.captured_piece.type == PieceType.EMPTY 
                                    else move.captured_piece.type.value,
                            'is_castling': move.is_castling,
                            'is_en_passant': move.is_en_passant,
                            'state_before': current_state
                        }
                        game_history.append(move_data)
                    
                    # Kiểm tra trạng thái kết thúc
                    self.check_game_state()
                    move_count += 1
                else:
                    # Nước đi không hợp lệ (không nên xảy ra nếu AI hoạt động đúng)
                    self.result["reason"] = "invalid_move"
                    self.game_over = True
                    
            except Exception as e:
                # Xử lý lỗi từ AI
                self.result["reason"] = f"ai_error: {str(e)}"
                self.game_over = True
        
        # Kiểm tra nếu đã đạt giới hạn nước đi
        if move_count >= max_moves and not self.game_over:
            self.game_over = True
            self.result["reason"] = "move_limit"
            
        # Tính FPS (nước đi / giây)
        elapsed_time = time.time() - start_time
        fps = move_count / elapsed_time if elapsed_time > 0 else 0
        
        # Cập nhật kết quả
        self.result["moves"] = move_count
        self.result["time_seconds"] = elapsed_time  
        self.result["moves_per_second"] = fps
        
        if collect_data:
            self.result["history"] = game_history
        
        return self.result
    
    def run_many_games(self, white_ai: ChessAI, black_ai: ChessAI, num_games=100, max_moves=200, 
                       collect_data=False, swap_sides=True, output_file=None):
        """
        Chạy nhiều trận đấu giữa các AI để thu thập thống kê
        
        Args:
            white_ai: AI quân trắng ban đầu
            black_ai: AI quân đen ban đầu
            num_games: Số trận đấu cần chạy
            max_moves: Số nước đi tối đa cho mỗi trận
            collect_data: Có thu thập dữ liệu chi tiết không
            swap_sides: Có đổi bên sau mỗi trận đấu không
            output_file: Đường dẫn file để lưu kết quả (tùy chọn)
            
        Returns:
            dict: Thống kê tổng hợp từ các trận đấu
        """
        stats = {
            "white_wins": 0,
            "black_wins": 0,
            "draws": 0,
            "total_moves": 0,
            "total_time": 0,
            "reasons": {},
            "games": []
        }
        
        start_time = time.time()
        
        for game_num in range(1, num_games + 1):
            # Xoay vòng AI nếu được yêu cầu
            if swap_sides and game_num % 2 == 0:
                current_white_ai, current_black_ai = black_ai, white_ai
            else:
                current_white_ai, current_black_ai = white_ai, black_ai
            
            # Chạy trận đấu
            result = self.run_game(current_white_ai, current_black_ai, max_moves, collect_data)
            
            # Cập nhật thống kê
            if result["winner"] == PieceColor.WHITE:
                stats["white_wins"] += 1
            elif result["winner"] == PieceColor.BLACK:
                stats["black_wins"] += 1
            else:
                stats["draws"] += 1
                
            stats["total_moves"] += result["moves"]
            stats["total_time"] += result.get("time_seconds", 0)
            
            # Đếm lý do kết thúc game
            reason = result["reason"]
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            
            # Lưu thông tin trận đấu
            game_info = {
                "game_num": game_num,
                "white_player": "white_ai" if not swap_sides or game_num % 2 == 1 else "black_ai",
                "black_player": "black_ai" if not swap_sides or game_num % 2 == 1 else "white_ai",
                "winner": result["winner"].value if result["winner"] else "draw",
                "reason": result["reason"],
                "moves": result["moves"],
                "time": result.get("time_seconds", 0)
            }
            stats["games"].append(game_info)
            
            # In tiến độ
            if game_num % 10 == 0 or game_num == 1:
                elapsed = time.time() - start_time
                avg_time = stats["total_time"] / game_num
                print(f"Game {game_num}/{num_games}: "
                      f"White: {stats['white_wins']}, Black: {stats['black_wins']}, "
                      f"Draw: {stats['draws']} | "
                      f"Avg time: {avg_time:.3f}s/game | "
                      f"Speed: {stats['total_moves']/stats['total_time']:.1f} moves/s")
        
        # Tính toán thống kê tổng hợp
        stats["white_win_percentage"] = (stats["white_wins"] / num_games) * 100
        stats["black_win_percentage"] = (stats["black_wins"] / num_games) * 100
        stats["draw_percentage"] = (stats["draws"] / num_games) * 100
        stats["avg_game_length"] = stats["total_moves"] / num_games
        stats["total_games"] = num_games
        stats["avg_moves_per_second"] = stats["total_moves"] / stats["total_time"] if stats["total_time"] > 0 else 0
        
        # Lưu kết quả nếu được yêu cầu
        if output_file:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
        
        return stats
    
    def _board_to_state(self):
        """
        Chuyển đổi bàn cờ thành biểu diễn state phù hợp cho AI
        """
        state = {}
        
        # Các quân cờ
        pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board.board[row][col]
                if piece.type != PieceType.EMPTY:
                    pieces.append({
                        'position': (row, col),
                        'type': piece.type.value,
                        'color': piece.color.value,
                        'has_moved': piece.has_moved
                    })
                    
        state['pieces'] = pieces
        state['turn'] = self.board.turn.value
        
        # Lượt đi hiện tại
        state['half_move_clock'] = self.board.half_move_clock
        state['move_number'] = len(self.board.move_history) // 2 + 1
        
        # Thông tin về chiếu
        state['white_in_check'] = self.board.is_check(PieceColor.WHITE)
        state['black_in_check'] = self.board.is_check(PieceColor.BLACK)
        
        return state