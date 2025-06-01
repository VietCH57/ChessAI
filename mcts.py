#!/usr/bin/env python
import numpy as np
import math
import copy
import torch
from chess_board import ChessBoard, Position, PieceColor, PieceType
from typing import Dict, List, Tuple, Optional

class MCTSNode:
    """Node trong cây tìm kiếm Monte Carlo"""
    
    def __init__(self, board_state: ChessBoard, move: Optional[Tuple] = None, parent=None, prior_prob: float = 0.0):
        self.board_state = board_state
        self.move = move  # Nước đi đưa đến state này
        self.parent = parent
        self.children: Dict[Tuple, 'MCTSNode'] = {}
        
        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
        
        # Expanded flag
        self.is_expanded = False
        
        # Valid moves cache
        self._valid_moves = None
    
    @property
    def q_value(self) -> float:
        """Q-value (average value)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def u_value(self) -> float:
        """Upper confidence bound"""
        if self.visit_count == 0:
            return float('inf')
        
        c_puct = 1.4  # Exploration constant
        return c_puct * self.prior_prob * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
    
    @property
    def uct_value(self) -> float:
        """UCT value for selection"""
        return self.q_value + self.u_value
    
    def get_valid_moves(self) -> List[Tuple[Position, Position]]:
        """Lấy các nước đi hợp lệ"""
        if self._valid_moves is None:
            self._valid_moves = []
            current_color = self.board_state.turn
            
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = self.board_state.get_piece(pos)
                    if piece.color == current_color:
                        moves = self.board_state.get_valid_moves(pos)
                        for move in moves:
                            self._valid_moves.append((pos, move.end_pos))
        
        return self._valid_moves
    
    def is_terminal(self) -> bool:
        """Kiểm tra nếu đây là trạng thái kết thúc"""
        current_color = self.board_state.turn
        return (self.board_state.is_checkmate(current_color) or 
                self.board_state.is_stalemate(current_color) or
                self.board_state.is_fifty_move_rule_draw() or
                self.board_state.is_threefold_repetition())
    
    def get_terminal_value(self) -> float:
        """Lấy giá trị cuối game"""
        current_color = self.board_state.turn
        
        if self.board_state.is_checkmate(current_color):
            # Checkmate - bên hiện tại thua
            return -1.0
        else:
            # Hòa cờ
            return 0.0
    
    def expand(self, policy_probs: np.ndarray):
        """Mở rộng node với prior probabilities từ neural network"""
        if self.is_expanded:
            return
        
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            self.is_expanded = True
            return
        
        # Tạo children cho tất cả nước đi hợp lệ
        for i, move in enumerate(valid_moves):
            # Tạo board state mới
            new_board = self.board_state.copy_board()
            new_board.move_piece(move[0], move[1])
            
            # Lấy prior probability (cần implement move encoding)
            prior_prob = policy_probs[i] if i < len(policy_probs) else 1.0 / len(valid_moves)
            
            # Tạo child node
            child = MCTSNode(new_board, move, self, prior_prob)
            self.children[move] = child
        
        self.is_expanded = True
    
    def select_child(self) -> 'MCTSNode':
        """Chọn child node tốt nhất theo UCT"""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda c: c.uct_value)
    
    def backup(self, value: float):
        """Backup value lên cây"""
        self.visit_count += 1
        self.total_value += value
        
        if self.parent is not None:
            # Đảo dấu value vì đây là zero-sum game
            self.parent.backup(-value)
    
    def get_action_probs(self, temperature: float = 1.0) -> np.ndarray:
        """Lấy action probabilities dựa trên visit counts"""
        valid_moves = self.get_valid_moves()
        action_probs = np.zeros(len(valid_moves))
        
        for i, move in enumerate(valid_moves):
            if move in self.children:
                action_probs[i] = self.children[move].visit_count
        
        if temperature == 0:
            # Chọn action tốt nhất
            best_action = np.argmax(action_probs)
            action_probs = np.zeros(len(action_probs))
            action_probs[best_action] = 1.0
        else:
            # Apply temperature
            action_probs = action_probs ** (1.0 / temperature)
            if action_probs.sum() > 0:
                action_probs = action_probs / action_probs.sum()
            else:
                action_probs = np.ones(len(action_probs)) / len(action_probs)
        
        return action_probs

class MCTS:
    """Monte Carlo Tree Search cho cờ vua"""
    
    def __init__(self, neural_network, simulations: int = 800, c_puct: float = 1.4):
        self.neural_network = neural_network
        self.simulations = simulations
        self.c_puct = c_puct
    
    def search(self, board_state: ChessBoard, temperature: float = 1.0) -> Tuple[Tuple[Position, Position], np.ndarray]:
        """
        Thực hiện MCTS search
        
        Returns:
            best_move: Nước đi tốt nhất
            action_probs: Phân phối xác suất các nước đi
        """
        root = MCTSNode(board_state.copy_board())
        
        # Thêm Dirichlet noise cho root node
        if not root.is_terminal():
            policy_probs, _ = self._evaluate(root.board_state)
            policy_probs = self._add_dirichlet_noise(policy_probs)
            root.expand(policy_probs)
        
        # Thực hiện simulations
        for _ in range(self.simulations):
            self._simulate(root)
        
        # Lấy kết quả
        valid_moves = root.get_valid_moves()
        action_probs = root.get_action_probs(temperature)
        
        if len(valid_moves) == 0:
            return None, action_probs
        
        # Chọn move dựa trên action_probs
        if temperature == 0:
            move_idx = np.argmax(action_probs)
        else:
            move_idx = np.random.choice(len(action_probs), p=action_probs)
        
        best_move = valid_moves[move_idx]
        
        return best_move, action_probs
    
    def _simulate(self, node: MCTSNode):
        """Thực hiện một simulation"""
        if node.is_terminal():
            value = node.get_terminal_value()
            node.backup(value)
            return
        
        # Selection
        if not node.is_expanded:
            # Leaf node - expand và evaluate
            policy_probs, value = self._evaluate(node.board_state)
            node.expand(policy_probs)
            node.backup(value)
        else:
            # Expanded node - select child và tiếp tục
            child = node.select_child()
            if child is not None:
                self._simulate(child)
    
    def _evaluate(self, board_state: ChessBoard) -> Tuple[np.ndarray, float]:
        """Đánh giá position với neural network"""
        # Chuyển board state thành tensor
        state_tensor = self._board_to_tensor(board_state)
        
        with torch.no_grad():
            if torch.cuda.is_available():
                state_tensor = state_tensor.cuda()
            
            policy_logits, value = self.neural_network(state_tensor.unsqueeze(0))
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.cpu().numpy()[0][0]
        
        return policy_probs, value
    
    def _board_to_tensor(self, board_state: ChessBoard) -> torch.Tensor:
        """Chuyển board state thành tensor"""
        # 12 kênh: 6 loại quân x 2 màu
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_types = [PieceType.PAWN, PieceType.KNIGHT, PieceType.BISHOP, 
                       PieceType.ROOK, PieceType.QUEEN, PieceType.KING]
        
        for row in range(8):
            for col in range(8):
                piece = board_state.get_piece(Position(row, col))
                if piece.type != PieceType.EMPTY:
                    piece_idx = piece_types.index(piece.type)
                    if piece.color == PieceColor.BLACK:
                        piece_idx += 6
                    tensor[piece_idx, row, col] = 1.0
        
        return torch.from_numpy(tensor)
    
    def _add_dirichlet_noise(self, policy_probs: np.ndarray, alpha: float = 0.3, epsilon: float = 0.25) -> np.ndarray:
        """Thêm Dirichlet noise cho exploration"""
        noise = np.random.dirichlet([alpha] * len(policy_probs))
        return (1 - epsilon) * policy_probs + epsilon * noise