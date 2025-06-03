import math
import numpy as np
import torch
import os
import threading
from typing import Dict, List, Optional, Tuple
from chess_board import ChessBoard, Position, PieceColor
from network import AlphaZeroNetwork, BoardEncoder, MoveEncoder
from config import AlphaZeroConfig

class MCTSNode:
    def __init__(self, board: ChessBoard, parent=None, move=None, prior=0.0):
        # Thay vì copy toàn bộ board, chỉ lưu move dẫn đến trạng thái này
        self.board_state = None  # Lazy evaluation - chỉ tạo khi cần
        self.move_sequence = []  # Dãy các nước đi dẫn đến state này
        self.board_hash = None   # Hash của board để kiểm tra nhanh
        
        if parent is None:
            # Root node - cần copy board
            self.board = board.copy_board()
            self.board_hash = board.get_board_hash() if hasattr(board, 'get_board_hash') else None
        else:
            # Child node - chỉ lưu move và reference đến parent
            self.board = None
            self.parent = parent
            self.move = move
            # Sẽ tạo board state khi cần bằng _create_board
        
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[tuple, MCTSNode] = {}
        self.is_expanded = False
        
    def get_board(self):
        """Lazy loading của board state - chỉ tạo khi cần"""
        if self.board is None:
            self._create_board()
        return self.board
    
    def _create_board(self):
        """Tạo board từ parent và move"""
        if self.parent is not None:
            self.board = self.parent.get_board().copy_board()
            if self.move is not None:
                from_pos = Position(self.move[0], self.move[1])
                to_pos = Position(self.move[2], self.move[3])
                self.board.move_piece(from_pos, to_pos)
    
    def is_leaf(self) -> bool:
        return not self.is_expanded
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest UCB score"""
        if not self.children:
            return None
        
        # Tối ưu bằng cách tính toán đồng thời tất cả UCB scores
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        total_values = np.array([child.total_value for child in self.children.values()])
        priors = np.array([child.prior for child in self.children.values()])
        
        # Xử lý các nút chưa visited
        with np.errstate(divide='ignore', invalid='ignore'):
            q_values = np.divide(total_values, visit_counts, 
                                out=np.ones_like(total_values)*float('inf'), 
                                where=visit_counts!=0)
            
        u_values = c_puct * priors * np.sqrt(max(1, self.visit_count)) / (1 + visit_counts)
        ucb_scores = q_values + u_values
        
        # Lấy index với UCB cao nhất
        best_idx = np.argmax(ucb_scores)
        
        # Trả về child với index đó
        return list(self.children.values())[best_idx]
    
    def expand(self, policy_probs: np.ndarray, move_encoder: MoveEncoder):
        """Expand node with children for all legal moves"""
        if self.is_expanded:
            return
        
        board = self.get_board()
        legal_moves = []
        
        # Tối ưu hóa việc lấy legal moves
        turn = board.turn
        for row in range(8):
            for col in range(8):
                from_pos = Position(row, col)
                piece = board.get_piece(from_pos)
                if piece and piece.color == turn:
                    try:
                        valid_moves = board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            legal_moves.append((from_pos, move.end_pos))
                    except Exception as e:
                        continue
        
        # Tạo children dựa trên policy network
        for from_pos, to_pos in legal_moves:
            try:
                # Chỉ lưu move key mà không copy board
                move_idx = move_encoder.encode_move(from_pos, to_pos)
                prior = policy_probs[move_idx] if move_idx >= 0 and move_idx < len(policy_probs) else 0.01
                
                # Tạo move key
                move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
                
                # Tạo child node mà không copy board ngay
                child = MCTSNode(None, parent=self, move=move_key, prior=prior)
                self.children[move_key] = child
            except Exception as e:
                continue
        
        self.is_expanded = True
    
    def backup(self, value: float):
        """Backup value through the tree"""
        self.visit_count += 1
        self.total_value += value
        
        if hasattr(self, 'parent') and self.parent:
            # Negate value when backing up (different perspectives)
            self.parent.backup(-value)
    
    def get_visit_counts(self) -> np.ndarray:
        """Get visit counts for all possible moves"""
        visit_counts = np.zeros(4096)  # 64*64 possible moves
        
        for move_key, child in self.children.items():
            from_row, from_col, to_row, to_col = move_key
            move_idx = from_row * 8 * 8 * 8 + from_col * 8 * 8 + to_row * 8 + to_col
            if move_idx < 4096:
                visit_counts[move_idx] = child.visit_count
                
        return visit_counts

class AlphaZeroMCTS:
    def __init__(self, network: AlphaZeroNetwork, config: AlphaZeroConfig):
        self.network = network
        self.config = config
        
        # Make sure network is on the right device
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.network = self.network.to(device)
        
        # Enable CUDNN benchmarking for faster convolutions
        if config.device == "cuda" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        
        # Batching
        self.eval_batch_size = 32
        self.pending_evaluations = []
        
        # Tree reuse
        self.root = None
        self.reuse_tree = True
        
        # Threading lock for parallel execution
        self.lock = threading.Lock()
        
    def search(self, board: ChessBoard, num_simulations: int, 
               temperature: float = 1.0, add_noise: bool = False) -> Tuple[np.ndarray, float]:
        """MCTS search with optimized tree traversal and evaluation batching"""
        # Check if we can reuse tree from previous search
        if self.reuse_tree and self.root is not None:
            board_hash = board.get_board_hash() if hasattr(board, 'get_board_hash') else None
            
            # Try to find a matching child node
            if board_hash is not None:
                for child in self.root.children.values():
                    child_board = child.get_board()
                    if child_board and child_board.get_board_hash() == board_hash:
                        # Found matching child - reuse this subtree
                        self.root = child
                        self.root.parent = None  # Detach from old parent
                        break
                else:
                    # No matching child found - create new root
                    self.root = MCTSNode(board)
            else:
                # Can't do hash check - create new root
                self.root = MCTSNode(board)
        else:
            # Create new root
            self.root = MCTSNode(board)
        
        # Number of simulations in each batch
        batch_size = min(self.eval_batch_size, num_simulations // 4)
        batch_size = max(1, batch_size)  # At least 1
        
        # Run simulations in batches
        completed_sims = 0
        while completed_sims < num_simulations:
            current_batch_size = min(batch_size, num_simulations - completed_sims)
            self._simulate_batch(self.root, current_batch_size)
            completed_sims += current_batch_size
        
        # Add Dirichlet noise at root for exploration
        if add_noise and self.root.is_expanded and self.root.children:
            self._add_dirichlet_noise(self.root)
        
        # Get visit counts for action probabilities
        visit_counts = self.root.get_visit_counts()
        
        # Convert visit counts to action probabilities based on temperature
        if temperature == 0:
            # Deterministic selection
            action_probs = np.zeros_like(visit_counts)
            if visit_counts.sum() > 0:
                best_action = np.argmax(visit_counts)
                action_probs[best_action] = 1.0
            else:
                action_probs = np.ones_like(visit_counts) / len(visit_counts)
        else:
            # Apply temperature scaling
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            if visit_counts_temp.sum() > 0:
                action_probs = visit_counts_temp / visit_counts_temp.sum()
            else:
                action_probs = np.ones_like(visit_counts) / len(visit_counts)
        
        # Return action probabilities and estimated value
        estimated_value = self.root.total_value / max(self.root.visit_count, 1)
        return action_probs, estimated_value
    
    def _simulate_batch(self, root: MCTSNode, batch_size: int):
        """Run multiple simulations in parallel and batch neural network evaluations"""
        # Collect leaf nodes
        leaf_nodes = []
        leaf_values = {}
        terminal_nodes = []
        
        # Find leaf nodes for evaluation
        for _ in range(batch_size):
            current = root
            path = [current]
            
            # Selection - traverse until leaf
            while not current.is_leaf() and not self._is_terminal(current.get_board()):
                current = current.select_child(self.config.c_puct)
                if current is None:
                    break
                path.append(current)
            
            if current is None:
                continue
            
            # Check if terminal node
            board = current.get_board()
            if self._is_terminal(board):
                value = self._get_terminal_value(board)
                terminal_nodes.append((current, value))
            else:
                leaf_nodes.append(current)
        
        # Batch evaluate and expand non-terminal leaf nodes
        if leaf_nodes:
            try:
                # Prepare batch input
                batch_states = []
                for node in leaf_nodes:
                    board_tensor = self.board_encoder.encode_board(node.get_board())
                    batch_states.append(board_tensor)
                
                batch_tensor = torch.FloatTensor(np.array(batch_states))
                
                # Move to correct device
                device = next(self.network.parameters()).device
                batch_tensor = batch_tensor.to(device)
                
                # Network evaluation
                self.network.eval()
                with torch.no_grad():
                    policy_logits, values = self.network(batch_tensor)
                    policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.cpu().numpy().flatten()
                
                # Expand nodes and backup values
                for i, node in enumerate(leaf_nodes):
                    node.expand(policy_probs[i], self.move_encoder)
                    node.backup(float(values[i]))
                    
            except Exception as e:
                # Fallback to individual evaluation
                for node in leaf_nodes:
                    try:
                        value = self._expand_and_evaluate(node)
                        node.backup(value)
                    except:
                        node.backup(0.0)
        
        # Process terminal nodes
        for node, value in terminal_nodes:
            node.backup(value)
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Single node evaluation"""
        try:
            # Get board representation
            board_tensor = self.board_encoder.encode_board(node.get_board())
            board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0)
            
            # Move to correct device
            device = next(self.network.parameters()).device
            board_tensor = batch_tensor.to(device)
            
            # Network evaluation
            self.network.eval()
            with torch.no_grad():
                policy_logits, value = self.network(board_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.cpu().numpy()[0][0]
            
            # Expand node
            node.expand(policy_probs, self.move_encoder)
            return float(value)
            
        except Exception as e:
            # Fallback to random policy
            uniform_probs = np.ones(4096) / 4096
            node.expand(uniform_probs, self.move_encoder)
            return 0.0
    
    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise for exploration"""
        if not root.children:
            return
            
        num_children = len(root.children)
        if num_children == 0:
            return
            
        try:
            # Generate Dirichlet noise
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_children)
            
            # Add noise to priors
            for i, child in enumerate(root.children.values()):
                child.prior = (1 - self.config.dirichlet_epsilon) * child.prior + \
                             self.config.dirichlet_epsilon * noise[i]
        except Exception as e:
            pass
    
    def _is_terminal(self, board: ChessBoard) -> bool:
        """Check for terminal game state"""
        try:
            return (board.is_checkmate(board.turn) or 
                    board.is_stalemate(board.turn) or
                    board.is_fifty_move_rule_draw() or
                    board.is_threefold_repetition())
        except Exception as e:
            return False
    
    def _get_terminal_value(self, board: ChessBoard) -> float:
        """Get value for terminal position"""
        try:
            if board.is_checkmate(board.turn):
                # Current player lost
                return -1.0
            else:
                # Draw
                return 0.0
        except Exception as e:
            return 0.0