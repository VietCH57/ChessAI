import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from chess_board import ChessBoard, Position, PieceColor
from network import AlphaZeroNetwork, BoardEncoder, MoveEncoder
from config import AlphaZeroConfig

class MCTSNode:
    def __init__(self, board: ChessBoard, parent=None, move=None, prior=0.0):
        self.board = board.copy_board()
        self.parent = parent
        self.move = move  # Move that led to this node
        self.prior = prior
        
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[tuple, MCTSNode] = {}
        self.is_expanded = False
        
    def is_leaf(self) -> bool:
        return not self.is_expanded
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest UCB score"""
        if not self.children:
            return None
            
        best_score = -float('inf')
        best_child = None
        
        for child in self.children.values():
            if child.visit_count == 0:
                ucb_score = float('inf')
            else:
                # UCB1 formula with prior
                q_value = child.total_value / child.visit_count
                u_value = c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
                ucb_score = q_value + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child
    
    def expand(self, policy_probs: np.ndarray, move_encoder: MoveEncoder):
        """Expand node with children for all legal moves"""
        if self.is_expanded:
            return
            
        legal_moves = []
        
        # Get all legal moves more efficiently
        for row in range(8):
            for col in range(8):
                from_pos = Position(row, col)
                piece = self.board.get_piece(from_pos)
                if piece and piece.color == self.board.turn:
                    try:
                        valid_moves = self.board.get_valid_moves(from_pos)
                        for move in valid_moves:
                            legal_moves.append((from_pos, move.end_pos))
                    except Exception as e:
                        print(f"Error getting moves for {from_pos.row},{from_pos.col}: {e}")
                        continue
        
        
        # Create children for legal moves
        for from_pos, to_pos in legal_moves:
            try:
                # Get policy probability for this move
                move_idx = move_encoder.encode_move(from_pos, to_pos)
                prior = policy_probs[move_idx] if move_idx >= 0 and move_idx < len(policy_probs) else 0.01
                
                # Create new board state
                new_board = self.board.copy_board()
                move_obj = new_board.move_piece(from_pos, to_pos)
                
                if move_obj:  # Valid move
                    move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
                    child = MCTSNode(new_board, parent=self, move=move_key, prior=prior)
                    self.children[move_key] = child
            except Exception as e:
                print(f"Error creating child for move {from_pos.row},{from_pos.col} -> {to_pos.row},{to_pos.col}: {e}")
                continue
        
        self.is_expanded = True
    
    def backup(self, value: float):
        """Backup value through the tree"""
        self.visit_count += 1
        self.total_value += value
        
        if self.parent:
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
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        
        # Batching for neural network evaluation
        self.eval_batch_size = 8
        self.pending_evaluations = []
        
    def search(self, board: ChessBoard, num_simulations: int, 
               temperature: float = 1.0, add_noise: bool = False) -> Tuple[np.ndarray, float]:
        """Optimized MCTS search with batched evaluation"""
        root = MCTSNode(board)
        
        # Process simulations in batches
        batch_size = min(self.eval_batch_size, num_simulations)
        completed_sims = 0
        
        while completed_sims < num_simulations:
            current_batch_size = min(batch_size, num_simulations - completed_sims)
            self._simulate_batch(root, current_batch_size)
            completed_sims += current_batch_size
            
        # Add Dirichlet noise
        if add_noise and root.is_expanded and root.children:
            self._add_dirichlet_noise(root)
        
        # Get action probabilities
        visit_counts = root.get_visit_counts()
        
        if temperature == 0:
            action_probs = np.zeros_like(visit_counts)
            if visit_counts.sum() > 0:
                best_action = np.argmax(visit_counts)
                action_probs[best_action] = 1.0
            else:
                action_probs = np.ones_like(visit_counts) / len(visit_counts)
        else:
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            if visit_counts_temp.sum() > 0:
                action_probs = visit_counts_temp / visit_counts_temp.sum()
            else:
                action_probs = np.ones_like(visit_counts) / len(visit_counts)
        
        estimated_value = root.total_value / max(root.visit_count, 1)
        return action_probs, estimated_value
    
    def _simulate_batch(self, root: MCTSNode, batch_size: int):
        """Run multiple simulations and batch neural network evaluations"""
        leaf_nodes = []
        
        # Collect leaf nodes from multiple simulations
        for _ in range(batch_size):
            leaf_node = self._traverse_to_leaf(root)
            if leaf_node and not self._is_terminal(leaf_node.board):
                leaf_nodes.append(leaf_node)
        
        # Batch evaluate all leaf nodes
        if leaf_nodes:
            self._batch_evaluate_and_expand(leaf_nodes)
    
    def _traverse_to_leaf(self, root: MCTSNode) -> Optional[MCTSNode]:
        """Traverse from root to a leaf node"""
        node = root
        path = [node]
        
        depth = 0
        while not node.is_leaf() and not self._is_terminal(node.board) and depth < 100:
            node = node.select_child(self.config.c_puct)
            if node is None:
                break
            path.append(node)
            depth += 1
        
        return node if node else None
    
    def _batch_evaluate_and_expand(self, leaf_nodes: List[MCTSNode]):
        """Evaluate multiple leaf nodes in a single batch"""
        if not leaf_nodes:
            return
        
        try:
            # Prepare batch input
            batch_states = []
            for node in leaf_nodes:
                board_tensor = self.board_encoder.encode_board(node.board)
                batch_states.append(board_tensor)
            
            batch_tensor = torch.FloatTensor(np.array(batch_states))
            if torch.cuda.is_available() and self.config.device == "cuda":
                batch_tensor = batch_tensor.cuda()
            
            # Batch neural network evaluation
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
            print(f"Error in batch evaluation: {e}")
            # Fallback to individual evaluation
            for node in leaf_nodes:
                try:
                    value = self._expand_and_evaluate(node)
                    node.backup(value)
                except:
                    node.backup(0.0)
    
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """Fallback single node evaluation"""
        try:
            board_tensor = self.board_encoder.encode_board(node.board)
            board_tensor = torch.FloatTensor(board_tensor).unsqueeze(0)
            
            if torch.cuda.is_available() and self.config.device == "cuda":
                board_tensor = board_tensor.cuda()
            
            self.network.eval()
            with torch.no_grad():
                policy_logits, value = self.network(board_tensor)
                policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
                value = value.cpu().numpy()[0][0]
            
            node.expand(policy_probs, self.move_encoder)
            return float(value)
            
        except Exception as e:
            print(f"Error in single evaluation: {e}")
            uniform_probs = np.ones(4096) / 4096
            node.expand(uniform_probs, self.move_encoder)
            return 0.0
    
    def _add_dirichlet_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root node priors"""
        if not root.children:
            return
            
        num_children = len(root.children)
        if num_children == 0:
            return
            
        try:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_children)
            for i, child in enumerate(root.children.values()):
                child.prior = (1 - self.config.dirichlet_epsilon) * child.prior + \
                             self.config.dirichlet_epsilon * noise[i]
        except Exception as e:
            print(f"Error adding Dirichlet noise: {e}")
    
    def _is_terminal(self, board: ChessBoard) -> bool:
        """Check if board position is terminal"""
        try:
            return (board.is_checkmate(board.turn) or 
                    board.is_stalemate(board.turn) or
                    board.is_fifty_move_rule_draw() or
                    board.is_threefold_repetition())
        except Exception as e:
            print(f"Error checking terminal state: {e}")
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
            print(f"Error getting terminal value: {e}")
            return 0.0