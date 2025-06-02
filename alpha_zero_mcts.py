import numpy as np
import torch
from typing import Tuple, List, Dict, Any
import time
from chess_board import ChessBoard, Position, PieceColor, Move

class MCTSNode:
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children = {}  # Dictionary of action -> MCTSNode
        self.expanded = False

    def value(self):
        """Calculate node value (Q)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def get_visit_count(self):
        """Get visit count"""
        return self.visit_count
        
    def update(self, value):
        """Update node statistics"""
        self.value_sum += value
        self.visit_count += 1
    
    def get_children_visit_counts(self):
        """Get visit counts for all children"""
        return {action: child.get_visit_count() for action, child in self.children.items()}
    
    def get_children_values(self):
        """Get Q values for all children"""
        return {action: child.value() for action, child in self.children.items()}


class AlphaZeroMCTS:
    def __init__(self, network, encoder, num_simulations=800, c_puct=1.0, device=None):
        """
        Monte Carlo Tree Search with AlphaZero policy/value network
        
        Args:
            network: Neural network for policy and value prediction
            encoder: Board encoder
            num_simulations: Number of simulations per move
            c_puct: Exploration constant in PUCT formula
            device: Computation device
        """
        self.network = network
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        
        # Ensure we're using the same device as the network
        self.device = device if device is not None else network.device
        print(f"MCTS initialized with device: {self.device}")
        
        # Performance tracking
        self.total_nodes_created = 0
        self.inference_time = 0
        self.simulation_time = 0

    def get_move_probabilities(self, board, temperature=1.0, show_stats=False):
        """
        Run MCTS and return move probabilities
        
        Args:
            board: Current board state
            temperature: Temperature for move selection (higher = more exploration)
            
        Returns:
            List of moves and their probabilities
        """
        if self.root is None:
            self.root = MCTSNode()
        
        # Run simulations
        start_time = time.time()
        
        # Tracking stats
        pre_sim_nodes = self.total_nodes_created
        
        for i in range(self.num_simulations):
            # Make a copy of the board for this simulation
            board_copy = board.copy_board()
            
            # Run a single simulation starting from the root
            self._simulate(board_copy, self.root)
        
        # Calculate time spent
        self.simulation_time += time.time() - start_time
        
        # Collect move probabilities based on visit counts
        moves = []
        visit_counts = []
        
        # Convert child visit counts to probabilities
        for move, child in self.root.children.items():
            moves.append(move)
            visit_counts.append(child.visit_count)
            
        if len(moves) == 0:
            print("WARNING: No moves found in MCTS tree")
            return [], []
        
        # Convert visit counts to probabilities
        visit_counts = np.array(visit_counts, dtype=np.float32)
        
        if temperature == 0:
            # Temperature 0 means greedy selection
            best_move_idx = np.argmax(visit_counts)
            probabilities = np.zeros_like(visit_counts)
            probabilities[best_move_idx] = 1.0
        else:
            # Apply temperature
            visit_count_distribution = np.power(visit_counts, 1.0 / temperature)
            probabilities = visit_count_distribution / np.sum(visit_count_distribution)
        
        # Show statistics if requested
        if show_stats:
            self._show_tree_stats(moves, visit_counts, probabilities)
            
            # Print performance stats
            nodes_created = self.total_nodes_created - pre_sim_nodes
            print(f"Nodes created: {nodes_created} ({nodes_created/self.num_simulations:.1f} per sim)")
            print(f"Inference time: {self.inference_time:.2f}s")
            print(f"Simulation time: {self.simulation_time:.2f}s")
            print(f"Time per simulation: {self.simulation_time/self.num_simulations*1000:.1f}ms")
        
        return moves, probabilities
        
    def _simulate(self, board, node):
        """Run a single simulation"""
        # If the game is over, return the result
        if board.is_game_over():
            result = board.get_result()
            
            # Convert result to value: perspective of current player is +1 for win, -1 for loss, 0 for draw
            player_to_move = board.turn
            value = 0
            if result == 1:  # White wins
                value = 1 if player_to_move == PieceColor.WHITE else -1
            elif result == 2:  # Black wins
                value = 1 if player_to_move == PieceColor.BLACK else -1
                
            # Negate value for backprop (next player's perspective)
            return -value
        
        # If the node is not expanded, expand it
        if not node.expanded:
            # Encode the board for neural network input
            encoded_state = self.encoder.encode_board(board)
            
            # Get policy logits and value from neural network
            start_time = time.time()
            policy_logits, value = self.get_action_value(encoded_state)
            self.inference_time += time.time() - start_time
            
            # Process and apply mask for legal moves
            # Initialize policy for all possible moves
            legal_moves_policy = {}
            
            # Find all legal moves
            for row in range(8):
                for col in range(8):
                    from_pos = Position(row, col)
                    piece = board.get_piece(from_pos)
                    
                    # Skip empty squares and opponent's pieces
                    if piece.color != board.turn:
                        continue
                    
                    # Get legal moves for this piece
                    moves = board.get_valid_moves(from_pos)
                    for move in moves:
                        to_pos = move.end_pos
                        action = (from_pos, to_pos)
                        
                        # Get index for this move in policy vector
                        policy_idx = self.encoder.action_to_index(from_pos, to_pos, move.promotion)
                        
                        # Get and normalize probability
                        if 0 <= policy_idx < len(policy_logits):
                            prob = np.exp(policy_logits[policy_idx])
                            legal_moves_policy[action] = prob
                        else:
                            # If index is out of bounds, assign a small probability
                            legal_moves_policy[action] = 1e-6
            
            # Normalize probabilities
            if legal_moves_policy:
                total_prob = sum(legal_moves_policy.values())
                if total_prob > 0:
                    legal_moves_policy = {k: v/total_prob for k, v in legal_moves_policy.items()}
            
            # Check for no legal moves
            if not legal_moves_policy:
                # This should never happen - either the game is over (checked above)
                # or there are legal moves
                print("WARNING: No legal moves found, but game not over")
                # Return default value
                return 0
            
            # Add new nodes to the tree for each legal move
            for action, prob in legal_moves_policy.items():
                node.children[action] = MCTSNode(prior=prob)
                self.total_nodes_created += 1
                
            # Mark node as expanded
            node.expanded = True
            
            # Return value from neural network
            return -value  # Negate value for backprop (next player's perspective)
        
        # Node is already expanded - select child according to PUCT formula
        action = self._select_child(node, board)
        
        # Make the selected move
        board.move_piece(action[0], action[1])
        
        # Recursive call to simulate from the child node
        value = self._simulate(board, node.children[action])
        
        # Update node statistics
        node.children[action].update(-value)  # Negate again for current player's perspective
        
        # Return value for backpropagation
        return value
        
    def _select_child(self, node, board):
        """
        Select child node according to PUCT formula
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b(N(s,b))) / (1 + N(s,a))
        - Q(s,a): mean action value
        - P(s,a): prior probability
        - N(s,a): visit count
        - c_puct: exploration constant
        """
        # Calculate total visit count
        total_visit_count = sum(child.visit_count for child in node.children.values())
        total_visit_count = max(total_visit_count, 1)  # Avoid division by zero
        
        best_score = -float('inf')
        best_action = None
        
        # Find action with highest PUCT score
        for action, child in node.children.items():
            # Get Q value
            q_value = child.value()
            
            # Calculate PUCT formula
            exploration_term = self.c_puct * child.prior * np.sqrt(total_visit_count) / (1 + child.visit_count)
            puct_score = q_value + exploration_term
            
            # Update best action if score is higher
            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                
        # Return best action
        return best_action
        
    def _show_tree_stats(self, moves, visit_counts, probabilities):
        """Show tree statistics"""
        # Sort indices by visit count (descending)
        sorted_indices = np.argsort(-np.array(visit_counts))
        
        print("\nTop MCTS moves:")
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            move = moves[idx]
            from_pos, to_pos = move
            visit_count = visit_counts[idx]
            prob = probabilities[idx] * 100  # convert to percentage
            
            # Get Q-value if available
            q_value = "N/A"
            if self.root.children and move in self.root.children:
                q_value = f"{self.root.children[move].value():.3f}"
            
            print(f"{from_pos} -> {to_pos}: visits={visit_count}, prob={prob:.1f}%, Q={q_value}")
    
    def get_action_value(self, encoded_state):
        """
        Get policy and value from neural network prediction
        
        Args:
            encoded_state: Encoded chess board state
            
        Returns:
            policy: Policy vector
            value: Value prediction
        """
        # Đảm bảo encoded_state là một tensor trên GPU
        if isinstance(encoded_state, np.ndarray):
            # Chuyển numpy array thành tensor và đưa lên GPU
            encoded_state_tensor = torch.FloatTensor(encoded_state).to(self.device)
        elif isinstance(encoded_state, torch.Tensor):
            # Đảm bảo tensor đã ở trên GPU
            encoded_state_tensor = encoded_state.to(self.device)
        else:
            raise TypeError(f"Encoded state must be numpy array or torch tensor, got {type(encoded_state)}")
            
        # Thêm batch dimension nếu cần
        if len(encoded_state_tensor.shape) == 3:
            encoded_state_tensor = encoded_state_tensor.unsqueeze(0)
        
        # Chạy forward pass trên GPU
        with torch.no_grad():
            self.network.eval()
            policy_logits, value = self.network(encoded_state_tensor)
            
            # Kiểm tra xem kết quả có đang ở trên GPU không
            if policy_logits.device != self.device:
                policy_logits = policy_logits.to(self.device)
            if value.device != self.device:
                value = value.to(self.device)
            
            # Chuyển kết quả về CPU và numpy để xử lý tiếp
            policy_logits = policy_logits.cpu().numpy()[0]  # Lấy batch đầu tiên
            value = value.cpu().numpy()[0][0]  # Lấy giá trị duy nhất
        
        return policy_logits, value
        
    def update_with_move(self, action):
        """
        Update the tree with the played move
        
        Args:
            action: The move that was played
            
        Returns:
            None
        """
        # If the action is in the root's children, keep that subtree
        if self.root and self.root.children and action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None  # Cut connection to parent to free memory
        else:
            # Reset the tree
            self.root = MCTSNode()