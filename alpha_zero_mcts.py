import math
import numpy as np
from chess_board import ChessBoard, Position, PieceType, PieceColor
from board_encoder import ChessEncoder

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search
    Tracks visit counts, total value, and prior probability
    """
    def __init__(self, prior_p):
        self.visit_count = 0
        self.prior_p = prior_p
        self.value_sum = 0.0
        self.children = {}  # Maps move to child node
    
    def expanded(self):
        """Node is expanded if it has children"""
        return len(self.children) > 0
    
    def value(self):
        """Returns the average value of this node"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
        
    def select(self, c_puct):
        """
        Select a child according to the PUCT formula used in AlphaZero
        
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b)))/(1+N(s,a))
        """
        # Find the best child based on UCB formula
        best_score = -float('inf')
        best_move = None
        
        # Sum of all child visit counts
        sum_visits = sum(child.visit_count for child in self.children.values())
        
        for move, child in self.children.items():
            # Calculate UCB score
            Q = child.value()  # Exploitation
            U = c_puct * child.prior_p * math.sqrt(sum_visits) / (1 + child.visit_count)  # Exploration
            
            # PUCT formula
            score = Q + U
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move, self.children[best_move]
    
    def expand(self, moves_probs):
        """
        Expand the node with moves and their prior probabilities
        
        Args:
            moves_probs: List of (move, prob) tuples
        """
        for move, prob in moves_probs:
            if move not in self.children:
                self.children[move] = MCTSNode(prior_p=prob)
    
    def update(self, value):
        """
        Update node statistics with a new value
        
        Args:
            value: Value from the perspective of the player who made the move
        """
        self.visit_count += 1
        self.value_sum += value


class AlphaZeroMCTS:
    """
    Monte Carlo Tree Search as used in AlphaZero
    """
    def __init__(self, network, encoder, num_simulations=800, c_puct=1.0):
        """
        Initialize the MCTS
        
        Args:
            network: Neural network for policy and value prediction
            encoder: Board encoder
            num_simulations: Number of simulations per move (800 in AlphaZero)
            c_puct: Exploration constant in PUCT formula
        """
        self.network = network
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        self.move_history = []

    def _simulate_one(self, board, current_node):
        """
        Perform one MCTS simulation
        
        Args:
            board: Current board state
            current_node: Current node in the search tree
            
        Returns:
            Leaf node value from the current player's perspective
        """
        # Check if game is over
        if board.is_checkmate(board.turn):
            # Return -1 if checkmate (loss from current player's perspective)
            return -1.0
        
        if board.is_stalemate(board.turn) or board.is_fifty_move_rule_draw() or board.is_threefold_repetition():
            # Return 0 if draw
            return 0.0
            
        # Check if node is a leaf node (not expanded)
        if not current_node.expanded():
            # Encode current board state
            encoded_state = self.encoder.encode_board(board, self.move_history)
            
            # Get policy and value prediction from neural network
            policy_logits, value = self.network.predict(encoded_state)
            
            # Get valid moves
            valid_moves = []
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    if piece.color == board.turn:
                        moves = board.get_valid_moves(pos)
                        for move in moves:
                            valid_moves.append((pos, move.end_pos))
            
            # Prepare moves with probabilities for expansion
            moves_probs = []
            for from_pos, to_pos in valid_moves:
                # Convert positions to indices in policy vector
                move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
                if move_key in self.encoder.move_to_index:
                    idx = self.encoder.move_to_index[move_key]
                    prob = np.exp(policy_logits[idx]) if idx < len(policy_logits) else 0
                    moves_probs.append(((from_pos, to_pos), prob))
            
            # Normalize probabilities
            if moves_probs:
                sum_probs = sum(prob for _, prob in moves_probs)
                if sum_probs > 0:
                    moves_probs = [(move, p/sum_probs) for move, p in moves_probs]
                else:
                    # If all probabilities are 0, use uniform distribution
                    moves_probs = [(move, 1.0/len(moves_probs)) for move, _ in moves_probs]
            
            # Expand current node
            current_node.expand(moves_probs)
            
            # Return predicted value from neural network
            # Negate because value is from the perspective of the player to move in the neural network,
            # but we need it from the perspective of the player who made the move to this position
            return -value
        
        # Node has been expanded before, continue simulation
        # Select best child according to PUCT formula
        move, child_node = current_node.select(self.c_puct)
        from_pos, to_pos = move
        
        # Make a copy of the board to avoid modifying the original
        board_copy = board.copy_board()
        
        # Sửa phần này: Dùng move_piece thay vì make_move
        try:
            board_copy.move_piece(from_pos, to_pos)
        except Exception as e:
            print(f"Error in move_piece during simulation: {e}")
            return -1.0  # Trả về giá trị xấu nếu gặp lỗi
        
        # Add board to history for repetition detection
        self.move_history.append(board_copy)
        
        # Recursively simulate from the child node
        value = -self._simulate_one(board_copy, child_node)  # Negate for alternating players
        
        # Remove the board from history
        self.move_history.pop()
        
        # Update current node statistics
        current_node.update(value)
        
        return value

    def get_move_probabilities(self, board, temperature=1.0):
        """
        Run simulations and get move probabilities based on visit counts
        
        Args:
            board: Current board state
            temperature: Temperature for exploration (1=explore, 0=best move)
            
        Returns:
            moves: List of possible moves
            probabilities: Probability distribution based on visit counts
        """
        # Initialize the root node if not already initialized
        if self.root is None:
            self.root = MCTSNode(prior_p=1.0)
        
        # Store the current board state for future simulations
        self.current_board = board.copy_board()
        self.move_history = []  # Reset move history
        
        # Run simulations
        for _ in range(self.num_simulations):
            board_copy = self.current_board.copy_board()
            self._simulate_one(board_copy, self.root)
        
        # Get move visits from root
        moves = []
        visit_counts = []
        
        for move, child in self.root.children.items():
            moves.append(move)
            visit_counts.append(child.visit_count)
        
        # Apply temperature
        if temperature == 0:
            # Choose the move with highest visit count
            best_idx = np.argmax(visit_counts)
            probabilities = np.zeros(len(moves))
            probabilities[best_idx] = 1.0
        else:
            # Apply temperature and normalize to get probabilities
            visit_counts = np.array(visit_counts) ** (1.0 / temperature)
            probabilities = visit_counts / np.sum(visit_counts)
        
        return moves, probabilities
    
        if not moves:
            print("WARNING: No moves found in MCTS tree")
            # Tìm các nước đi hợp lệ để debug
            valid_moves = []
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    if piece.color == board.turn:
                        moves_list = board.get_valid_moves(pos)
                        valid_moves.extend([(pos, move.end_pos) for move in moves_list])
            print(f"Board has {len(valid_moves)} valid moves")
        
        return moves, probabilities

    def update_with_move(self, last_move):
        """
        Update the tree with the selected move, recycling the subtree if possible
        
        Args:
            last_move: Last move made in the game
        """
        if self.root and last_move in self.root.children:
            # Reuse the subtree for the played move
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            # Reset the tree for a new search
            self.root = None