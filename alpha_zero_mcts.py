import math
import numpy as np
import torch
from chess_board import ChessBoard, Position, PieceType, PieceColor
from board_encoder import ChessEncoder

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search
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
        """Select a child according to the PUCT formula"""
        # Add virtual loss to discourage other threads from selecting the same node
        self.visit_count += 1  # Virtual loss
        
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
        
        if best_move is None:
            # This shouldn't happen if the node has children
            self.visit_count -= 1  # Remove virtual loss
            raise ValueError("No best move found in select() - node has no children")
            
        return best_move, self.children[best_move]
    
    def expand(self, moves_probs):
        """Expand the node with moves and their prior probabilities"""
        for move, prob in moves_probs:
            if move not in self.children:
                self.children[move] = MCTSNode(prior_p=prob)
    
    def update(self, value):
        """Update node statistics with a new value"""
        self.visit_count += 1
        self.value_sum += value


class AlphaZeroMCTS:
    """Monte Carlo Tree Search for AlphaZero"""
    def __init__(self, network, encoder, num_simulations=800, c_puct=1.0):
        self.network = network
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        self.move_history = []
        self.prediction_cache = {}  # Cache for network predictions
    
    def _simulate_one(self, board, current_node):
        """Perform one MCTS simulation"""
        # Cache state hash once before checking - more efficient
        encoded_state = self.encoder.encode_board(board, self.move_history)
        state_hash = hash(str(encoded_state.tobytes()))
        
        # Check if game is over
        if board.is_checkmate(board.turn):
            return -1.0  # Loss from current player's perspective
        
        if board.is_stalemate(board.turn) or board.is_fifty_move_rule_draw() or board.is_threefold_repetition():
            return 0.0  # Draw
            
        # Check if node is a leaf node (not expanded)
        if not current_node.expanded():
            # Encode current board state
            encoded_state = self.encoder.encode_board(board, self.move_history)
            
            # Get policy and value prediction from neural network
            state_hash = hash(str(encoded_state.tobytes()))
            if state_hash in self.prediction_cache:
                policy_logits, value = self.prediction_cache[state_hash]
            else:
                try:
                    policy_logits, value = self.network.predict(encoded_state)
                    self.prediction_cache[state_hash] = (policy_logits, value)
                except Exception as e:
                    print(f"ERROR in network prediction: {e}")
                    # Return a default value to avoid crashing
                    return 0.0
            
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
            
            # Debug if no valid moves
            if not valid_moves:
                print(f"WARNING: No valid moves found for {board.turn}!")
                print(f"Board state: {board.board}")
                return 0.0  # Return draw value
            
            # Prepare moves with probabilities for expansion
            moves_probs = []
            for from_pos, to_pos in valid_moves:
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
            else:
                # This shouldn't happen if valid_moves is non-empty
                print("WARNING: No moves with probabilities available!")
                return 0.0
            
            # Expand current node
            current_node.expand(moves_probs)
            
            # Return predicted value
            return -value
        
        # Node has been expanded before, continue simulation
        try:
            # Select best child according to PUCT formula
            move, child_node = current_node.select(self.c_puct)
            from_pos, to_pos = move
            
            # Make a copy of the board to avoid modifying the original
            board_copy = board.copy_board()
            
            # Make the move
            result = board_copy.move_piece(from_pos, to_pos)
            if not result:
                print(f"WARNING: Invalid move in simulation: {from_pos.row},{from_pos.col} -> {to_pos.row},{to_pos.col}")
                return 0.0
            
            # Add board to history for repetition detection
            self.move_history.append(board_copy)
            
            # Recursively simulate from the child node
            value = -self._simulate_one(board_copy, child_node)
            
            # Remove the board from history
            self.move_history.pop()
            
            # Update current node statistics
            child_node.update(value)
            
            return value
            
        except Exception as e:
            print(f"ERROR in simulation: {e}")
            return 0.0

    def get_move_probabilities(self, board, temperature=1.0):
        """Run simulations and get move probabilities"""  
        try:
            # Initialize the root node if not already initialized
            if self.root is None:
                self.root = MCTSNode(prior_p=1.0)
            
            # Store the current board state for future simulations
            self.current_board = board.copy_board()
            self.move_history = []  # Reset move history
            self.prediction_cache = {}  # Clear prediction cache
            
            # Debug to check if the board has valid moves
            total_valid_moves = 0
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    if piece.color == board.turn:
                        moves = board.get_valid_moves(pos)
                        total_valid_moves += len(moves)
            
            print(f"Board has {total_valid_moves} valid moves before simulations")
            
            # Run simulations
            for i in range(self.num_simulations):
                board_copy = self.current_board.copy_board()
                self._simulate_one(board_copy, self.root)
            
            # Get move visits from root
            moves = []
            visit_counts = []
            
            for move, child in self.root.children.items():
                moves.append(move)
                visit_counts.append(child.visit_count)
            
            # Debug info
            if moves:
                top_moves = sorted(zip(moves, visit_counts), key=lambda x: x[1], reverse=True)[:5]
            else:
                print("WARNING: No moves found after simulations")
            # Add Dirichlet noise at root node for exploration (only during training)
            if add_exploration_noise and moves:
                noise = np.random.dirichlet([0.3] * len(moves))
                for i, (move, child) in enumerate(self.root.children.items()):
                    child.prior_p = 0.75 * child.prior_p + 0.25 * noise[i]

            
            # If no moves found, fall back to all valid moves with uniform distribution
            if not moves:
                valid_moves = []
                for row in range(8):
                    for col in range(8):
                        pos = Position(row, col)
                        piece = board.get_piece(pos)
                        if piece.color == board.turn:
                            moves_list = board.get_valid_moves(pos)
                            valid_moves.extend([(pos, move.end_pos) for move in moves_list])
                
                print(f"Falling back to {len(valid_moves)} valid moves with uniform distribution")
                moves = valid_moves
                visit_counts = [1] * len(valid_moves)
            
            # Apply temperature
            if len(moves) == 0:
                # No moves available - should never happen if the board state is valid
                print("CRITICAL ERROR: No moves available!")
                raise ValueError("No moves available to select from")
            
            if temperature == 0 or len(moves) == 1:
                # Choose the move with highest visit count
                best_idx = np.argmax(visit_counts)
                probabilities = np.zeros(len(moves))
                probabilities[best_idx] = 1.0
            else:
                # Apply temperature and normalize to get probabilities
                visit_counts = np.array(visit_counts) ** (1.0 / temperature)
                probabilities = visit_counts / np.sum(visit_counts)
            
            return moves, probabilities
            
        except Exception as e:
            print(f"ERROR in get_move_probabilities: {e}")
            raise