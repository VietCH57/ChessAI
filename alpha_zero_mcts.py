import math
import numpy as np
import torch
from chess_board import ChessBoard, Position
from board_encoder import ChessEncoder

class MCTSNode:
    def __init__(self, prior_p):
        self.visit_count = 0
        self.prior_p = prior_p
        self.value_sum = 0.0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def select(self, c_puct):
        best_score = -float('inf')
        best_move = None
        sum_visits = sum(child.visit_count for child in self.children.values()) + 1e-8

        for move, child in self.children.items():
            Q = child.value()
            U = c_puct * child.prior_p * math.sqrt(sum_visits) / (1 + child.visit_count)
            score = Q + U
            if score > best_score:
                best_score = score
                best_move = move

        return best_move, self.children[best_move]

    def expand(self, moves_probs):
        for move, prob in moves_probs:
            if move not in self.children:
                self.children[move] = MCTSNode(prior_p=prob)

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

class AlphaZeroMCTS:
    def __init__(self, network, encoder, num_simulations=800, c_puct=1.0):
        self.network = network
        self.encoder = encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.root = None
        self.move_history = []
        self.prediction_cache = {}

    def _simulate_one(self, board, current_node):
        if board.is_checkmate(board.turn):
            return -1.0
        if board.is_stalemate(board.turn) or board.is_fifty_move_rule_draw() or board.is_threefold_repetition():
            return 0.0

        if not current_node.expanded():
            encoded_state = self.encoder.encode_board(board, self.move_history)
            state_hash = hash(encoded_state.tobytes())

            if state_hash in self.prediction_cache:
                policy_logits, value = self.prediction_cache[state_hash]
            else:
                try:
                    policy_logits, value = self.network.predict(encoded_state)
                    self.prediction_cache[state_hash] = (policy_logits, value)
                except Exception:
                    return 0.0

            valid_moves = []
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    if piece and piece.color == board.turn:
                        moves = board.get_valid_moves(pos)
                        for move in moves:
                            valid_moves.append((pos, move.end_pos))

            if not valid_moves:
                return 0.0

            moves_probs = []
            for from_pos, to_pos in valid_moves:
                move_key = (from_pos.row, from_pos.col, to_pos.row, to_pos.col)
                idx = self.encoder.move_to_index.get(move_key, None)
                if idx is not None and idx < len(policy_logits):
                    prob = np.exp(policy_logits[idx])
                    moves_probs.append(((from_pos, to_pos), prob))

            if moves_probs:
                sum_probs = sum(prob for _, prob in moves_probs)
                if sum_probs > 0:
                    moves_probs = [(move, p / sum_probs) for move, p in moves_probs]
                else:
                    moves_probs = [(move, 1.0 / len(moves_probs)) for move, _ in moves_probs]
            else:
                return 0.0

            current_node.expand(moves_probs)
            return -value

        try:
            move, child_node = current_node.select(self.c_puct)
            from_pos, to_pos = move
            board_copy = board.copy_board()
            result = board_copy.move_piece(from_pos, to_pos)
            if not result:
                return 0.0

            self.move_history.append(board_copy)
            value = -self._simulate_one(board_copy, child_node)
            self.move_history.pop()
            child_node.update(value)
            return value
        except Exception:
            return 0.0

    def get_move_probabilities(self, board, temperature=1.0):
        if self.root is None:
            self.root = MCTSNode(prior_p=1.0)

        self.current_board = board.copy_board()
        self.move_history = []
        self.prediction_cache = {}

        for _ in range(self.num_simulations):
            board_copy = self.current_board.copy_board()
            self._simulate_one(board_copy, self.root)

        moves = []
        visit_counts = []

        for move, child in self.root.children.items():
            moves.append(move)
            visit_counts.append(child.visit_count)

        if not moves:
            valid_moves = []
            for row in range(8):
                for col in range(8):
                    pos = Position(row, col)
                    piece = board.get_piece(pos)
                    if piece and piece.color == board.turn:
                        moves_list = board.get_valid_moves(pos)
                        valid_moves.extend([(pos, move.end_pos) for move in moves_list])

            moves = valid_moves
            visit_counts = [1] * len(valid_moves)

        if temperature == 0 or len(moves) == 1:
            best_idx = np.argmax(visit_counts)
            probabilities = np.zeros(len(moves))
            probabilities[best_idx] = 1.0
        else:
            visit_counts = np.array(visit_counts) ** (1.0 / temperature)
            probabilities = visit_counts / np.sum(visit_counts)

        return moves, probabilities
