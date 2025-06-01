import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from chess_board import ChessBoard, Position, PieceType, PieceColor

class ResidualBlock(nn.Module):
    def __init__(self, num_filters=256):
        """Residual block as per AlphaZero paper"""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZeroNetwork(nn.Module):
    def __init__(self, num_res_blocks=20, num_filters=256):
        """
        AlphaZero neural network with:
        - Input: 8×8×119 planes representing board state
        - Body: 20 residual blocks with 256 filters
        - Heads:
            - Policy head: outputs logits for all legal moves
            - Value head: outputs scalar ∈ [−1, 1]
        """
        super(AlphaZeroNetwork, self).__init__()

        # Input layer: 119 input planes -> 256 filters
        self.conv_input = nn.Conv2d(119, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual blocks: 20 blocks with 256 filters each
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head: outputs logits for all possible moves (1968 = 8×8×8×8 + 8×8×3)
        # - 8×8×8×8 covers all possible from-to square combinations
        # - 8×8×3 covers underpromotions (knight, bishop, rook) - queen promotion is part of the from-to
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 1968)

        # Value head: outputs scalar ∈ [−1, 1]
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """Forward pass through the network"""
        # Input layer
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 8 * 8)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value

    def predict(self, encoded_state):
        """
        Make a prediction given an encoded board state
        Returns policy logits and value
        """
        # Convert numpy array to torch tensor
        x = torch.FloatTensor(encoded_state)
        
        # Add batch dimension if not already present
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            
        return policy_logits.numpy(), value.item()