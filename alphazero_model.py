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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) 
        out += residual 
        out = F.relu(out)  
        return out

class AlphaZeroNetwork(nn.Module):
    def __init__(self, num_res_blocks=20, num_filters=256, device=None):
        """
        AlphaZero neural network with:
        - Input: 8×8×119 planes representing board state
        - Body: Configurable residual blocks with configurable filters
        - Heads:
            - Policy head: outputs logits for all legal moves
            - Value head: outputs scalar ∈ [−1, 1]
        """
        super(AlphaZeroNetwork, self).__init__()
        
        # Set device (GPU/CPU) - Đảm bảo sử dụng GPU nếu có
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero network initialized on: {self.device}")

        # Input layer: 119 input planes -> num_filters
        self.conv_input = nn.Conv2d(119, num_filters, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_filters)

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # Policy head: outputs logits for all possible moves (1968 = 8×8×8×8 + 8×8×3)
        self.policy_conv = nn.Conv2d(num_filters, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 1968)

        # Value head: outputs scalar ∈ [−1, 1]
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        # Kiểm tra trước khi chuyển model sang device
        print(f"Moving model to device: {self.device}")
        
        # Di chuyển model sang device
        self.to(self.device)
        
        # Xác minh model đã được chuyển đúng sang device
        print(f"Model device after moving: {next(self.parameters()).device}")

    def forward(self, x):
        """Forward pass through the network"""
        # Đảm bảo input ở đúng device
        if x.device != self.device:
            x = x.to(self.device)
            
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
        # Convert numpy array to torch tensor and move to device
        x = torch.FloatTensor(encoded_state).to(self.device)
        
        # Add batch dimension if not already present
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # Forward pass
        with torch.no_grad():
            self.eval()  # Set to evaluation mode
            policy_logits, value = self.forward(x)
            
        # Move results back to CPU for numpy conversion
        return policy_logits.cpu().numpy(), value.item()

    def save_checkpoint(self, filepath, optimizer=None, iteration=None, config=None):
        """
        Save model checkpoint with optimizer state
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': config if config else {},
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if iteration is not None:
            checkpoint['iteration'] = iteration
            
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath, optimizer=None):
        """
        Load model checkpoint and optimizer state if provided
        Returns the iteration number if available
        """
        # Map location allows loading models saved from any device
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Di chuyển trạng thái optimizer sang đúng device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
        
        iteration = checkpoint.get('iteration', 0)
        config = checkpoint.get('config', {})
        
        return iteration, config