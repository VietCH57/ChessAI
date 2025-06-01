#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt

class ChessBoardDataset(Dataset):
    """Dataset cho dữ liệu cờ vua (state, policy, value)"""
    def __init__(self, dataset):
        self.X = dataset[:, 0]  # states
        self.y_p = dataset[:, 1]  # policies
        self.y_v = dataset[:, 2]  # values
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Chuyển đổi từ (H, W, C) sang (C, H, W) cho PyTorch
        state = self.X[idx]
        if len(state.shape) == 3:
            state = state.transpose(2, 0, 1)
        return state, self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    """Convolutional block với batch normalization"""
    def __init__(self, input_channels=12, filters=256):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, filters, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)

    def forward(self, x):
        x = x.view(-1, 12, 8, 8)  # batch_size x channels x board_x x board_y
        x = F.relu(self.bn1(self.conv1(x)))
        return x

class ResidualBlock(nn.Module):
    """Residual block với batch normalization"""
    def __init__(self, filters=256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class PolicyValueHead(nn.Module):
    """Combined policy and value head"""
    def __init__(self, filters=256, num_actions=4096):
        super(PolicyValueHead, self).__init__()
        
        # Policy head
        self.policy_conv = nn.Conv2d(filters, 128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_fc = nn.Linear(8*8*128, num_actions)
        
        # Value head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8*8, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 8*8*128)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1).exp()
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8*8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class AlphaZeroNet(nn.Module):
    """AlphaZero neural network architecture"""
    def __init__(self, input_channels=12, filters=256, num_res_blocks=19, num_actions=4096):
        super(AlphaZeroNet, self).__init__()
        
        self.conv_block = ConvBlock(input_channels, filters)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(num_res_blocks)
        ])
        
        self.policy_value_head = PolicyValueHead(filters, num_actions)
    
    def forward(self, x):
        x = self.conv_block(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy, value = self.policy_value_head(x)
        return policy, value

class AlphaZeroLoss(nn.Module):
    """Loss function for AlphaZero"""
    def __init__(self):
        super(AlphaZeroLoss, self).__init__()

    def forward(self, value_pred, value_target, policy_pred, policy_target):
        value_loss = torch.mean((value_target - value_pred.view(-1)) ** 2)
        policy_loss = -torch.mean(torch.sum(policy_target * torch.log(policy_pred + 1e-8), 1))
        total_loss = value_loss + policy_loss
        return total_loss, value_loss, policy_loss

def train_network(net, dataset, epochs=100, batch_size=32, learning_rate=0.001, 
                 save_dir="./model_data/", device="cuda"):
    """Huấn luyện neural network"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    net.train()
    criterion = AlphaZeroLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    train_dataset = ChessBoardDataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_value_loss = 0.0
        epoch_policy_loss = 0.0
        num_batches = 0
        
        for batch_idx, (states, policies, values) in enumerate(train_loader):
            states = states.to(device).float()
            policies = policies.to(device).float()
            values = values.to(device).float()
            
            optimizer.zero_grad()
            
            policy_pred, value_pred = net(states)
            loss, value_loss, policy_loss = criterion(value_pred, values, policy_pred, policies)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_value_loss += value_loss.item()
            epoch_policy_loss += policy_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, '
                      f'Value Loss: {value_loss.item():.4f}, '
                      f'Policy Loss: {policy_loss.item():.4f}')
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        avg_policy_loss = epoch_policy_loss / num_batches
        
        losses.append(avg_loss)
        
        print(f'Epoch {epoch+1} completed - '
              f'Avg Loss: {avg_loss:.4f}, '
              f'Avg Value Loss: {avg_value_loss:.4f}, '
              f'Avg Policy Loss: {avg_policy_loss:.4f}')
        
        # Lưu checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Lưu model cuối cùng
    final_checkpoint = {
        'epoch': epochs,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': losses[-1] if losses else 0
    }
    torch.save(final_checkpoint, os.path.join(save_dir, 'final_model.pth'))
    
    # Vẽ biểu đồ loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, f'training_loss_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png'))
    plt.close()
    
    return losses