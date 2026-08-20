"""
Neural network architecture for Go AlphaZero.
"""

import math

import torch.nn.functional as F
from torch import nn

from config import (
    BOARD_SIZE,
    DEFAULT_BLOCKS,
    DEFAULT_CHANNELS,
    INPUT_CHANNELS,
    NUM_ACTIONS,
)


class ResBlock(nn.Module):
    """Nested bottleneck residual block without BatchNorm."""
    def __init__(self, channels: int):
        super().__init__()
        bottleneck_channels = channels // 2
        
        # Bottleneck reduction
        self.conv1 = nn.Conv2d(channels, bottleneck_channels, 1, bias=False)
        
        # First nested block
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        
        # Second nested block
        self.conv4 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.conv5 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        
        # Bottleneck expansion
        self.conv6 = nn.Conv2d(bottleneck_channels, channels, 1, bias=False)
        
        # Fixed scaling factors
        self.scale_conv4 = 1.0 / math.sqrt(2)
        self.scale_conv6 = 1.0 / math.sqrt(3)
    
    def forward(self, x):
        identity = x
        
        # Reduce to bottleneck
        out = F.gelu(x)
        out = self.conv1(out)
        
        # First nested residual block
        nested_identity1 = out
        out = F.gelu(out)
        out = self.conv2(out)
        out = F.gelu(out)
        out = self.conv3(out)
        out = out + nested_identity1
        
        # Second nested residual block
        nested_identity2 = out
        out = F.gelu(out)
        out = self.conv4(out) * self.scale_conv4
        out = F.gelu(out)
        out = self.conv5(out)
        out = out + nested_identity2
        
        # Expand back to original channels with scaling
        out = F.gelu(out)
        out = self.conv6(out) * self.scale_conv6
        
        # Main skip connection
        out = out + identity
        
        return out


class PolicyHead(nn.Module):
    """Policy head for predicting action probabilities."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 2, 1, bias=False),
            nn.BatchNorm2d(2, momentum=0.1),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, NUM_ACTIONS)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """Value head for predicting position evaluation."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.1),
            nn.GELU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class GoNetwork(nn.Module):
    """Enhanced dual-head network with preactivation nested bottleneck blocks."""
    
    def __init__(self, in_channels: int = INPUT_CHANNELS, blocks: int = DEFAULT_BLOCKS, 
                 channels: int = DEFAULT_CHANNELS):
        super().__init__()
        
        # Initial convolution with proper initialization
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels, momentum=0.1),
            nn.GELU(),
        )
        
        # Residual tower with preactivation nested bottleneck blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(channels)
            for i in range(blocks)
        ])
        
        # Single BatchNorm at the end of the trunk
        self.trunk_bn = nn.BatchNorm2d(channels, momentum=0.1)
        
        # Policy and value heads
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

        self._initialize_weights()
    
    def _initialize_weights(self):
        """Improved initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for GELU/ReLU-like activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Trunk
        x = self.stem(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.trunk_bn(x)
        
        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
