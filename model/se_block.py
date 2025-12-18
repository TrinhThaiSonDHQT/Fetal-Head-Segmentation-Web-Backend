"""
Squeeze-and-Excitation (SE) Block for channel-wise attention
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    
    This module adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    
    Args:
        channels (int): Number of input/output channels
        reduction_ratio (int): Ratio for channel reduction in SE block (default=16)
    """
    
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        
        # Ensure reduction doesn't make channels less than 1
        reduced_channels = max(channels // reduction_ratio, 1)
        
        # Squeeze: Global average pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: Two FC layers with ReLU and Sigmoid
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through SE Block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Channel-wise recalibrated tensor of shape (B, C, H, W)
        """
        # Squeeze: Global information embedding
        squeezed = self.squeeze(x)
        
        # Excitation: Adaptive recalibration
        attention = self.excitation(squeezed)
        
        # Scale: Apply channel attention
        return x * attention
