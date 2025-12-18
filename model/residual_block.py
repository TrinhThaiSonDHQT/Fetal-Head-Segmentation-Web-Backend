"""
Residual Block with Squeeze-and-Excitation (SE) Mechanism
"""
import torch
import torch.nn as nn
from .se_block import SEBlock


class ResidualBlockSE(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation mechanism.
    
    Architecture:
        Input -> Conv1 -> BatchNorm -> ReLU -> Conv2 -> BatchNorm -> SE -> (+) -> ReLU -> Output
                                                                          |
                                                                       Identity
    
    If in_channels != out_channels, a 1x1 convolution is applied to the skip connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        reduction_ratio (int): Ratio for channel reduction in SE block (default=16)
    """
    
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(ResidualBlockSE, self).__init__()
        
        # Main path: Two 3x3 convolutions with BatchNorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-and-Excitation block
        self.se = SEBlock(out_channels, reduction_ratio)
        
        # Skip connection: 1x1 conv if channel dimensions don't match
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through Residual Block with SE.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W)
        """
        # Save identity for skip connection
        identity = self.skip_connection(x)
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE mechanism
        out = self.se(out)
        
        # Add skip connection
        out = out + identity
        
        # Final activation
        out = self.relu(out)
        
        return out


class DoubleResidualBlockSE(nn.Module):
    """
    Double Residual Block with SE for stronger feature extraction.
    
    Stacks two ResidualBlockSE modules sequentially.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        reduction_ratio (int): Ratio for channel reduction in SE blocks (default=16)
    """
    
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(DoubleResidualBlockSE, self).__init__()
        
        self.block1 = ResidualBlockSE(in_channels, out_channels, reduction_ratio)
        self.block2 = ResidualBlockSE(out_channels, out_channels, reduction_ratio)
    
    def forward(self, x):
        """
        Forward pass through Double Residual Block with SE.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W)
        """
        x = self.block1(x)
        x = self.block2(x)
        return x
