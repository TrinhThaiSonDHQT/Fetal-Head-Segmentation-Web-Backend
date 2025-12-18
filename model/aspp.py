"""
Atrous Spatial Pyramid Pooling (ASPP) Module

ASPP captures multi-scale contextual information using parallel atrous convolutions
with different dilation rates. This implementation is based on DeepLabV3+.

Components:
1. 1x1 convolution (captures point-wise features)
2. Multiple 3x3 atrous convolutions with different dilation rates
3. Global average pooling branch (captures image-level features)
4. Fusion of all branches via 1x1 convolution

Reference:
    Chen, L. C., et al. (2018). Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation (DeepLabV3+). ECCV 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) Module.
    
    Captures multi-scale contextual information through parallel branches:
    - 1x1 convolution
    - 3x3 atrous convolutions with different dilation rates
    - Global average pooling
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        atrous_rates (list): List of dilation rates for atrous convolutions
                            Default: [6, 12, 18] (DeepLabV3+ standard)
        dropout_rate (float): Dropout rate for regularization (default: 0.5)
    
    Input shape: (B, in_channels, H, W)
    Output shape: (B, out_channels, H, W)
    """
    
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18], dropout_rate=0.5, use_groupnorm=True):
        super(ASPP, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates
        self.use_groupnorm = use_groupnorm
        
        # Branch 1: 1x1 convolution (point-wise feature extraction)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2-4: 3x3 atrous convolutions with different dilation rates
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Branch 5: Global average pooling branch (image-level features)
        # Use GroupNorm (default) instead of BatchNorm to handle batch_size=1 cases
        if use_groupnorm:
            # GroupNorm: works with any batch size
            norm_layer = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        else:
            # BatchNorm: requires batch_size > 1 (for backward compatibility)
            norm_layer = nn.BatchNorm2d(out_channels)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling to (B, C, 1, 1)
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer,
            nn.ReLU(inplace=True)
        )
        
        # Fusion: Concatenate all branches and project to output channels
        # Total branches: 1x1 conv + len(atrous_rates) atrous convs + global pooling
        num_branches = len(atrous_rates) + 2
        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * num_branches,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)  # Regularization
        )
    
    def forward(self, x):
        """
        Forward pass through ASPP module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        h, w = x.shape[2:]
        
        # Collect features from all branches
        features = []
        
        # Branch 1: 1x1 convolution
        features.append(self.conv1x1(x))
        
        # Branch 2-4: Atrous convolutions
        for atrous_conv in self.atrous_convs:
            features.append(atrous_conv(x))
        
        # Branch 5: Global average pooling
        global_feat = self.global_avg_pool(x)
        # Upsample back to original spatial dimensions
        global_feat = F.interpolate(
            global_feat,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        features.append(global_feat)
        
        # Concatenate all branches along channel dimension
        x = torch.cat(features, dim=1)
        
        # Project to output channels
        x = self.project(x)
        
        return x


if __name__ == "__main__":
    """Test ASPP module"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with bottleneck dimensions (after 4 pooling operations: 256/16 = 16)
    in_channels = 512
    out_channels = 1024
    batch_size = 2
    spatial_size = 16  # 256 / (2^4) = 16
    
    # Create ASPP module
    aspp = ASPP(
        in_channels=in_channels,
        out_channels=out_channels,
        atrous_rates=[6, 12, 18],
        dropout_rate=0.5
    ).to(device)
    
    # Test forward pass
    dummy_input = torch.randn(batch_size, in_channels, spatial_size, spatial_size).to(device)
    output = aspp(dummy_input)
    
    # Count parameters
    total_params = sum(p.numel() for p in aspp.parameters())
    trainable_params = sum(p.numel() for p in aspp.parameters() if p.requires_grad)
    
    print("=" * 70)
    print("ASPP Module Test")
    print("=" * 70)
    print(f"Input shape:          {dummy_input.shape}")
    print(f"Output shape:         {output.shape}")
    print(f"Expected output:      torch.Size([{batch_size}, {out_channels}, {spatial_size}, {spatial_size}])")
    print(f"Shape match:          {output.shape == torch.Size([batch_size, out_channels, spatial_size, spatial_size])}")
    print("-" * 70)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Module size:          ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 70)
