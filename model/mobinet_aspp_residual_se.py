"""
MobileNetV2-Based U-Net with ASPP and SE Mechanisms

This model combines efficiency and accuracy by using:
- MobileNetV2 encoder (pre-trained on ImageNet, frozen for transfer learning)
- ASPP bottleneck (multi-scale context)
- Residual decoder blocks with SE (trainable)
- SE on skip connections

Architecture:
    - Encoder: MobileNetV2 backbone (frozen, pre-trained)
    - Bottleneck: ASPP module for multi-scale feature extraction
    - Decoder: 4 upsampling stages with ResidualBlockSE (trainable)
    - SE applied to skip connections before concatenation

Key Features:
    - Efficient: MobileNetV2 uses depthwise separable convolutions
    - Transfer Learning: Pre-trained weights from ImageNet
    - Multi-scale Context: ASPP at bottleneck
    - Channel Attention: SE blocks throughout decoder and skip connections

Reference:
    - MobileNetV2: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals
      and Linear Bottlenecks. CVPR 2018.
    - DeepLabV3+: Chen, L. C., et al. (2018). Encoder-Decoder with Atrous
      Separable Convolution for Semantic Image Segmentation. ECCV 2018.
"""
import torch
import torch.nn as nn

# Optimize imports - only import what we need to speed up loading
try:
    # Faster import - only load MobileNetV2 instead of all models
    from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
except ImportError:
    # Fallback for older torchvision versions
    import torchvision.models as models
    from torchvision.models import MobileNet_V2_Weights
    mobilenet_v2 = models.mobilenet_v2

# Use proper relative imports from parent package
from .residual_block import ResidualBlockSE
from .se_block import SEBlock
from .aspp import ASPP


class MobileNetV2ASPPResidualSEUNet(nn.Module):
    """
    MobileNetV2-based U-Net with ASPP bottleneck and SE mechanisms.
    
    Architecture Overview:
        - Encoder: MobileNetV2 (pre-trained, frozen)
        - Bottleneck: ASPP module for multi-scale context
        - Decoder: 4 stages with ResidualBlockSE (trainable)
        - Skip connections: SE blocks applied before concatenation
    
    MobileNetV2 Feature Extraction Points:
        - Initial Conv (custom): 32 channels  @ H×W     (full resolution, trainable)
        - Block 1: features[1]   -> 16 channels  @ H/2×W/2
        - Block 3: features[3]   -> 24 channels  @ H/4×W/4
        - Block 6: features[6]   -> 32 channels  @ H/8×W/8
        - Block 13: features[13] -> 96 channels  @ H/16×W/16
        - Block 18: features[18] -> 1280 channels @ H/32×W/32
    
    Args:
        in_channels (int): Number of input channels (default=1 for grayscale)
        out_channels (int): Number of output channels (default=1 for binary segmentation)
        pretrained (bool): Use ImageNet pre-trained weights (default=True). Deprecated, use weights instead.
        weights (MobileNet_V2_Weights): Pre-trained weights to use (default=MobileNet_V2_Weights.IMAGENET1K_V1)
        freeze_encoder (bool): Freeze MobileNetV2 encoder (default=True)
        reduction_ratio (int): Reduction ratio for SE blocks (default=16)
        atrous_rates (list): Dilation rates for ASPP (default=[6, 12, 18])
        aspp_dropout (float): Dropout rate in ASPP module (default=0.5)
        aspp_use_groupnorm (bool): Use GroupNorm in ASPP global pooling (default=True)
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        pretrained=None,  # Deprecated: kept for backward compatibility
        weights=MobileNet_V2_Weights.IMAGENET1K_V1,
        freeze_encoder=True,
        reduction_ratio=16,
        atrous_rates=[6, 12, 18],
        aspp_dropout=0.5,
        aspp_use_groupnorm=True
    ):
        super(MobileNetV2ASPPResidualSEUNet, self).__init__()
        
        # Handle backward compatibility: if pretrained is explicitly set, use it
        if pretrained is not None:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        
        # ==================== ENCODER (MobileNetV2) ====================
        # Load pre-trained MobileNetV2
        mobilenet = mobilenet_v2(weights=weights)
        
        # MobileNetV2 expects 3-channel RGB input, but we have 1-channel grayscale
        # Solution: Replace first conv layer to accept 1-channel input
        if in_channels != 3:
            # Get original first conv weights (3, 32, 3, 3)
            original_conv = mobilenet.features[0][0]
            
            # Create new conv layer for grayscale input
            new_conv = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new conv weights
            if weights is not None:
                # Average RGB weights to initialize grayscale channel
                with torch.no_grad():
                    new_conv.weight[:, :, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
            
            # Replace first conv layer
            mobilenet.features[0][0] = new_conv
        
        # Extract encoder features at different scales
        self.encoder = mobilenet.features
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Since MobileNetV2's first layer downsamples immediately (stride=2),
        # we add a simple initial conv layer that maintains resolution
        # to create the first skip connection at full resolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # MobileNetV2 channel counts at different stages
        # Extract features at indices: [1, 3, 6, 13, 18]
        # Plus init_conv provides the first skip connection (32 channels)
        # Resolutions: [1/1, 1/2, 1/4, 1/8, 1/16, 1/32]
        # Note: MobileNetV2 downsamples to H/32 (8x8 for 256x256 input)
        self.encoder_channels = [32, 16, 24, 32, 96, 1280]
        
        # ==================== BOTTLENECK (ASPP) ====================
        # ASPP for multi-scale contextual feature extraction
        # Input: 1280 channels from MobileNetV2's last block (layer 18)
        # Output: 512 channels (standard bottleneck size)
        self.bottleneck_channels = 512
        self.bottleneck_aspp = ASPP(
            in_channels=self.encoder_channels[5],  # 1280 from layer 18
            out_channels=self.bottleneck_channels,  # 512
            atrous_rates=atrous_rates,
            dropout_rate=aspp_dropout,
            use_groupnorm=aspp_use_groupnorm
        )
        
        # ==================== DECODER ====================
        # Decoder channel progression - 5 stages to match encoder
        # [1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1]
        decoder_channels = [256, 128, 64, 32, 32]
        
        # Decoder stage 5 (1/32 -> 1/16): Upsample from bottleneck
        self.up5 = nn.ConvTranspose2d(
            self.bottleneck_channels,  # 512
            decoder_channels[0],       # 256
            kernel_size=2,
            stride=2
        )
        self.skip5_se = SEBlock(self.encoder_channels[4], reduction_ratio)  # 96 channels from layer 13
        # Input to dec5: 256 (upsampled) + 96 (skip from enc4/layer13) = 352
        self.dec5 = ResidualBlockSE(
            decoder_channels[0] + self.encoder_channels[4],  # 352
            decoder_channels[0],                              # 256
            reduction_ratio
        )
        
        # Decoder stage 4 (1/16 -> 1/8)
        self.up4 = nn.ConvTranspose2d(
            decoder_channels[0],  # 256
            decoder_channels[1],  # 128
            kernel_size=2,
            stride=2
        )
        self.skip4_se = SEBlock(self.encoder_channels[3], reduction_ratio)  # 32 channels
        # Input to dec4: 128 (upsampled) + 32 (skip) = 160
        self.dec4 = ResidualBlockSE(
            decoder_channels[1] + self.encoder_channels[3],  # 160
            decoder_channels[1],                              # 128
            reduction_ratio
        )
        
        # Decoder stage 3 (1/8 -> 1/4)
        self.up3 = nn.ConvTranspose2d(
            decoder_channels[1],  # 128
            decoder_channels[2],  # 64
            kernel_size=2,
            stride=2
        )
        self.skip3_se = SEBlock(self.encoder_channels[2], reduction_ratio)  # 24 channels
        # Input to dec3: 64 (upsampled) + 24 (skip) = 88
        self.dec3 = ResidualBlockSE(
            decoder_channels[2] + self.encoder_channels[2],  # 88
            decoder_channels[2],                              # 64
            reduction_ratio
        )
        
        # Decoder stage 2 (1/4 -> 1/2)
        self.up2 = nn.ConvTranspose2d(
            decoder_channels[2],  # 64
            decoder_channels[3],  # 32
            kernel_size=2,
            stride=2
        )
        self.skip2_se = SEBlock(self.encoder_channels[1], reduction_ratio)  # 16 channels
        # Input to dec2: 32 (upsampled) + 16 (skip) = 48
        self.dec2 = ResidualBlockSE(
            decoder_channels[3] + self.encoder_channels[1],  # 48
            decoder_channels[3],                              # 32
            reduction_ratio
        )
        
        # Decoder stage 1 (1/2 -> 1/1)
        self.up1 = nn.ConvTranspose2d(
            decoder_channels[3],  # 32
            decoder_channels[4],  # 32
            kernel_size=2,
            stride=2
        )
        self.skip1_se = SEBlock(self.encoder_channels[0], reduction_ratio)  # 32 channels
        # Input to dec1: 32 (upsampled) + 32 (skip) = 64
        self.dec1 = ResidualBlockSE(
            decoder_channels[4] + self.encoder_channels[0],  # 64
            decoder_channels[4],                              # 32
            reduction_ratio
        )
        
        # ==================== OUTPUT ====================
        self.out_conv = nn.Conv2d(decoder_channels[4], out_channels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through MobileNetV2 ASPP Residual SE U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Output segmentation map of shape (B, C_out, H, W)
        """
        # ==================== ENCODER (MobileNetV2) ====================
        # Initial conv (maintains full resolution) - for first skip connection
        enc0 = self.init_conv(x)  # (B, 32, H, W)
        
        # Extract features at different scales from MobileNetV2
        # MobileNetV2 features indices: [1, 3, 6, 13, 18]
        encoder_features = []
        
        # Pass through MobileNetV2 and collect intermediate features
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # Collect features at specific indices
            if i in [1, 3, 6, 13, 18]:
                encoder_features.append(x)
        
        # Unpack encoder features
        enc1, enc2, enc3, enc4, enc5 = encoder_features
        # enc0: (B, 32, H, W)        - from init_conv
        # enc1: (B, 16, H/2, W/2)    - from MobileNetV2 layer 1
        # enc2: (B, 24, H/4, W/4)    - from MobileNetV2 layer 3
        # enc3: (B, 32, H/8, W/8)    - from MobileNetV2 layer 6
        # enc4: (B, 96, H/16, W/16)  - from MobileNetV2 layer 13
        # enc5: (B, 1280, H/32, W/32) - from MobileNetV2 layer 18
        
        # ==================== BOTTLENECK (ASPP) ====================
        # ASPP for multi-scale contextual feature extraction
        x = self.bottleneck_aspp(enc5)  # (B, 512, H/32, W/32)
        
        # ==================== DECODER ====================
        # Decoder stage 5 (1/32 -> 1/16)
        x = self.up5(x)  # (B, 256, H/16, W/16)
        enc4 = self.skip5_se(enc4)  # Apply SE to skip connection
        x = torch.cat([x, enc4], dim=1)  # (B, 352, H/16, W/16)
        x = self.dec5(x)  # (B, 256, H/16, W/16)
        
        # Decoder stage 4 (1/16 -> 1/8)
        x = self.up4(x)  # (B, 128, H/8, W/8)
        enc3 = self.skip4_se(enc3)  # Apply SE to skip connection
        x = torch.cat([x, enc3], dim=1)  # (B, 160, H/8, W/8)
        x = self.dec4(x)  # (B, 128, H/8, W/8)
        
        # Decoder stage 3 (1/8 -> 1/4)
        x = self.up3(x)  # (B, 64, H/4, W/4)
        enc2 = self.skip3_se(enc2)  # Apply SE to skip connection
        x = torch.cat([x, enc2], dim=1)  # (B, 88, H/4, W/4)
        x = self.dec3(x)  # (B, 64, H/4, W/4)
        
        # Decoder stage 2 (1/4 -> 1/2)
        x = self.up2(x)  # (B, 32, H/2, W/2)
        enc1 = self.skip2_se(enc1)  # Apply SE to skip connection
        x = torch.cat([x, enc1], dim=1)  # (B, 48, H/2, W/2)
        x = self.dec2(x)  # (B, 32, H/2, W/2)
        
        # Decoder stage 1 (1/2 -> 1/1)
        x = self.up1(x)  # (B, 32, H, W)
        enc0 = self.skip1_se(enc0)  # Apply SE to skip connection
        x = torch.cat([x, enc0], dim=1)  # (B, 64, H, W)
        x = self.dec1(x)  # (B, 32, H, W)
        
        # ==================== OUTPUT ====================
        x = self.out_conv(x)  # (B, out_channels, H, W)
        # x = self.sigmoid(x)
        
        return x


def count_parameters(model):
    """
    Count the total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params, frozen_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MobileNetV2ASPPResidualSEUNet(
        in_channels=1,
        out_channels=1,
        weights=MobileNet_V2_Weights.IMAGENET1K_V1,
        freeze_encoder=True,
        reduction_ratio=16,
        atrous_rates=[6, 12, 18],
        aspp_dropout=0.5,
        aspp_use_groupnorm=True
    ).to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    output = model(dummy_input)
    
    # Print model information
    print("=" * 80)
    print("MobileNetV2-Based ASPP Residual SE U-Net Model Summary")
    print("=" * 80)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("-" * 80)
    
    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    print(f"Frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
    print(f"Total model size:     ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"Trainable size:       ~{trainable_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 80)
    
    # Architecture summary
    print("\nArchitecture Highlights:")
    print("  • Encoder: MobileNetV2 (pre-trained on ImageNet, frozen)")
    print("  • Bottleneck: ASPP (multi-scale context, rates=[6, 12, 18])")
    print("  • Decoder: 4 stages with ResidualBlockSE (trainable)")
    print("  • Skip connections: SE blocks applied before concat")
    print("\nKey Advantages:")
    print("  ✓ Efficient: MobileNetV2 uses depthwise separable convolutions")
    print("  ✓ Transfer Learning: Pre-trained weights from ImageNet")
    print("  ✓ Fast Training: Frozen encoder reduces trainable parameters")
    print("  ✓ Multi-scale Context: ASPP captures features at multiple scales")
    print("  ✓ Channel Attention: SE blocks enhance feature representation")
    print("=" * 80)
