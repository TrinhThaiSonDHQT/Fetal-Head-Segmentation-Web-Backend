"""
Model Loader for Fetal Head Segmentation

Loads the PyTorch model and prepares it for inference.
Handles device detection (CPU/GPU) and model initialization.
"""
import torch
from pathlib import Path
import sys
import os

# Add backend directory to path for model imports
sys.path.insert(0, os.path.dirname(__file__))

from model.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet


class ModelLoader:
    """
    Loads and manages the PyTorch segmentation model.
    
    Attributes:
        device (torch.device): Device for inference (cuda/cpu)
        model (nn.Module): Loaded PyTorch model in eval mode
        model_path (Path): Path to model weights file
    """
    
    def __init__(self, model_path):
        """
        Initialize ModelLoader.
        
        Args:
            model_path (str or Path): Path to model weights (.pth file)
        """
        self.model_path = Path(model_path)
        self.device = self._detect_device()
        self.model = self._load_model()
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Device: {self.device}")
        print(f"✓ Model weights: {self.model_path.name}")
    
    def _detect_device(self):
        """
        Detect available device (CUDA GPU or CPU).
        
        Returns:
            torch.device: Device for inference
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("✓ No GPU detected, using CPU")
        
        return device
    
    def _load_model(self):
        """
        Load model architecture and weights.
        Supports both original and quantized models.
        
        Returns:
            nn.Module: Loaded model in evaluation mode
        """
        try:
            # Detect if quantized model
            is_quantized = 'quantized' in str(self.model_path).lower()
            
            # Load the checkpoint (weights_only=False for compatibility)
            checkpoint = torch.load(
                self.model_path, 
                map_location=self.device,
                weights_only=False  # Required for models trained with older PyTorch versions
            )
            
            # Create model architecture
            model = MobileNetV2ASPPResidualSEUNet(
                in_channels=1,
                out_channels=1,
                weights=None,  # We're loading our trained weights
                freeze_encoder=False,  # No need to freeze during inference
                reduction_ratio=16,
                atrous_rates=[6, 12, 18],
                aspp_dropout=0.5,
                aspp_use_groupnorm=True
            )
            
            # Load weights - handle both checkpoint formats
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Checkpoint with metadata (epoch, optimizer, etc.)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                if 'best_dice' in checkpoint:
                    print(f"✓ Best Dice Score: {checkpoint['best_dice']:.4f}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            if is_quantized:
                print("✓ Quantized model detected (INT8 - reduced memory)")
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def get_model(self):
        """
        Get the loaded model.
        
        Returns:
            nn.Module: Model in eval mode
        """
        return self.model
    
    def get_device(self):
        """
        Get the device being used.
        
        Returns:
            torch.device: Current device
        """
        return self.device
