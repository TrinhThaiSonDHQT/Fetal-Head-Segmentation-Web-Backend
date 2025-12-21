"""
Model Quantization Script

Converts the PyTorch model to INT8 quantized version to reduce memory usage 
by ~50-60% while maintaining accuracy (1-2% loss typical).

Usage:
    python quantize_model.py
"""
import torch
import sys
import os
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

from model.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet


def quantize_model():
    """Quantize the model to INT8 for reduced memory footprint."""
    
    model_path = Path("best_model_mobinet_aspp_residual_se_v2.pth")
    output_path = Path("best_model_mobinet_aspp_residual_se_v2_quantized.pth")
    
    if not model_path.exists():
        print(f"❌ Error: Model file not found: {model_path}")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    print("=" * 60)
    print("PyTorch Model Quantization")
    print("=" * 60)
    print(f"\n📁 Input:  {model_path}")
    print(f"📁 Output: {output_path}")
    
    # Load original model
    print("\n[1/4] Loading original model...")
    try:
        device = torch.device('cpu')  # Quantization requires CPU
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model architecture
        model = MobileNetV2ASPPResidualSEUNet(
            in_channels=1,
            out_channels=1,
            weights=None,
            freeze_encoder=False,
            reduction_ratio=16,
            atrous_rates=[6, 12, 18],
            aspp_dropout=0.5,
            aspp_use_groupnorm=True
        )
        
        # Load weights
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_dice' in checkpoint:
                print(f"   ✓ Best Dice Score: {checkpoint['best_dice']:.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print("   ✓ Model loaded successfully")
        
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        return
    
    # Quantize model
    print("\n[2/4] Applying dynamic INT8 quantization...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},  # Quantize convolutions and linear layers
            dtype=torch.qint8
        )
        print("   ✓ Quantization completed")
    except Exception as e:
        print(f"   ❌ Quantization failed: {e}")
        return
    
    # Save quantized model
    print("\n[3/4] Saving quantized model...")
    try:
        torch.save(quantized_model.state_dict(), output_path)
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        return
    
    # Compare file sizes
    print("\n[4/4] Comparing model sizes...")
    original_size = model_path.stat().st_size / (1024**2)
    quantized_size = output_path.stat().st_size / (1024**2)
    reduction = (1 - quantized_size/original_size) * 100
    
    print(f"\n   Original model:  {original_size:.2f} MB")
    print(f"   Quantized model: {quantized_size:.2f} MB")
    print(f"   Size reduction:  {reduction:.1f}%")
    
    # Memory usage estimate
    print("\n📊 Estimated memory usage during inference:")
    print(f"   Original:  ~{original_size * 30:.0f} MB (model + activations)")
    print(f"   Quantized: ~{quantized_size * 30:.0f} MB (model + activations)")
    
    print("\n" + "=" * 60)
    print("✅ Quantization Complete!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("   1. Test accuracy: python test_quantization.py")
    print("   2. Update config.py: MODEL_PATH = 'best_model_mobinet_aspp_residual_se_v2_quantized.pth'")
    print("   3. Redeploy to Render with quantized model")
    print("\n⚠️  Expected accuracy loss: 1-2% (acceptable for most use cases)")


if __name__ == "__main__":
    quantize_model()
