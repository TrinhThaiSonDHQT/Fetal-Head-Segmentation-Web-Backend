"""
Aggressive Model Optimization for Render Deployment

Combines multiple techniques to reduce memory to <512 MB:
1. Prune model weights (remove near-zero weights)
2. Apply quantization
3. Remove unnecessary checkpoint metadata

Usage:
    python optimize_model_aggressive.py
"""
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from model.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet


def prune_model(model, amount=0.3):
    """Remove weights below threshold (pruning)."""
    import torch.nn.utils.prune as prune
    
    print(f"   Pruning {amount*100:.0f}% of weights...")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
    
    return model


def optimize_model_aggressive():
    """Apply aggressive optimization to fit <512 MB memory."""
    
    model_path = Path("best_model_mobinet_aspp_residual_se_v2.pth")
    output_path = Path("best_model_optimized_render.pth")
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print("=" * 60)
    print("Aggressive Model Optimization for Render")
    print("=" * 60)
    
    # Load model
    print("\n[1/5] Loading original model...")
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = MobileNetV2ASPPResidualSEUNet(
        in_channels=1, out_channels=1, weights=None,
        freeze_encoder=False, reduction_ratio=16,
        atrous_rates=[6, 12, 18], aspp_dropout=0.5,
        aspp_use_groupnorm=True
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   ✓ Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Step 1: Prune model
    print("\n[2/5] Pruning model weights...")
    model = prune_model(model, amount=0.4)  # Remove 40% of smallest weights
    print("   ✓ Pruning complete")
    
    # Step 2: Quantize
    print("\n[3/5] Applying INT8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Conv2d, torch.nn.Linear},
        dtype=torch.qint8
    )
    print("   ✓ Quantization complete")
    
    # Step 3: Save only state dict (no metadata)
    print("\n[4/5] Saving optimized model (state_dict only)...")
    torch.save(quantized_model.state_dict(), output_path)
    print(f"   ✓ Saved: {output_path}")
    
    # Compare sizes
    print("\n[5/5] Size comparison...")
    original_size = model_path.stat().st_size / (1024**2)
    optimized_size = output_path.stat().st_size / (1024**2)
    reduction = (1 - optimized_size/original_size) * 100
    
    print(f"\n   Original:  {original_size:.1f} MB")
    print(f"   Optimized: {optimized_size:.1f} MB")
    print(f"   Reduction: {reduction:.1f}%")
    
    print(f"\n📊 Estimated memory usage:")
    print(f"   Original:  ~{original_size * 3:.0f} MB (model + overhead)")
    print(f"   Optimized: ~{optimized_size * 3:.0f} MB (model + overhead)")
    
    if optimized_size * 3 < 400:
        print("\n✅ Should fit in Render's 512 MB limit!")
    else:
        print("\n⚠️  May still be too large for Render free tier")
        print("   Consider:")
        print("   - Disable TTA (reduces runtime memory)")
        print("   - Upgrade to Render paid plan ($21/month for 2GB)")
    
    print("\n" + "=" * 60)
    print("✅ Optimization Complete!")
    print("=" * 60)
    print(f"\n📝 Update config.py:")
    print(f"   MODEL_PATH = '{output_path.name}'")
    print(f"\n⚠️  Expected accuracy loss: 2-4% (test before deploying)")


if __name__ == "__main__":
    optimize_model_aggressive()
