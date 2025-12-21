"""
Test Quantization Accuracy

Compares predictions from original vs quantized model on sample images
to verify acceptable accuracy loss (<2%).

Usage:
    python test_quantization.py [image_path]
"""
import torch
import numpy as np
import cv2
import sys
import os
from pathlib import Path
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from model.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet


def load_model(model_path, quantized=False):
    """Load model (original or quantized)."""
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
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def preprocess_image(image_path):
    """Preprocess image for inference."""
    img = Image.open(image_path).convert('L')  # Grayscale
    img = np.array(img)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor, img


def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient."""
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)
    
    intersection = np.sum(pred_binary * target_binary)
    dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(target_binary) + 1e-7)
    return dice


def test_quantization(image_path=None):
    """Compare original vs quantized model predictions."""
    
    original_path = Path("best_model_mobinet_aspp_residual_se_v2.pth")
    quantized_path = Path("best_model_mobinet_aspp_residual_se_v2_quantized.pth")
    
    # Check if models exist
    if not original_path.exists():
        print(f"❌ Original model not found: {original_path}")
        return
    
    if not quantized_path.exists():
        print(f"❌ Quantized model not found: {quantized_path}")
        print("   Run 'python quantize_model.py' first")
        return
    
    print("=" * 60)
    print("Quantization Accuracy Test")
    print("=" * 60)
    
    # Load models
    print("\n[1/3] Loading models...")
    original_model = load_model(original_path, quantized=False)
    quantized_model = load_model(quantized_path, quantized=True)
    print("   ✓ Both models loaded")
    
    # Get test image
    if image_path is None:
        print("\n⚠️  No test image provided")
        print("   Usage: python test_quantization.py <path_to_ultrasound_image>")
        print("   Skipping accuracy comparison (models loaded successfully)")
        return
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"\n❌ Image not found: {image_path}")
        return
    
    print(f"\n[2/3] Testing on image: {image_path.name}")
    
    # Preprocess
    input_tensor, _ = preprocess_image(image_path)
    
    # Run inference
    print("   Running inference...")
    with torch.no_grad():
        original_output = torch.sigmoid(original_model(input_tensor))
        quantized_output = torch.sigmoid(quantized_model(input_tensor))
    
    # Convert to numpy
    original_mask = original_output.squeeze().cpu().numpy()
    quantized_mask = quantized_output.squeeze().cpu().numpy()
    
    # Calculate metrics
    print("\n[3/3] Comparing predictions...")
    
    # Dice similarity between predictions
    dice = calculate_dice(quantized_mask, original_mask, threshold=0.5)
    
    # Pixel-wise difference
    pixel_diff = np.abs(original_mask - quantized_mask)
    mean_diff = np.mean(pixel_diff)
    max_diff = np.max(pixel_diff)
    
    # Accuracy loss
    accuracy_loss = (1 - dice) * 100
    
    print(f"\n📊 Results:")
    print(f"   Dice Similarity:    {dice:.4f} ({dice*100:.2f}%)")
    print(f"   Accuracy Loss:      {accuracy_loss:.2f}%")
    print(f"   Mean Pixel Diff:    {mean_diff:.4f}")
    print(f"   Max Pixel Diff:     {max_diff:.4f}")
    
    # Verdict
    print("\n" + "=" * 60)
    if accuracy_loss < 2.0:
        print("✅ PASS: Accuracy loss < 2% - Safe to deploy!")
    elif accuracy_loss < 5.0:
        print("⚠️  CAUTION: Accuracy loss 2-5% - Consider testing more images")
    else:
        print("❌ FAIL: Accuracy loss > 5% - Not recommended for production")
    print("=" * 60)
    
    # Save comparison image
    try:
        comparison = np.hstack([
            (original_mask * 255).astype(np.uint8),
            (quantized_mask * 255).astype(np.uint8),
            (pixel_diff * 255).astype(np.uint8)
        ])
        cv2.imwrite('quantization_comparison.png', comparison)
        print("\n💾 Comparison saved: quantization_comparison.png")
        print("   (Left: Original | Middle: Quantized | Right: Difference)")
    except:
        pass


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_quantization(image_path)
