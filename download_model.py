"""
Model Downloader

Downloads model from GitHub releases or external URL at startup.
This reduces Docker image size by excluding the 272 MB model file.
"""
import requests
import os
from pathlib import Path


def download_model_if_needed():
    """Download model file if it doesn't exist locally."""
    
    model_path = Path('best_model_mobinet_aspp_residual_se_v2.pth')
    
    if model_path.exists():
        print(f"✓ Model already exists: {model_path.name}")
        return
    
    print(f"📥 Downloading model file ({model_path.name})...")
    
    # Option 1: Download from GitHub Releases
    # Replace with your actual GitHub release URL
    model_url = os.getenv(
        'MODEL_DOWNLOAD_URL',
        'https://github.com/TrinhThaiSonDHQT/Fetal-Head-Segmentation-Web-Backend/releases/download/v1.0.0/best_model_mobinet_aspp_residual_se_v2.pth'
    )
    
    try:
        response = requests.get(model_url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"   Progress: {progress:.1f}%", end='\r')
        
        print(f"\n✓ Model downloaded successfully: {model_path.name}")
        
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        print(f"   Please download manually from: {model_url}")
        raise


if __name__ == "__main__":
    download_model_if_needed()
