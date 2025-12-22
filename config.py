"""
Simple configuration constants for local development.

Note: This file is kept for potential future use but is not actively used in the simplified local setup.
"""
from pathlib import Path

# Model path
MODEL_PATH = Path('best_model_mobinet_aspp_residual_se_v2.pth')

# File upload limit
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
