import numpy as np
import cv2
from typing import Tuple


class UltrasoundDetector:
    """
    Lightweight detector to identify if an image is likely an ultrasound.
    Uses heuristics based on ultrasound image characteristics.
    """
    
    def __init__(self):
        # Ultrasound images typically have these characteristics
        self.confidence_threshold = 0.5
    
    def is_ultrasound(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Determine if image is likely an ultrasound.
        
        Args:
            image: Input image (RGB or grayscale)
        
        Returns:
            (is_ultrasound, confidence_score)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate features
        features = self._extract_features(gray)
        
        # Compute confidence score
        confidence = self._compute_confidence(features)
        
        is_us = confidence >= self.confidence_threshold
        
        return is_us, confidence
    
    def _extract_features(self, gray_image: np.ndarray) -> dict:
        """Extract features characteristic of ultrasound images."""
        h, w = gray_image.shape
        
        # Feature 1: Grayscale distribution (ultrasounds are predominantly dark/mid-tones)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Check if most pixels are in dark/mid range (0-180)
        dark_mid_ratio = hist[0:180].sum()
        
        # Feature 2: Contrast (ultrasounds have moderate contrast)
        contrast = gray_image.std() / 255.0
        
        # Feature 3: Edge density (ultrasounds have specific edge patterns)
        edges = cv2.Canny(gray_image, 50, 150)
        edge_density = np.count_nonzero(edges) / (h * w)
        
        # Feature 4: Corner regions are often black in ultrasounds (cone shape)
        corner_darkness = self._check_corner_darkness(gray_image)
        
        # Feature 5: Texture uniformity (ultrasounds have speckle noise pattern)
        texture_score = self._calculate_texture_uniformity(gray_image)
        
        return {
            'dark_mid_ratio': dark_mid_ratio,
            'contrast': contrast,
            'edge_density': edge_density,
            'corner_darkness': corner_darkness,
            'texture_score': texture_score
        }
    
    def _check_corner_darkness(self, gray_image: np.ndarray) -> float:
        """Check if corner regions are predominantly dark (typical in ultrasound)."""
        h, w = gray_image.shape
        corner_size = min(h, w) // 6
        
        # Sample corner regions
        corners = [
            gray_image[0:corner_size, 0:corner_size],  # Top-left
            gray_image[0:corner_size, -corner_size:],  # Top-right
            gray_image[-corner_size:, 0:corner_size],  # Bottom-left
            gray_image[-corner_size:, -corner_size:],  # Bottom-right
        ]
        
        # Calculate mean darkness (lower = darker)
        darkness_scores = [1.0 - (corner.mean() / 255.0) for corner in corners]
        return np.mean(darkness_scores)
    
    def _calculate_texture_uniformity(self, gray_image: np.ndarray) -> float:
        """Calculate texture uniformity (ultrasounds have speckle pattern)."""
        # Use local binary patterns or simple variance measure
        # Higher variance in local patches = more texture
        kernel_size = 8
        h, w = gray_image.shape
        
        variances = []
        for i in range(0, h - kernel_size, kernel_size):
            for j in range(0, w - kernel_size, kernel_size):
                patch = gray_image[i:i+kernel_size, j:j+kernel_size]
                variances.append(patch.var())
        
        # Normalize - use smaller denominator for better scaling
        texture_score = np.mean(variances) / 1000.0 if variances else 0.0
        return min(texture_score, 1.0)
    
    def _compute_confidence(self, features: dict) -> float:
        """
        Compute confidence score that image is an ultrasound.
        Weighted combination of features.
        """
        score = 0.0
        
        # Dark/mid-tone ratio (ultrasounds are typically 60-85% dark/mid)
        if 0.6 <= features['dark_mid_ratio'] <= 0.95:
            score += 0.25
        
        # Moderate contrast (ultrasounds: 0.15-0.35)
        if 0.10 <= features['contrast'] <= 0.40:
            score += 0.20
        
        # Edge density (ultrasounds: 0.05-0.20)
        if 0.03 <= features['edge_density'] <= 0.25:
            score += 0.20
        
        # Dark corners (ultrasounds: > 0.4)
        if features['corner_darkness'] > 0.3:
            score += 0.20
        
        # Texture (ultrasounds have speckle: 0.05-0.40)
        if 0.03 <= features['texture_score'] <= 0.60:
            score += 0.15
        
        return min(score, 1.0)
