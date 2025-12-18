import numpy as np
import cv2
from typing import Dict, Tuple

class QualityChecker:
    """Analyzes segmentation quality to detect invalid/poor results."""
    
    def __init__(self):
        # Thresholds based on fetal head characteristics
        self.MIN_AREA_RATIO = 0.05  # Mask should be at least 5% of image
        self.MAX_AREA_RATIO = 0.60  # Mask shouldn't exceed 60% of image
        self.MIN_CIRCULARITY = 0.60  # Fetal head is roughly circular
        self.MIN_EDGE_SHARPNESS = 0.002  # Edges should be reasonably sharp (lowered for real masks)
    
    def analyze_mask(self, mask: np.ndarray) -> Dict:
        """
        Analyze segmentation mask quality.
        
        Args:
            mask: Binary mask (0-255, 0-1, or any binary format)
        
        Returns:
            Dictionary with quality metrics
        """
        # Normalize mask to 0-255 binary format
        if mask.max() <= 1:
            # Float mask in range [0, 1]
            mask = (mask * 255).astype(np.uint8)
        elif mask.max() < 255:
            # Binary mask with values other than 255 (e.g., 0 and 38)
            # Convert to proper binary: 0 or 255
            mask = ((mask > 0) * 255).astype(np.uint8)
        
        # Calculate metrics
        area_ratio = self._calculate_area_ratio(mask)
        circularity = self._calculate_circularity(mask)
        edge_sharpness = self._calculate_edge_sharpness(mask)
        
        # Determine if shape is valid
        is_valid_shape = self._is_valid_fetal_head_shape(
            area_ratio, circularity, edge_sharpness
        )
        
        return {
            'mask_area_ratio': round(area_ratio, 4),
            'mask_circularity': round(circularity, 4),
            'edge_sharpness': round(edge_sharpness, 4),
            'is_valid_shape': bool(is_valid_shape)
        }
    
    def _calculate_area_ratio(self, mask: np.ndarray) -> float:
        """Calculate ratio of mask area to total image area."""
        total_pixels = mask.shape[0] * mask.shape[1]
        mask_pixels = np.count_nonzero(mask > 127)
        return mask_pixels / total_pixels
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """
        Calculate how circular/elliptical the mask is.
        Circularity = 4π × Area / Perimeter²
        Perfect circle = 1.0, lower values = less circular
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return 0.0
        
        # Get largest contour (main mask)
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return min(circularity, 1.0)  # Cap at 1.0
    
    def _calculate_edge_sharpness(self, mask: np.ndarray) -> float:
        """
        Calculate edge sharpness using gradient magnitude.
        Sharp edges = higher values
        """
        # Calculate gradients
        grad_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize to 0-1
        if gradient_magnitude.max() > 0:
            sharpness = gradient_magnitude.mean() / gradient_magnitude.max()
        else:
            sharpness = 0.0
        
        return sharpness
    
    def _is_valid_fetal_head_shape(
        self, area_ratio: float, circularity: float, edge_sharpness: float
    ) -> bool:
        """Determine if metrics indicate a valid fetal head segmentation."""
        return (
            self.MIN_AREA_RATIO <= area_ratio <= self.MAX_AREA_RATIO and
            circularity >= self.MIN_CIRCULARITY and
            edge_sharpness >= self.MIN_EDGE_SHARPNESS
        )
    
    def compute_segmentation_confidence(self, probability_map: np.ndarray) -> float:
        """
        Compute confidence score for segmentation quality (0-1).
        
        Uses mean probability in the predicted segmented region as confidence.
        This directly reflects the model's confidence in its prediction.
        
        Args:
            probability_map: Model's probability map (values 0-1)
        
        Returns:
            Confidence score (0-1) where:
            - 0.8-1.0: High confidence, model very certain
            - 0.6-0.8: Medium confidence, model fairly certain
            - 0.0-0.6: Low confidence, model uncertain
        """
        # Get binary mask from probability map
        mask_binary = (probability_map > 0.5).astype(np.uint8)
        
        # If no pixels predicted, return 0 confidence
        if mask_binary.sum() == 0:
            return 0.0
        
        # Calculate mean probability in predicted region
        masked_probs = probability_map[mask_binary > 0]
        mean_confidence = masked_probs.mean()
        
        return float(mean_confidence)
    
    def generate_warnings(self, metrics: Dict, confidence_score: float = None) -> list:
        """
        Generate human-readable warnings based on quality metrics.
        
        Args:
            metrics: Quality metrics dictionary
            confidence_score: Optional confidence score to include in warnings
        """
        warnings = []
        
        area_ratio = metrics['mask_area_ratio']
        circularity = metrics['mask_circularity']
        edge_sharpness = metrics['edge_sharpness']
        
        if area_ratio < self.MIN_AREA_RATIO:
            warnings.append(
                "Segmentation area is very small. This may not be a valid ultrasound image."
            )
        elif area_ratio > self.MAX_AREA_RATIO:
            warnings.append(
                "Segmentation area is unusually large. Image quality may be poor."
            )
        
        if circularity < self.MIN_CIRCULARITY:
            warnings.append(
                "Detected shape is not circular/elliptical. "
                "This may not be a fetal head ultrasound."
            )
        
        if edge_sharpness < self.MIN_EDGE_SHARPNESS:
            warnings.append(
                "Segmentation edges are unclear. Image may be low quality or incorrect."
            )
        
        if not metrics['is_valid_shape']:
            warnings.append(
                "⚠️ The uploaded image may not be a fetal head ultrasound. "
                "Results may be inaccurate."
            )
        
        # Add confidence-based warning
        if confidence_score is not None and confidence_score < 0.5:
            warnings.append(
                f"⚠️ Low segmentation confidence ({confidence_score*100:.1f}%). "
                "Results may not be reliable."
            )
        
        return warnings
