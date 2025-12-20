"""
Inference Module for Fetal Head Segmentation

Handles image preprocessing, model inference, and post-processing
for fetal head segmentation from ultrasound images.
"""
import torch
import cv2
import numpy as np
from PIL import Image
import time
from typing import Dict
import albumentations as A
from utils import QualityChecker, UltrasoundDetector


class InferenceEngine:
    """
    Handles the complete inference pipeline:
    1. Preprocess input image
    2. Run model inference
    3. Post-process segmentation mask
    """
    
    def __init__(self, model, device):
        """
        Initialize InferenceEngine.
        
        Args:
            model (nn.Module): Loaded PyTorch model
            device (torch.device): Device for inference
        """
        self.model = model
        self.device = device
        self.input_size = (256, 256)  # Model expects 256x256 input
        self.quality_checker = QualityChecker()
        self.us_detector = UltrasoundDetector()
        
        # Define TTA transforms (aligned with training augmentations)
        self._setup_tta_transforms()
    
    def _setup_tta_transforms(self):
        """
        Setup Test-Time Augmentation transforms.
        Uses geometric augmentations for robustness:
        - Original image
        - Horizontal flip
        - Vertical flip  
        - Both flips combined
        - Slight rotation (+10°)
        - Slight rotation (-10°)
        - Scale up (1.1x)
        - Scale down (0.9x)
        """
        self.tta_transforms = [
            # Original (no augmentation)
            {
                'name': 'original',
                'forward': A.Compose([]),
                'reverse': A.Compose([])
            },
            # Horizontal flip
            {
                'name': 'hflip',
                'forward': A.Compose([A.HorizontalFlip(p=1.0)]),
                'reverse': A.Compose([A.HorizontalFlip(p=1.0)])
            },
            # Vertical flip
            {
                'name': 'vflip',
                'forward': A.Compose([A.VerticalFlip(p=1.0)]),
                'reverse': A.Compose([A.VerticalFlip(p=1.0)])
            },
            # Both flips combined
            {
                'name': 'hvflip',
                'forward': A.Compose([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0)
                ]),
                'reverse': A.Compose([
                    A.VerticalFlip(p=1.0),
                    A.HorizontalFlip(p=1.0)
                ])
            },
            # # Rotate +10 degrees
            # {
            #     'name': 'rot+10',
            #     'forward': A.Compose([A.Rotate(limit=(10, 10), border_mode=cv2.BORDER_REFLECT, p=1.0)]),
            #     'reverse': A.Compose([A.Rotate(limit=(-10, -10), border_mode=cv2.BORDER_REFLECT, p=1.0)])
            # },
            # # Rotate -10 degrees
            # {
            #     'name': 'rot-10',
            #     'forward': A.Compose([A.Rotate(limit=(-10, -10), border_mode=cv2.BORDER_REFLECT, p=1.0)]),
            #     'reverse': A.Compose([A.Rotate(limit=(10, 10), border_mode=cv2.BORDER_REFLECT, p=1.0)])
            # },
            # # Scale up 1.1x
            # {
            #     'name': 'scale1.1',
            #     'forward': A.Compose([A.Affine(scale=1.1, p=1.0)]),
            #     'reverse': A.Compose([A.Affine(scale=1/1.1, p=1.0)])
            # },
            # # Scale down 0.9x
            # {
            #     'name': 'scale0.9',
            #     'forward': A.Compose([A.Affine(scale=0.9, p=1.0)]),
            #     'reverse': A.Compose([A.Affine(scale=1/0.9, p=1.0)])
            # },
        ]
    
    def preprocess(self, image):
        """
        Preprocess image for model input.
        
        Steps:
        1. Convert to grayscale if needed
        2. Resize to 256x256
        3. Normalize to [0, 1]
        4. Convert to tensor (1, 1, 256, 256)
        
        Args:
            image (np.ndarray or PIL.Image): Input image
        
        Returns:
            torch.Tensor: Preprocessed image tensor
            tuple: Original image size (height, width)
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Store original size
        original_size = image.shape[:2]  # (height, width)
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize to model input size
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor: (H, W) -> (1, 1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor, original_size
    
    def run_inference(self, image_tensor):
        """
        Run model inference on preprocessed image.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image (1, 1, 256, 256)
        
        Returns:
            torch.Tensor: Raw model output (1, 1, 256, 256)
            float: Inference time in milliseconds
        """
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model(image_tensor)
            
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return output, inference_time
    
    def postprocess(self, output, original_size, threshold=0.5):
        """
        Post-process model output to binary mask.
        
        Steps:
        1. Apply sigmoid activation
        2. Apply threshold
        3. Resize to original image size
        4. Convert to numpy array
        
        Args:
            output (torch.Tensor): Raw model output (1, 1, 256, 256)
            original_size (tuple): Original image size (height, width)
            threshold (float): Threshold for binary mask (default: 0.5)
        
        Returns:
            tuple: (binary_mask, probability_map)
                - binary_mask (np.ndarray): Binary mask (H, W) with values 0 or 255
                - probability_map (np.ndarray): Probability map (H, W) with values [0, 1]
        """
        # Apply sigmoid to get probabilities
        prob_mask = torch.sigmoid(output)
        
        # Apply threshold
        binary_mask = (prob_mask > threshold).float()
        
        # Convert to numpy and remove batch/channel dims: (1, 1, H, W) -> (H, W)
        mask = binary_mask.squeeze().cpu().numpy()
        prob_map = prob_mask.squeeze().cpu().numpy()
        
        # Resize to original size
        mask = cv2.resize(mask, (original_size[1], original_size[0]), 
                         interpolation=cv2.INTER_NEAREST)
        prob_map = cv2.resize(prob_map, (original_size[1], original_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
        
        # Convert binary mask to 0-255 range
        mask = (mask * 255).astype(np.uint8)
        
        return mask, prob_map
    
    def _apply_tta_augmentation(self, image_np, transform_dict):
        """
        Apply TTA augmentation to numpy image.
        
        Args:
            image_np: Grayscale image as numpy array (H, W)
            transform_dict: Dictionary with 'forward' and 'reverse' transforms
        
        Returns:
            Augmented image as numpy array
        """
        # Albumentations expects (H, W, C) for grayscale, add channel dimension
        if len(image_np.shape) == 2:
            image_np = np.expand_dims(image_np, axis=-1)
        
        # Apply forward transform
        augmented = transform_dict['forward'](image=image_np)
        
        return augmented['image']
    
    def _reverse_tta_augmentation(self, mask_np, transform_dict):
        """
        Reverse TTA augmentation on mask.
        
        Args:
            mask_np: Mask as numpy array (H, W) or (H, W, 1)
            transform_dict: Dictionary with 'forward' and 'reverse' transforms
        
        Returns:
            Reversed mask as numpy array (H, W)
        """
        # Ensure mask has channel dimension for Albumentations
        if len(mask_np.shape) == 2:
            mask_np = np.expand_dims(mask_np, axis=-1)
        
        # Apply reverse transform
        reversed_aug = transform_dict['reverse'](image=mask_np)
        
        # Remove channel dimension
        result = reversed_aug['image']
        if len(result.shape) == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        
        return result
    
    def predict_with_tta(self, image, threshold=0.5, use_all_augmentations=True):
        """
        Predict with Test-Time Augmentation for improved robustness and confidence.
        
        TTA Process:
        1. Apply multiple augmentations to input image
        2. Run inference on each augmented version
        3. Reverse augmentations on predictions
        4. Average all predictions for final result
        5. Compute prediction variance as additional confidence metric
        
        Args:
            image (np.ndarray or PIL.Image): Input ultrasound image
            threshold (float): Threshold for binary segmentation (default: 0.5)
            use_all_augmentations (bool): If True, use all 8 augmentations (flips, rotations, scales).
                                         If False, use only 4 (original + 3 flips) for faster inference.
        
        Returns:
            dict: {
                'mask': Averaged binary segmentation mask (np.ndarray),
                'probability_map': Averaged probability map (np.ndarray),
                'inference_time': Total time in milliseconds (float),
                'tta_variance': Variance across predictions (float, lower = more consistent),
                'tta_confidence': TTA-based confidence score (float, 0-1, higher = better)
            }
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Store original size
        original_size = image.shape[:2]
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image.copy()
        
        # Select augmentations
        if use_all_augmentations:
            transforms_to_use = self.tta_transforms  # All 4
        else:
            transforms_to_use = self.tta_transforms[:1]  # Original
        
        # Storage for predictions
        all_prob_maps = []
        total_time = 0
        
        # Run inference for each augmentation
        for aug_dict in transforms_to_use:
            start_time = time.time()
            
            # Apply augmentation to image
            aug_image = self._apply_tta_augmentation(image_gray, aug_dict)
            
            # Remove channel dimension if added
            if len(aug_image.shape) == 3 and aug_image.shape[2] == 1:
                aug_image = aug_image[:, :, 0]
            
            # Preprocess augmented image
            image_tensor, _ = self.preprocess(aug_image)
            
            # Run inference
            output, _ = self.run_inference(image_tensor)
            
            # Get probability map (before thresholding)
            prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Reverse augmentation on probability map
            prob_map_reversed = self._reverse_tta_augmentation(prob_mask, aug_dict)
            
            # Resize to original size
            prob_map_reversed = cv2.resize(
                prob_map_reversed, 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_LINEAR
            )
            
            all_prob_maps.append(prob_map_reversed)
            total_time += (time.time() - start_time) * 1000
        
        # Average probability maps
        avg_prob_map = np.mean(all_prob_maps, axis=0)
        
        # Compute variance (measure of prediction consistency)
        prob_variance = np.var(all_prob_maps, axis=0).mean()
        
        # Compute TTA confidence (lower variance = higher confidence)
        # Normalize variance to [0, 1] range, then invert
        max_expected_variance = 0.05  # Typical max variance for good predictions
        normalized_variance = min(prob_variance, max_expected_variance) / max_expected_variance
        tta_confidence = 1.0 - normalized_variance
        
        # Apply threshold to get binary mask
        binary_mask = (avg_prob_map > threshold).astype(np.uint8) * 255
        
        return {
            'mask': binary_mask,
            'probability_map': avg_prob_map,
            'inference_time': total_time,
            'tta_variance': float(prob_variance),
            'tta_confidence': float(tta_confidence)
        }
    
    def process_image(self, image: np.ndarray, threshold: float = 0.5, use_tta: bool = True) -> Dict:
        """
        Complete inference pipeline with validation.
        
        This method integrates:
        1. Ultrasound image detection
        2. Model inference (with optional TTA)
        3. Mask quality analysis
        4. Segmentation confidence calculation
        5. Warning generation
        
        Args:
            image: Input image (RGB/grayscale numpy array)
            threshold: Segmentation threshold (default: 0.5)
            use_tta: Whether to use Test-Time Augmentation (default: True)
        
        Returns:
            Dictionary with results and validation info:
            {
                'mask': Binary segmentation mask,
                'probability_map': Probability map,
                'inference_time': Inference time in ms,
                'is_valid_ultrasound': Whether input is likely ultrasound,
                'confidence_score': Segmentation quality confidence (0-1),
                'quality_metrics': Dict with mask quality metrics,
                'warnings': List of warning messages,
                'tta_variance': (if use_tta=True) Prediction variance,
                'tta_confidence': (if use_tta=True) TTA-based confidence
            }
        """
        # Step 1: Check if image is ultrasound
        is_ultrasound, us_confidence = self.us_detector.is_ultrasound(image)
        
        # Step 2: Run inference pipeline (with or without TTA)
        if use_tta:
            result = self.predict_with_tta(image, threshold, use_all_augmentations=True)
            # result = self.predict_with_tta(image, threshold, use_all_augmentations=False)
        else:
            result = self.predict(image, threshold)
        
        mask = result['mask']
        prob_map = result['probability_map']
        
        # Step 3: Analyze mask quality
        quality_metrics = self.quality_checker.analyze_mask(mask)
        
        # Step 4: Compute segmentation confidence (mean probability in predicted region)
        segmentation_confidence = self.quality_checker.compute_segmentation_confidence(prob_map)
        
        # Step 5: Generate warnings
        warnings = []
        
        if not is_ultrasound:
            warnings.append(
                f"⚠️ This image may not be an ultrasound. "
                f"Detection confidence: {us_confidence*100:.1f}%"
            )
        
        quality_warnings = self.quality_checker.generate_warnings(
            quality_metrics, segmentation_confidence
        )
        warnings.extend(quality_warnings)
        
        # Build response
        response = {
            'mask': mask,
            'probability_map': prob_map,
            'inference_time': result['inference_time'],
            'is_valid_ultrasound': is_ultrasound,
            'confidence_score': float(segmentation_confidence),
            'quality_metrics': quality_metrics,
            'warnings': warnings
        }
        
        # Add TTA-specific metrics if used
        if use_tta:
            response['tta_variance'] = result['tta_variance']
            response['tta_confidence'] = result['tta_confidence']
        
        return response
    
    def predict(self, image, threshold=0.5):
        """
        Complete inference pipeline.
        
        Args:
            image (np.ndarray or PIL.Image): Input ultrasound image
            threshold (float): Threshold for binary segmentation
        
        Returns:
            dict: {
                'mask': Binary segmentation mask (np.ndarray),
                'probability_map': Probability map (np.ndarray),
                'inference_time': Time in milliseconds (float)
            }
        """
        # Preprocess
        image_tensor, original_size = self.preprocess(image)
        
        # Inference
        output, inference_time = self.run_inference(image_tensor)
        
        # Post-process
        mask, prob_map = self.postprocess(output, original_size, threshold)
        
        return {
            'mask': mask,
            'probability_map': prob_map,
            'inference_time': inference_time
        }
