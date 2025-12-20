"""
Segmentation Service

Business logic for image segmentation operations.
"""
from pathlib import Path
import numpy as np
import random
import time
from PIL import Image

from core import ModelLoader, InferenceEngine
from utils import pil_to_numpy, create_overlay, image_to_base64


class SegmentationService:
    """Service for handling image segmentation operations."""
    
    def __init__(self, model_path: Path):
        """
        Initialize segmentation service.
        
        Args:
            model_path: Path to model weights file
        """
        self.model_path = model_path
        self.model_loader = None
        self.inference_engine = None
        
    def initialize_model(self):
        """Initialize model loader and inference engine."""
        if self.model_loader is None:
            print("Initializing model...")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model_loader = ModelLoader(self.model_path)
            self.inference_engine = InferenceEngine(
                self.model_loader.model, 
                self.model_loader.device
            )
            print("✓ Model ready for inference")
    
    def is_model_loaded(self):
        """Check if model is loaded."""
        return self.model_loader is not None
    
    def get_device_info(self):
        """Get device information."""
        if self.model_loader:
            return str(self.model_loader.device)
        return None
    
    def process_uploaded_image(self, image_file, use_tta=True):
        """
        Process an uploaded image file and generate segmentation.
        
        Args:
            image_file: File object from request.files
            use_tta: Whether to use Test-Time Augmentation
            
        Returns:
            dict: Processing results including base64 images, metrics, and validation
            
        Raises:
            ValueError: If image is invalid or corrupted
            RuntimeError: If inference fails
        """
        # Read and validate image
        try:
            image = Image.open(image_file.stream)
            image.verify()
            # Re-open after verify (verify() closes the file)
            image_file.stream.seek(0)
            image = Image.open(image_file.stream)
        except (IOError, OSError):
            raise ValueError('Corrupted or invalid image file. Please upload a valid image.')
        except Exception as e:
            raise ValueError(f'Failed to read image: {str(e)}')
        
        # Convert to RGB if needed
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise ValueError(f'Failed to convert image format: {str(e)}')
        
        # Convert to numpy
        try:
            image_np = pil_to_numpy(image)
        except Exception as e:
            raise ValueError(f'Failed to process image data: {str(e)}')
        
        # Run inference
        try:
            result = self.inference_engine.process_image(image_np, use_tta=use_tta)
        except RuntimeError:
            raise RuntimeError(
                'Model inference failed. This may be due to GPU memory issues or invalid image dimensions.'
            )
        except Exception as e:
            raise RuntimeError(f'Inference error: {str(e)}')
        
        # Extract results
        mask = result['mask']
        inference_time = result['inference_time']
        
        # Create visualization
        try:
            visualization = create_overlay(image_np, mask)
        except Exception as e:
            raise RuntimeError(f'Failed to create visualization: {str(e)}')
        
        # Convert to base64 for JSON response
        original_b64 = image_to_base64(image_np)
        segmentation_b64 = image_to_base64(visualization)
        
        response_data = {
            'success': True,
            'original': original_b64,
            'segmentation': segmentation_b64,
            'inference_time': round(inference_time, 2),
            'is_valid_ultrasound': result['is_valid_ultrasound'],
            'confidence_score': float(result['confidence_score']),
            'quality_metrics': result['quality_metrics'],
            'warnings': result['warnings']
        }
        
        # Add TTA-specific metrics if used
        if use_tta and 'tta_variance' in result:
            response_data['tta_variance'] = result['tta_variance']
            response_data['tta_confidence'] = result['tta_confidence']
        
        return response_data
    
    def run_benchmark(self, num_images=100, use_tta=False):
        """
        Run benchmark to measure inference performance.
        
        Args:
            num_images: Number of images to benchmark
            use_tta: Whether to use Test-Time Augmentation
            
        Returns:
            dict: Benchmark statistics
            
        Raises:
            FileNotFoundError: If dataset path not found
            RuntimeError: If all images fail to process
        """
        # Get path to dataset
        project_root = Path(__file__).parent.parent.parent
        dataset_path = project_root / 'shared' / 'dataset_v5' / 'training_set' / 'images'
        
        if not dataset_path.exists():
            raise FileNotFoundError(f'Dataset path not found: {dataset_path}')
        
        # Get all image files
        image_files = list(dataset_path.glob('*.png')) + list(dataset_path.glob('*.jpg'))
        
        if len(image_files) == 0:
            raise FileNotFoundError('No images found in dataset')
        
        # Randomly sample images
        num_images = min(num_images, len(image_files))
        sampled_images = random.sample(image_files, num_images)
        
        # Track inference times
        inference_times = []
        failed_images = 0
        
        print(f"\n{'='*60}")
        print(f"Starting benchmark: {num_images} images, TTA={use_tta}")
        print(f"{'='*60}")
        
        benchmark_start = time.time()
        
        # Process each image
        for idx, img_path in enumerate(sampled_images, 1):
            try:
                # Load image
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Convert to numpy
                image_np = pil_to_numpy(image)
                
                # Run inference
                result = self.inference_engine.process_image(image_np, use_tta=use_tta)
                inference_times.append(result['inference_time'])
                
                # Progress indicator
                if idx % 10 == 0:
                    print(f"Processed {idx}/{num_images} images...")
                
            except Exception as e:
                print(f"Failed to process {img_path.name}: {str(e)}")
                failed_images += 1
                continue
        
        benchmark_end = time.time()
        total_time = benchmark_end - benchmark_start
        
        # Calculate statistics
        if len(inference_times) == 0:
            raise RuntimeError('All images failed to process')
        
        avg_time = np.mean(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        std_dev = np.std(inference_times)
        
        print(f"\n{'='*60}")
        print(f"Benchmark Complete!")
        print(f"{'='*60}")
        print(f"Total images: {len(inference_times)}/{num_images}")
        print(f"Average inference time: {avg_time:.2f} ms")
        print(f"Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"Std Dev: {std_dev:.2f} ms")
        print(f"Total benchmark time: {total_time:.2f} s")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'avg_inference_time': round(avg_time, 2),
            'total_images': len(inference_times),
            'failed_images': failed_images,
            'total_time': round(total_time, 2),
            'min_time': round(min_time, 2),
            'max_time': round(max_time, 2),
            'std_dev': round(std_dev, 2),
            'use_tta': use_tta,
            'throughput_fps': round(len(inference_times) / total_time, 2)
        }
