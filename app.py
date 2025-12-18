"""
Flask REST API for Fetal Head Segmentation

Provides endpoints for:
- Health check
- Image upload and segmentation
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

from model_loader import ModelLoader
from inference import InferenceEngine
from utils import pil_to_numpy, create_overlay, image_to_base64, numpy_to_pil
from PIL import Image
import numpy as np
import random
import time
import glob

# Load environment variables
load_dotenv()

# Global model loader (singleton pattern)
model_loader = None
inference_engine = None


def setup_logging(app):
    """Configure application logging."""
    if not app.debug:
        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_dir / 'app.log',
            maxBytes=10240000,  # 10 MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Fetal Head Segmentation API startup')


def create_app():
    """Application factory pattern for production deployment."""
    # Initialize Flask app
    app = Flask(__name__)
    
    # Configuration from environment variables
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.config['REQUEST_TIMEOUT'] = int(os.getenv('REQUEST_TIMEOUT', 30))
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # CORS Configuration
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    CORS(app, resources={r'/api/*': {'origins': cors_origins}})
    
    # Setup logging
    setup_logging(app)
    
    # Initialize model
    initialize_model()
    
    # Register routes
    register_routes(app)
    register_error_handlers(app)
    
    return app


# Configuration
MODEL_PATH = Path(os.getenv('MODEL_PATH', 'best_model_mobinet_aspp_residual_se_v2.pth'))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = Path(__file__).parent / MODEL_PATH

DEMO_FRAMES_DIR = Path(__file__).parent.parent / 'frontend' / 'public' / 'demo_videos'


def initialize_model():
    """Initialize model loader and inference engine."""
    global model_loader, inference_engine
    
    if model_loader is None:
        print("Initializing model...")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        model_loader = ModelLoader(MODEL_PATH)
        inference_engine = InferenceEngine(model_loader.model, model_loader.device)
        print("✓ Model ready for inference")


def register_routes(app):
    """Register all API routes."""
    app.add_url_rule('/api/health', 'health_check', health_check, methods=['GET'])
    app.add_url_rule('/api/upload', 'upload_image', upload_image, methods=['POST'])
    app.add_url_rule('/api/benchmark', 'benchmark_inference', benchmark_inference, methods=['GET'])


def register_error_handlers(app):
    """Register error handlers."""
    app.register_error_handler(404, not_found)
    app.register_error_handler(413, request_entity_too_large)
    app.register_error_handler(500, internal_error)
    app.register_error_handler(408, request_timeout)


def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status and model loading state
    """
    print("Health check endpoint called")  # Debug
    try:
        print("Building response...")  # Debug
        model_status = model_loader is not None
        device_info = str(model_loader.device) if model_loader else None
        
        response_data = {
            'status': 'healthy',
            'model_loaded': model_status,
            'device': device_info
        }
        print(f"Response data: {response_data}")  # Debug
        
        result = jsonify(response_data)
        print("Jsonify successful, returning...")  # Debug
        return result
    except Exception as e:
        print(f"ERROR in health_check: {e}")  # Debug
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/upload', methods=['POST'])
    """
    Upload and segment ultrasound image.
    
    Expects:
        - Form data with 'image' file field
        - Optional 'use_tta' field (boolean, default: true)
    
    Returns:
        JSON with:
        - success: bool
        - original: Base64 encoded original image
        - segmentation: Base64 encoded overlay visualization
        - inference_time: Processing time in milliseconds
        - tta_variance: (if TTA enabled) Prediction variance
        - tta_confidence: (if TTA enabled) TTA-based confidence
        - error: Error message (if failed)
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Get TTA flag (default: True)
        use_tta = request.form.get('use_tta', 'true').lower() == 'true'
        
        # Read and validate image
        try:
            image = Image.open(file.stream)
            # Verify image is not corrupted by loading it
            image.verify()
            # Re-open after verify (verify() closes the file)
            file.stream.seek(0)
            image = Image.open(file.stream)
        except (IOError, OSError) as e:
            return jsonify({
                'success': False,
                'error': 'Corrupted or invalid image file. Please upload a valid image.'
            }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to read image: {str(e)}'
            }), 400
        
        # Convert to RGB if needed (some images might be RGBA or other formats)
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to convert image format: {str(e)}'
            }), 400
        
        # Convert to numpy for processing
        try:
            image_np = pil_to_numpy(image)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to process image data: {str(e)}'
            }), 400
        
        # Run inference with validation (TTA enabled by default)
        try:
            result = inference_engine.process_image(image_np, use_tta=use_tta)
        except RuntimeError as e:
            return jsonify({
                'success': False,
                'error': 'Model inference failed. This may be due to GPU memory issues or invalid image dimensions.'
            }), 500
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Inference error: {str(e)}'
            }), 500
        
        # Extract results
        mask = result['mask']  # Binary mask (H, W)
        inference_time = result['inference_time']  # Time in ms
        
        # Create visualization overlay
        try:
            visualization = create_overlay(image_np, mask)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to create visualization: {str(e)}'
            }), 500
        
        # Convert to base64 for JSON response
        original_b64 = image_to_base64(image_np)
        segmentation_b64 = image_to_base64(visualization)
        
        response_data = {
            'success': True,
            'original': original_b64,
            'segmentation': segmentation_b64,
            'inference_time': round(inference_time, 2),
            
            # Add validation data
            'is_valid_ultrasound': result['is_valid_ultrasound'],
            'confidence_score': float(result['confidence_score']),
            'quality_metrics': result['quality_metrics'],
            'warnings': result['warnings']
        }
        
        # Add TTA-specific metrics if used
        if use_tta and 'tta_variance' in result:
            response_data['tta_variance'] = result['tta_variance']
            response_data['tta_confidence'] = result['tta_confidence']
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'warnings': ['An error occurred during processing']
        }), 500


@app.route('/api/benchmark', methods=['GET'])
    """
    Benchmark endpoint to measure average inference time.
    
    This endpoint processes ~100 random images from the dataset_v5 training set
    and returns the average inference time per image. Used for testing performance.
    
    Query Parameters:
        - num_images: Number of images to benchmark (default: 100, max: 500)
        - use_tta: Whether to use Test-Time Augmentation (default: false)
    
    Returns:
        JSON with:
        - success: bool
        - avg_inference_time: Average time per image in milliseconds
        - total_images: Number of images processed
        - total_time: Total processing time in seconds
        - min_time: Minimum inference time
        - max_time: Maximum inference time
        - std_dev: Standard deviation of inference times
        - use_tta: Whether TTA was enabled
    """
    try:
        # Get query parameters
        num_images = min(int(request.args.get('num_images', 100)), 500)
        use_tta = request.args.get('use_tta', 'false').lower() == 'true'
        
        # Get path to dataset_v5 training images
        project_root = Path(__file__).parent.parent.parent
        dataset_path = project_root / 'shared' / 'dataset_v5' / 'training_set' / 'images'
        
        if not dataset_path.exists():
            return jsonify({
                'success': False,
                'error': f'Dataset path not found: {dataset_path}'
            }), 404
        
        # Get all image files
        image_files = list(dataset_path.glob('*.png')) + list(dataset_path.glob('*.jpg'))
        
        if len(image_files) == 0:
            return jsonify({
                'success': False,
                'error': 'No images found in dataset'
            }), 404
        
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
                result = inference_engine.process_image(image_np, use_tta=use_tta)
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
            return jsonify({
                'success': False,
                'error': 'All images failed to process'
            }), 500
        
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
        
        return jsonify({
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
        })
    
    except Exception as e:
        print(f"Benchmark error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(413)
    """Handle file too large errors."""
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16 MB.'
    }), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


@app.errorhandler(408)
def request_timeout(error):
    return jsonify({
        'success': False,
        'error': 'Request timeout. The operation took too long to complete.'
    }), 408


if __name__ == '__main__':
    # Initialize model before starting server
# Create app instance for production servers (Gunicorn/Waitress)
app = create_app()


if __name__ == '__main__':
    # Development server only
    print("\n" + "="*60)
    print("Starting Fetal Head Segmentation API Server (DEVELOPMENT)")
    print("="*60)
    print(f"Server: http://localhost:5000")
    print(f"Health: http://localhost:5000/api/health")
    print(f"Upload: POST http://localhost:5000/api/upload")
    print(f"Benchmark: GET http://localhost:5000/api/benchmark")
    print("="*60)
    print("⚠ WARNING: This is the development server. Use Gunicorn/Waitress for production.")
    print("="*60 + "\n")
    
    try:
        app.run(debug=Trushed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Server shutting down...")
