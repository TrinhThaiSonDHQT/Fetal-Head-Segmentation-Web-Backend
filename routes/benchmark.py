"""
Benchmark Routes

Endpoints for performance testing and benchmarking.
"""
from flask import Blueprint, request, jsonify, current_app

benchmark_bp = Blueprint('benchmark', __name__)


@benchmark_bp.route('/benchmark', methods=['GET'])
def benchmark_inference():
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
        
        # Get segmentation service
        segmentation_service = current_app.config['SEGMENTATION_SERVICE']
        
        # Run benchmark
        result = segmentation_service.run_benchmark(num_images, use_tta)
        
        return jsonify(result)
    
    except FileNotFoundError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 404
    
    except RuntimeError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
    except Exception as e:
        print(f"Benchmark error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
