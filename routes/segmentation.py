"""
Segmentation Routes

Endpoints for image upload and segmentation.
"""
from flask import Blueprint, request, jsonify, current_app

segmentation_bp = Blueprint('segmentation', __name__)


@segmentation_bp.route('/upload', methods=['POST'])
def upload_image():
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
        
        # Get segmentation service
        segmentation_service = current_app.config['SEGMENTATION_SERVICE']
        
        # Process image
        result = segmentation_service.process_uploaded_image(file, use_tta=use_tta)
        
        return jsonify(result)
    
    except ValueError as e:
        # Validation errors (400)
        return jsonify({
            'success': False,
            'error': str(e),
            'warnings': []
        }), 400
    
    except RuntimeError as e:
        # Processing errors (500)
        return jsonify({
            'success': False,
            'error': str(e),
            'warnings': []
        }), 500
    
    except Exception as e:
        # Unexpected errors
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred during processing',
            'warnings': []
        }), 500
