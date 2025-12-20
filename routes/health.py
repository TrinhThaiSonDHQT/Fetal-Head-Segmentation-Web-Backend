"""
Health Check Routes

Endpoints for checking API health and model status.
"""
from flask import Blueprint, jsonify, current_app

health_bp = Blueprint('health', __name__)


@health_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        JSON response with status and model loading state
    """
    print("Health check endpoint called")
    try:
        segmentation_service = current_app.config['SEGMENTATION_SERVICE']
        
        model_status = segmentation_service.is_model_loaded()
        device_info = segmentation_service.get_device_info()
        
        response_data = {
            'status': 'healthy',
            'model_loaded': model_status,
            'device': device_info
        }
        
        return jsonify(response_data)
    except Exception as e:
        print(f"ERROR in health_check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
