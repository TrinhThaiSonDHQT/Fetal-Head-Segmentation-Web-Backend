"""
Flask REST API for Fetal Head Segmentation - Local Development
"""
from flask import Flask
from flask_cors import CORS
from pathlib import Path

from services import SegmentationService
from middleware import register_error_handlers
from routes import health_bp, segmentation_bp, benchmark_bp

# Initialize Flask app
app = Flask(__name__)

# Simple configuration for local development
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['SECRET_KEY'] = 'dev-secret-key'

# CORS - Allow requests from frontend dev server
CORS(app, resources={r'/api/*': {'origins': 'http://localhost:3000'}})

# Initialize segmentation service
model_path = Path('best_model_mobinet_aspp_residual_se_v2.pth')
segmentation_service = SegmentationService(model_path)
segmentation_service.initialize_model()

# Store service in app config for access in routes
app.config['SEGMENTATION_SERVICE'] = segmentation_service

# Register blueprints
app.register_blueprint(health_bp, url_prefix='/api')
app.register_blueprint(segmentation_bp, url_prefix='/api')
app.register_blueprint(benchmark_bp, url_prefix='/api')

# Register error handlers
register_error_handlers(app)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Fetal Head Segmentation API - Local Development Server")
    print("="*60)
    print(f"Server: http://localhost:5000")
    print(f"Health: http://localhost:5000/api/health")
    print(f"Upload: POST http://localhost:5000/api/upload")
    print(f"Benchmark: GET http://localhost:5000/api/benchmark")
    print("="*60 + "\n")
    
    try:
        app.run(debug=True, host='127.0.0.1', port=5000)
    except KeyboardInterrupt:
        print("\n✓ Server stopped by user")
    except Exception as e:
        print(f"\n✗ Server crashed: {e}")
        import traceback
        traceback.print_exc()
