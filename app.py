"""
Flask REST API for Fetal Head Segmentation

Application factory and initialization.
"""
from flask import Flask
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler

from config import get_config
from services import SegmentationService
from middleware import register_error_handlers
from routes import health_bp, segmentation_bp, benchmark_bp


def setup_logging(app):
    """Configure application logging."""
    if not app.debug:
        # Create logs directory if it doesn't exist
        app.config['LOG_DIR'].mkdir(exist_ok=True)
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            app.config['LOG_DIR'] / 'app.log',
            maxBytes=app.config['LOG_MAX_BYTES'],
            backupCount=app.config['LOG_BACKUP_COUNT']
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Fetal Head Segmentation API startup')


def create_app(config_name=None):
    """
    Application factory pattern for production deployment.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
    
    Returns:
        Flask application instance
    """
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    
    # CORS Configuration
    CORS(app, resources={r'/api/*': {'origins': app.config['CORS_ORIGINS']}})
    
    # Setup logging
    setup_logging(app)
    
    # Initialize segmentation service
    model_path = config_class.get_model_path()
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
    
    return app


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
        app.run(debug=True, host='127.0.0.1', port=5000)
    except Exception as e:
        print(f"Server crashed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Server shutting down...")
