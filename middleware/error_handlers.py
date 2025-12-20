"""
Error Handlers

Centralized error handling for the application.
"""
from flask import jsonify


def register_error_handlers(app):
    """Register all error handlers with the Flask app."""
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle file too large errors."""
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 16 MB.'
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle internal server errors."""
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(408)
    def request_timeout(error):
        """Handle request timeout errors."""
        return jsonify({
            'success': False,
            'error': 'Request timeout. The operation took too long to complete.'
        }), 408
