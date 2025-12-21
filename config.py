"""
Application Configuration

Manages configuration from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Base configuration class."""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # File Upload
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16 MB
    SEND_FILE_MAX_AGE_DEFAULT = 0
    
    # Request Handling
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 30))
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
    
    # Model
    # Use original model on Railway (8GB RAM), quantized on Render (512MB RAM)
    MODEL_PATH = Path(os.getenv('MODEL_PATH', 'best_model_mobinet_aspp_residual_se_v2.pth'))
    
    # Logging
    LOG_DIR = Path('logs')
    LOG_MAX_BYTES = 10240000  # 10 MB
    LOG_BACKUP_COUNT = 10
    
    @classmethod
    def get_model_path(cls):
        """Get absolute path to model file."""
        if cls.MODEL_PATH.is_absolute():
            return cls.MODEL_PATH
        return Path(__file__).parent / cls.MODEL_PATH


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    

# Configuration mapping
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(config_name=None):
    """
    Get configuration object based on environment.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, uses FLASK_ENV environment variable
    
    Returns:
        Configuration class
    """
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    return config_by_name.get(config_name, DevelopmentConfig)
