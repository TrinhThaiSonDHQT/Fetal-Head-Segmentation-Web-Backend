"""
Routes Package

Blueprint modules for organizing API endpoints.
"""
from .health import health_bp
from .segmentation import segmentation_bp
from .benchmark import benchmark_bp

__all__ = ['health_bp', 'segmentation_bp', 'benchmark_bp']
