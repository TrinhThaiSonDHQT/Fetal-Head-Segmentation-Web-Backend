"""
Middleware Package

Error handlers and middleware components.
"""
from .error_handlers import register_error_handlers

__all__ = ['register_error_handlers']
