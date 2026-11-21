"""Headless ASGI entry: python -m uvicorn core.api:create_app --factory."""
from core.WebGui import create_app

__all__ = ['create_app']
