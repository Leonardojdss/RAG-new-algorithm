"""
Models package - Modelos SQLAlchemy para o sistema RAG
"""

from .database_models import (
    Base,
    DbOriginText,
    DbCorrelationEmbedding,
    CorrelationType
)

__all__ = [
    'Base',
    'DbOriginText',
    'DbCorrelationEmbedding',
    'CorrelationType',
    'OriginTextRepository',
    'CorrelationEmbeddingRepository',
    'RAGRepository'
]
