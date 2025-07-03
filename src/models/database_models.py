"""
Modelos SQLAlchemy para as tabelas do sistema RAG
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, CheckConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Float

Base = declarative_base()

# Definindo o tipo Vector para compatibilidade
def Vector(dimensions):
    """
    Função para criar um tipo Vector compatível
    Se pgvector estiver disponível, usa Vector, senão usa ARRAY
    """
    try:
        from pgvector.sqlalchemy import Vector as PgVector
        return PgVector(dimensions)
    except ImportError:
        # Fallback para ARRAY de Float se pgvector não estiver disponível
        return ARRAY(Float)


class DbOriginText(Base):
    """
    Modelo para a tabela db_origin_text
    Armazena os textos originais do sistema
    """
    __tablename__ = 'db_origin_text'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    data = Column(Text, nullable=False)
    
    # Relacionamento com embeddings
    embeddings = relationship("DbCorrelationEmbedding", back_populates="origin_text", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DbOriginText(id={self.id}, data='{self.data[:50]}...')>"


class DbCorrelationEmbedding(Base):
    """
    Modelo para a tabela db_correlation_embedding
    Armazena os embeddings com diferentes tipos de correlação
    """
    __tablename__ = 'db_correlation_embedding'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    id_text_origin = Column(Integer, ForeignKey('db_origin_text.id'), nullable=False)
    correlation_type = Column(
        String(50), 
        nullable=False,
        # CheckConstraint para validar os valores permitidos
    )
    text_content = Column(Text, nullable=False)
    vector = Column(Vector(3072), nullable=True)
    
    # Constraint para validar correlation_type
    __table_args__ = (
        CheckConstraint(
            "correlation_type IN ('Similaridade semântica', 'Relacionamento Semântico', 'Contexto Compartilhado')",
            name='check_correlation_type'
        ),
    )
    
    # Relacionamento com texto original
    origin_text = relationship("DbOriginText", back_populates="embeddings")
    
    def __repr__(self):
        return f"<DbCorrelationEmbedding(id={self.id}, id_text_origin={self.id_text_origin}, correlation_type='{self.correlation_type}')>"


class CorrelationType:
    SIMILARIDADE_SEMANTICA = "Similaridade semântica"
    RELACIONAMENTO_SEMANTICO = "Relacionamento Semântico"
    CONTEXTO_COMPARTILHADO = "Contexto Compartilhado"
    
    @classmethod
    def get_all_types(cls):
        return [
            cls.SIMILARIDADE_SEMANTICA,
            cls.RELACIONAMENTO_SEMANTICO,
            cls.CONTEXTO_COMPARTILHADO
        ]
