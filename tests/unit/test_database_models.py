import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models.database_models import (
    Base,
    DbOriginText,
    DbCorrelationEmbedding,
    CorrelationType,
    Vector
)


class TestDatabaseModels:
    """Test cases for database models"""
    
    @pytest.fixture
    def engine(self):
        """Create an in-memory SQLite database for testing"""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def session(self, engine):
        """Create a database session for testing"""
        Session = sessionmaker(bind=engine)
        session = Session()
        yield session
        session.close()

    def test_db_origin_text_creation(self, session):
        """Test creating a DbOriginText instance"""
        origin_text = DbOriginText(data="Test original text content")
        session.add(origin_text)
        session.commit()
        
        assert origin_text.id is not None
        assert origin_text.data == "Test original text content"

    def test_db_origin_text_repr(self):
        """Test DbOriginText string representation"""
        origin_text = DbOriginText(id=1, data="This is a very long text that should be truncated in the repr method")
        repr_str = repr(origin_text)
        
        assert "DbOriginText(id=1" in repr_str
        assert "This is a very long text that should be truncated" in repr_str
        assert "...'" in repr_str

    def test_correlation_type_enum_values(self):
        """Test CorrelationType enum values"""
        assert CorrelationType.SIMILARIDADE_SEMANTICA == "Similaridade semântica"
        assert CorrelationType.RELACIONAMENTO_SEMANTICO == "Relacionamento Semântico"
        assert CorrelationType.CONTEXTO_COMPARTILHADO == "Contexto Compartilhado"

    def test_vector_function_with_pgvector(self):
        """Test Vector function when pgvector is available"""
        # This test assumes pgvector might not be available
        try:
            vector_type = Vector(128)
            # If pgvector is available, this should work
            assert vector_type is not None
        except ImportError:
            # If pgvector is not available, Vector should fallback to ARRAY
            vector_type = Vector(128)
            assert vector_type is not None

    def test_db_origin_text_empty_data(self, session):
        """Test DbOriginText with empty data"""
        origin_text = DbOriginText(data="")
        session.add(origin_text)
        session.commit()
        
        assert origin_text.id is not None
        assert origin_text.data == ""

    def test_correlation_type_get_all_types(self):
        """Test CorrelationType get_all_types method"""
        all_types = CorrelationType.get_all_types()
        
        assert len(all_types) == 3
        assert CorrelationType.SIMILARIDADE_SEMANTICA in all_types
        assert CorrelationType.RELACIONAMENTO_SEMANTICO in all_types
        assert CorrelationType.CONTEXTO_COMPARTILHADO in all_types

    def test_database_model_simple_creation(self):
        """Test simple database model creation without session complexity"""
        # Test basic model instantiation without database connection
        origin_text = DbOriginText()
        origin_text.data = "Test data"
        
        assert origin_text.data == "Test data"
        
        # Test basic embedding model
        embedding = DbCorrelationEmbedding()
        embedding.text_content = "Test content"
        embedding.correlation_type = CorrelationType.SIMILARIDADE_SEMANTICA
        
        assert embedding.text_content == "Test content"
        assert embedding.correlation_type == CorrelationType.SIMILARIDADE_SEMANTICA

    def test_database_model_attributes(self):
        """Test that database models have expected attributes"""
        # Test DbOriginText attributes
        origin_text = DbOriginText()
        assert hasattr(origin_text, 'id')
        assert hasattr(origin_text, 'data')
        assert hasattr(origin_text, 'embeddings')
        
        # Test DbCorrelationEmbedding attributes  
        embedding = DbCorrelationEmbedding()
        assert hasattr(embedding, 'id')
        assert hasattr(embedding, 'id_text_origin')
        assert hasattr(embedding, 'correlation_type')
        assert hasattr(embedding, 'text_content')
        assert hasattr(embedding, 'vector')
        assert hasattr(embedding, 'origin_text')
