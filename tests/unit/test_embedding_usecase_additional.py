import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
from src.usecase.embedding_usecase import (
    embedding_save_usecase,
    embedding_search_usecase
)
from src.models.database_models import CorrelationType


class TestEmbeddingSaveUseCase:
    """Test cases for embedding_save_usecase function"""
    
    @patch('src.usecase.embedding_usecase.save_original_text')
    @patch('src.usecase.embedding_usecase.save_embedding_to_postgresql')
    def test_embedding_save_usecase_single_chunk(self, mock_save_embedding, mock_save_original):
        """Test embedding save use case with single chunk"""
        mock_save_original.return_value = 123
        
        embedding_data = [
            {
                "id_text_origin": 1,
                "type": "similaridade_semantica",
                "text": "test content",
                "embedding": [0.1, 0.2, 0.3],
                "chunk_index": 0,
                "original_chunk": "original text",
                "chunk_metadata": {"total_chunks": 1}
            }
        ]
        
        embedding_json = json.dumps(embedding_data)
        result = embedding_save_usecase(embedding_json)
        
        assert result == [123]
        mock_save_original.assert_called_once_with("original text")
        mock_save_embedding.assert_called_once()
        
        # Verify the processed embedding data
        call_args = mock_save_embedding.call_args
        assert call_args[0][0] == 123  # id_text_origin
        processed_embeddings = call_args[0][1]
        assert len(processed_embeddings) == 1
        assert processed_embeddings[0]["correlation_type"] == CorrelationType.SIMILARIDADE_SEMANTICA
        assert "[Chunk 0 de 1]" in processed_embeddings[0]["text_content"]

    @patch('src.usecase.embedding_usecase.save_original_text')
    @patch('src.usecase.embedding_usecase.save_embedding_to_postgresql')
    def test_embedding_save_usecase_multiple_chunks(self, mock_save_embedding, mock_save_original):
        """Test embedding save use case with multiple chunks"""
        mock_save_original.side_effect = [123, 124]
        
        embedding_data = [
            {
                "type": "similaridade_semantica",
                "text": "test content 1",
                "embedding": [0.1, 0.2, 0.3],
                "chunk_index": 0,
                "original_chunk": "original text 1",
                "chunk_metadata": {"total_chunks": 2}
            },
            {
                "type": "relacionamento_semantico",
                "text": "test content 2",
                "embedding": [0.4, 0.5, 0.6],
                "chunk_index": 1,
                "original_chunk": "original text 2",
                "chunk_metadata": {"total_chunks": 2}
            }
        ]
        
        embedding_json = json.dumps(embedding_data)
        result = embedding_save_usecase(embedding_json)
        
        assert result == [123, 124]
        assert mock_save_original.call_count == 2
        assert mock_save_embedding.call_count == 2

    @patch('src.usecase.embedding_usecase.save_original_text')
    @patch('src.usecase.embedding_usecase.save_embedding_to_postgresql')
    def test_embedding_save_usecase_multiple_embeddings_same_chunk(self, mock_save_embedding, mock_save_original):
        """Test embedding save use case with multiple embeddings for same chunk"""
        mock_save_original.return_value = 123
        
        embedding_data = [
            {
                "type": "similaridade_semantica",
                "text": "test content 1",
                "embedding": [0.1, 0.2, 0.3],
                "chunk_index": 0,
                "original_chunk": "original text",
                "chunk_metadata": {"total_chunks": 1}
            },
            {
                "type": "relacionamento_semantico",
                "text": "test content 2",
                "embedding": [0.4, 0.5, 0.6],
                "chunk_index": 0,
                "original_chunk": "original text",
                "chunk_metadata": {"total_chunks": 1}
            }
        ]
        
        embedding_json = json.dumps(embedding_data)
        result = embedding_save_usecase(embedding_json)
        
        assert result == [123]
        mock_save_original.assert_called_once_with("original text")
        mock_save_embedding.assert_called_once()
        
        # Verify multiple embeddings were processed for same chunk
        call_args = mock_save_embedding.call_args
        processed_embeddings = call_args[0][1]
        assert len(processed_embeddings) == 2

    @patch('src.usecase.embedding_usecase.save_original_text')
    @patch('src.usecase.embedding_usecase.save_embedding_to_postgresql')
    def test_embedding_save_usecase_unknown_type(self, mock_save_embedding, mock_save_original):
        """Test embedding save use case with unknown correlation type"""
        mock_save_original.return_value = 123
        
        embedding_data = [
            {
                "type": "unknown_type",
                "text": "test content",
                "embedding": [0.1, 0.2, 0.3],
                "chunk_index": 0,
                "original_chunk": "original text"
            }
        ]
        
        embedding_json = json.dumps(embedding_data)
        result = embedding_save_usecase(embedding_json)
        
        assert result == [123]
        call_args = mock_save_embedding.call_args
        processed_embeddings = call_args[0][1]
        assert processed_embeddings[0]["correlation_type"] == "unknown_type"

    @patch('src.usecase.embedding_usecase.save_original_text')
    @patch('src.usecase.embedding_usecase.save_embedding_to_postgresql')
    def test_embedding_save_usecase_no_chunk_metadata(self, mock_save_embedding, mock_save_original):
        """Test embedding save use case without chunk metadata"""
        mock_save_original.return_value = 123
        
        embedding_data = [
            {
                "type": "similaridade_semantica",
                "text": "test content",
                "embedding": [0.1, 0.2, 0.3],
                "chunk_index": 0,
                "original_chunk": "original text"
            }
        ]
        
        embedding_json = json.dumps(embedding_data)
        result = embedding_save_usecase(embedding_json)
        
        assert result == [123]
        call_args = mock_save_embedding.call_args
        processed_embeddings = call_args[0][1]
        assert "[Chunk 0 de N/A]" in processed_embeddings[0]["text_content"]

    def test_embedding_save_usecase_invalid_json(self):
        """Test embedding save use case with invalid JSON"""
        with pytest.raises(json.JSONDecodeError):
            embedding_save_usecase("invalid json")


class TestEmbeddingSearchUseCase:
    """Test cases for embedding_search_usecase function"""
    
    @patch('src.usecase.embedding_usecase.search_vetorial')
    def test_embedding_search_usecase_success(self, mock_search_vetorial):
        """Test successful embedding search use case"""
        expected_results = [
            (0.123, 'text content', 'similaridade_semantica', 'original text'),
            (0.456, 'text content 2', 'relacionamento_semantico', 'original text 2')
        ]
        mock_search_vetorial.return_value = expected_results
        
        result = embedding_search_usecase("test question", 5)
        
        assert result == expected_results
        mock_search_vetorial.assert_called_once_with("test question", 5)

    @patch('src.usecase.embedding_usecase.search_vetorial')
    def test_embedding_search_usecase_with_exception(self, mock_search_vetorial):
        """Test embedding search use case with exception"""
        mock_search_vetorial.side_effect = Exception("Search failed")
        
        result = embedding_search_usecase("test question", 5)
        
        assert "Error in embedding search use case: Search failed" in result
        mock_search_vetorial.assert_called_once_with("test question", 5)

    @patch('src.usecase.embedding_usecase.search_vetorial')
    def test_embedding_search_usecase_default_top_k(self, mock_search_vetorial):
        """Test embedding search use case with default top_k"""
        mock_search_vetorial.return_value = []
        
        result = embedding_search_usecase("test question")
        
        mock_search_vetorial.assert_called_once_with("test question", 5)
        assert result == []

    @patch('src.usecase.embedding_usecase.search_vetorial')
    def test_embedding_search_usecase_custom_top_k(self, mock_search_vetorial):
        """Test embedding search use case with custom top_k"""
        mock_search_vetorial.return_value = []
        
        result = embedding_search_usecase("test question", 10)
        
        mock_search_vetorial.assert_called_once_with("test question", 10)
        assert result == []
