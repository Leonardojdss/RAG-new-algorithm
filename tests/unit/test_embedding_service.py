import pytest
from unittest.mock import patch, MagicMock
from src.service.embedding_service import (
    generate_text_semantic_service,
    embedding_service,
    save_original_text,
    save_embedding_to_postgresql,
    search_vetorial
)
from src.models.database_models import DbOriginText, DbCorrelationEmbedding
import json

# Mock the OpenAI client
@patch('src.service.embedding_service.client')
def test_generate_text_semantic_service_success(mock_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = '{"key": "value"}'
    mock_client.chat.completions.create.return_value = mock_completion

    response = generate_text_semantic_service("test input", "test prompt")
    assert response == {"key": "value"}

@patch('src.service.embedding_service.client')
def test_generate_text_semantic_service_json_error(mock_client):
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = 'not a json'
    mock_client.chat.completions.create.return_value = mock_completion

    response = generate_text_semantic_service("test input", "test prompt")
    assert response is None

@patch('src.service.embedding_service.client')
def test_embedding_service_success(mock_client):
    mock_embedding = MagicMock()
    mock_embedding.data[0].embedding = [0.1, 0.2, 0.3]
    mock_client.embeddings.create.return_value = mock_embedding

    embedding = embedding_service("test input")
    assert embedding == [0.1, 0.2, 0.3]

def test_embedding_service_empty_input():
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        embedding_service("")

@patch('src.service.embedding_service.get_db_session')
def test_save_original_text(mock_get_db_session):
    mock_session = MagicMock()
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    
    text_to_save = "original text"
    saved_id = save_original_text(text_to_save)

    mock_session.add.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.refresh.assert_called_once()

@patch('src.service.embedding_service.get_db_session')
def test_save_embedding_to_postgresql(mock_get_db_session):
    mock_session = MagicMock()
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    
    embedding_data = [
        {"correlation_type": "type1", "text_content": "content1", "embedding": [0.1]},
        {"correlation_type": "type2", "text_content": "content2", "embedding": [0.2]}
    ]
    save_embedding_to_postgresql(1, embedding_data)

    assert mock_session.add.call_count == 2
    mock_session.commit.assert_called_once()

@patch('src.service.embedding_service.embedding_service')
@patch('src.service.embedding_service.get_db_session')
def test_search_vetorial_success(mock_get_db_session, mock_embedding_service):
    mock_embedding_service.return_value = [0.1, 0.2, 0.3]
    mock_session = MagicMock()
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    mock_session.execute.return_value.fetchall.return_value = [(0.123, 'text', 'type', 'origin')]

    results = search_vetorial("question", 5)
    assert results == [(0.123, 'text', 'type', 'origin')]

def test_search_vetorial_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        search_vetorial("", 5)

def test_search_vetorial_invalid_top_k():
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_vetorial("question", 0)
    with pytest.raises(ValueError, match="top_k cannot exceed 1000"):
        search_vetorial("question", 1001)

# Additional test cases for better coverage
@patch('src.service.embedding_service.client')
def test_generate_text_semantic_service_exception(mock_client):
    """Test generate_text_semantic_service with exception - API call exception should propagate"""
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    
    # The function doesn't catch API call exceptions, so it should raise
    with pytest.raises(Exception, match="API Error"):
        generate_text_semantic_service("test input", "test prompt")

@patch('src.service.embedding_service.client')
def test_embedding_service_exception(mock_client):
    """Test embedding_service with exception"""
    mock_client.embeddings.create.side_effect = Exception("API Error")
    
    with pytest.raises(Exception, match="API Error"):
        embedding_service("test input")

@patch('src.service.embedding_service.get_db_session')
def test_save_original_text_exception(mock_get_db_session):
    """Test save_original_text with database exception"""
    mock_session = MagicMock()
    mock_session.commit.side_effect = Exception("Database error")
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    
    with pytest.raises(Exception, match="Database error"):
        save_original_text("test text")

@patch('src.service.embedding_service.get_db_session')
def test_save_embedding_to_postgresql_exception(mock_get_db_session):
    """Test save_embedding_to_postgresql with database exception"""
    mock_session = MagicMock()
    mock_session.commit.side_effect = Exception("Database error")
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    
    embedding_data = [
        {"correlation_type": "type1", "text_content": "content1", "embedding": [0.1]}
    ]
    
    with pytest.raises(Exception, match="Database error"):
        save_embedding_to_postgresql(1, embedding_data)

@patch('src.service.embedding_service.embedding_service')
@patch('src.service.embedding_service.get_db_session')
def test_search_vetorial_database_exception(mock_get_db_session, mock_embedding_service):
    """Test search_vetorial with database exception - should print error and return None"""
    mock_embedding_service.return_value = [0.1, 0.2, 0.3]
    mock_session = MagicMock()
    mock_session.execute.side_effect = Exception("Database error")
    mock_get_db_session.return_value.__enter__.return_value = mock_session

    # The function catches exceptions and prints error, then returns None
    result = search_vetorial("question", 5)
    assert result is None

@patch('src.service.embedding_service.embedding_service')
def test_search_vetorial_embedding_exception(mock_embedding_service):
    """Test search_vetorial with embedding service exception - should print error and return None"""
    mock_embedding_service.side_effect = Exception("Embedding error")

    # The function catches exceptions and prints error, then returns None
    result = search_vetorial("question", 5)
    assert result is None

def test_search_vetorial_large_top_k():
    """Test search_vetorial with top_k at maximum limit"""
    with pytest.raises(ValueError, match="top_k cannot exceed 1000"):
        search_vetorial("question", 1001)

def test_search_vetorial_negative_top_k():
    """Test search_vetorial with negative top_k"""
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        search_vetorial("question", -1)

@patch('src.service.embedding_service.get_db_session')
def test_save_original_text_with_session_refresh(mock_get_db_session):
    """Test save_original_text ensuring session refresh is called"""
    mock_session = MagicMock()
    mock_origin_text = MagicMock()
    mock_origin_text.id = 123
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    
    # Mock the DbOriginText creation
    with patch('src.service.embedding_service.DbOriginText', return_value=mock_origin_text):
        saved_id = save_original_text("test text")
    
    assert saved_id == 123
    mock_session.refresh.assert_called_once_with(mock_origin_text)

@patch('src.service.embedding_service.get_db_session')
def test_save_embedding_to_postgresql_multiple_embeddings(mock_get_db_session):
    """Test save_embedding_to_postgresql with multiple embeddings"""
    mock_session = MagicMock()
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    
    embedding_data = [
        {"correlation_type": "type1", "text_content": "content1", "embedding": [0.1]},
        {"correlation_type": "type2", "text_content": "content2", "embedding": [0.2]},
        {"correlation_type": "type3", "text_content": "content3", "embedding": [0.3]}
    ]
    
    save_embedding_to_postgresql(1, embedding_data)
    
    assert mock_session.add.call_count == 3
    mock_session.commit.assert_called_once()

@patch('src.service.embedding_service.embedding_service')
@patch('src.service.embedding_service.get_db_session')
def test_search_vetorial_no_results(mock_get_db_session, mock_embedding_service):
    """Test search_vetorial when no results are found"""
    mock_embedding_service.return_value = [0.1, 0.2, 0.3]
    mock_session = MagicMock()
    mock_get_db_session.return_value.__enter__.return_value = mock_session
    mock_session.execute.return_value.fetchall.return_value = []

    results = search_vetorial("question", 5)
    assert results == []
