import pytest
from fastapi.testclient import TestClient
from src.main import app  # Assuming your FastAPI app is in src/main.py
from unittest.mock import patch

client = TestClient(app)

@patch('src.controller.api.router.embedding_usecase')
@patch('src.controller.api.router.embedding_save_usecase')
def test_create_embedding_success(mock_embedding_save_usecase, mock_embedding_usecase):
    # Mock the use cases
    mock_embedding_usecase.return_value = "mocked_embedding"
    mock_embedding_save_usecase.return_value = ["id1", "id2"]

    response = client.post("/new_rag/embedding", json={"text": "some text", "index": 2})
    
    assert response.status_code == 200
    assert response.json() == {
        "message": "Embedding created successfully",
        "text_ids": ["id1", "id2"],
        "total_chunks": 2,
    }
    mock_embedding_usecase.assert_called_once_with("some text", 2)
    mock_embedding_save_usecase.assert_called_once_with("mocked_embedding")

@patch('src.controller.api.router.embedding_usecase')
def test_create_embedding_value_error(mock_embedding_usecase):
    mock_embedding_usecase.side_effect = ValueError("Test error")

    response = client.post("/new_rag/embedding", json={"text": "some text", "index": 2})

    assert response.status_code == 400
    assert "Invalid input: Test error" in response.json()["detail"]

@patch('src.controller.api.router.embedding_search_usecase')
def test_search_embedding_success(mock_embedding_search_usecase):
    mock_embedding_search_usecase.return_value = [{"text": "result1"}, {"text": "result2"}]

    response = client.get("/new_rag/search_vetorial?question=my_question&top_k=2")

    assert response.status_code == 200
    assert response.json() == {"results": [{"text": "result1"}, {"text": "result2"}]}
    mock_embedding_search_usecase.assert_called_once_with("my_question", 2)

@patch('src.controller.api.router.embedding_search_usecase')
def test_search_embedding_http_exception(mock_embedding_search_usecase):
    mock_embedding_search_usecase.return_value = "Internal server error occurred."

    response = client.get("/new_rag/search_vetorial?question=my_question&top_k=2")

    assert response.status_code == 500
    assert "Internal server error occurred." in response.json()["detail"]

@patch('src.controller.api.router.embedding_search_usecase')
def test_search_embedding_exception(mock_embedding_search_usecase):
    mock_embedding_search_usecase.side_effect = Exception("Generic error")

    response = client.get("/new_rag/search_vetorial?question=my_question&top_k=2")

    assert response.status_code == 500
    assert "Error in search: Generic error" in response.json()["detail"]
