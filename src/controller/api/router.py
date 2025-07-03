from fastapi import APIRouter, HTTPException
from src.usecase.embedding_usecase import embedding_usecase, embedding_save_usecase, embedding_search_usecase
from pydantic import BaseModel
from typing import Optional

class TextRequest(BaseModel):
    text: str
    index: Optional[int] = 5

class TextRequestWithStrategy(BaseModel):
    text: str
    index: Optional[int] = 5

class ChunkPreviewRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

router = APIRouter()

@router.post("/embedding")
async def create_embedding(text_request: TextRequest):
    """
    Create embeddings with custom chunking parameters.
    """
    try:
        text = text_request.text
        index = text_request.index
        
        embedding = embedding_usecase(text, index)
        embedding_save = embedding_save_usecase(embedding)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    return {
        "message": "Embedding created successfully",
        "text_ids": embedding_save,  
        "total_chunks": len(embedding_save) if isinstance(embedding_save, list) else 1,
    }

@router.get("/search_vetorial")
async def search_embedding(question: str, top_k: int = 5):
    try:
        results = embedding_search_usecase(question, top_k)
        if isinstance(results, str):
            raise HTTPException(status_code=500, detail=results)
        return {
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in search: {e}")