from src.infrastructure.connection_openai import OpenAIConnection
from src.infrastructure.connection_postgresql import get_db_session
from src.models.database_models import DbOriginText, DbCorrelationEmbedding
from sqlalchemy import text   

import json

client = OpenAIConnection().get_client()

def generate_text_semantic_service(input_text: str, prompt_assistant: str) -> dict:
    """
    Generate a text using the OpenAI API with semantic understanding.
    """
    completion = client.chat.completions.create(
        model="gpt-4.1-nano", # Replace with your model deployment name.
        messages=[
            {"role": "user", "content": f"{input_text}"},
            {"role": "system", "content": f"{prompt_assistant}"},
        ],
        #response_format={"type": "json_object"},
    ) 
    try:
        return_response = completion.choices[0].message.content
        json_response = json.loads(return_response)
        return json_response
    except (AttributeError, IndexError, json.JSONDecodeError) as e:
        print(f"Json not formated correct: {e}")
        return None
    
def embedding_service(input_text: str):
    """
    Generate a embedding using the OpenAI API and model embedding large 3 with 3072 dimensions.
    """
    if not input_text or not input_text.strip():
        raise ValueError("Input text cannot be empty")
    
    embedding = client.embeddings.create(
        model="text-embedding-3-large",
        input=input_text,
    )
    
    return embedding.data[0].embedding

def save_original_text(text: str) -> int:
    """
    Save the original text to PostgreSQL database and return the ID.
    """
    with get_db_session() as session:
        origin_text = DbOriginText(data=text)
        session.add(origin_text)
        session.commit()
        session.refresh(origin_text)
        return origin_text.id

def save_embedding_to_postgresql(id_text_origin: int, embedding_data: list):
    """
    Save the embedding data to PostgreSQL database.
    """
    with get_db_session() as session:
        for data in embedding_data:
            embedding = DbCorrelationEmbedding(
                id_text_origin=id_text_origin,
                correlation_type=data["correlation_type"],
                text_content=data["text_content"],  # Novo campo
                vector=data["embedding"]
            )
            session.add(embedding)
        session.commit()

def search_vetorial(question: str, top_k: int):
    """
    Search for similar embeddings in the PostgreSQL database using vector similarity.
    
    Args:
        question (str): The question to search for
        top_k (int): Number of top results to return (default: 30)
    
    Returns:
        list: List of tuples containing (distance, text_content, correlation_type, origin_text_data)
              or None if error occurs
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    
    if top_k > 1000:  # Limite máximo para evitar sobrecarga
        raise ValueError("top_k cannot exceed 1000")
    
    try:
        question_embedding = embedding_service(question)
        
        with get_db_session() as session:
            # Usar parâmetros seguros para pgvector - converter para lista Python
            # O SQLAlchemy irá tratar a conversão de forma segura
            
            query = text("""
                SELECT 
                    ce.vector <=> CAST(:question_vector AS vector) AS distance,
                    ce.text_content,
                    ce.correlation_type,
                    ot.data as origin_text_data,
                    ce.id as embedding_id,
                    ot.id as origin_text_id
                FROM db_correlation_embedding ce
                INNER JOIN db_origin_text ot ON ce.id_text_origin = ot.id
                WHERE ce.vector IS NOT NULL
                ORDER BY distance ASC
                LIMIT :limit_count
            """)
            
            embedding_str = '[' + ','.join(str(float(x)) for x in question_embedding) + ']'
            
            result = session.execute(
                query, 
                {
                    'question_vector': embedding_str,
                    'limit_count': top_k
                }
            )
            
            results = result.fetchall()
            
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'distance': float(row[0]),
                    'text_content': row[1],
                    'correlation_type': row[2],
                    'origin_text_data': row[3],
                    'embedding_id': row[4],
                    'origin_text_id': row[5]
                })
            
            return formatted_results
            
    except Exception as e:
        print(f"Error in vector search: {e}")
        return None



# Teste local
# print(json.dumps(generate_text_semantic("Responsa em json, quanto é 2 + 2", "você é uma matematico"), indent=2, ensure_ascii=False))
# print(save_original_text("Responsa em json, quanto é 2 + 2"))

# Teste da busca vetorial
# results = search_vetorial("infraestrutura global.", 10)
# print(json.dumps(results, indent=2, ensure_ascii=False))