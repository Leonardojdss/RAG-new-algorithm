from src.service.embedding_service import generate_text_semantic_service, embedding_service, save_original_text, save_embedding_to_postgresql, search_vetorial
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models.database_models import CorrelationType
import json

def create_overlapping_chunks(texts, overlap_size):
    """
    Aplica overlap manual entre os chunks de texto.
    Função criada para contornar o bug da função de overlap da langchain.
    
    Args:
        texts: Lista de strings com os chunks de texto
        overlap_size: Tamanho do overlap em caracteres
    
    Returns:
        Lista de strings com os chunks com overlap aplicado
    """
    overlapping_texts = []
    for i in range(len(texts)):
        current_chunk = texts[i]
        overlap_prefix = ''
        overlap_suffix = ''
        
        if i > 0:
            previous_chunk = texts[i-1]
            overlap_prefix = previous_chunk[-overlap_size:]
        
        if i < len(texts) - 1:
            next_chunk = texts[i+1]
            overlap_suffix = next_chunk[:overlap_size]
        
        combined_chunk = overlap_prefix + current_chunk + overlap_suffix
        overlapping_texts.append(combined_chunk)
    
    return overlapping_texts

def embedding_usecase(input_text: str, index: int, chunk_size: int = 500, overlap_size: int = 100):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,  # função com de overlap da langchain não funciona
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    initial_chunks = text_splitter.split_text(input_text)
    
    text_chunks = create_overlapping_chunks(initial_chunks, overlap_size)
    
    type_relationship = ["similaridade_semantica", "relacionamento_semantico", "contexto_compartilhado"]
    
    all_texts = []
    
    for chunk_index, chunk_text in enumerate(text_chunks):
        for text in type_relationship:
            with open(f'src/prompt/{text}.txt', 'r', encoding='utf-8') as file:
                prompt_assistant = file.read()
            
            input_text_all = generate_text_semantic_service(chunk_text, prompt_assistant)
            
            result_data = {
                "type": text,
                "data": input_text_all,
                "chunk_index": chunk_index,
                "original_chunk": chunk_text
            }
            all_texts.append(result_data)

    count = 0
    embedding_json = []
    
    for text_data in all_texts:
        text_type = text_data["type"]
        results = text_data["data"]
        chunk_index = text_data["chunk_index"]
        original_chunk = text_data["original_chunk"]
        
        type_count = 0
        
        if isinstance(results, dict):
            for key, text_content in results.items():
                if type_count >= index:
                    break
                    
                text_embedding = embedding_service(text_content)
                
                embedding_data = {
                    "id_text_origin": "",
                    "type": text_type,
                    "text": text_content,
                    "embedding": text_embedding,
                    "chunk_index": chunk_index,
                    "original_chunk": original_chunk,
                    "chunk_metadata": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap_size,
                        "total_chunks": len(text_chunks)
                    }
                }
                
                embedding_json.append(embedding_data)
                count += 1
                type_count += 1

    with open('embedding_temp.json', 'w', encoding='utf-8') as temp_file:
        json.dump(embedding_json, temp_file, indent=2, ensure_ascii=False)

    embedding_json_output = json.dumps(embedding_json, indent=2, ensure_ascii=False)    
    return embedding_json_output

def embedding_save_usecase(embedding_json: str):
    """
    Save the original text and embeddings to the database.
    """
    
    type_mapping = {
        "similaridade_semantica": CorrelationType.SIMILARIDADE_SEMANTICA,
        "relacionamento_semantico": CorrelationType.RELACIONAMENTO_SEMANTICO,
        "contexto_compartilhado": CorrelationType.CONTEXTO_COMPARTILHADO
    }
    
    embedding_data = json.loads(embedding_json)
    
    chunks_data = {}
    for data in embedding_data:
        chunk_index = data.get("chunk_index", 0)
        if chunk_index not in chunks_data:
            chunks_data[chunk_index] = {
                "original_chunk": data.get("original_chunk", ""),
                "embeddings": []
            }
        chunks_data[chunk_index]["embeddings"].append(data)
    
    saved_ids = []
    
    for chunk_index, chunk_info in chunks_data.items():
        chunk_text = chunk_info["original_chunk"]
        id_text_origin = save_original_text(chunk_text)
        
        processed_embeddings = []
        for data in chunk_info["embeddings"]:
            text_content = data["text"]
            
            if "chunk_index" in data and "original_chunk" in data:
                chunk_info_text = f"\n[Chunk {data['chunk_index']} de {data.get('chunk_metadata', {}).get('total_chunks', 'N/A')}]"
                text_content = f"{text_content}{chunk_info_text}"
            
            processed_embedding = {
                "correlation_type": type_mapping.get(data["type"], data["type"]),
                "text_content": text_content,
                "embedding": data["embedding"]
            }
            processed_embeddings.append(processed_embedding)
        
        save_embedding_to_postgresql(id_text_origin, processed_embeddings)
        saved_ids.append(id_text_origin)
    
    return saved_ids

def embedding_search_usecase(question: str, top_k: int = 5):
    """
    Use case to search for embeddings based on a question.
    
    Args:
        question (str): The question to search for.
        top_k (int): The number of top results to return.
    
    Returns:
        list: List of tuples containing (distance, text_content, correlation_type, origin_text_data)
              or None if error occurs
    """
    try:
        results = search_vetorial(question, top_k)
        return results
    except Exception as e:
        result_error = f"Error in embedding search use case: {e}"
        return result_error
