from embedding import *
from llm_prompt import generate_response
from pdf_ingestion  import extract_text_from_pdf
from preprocessing import chunk_text    
from retrival import retrieve_chunks
import numpy as np
from sentence_transformers import SentenceTransformer

def chatbot_response(pdf_path, query, user_profile):
    raw_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(raw_text)
    embeddings = embed_chunks(chunks)
    index = create_faiss_index(np.array(embeddings))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    relevant_chunks = retrieve_chunks(query, model, index, chunks)
    context = "\n\n".join(relevant_chunks)
    answer = generate_response(context, query, user_profile)
    
    # Logging (basic)
    print("User:", user_profile)
    print("Query:", query)
    print("Matched Topic:", relevant_chunks[0][:60])
    print("Includes Weak Topic:", "Physics" in relevant_chunks[0])
    
    return {
        "query": query,
        "retrieved_content": context,
        "answer": answer
    }
