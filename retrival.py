def retrieve_chunks(query, model, index, chunks, k=5):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]
