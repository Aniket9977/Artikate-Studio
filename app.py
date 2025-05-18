import os
import streamlit as st
from PyPDF2 import PdfReader
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_DIR = "book\Science Class IX"

# --- Step 1: Load and Combine PDFs ---
@st.cache_data
def load_all_pdfs(pdf_dir):
    all_text = []
    for filename in sorted(os.listdir(pdf_dir)):
        if filename.endswith(".pdf"):
            reader = PdfReader(os.path.join(pdf_dir, filename))
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    all_text.append(page_text)
    return "\n".join(all_text)

# --- Step 2: Chunk Content ---
def chunk_text(text, chunk_size=500):
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- Step 3: Embedding & Indexing ---
@st.cache_resource
def embed_and_index(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index, chunks

def retrieve_chunks(query, model, index, chunks, k=5):
    query_emb = model.encode([query])
    _, I = index.search(np.array(query_emb), k)
    return [chunks[i] for i in I[0]]

# --- Step 4: Generate LLM Response ---
def generate_response(context, question, profile):
    prefix = "Explain in a simple tone as the student struggles with Physics." if profile == "weak" else "Explain in a concise and technical tone."

    prompt = f"""{prefix}

Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("📗 Class 9 Science Tutor (NCERT)")
st.markdown("Ask questions from the entire Class 9 Science syllabus")

profile = st.selectbox("Select your learning level:", ["weak", "strong"], format_func=lambda x: "Weak in Physics" if x=="weak" else "Strong in Physics")

user_query = st.text_input("🔍 Ask your question:")

# Load, Chunk, Embed
with st.spinner("Loading textbook..."):
    raw_text = load_all_pdfs(PDF_DIR)
    chunks = chunk_text(raw_text)
    model, index, chunks = embed_and_index(chunks)

# Run query
if user_query:
    with st.spinner("Thinking..."):
        top_chunks = retrieve_chunks(user_query, model, index, chunks)
        context = "\n\n".join(top_chunks)
        answer = generate_response(context, user_query, profile)

        st.markdown("### ✅ Answer")
        st.write(answer)

        with st.expander("📚 Context Chunks Used"):
            for i, chunk in enumerate(top_chunks, 1):
                st.markdown(f"**Chunk {i}:**\n{chunk}")

        with st.expander("📌 Log Info"):
            st.json({
                "Query": user_query,
                "Profile": profile,
                "Chunks Retrieved": len(top_chunks),
                "Weak Topic": any("physics" in c.lower() for c in top_chunks)
            })
