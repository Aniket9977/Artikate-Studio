# Artikate-Studio

# 🧠 Class 9 Science Chatbot (LLM-Powered Tutoring Assistant)

This project is a personalized educational chatbot designed to help Class 9 students understand Science concepts from the official NCERT curriculum. The chatbot leverages a Large Language Model (LLM) and a PDF-based retrieval system to answer natural language questions with contextual, syllabus-based, and personalized responses.

---

## 📂 Project Structure

project/
├── app.py # Streamlit application
├── .env # Environment variables for API keys
├── requirements.txt # Python dependencies
├── README.md # This file
└── book/
└── Science Class IX/ # Folder containing split chapter PDFs
├── iesc101.pdf
├── iesc102.pdf
├── ...
└── iesc112.pdf




---

## 🚀 Features

- ✅ Ingests **multiple pre-uploaded PDFs** (chapter-wise Class 9 NCERT Science textbook)
- ✅ **Intelligently chunks and indexes** content using `sentence-transformers` and `FAISS`
- ✅ Answers student questions using **OpenAI GPT-4** with context grounding
- ✅ Personalizes responses based on **student learning profile**
- ✅ Built with **Streamlit UI** for a smooth user experience

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) - Interactive UI
- [OpenAI API](https://platform.openai.com/) - GPT-4 for answer generation
- [Sentence Transformers](https://www.sbert.net/) - Text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [PyPDF2](https://pythonhosted.org/PyPDF2/) - PDF text extraction
- [python-dotenv](https://github.com/theskumar/python-dotenv) - Manage API keys

---

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-org/class9-science-chatbot.git
cd class9-science-chatbot
```

2. **Install dependencies**
```bash 

pip install -r requirements.txt
```
3. **Run the Streamlit app**

```bash

streamlit run app.py
```


## Possible Improvements

Add OCR for scanned PDFs

Support for follow-up questions using chat history

Chapter-based dropdown selection

Support multi-subject textbooks

Save and analyze student queries for progress tracking