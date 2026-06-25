# Document Q&A RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets users upload a PDF and ask natural-language questions about its contents, with answers grounded in the actual document — not hallucinated.

## How it works

1. **PDF ingestion** — extracts text using `pdfplumber`
2. **Chunking** — splits text into overlapping 200-word chunks to preserve context across boundaries
3. **Embeddings** — converts each chunk into a 384-dimensional vector using `sentence-transformers` (`all-MiniLM-L6-v2`)
4. **Vector storage** — stores chunks and embeddings in `ChromaDB` for fast similarity search
5. **Retrieval** — converts the user's question into an embedding and finds the most relevant chunks
6. **Generation** — sends the question + retrieved chunks to Groq's `llama-3.3-70b-versatile` model, which generates an answer grounded only in the provided context

## Tech stack

- Python
- Streamlit (UI)
- pdfplumber (PDF text extraction)
- sentence-transformers (embeddings)
- ChromaDB (vector database)
- Groq API (LLM generation)

## Design decisions

- **pdfplumber over pypdf**: initial testing with `pypdf` produced text with missing spaces between words, which would have degraded embedding quality. Switched to `pdfplumber` for more reliable text extraction.
- **Groq over a paid API**: chose Groq for free, fast inference using open models, avoiding API costs while keeping answer quality high.
- **Manual RAG pipeline over LangChain**: built retrieval and chunking manually with ChromaDB and sentence-transformers, for full control and to deeply understand each pipeline step.

## Known limitations

- Short documents (e.g. resumes) sometimes split key details across chunk boundaries, occasionally causing partial answers to multi-part questions.
- Currently supports one document at a time.

## Running locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`

You'll need a free Groq API key in a `.env` file:
\`\`\`
GROQ_API_KEY=your_key_here
\`\`\`