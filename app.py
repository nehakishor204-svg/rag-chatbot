import streamlit as st
from ingest import load_pdf, chunk_text, embed_chunks, store_chunks, retrieve, collection
from generate import ask_groq

st.title("📄 Document Q&A Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save the uploaded file temporarily so pdfplumber can read it
    with open("temp_upload.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Processing document..."):
        text = load_pdf("temp_upload.pdf")
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        store_chunks(chunks, embeddings, doc_id=uploaded_file.name)

    st.success(f"Processed {len(chunks)} chunks from {uploaded_file.name}")

query = st.text_input("Ask a question about the document")

if query:
    with st.spinner("Thinking..."):
        relevant_chunks = retrieve(query)
        answer = ask_groq(query, relevant_chunks)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Show retrieved context"):
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"**Chunk {i+1}:**")
            st.write(chunk)