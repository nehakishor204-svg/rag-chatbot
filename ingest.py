import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")


def load_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def embed_chunks(chunks):
    return model.encode(chunks).tolist()


def store_chunks(chunks, embeddings, doc_id):
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


def retrieve(query, top_k=3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results['documents'][0]


if __name__ == "__main__":
    extracted_text = load_pdf("resume_neha.pdf")
    chunks = chunk_text(extracted_text)
    embeddings = embed_chunks(chunks)
    store_chunks(chunks, embeddings, doc_id="resume")

    print(f"Total chunks: {len(chunks)}")
    print(f"Stored {collection.count()} items in ChromaDB")

    # Test retrieval
    query = "What projects has this person worked on?"
    results = retrieve(query)
    print(f"\nTop matches for: '{query}'\n")
    for i, r in enumerate(results):
        print(f"--- Match {i+1} ---")
        print(r)
        print()