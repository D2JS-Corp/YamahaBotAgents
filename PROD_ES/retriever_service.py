"""
Simple RAG Retriever Service using sentence-transformers and FAISS
Reads from information.txt and provides semantic search endpoint
"""
import os
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="RAG Retriever Service")

# Global variables
embeddings_model = None
index = None
text_chunks = []
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between chunks

class SearchResult(BaseModel):
    id: int
    score: float
    text: str

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks

def load_information(file_path: str = "information.txt"):
    """Load and process the information.txt file"""
    global text_chunks, embeddings_model, index

    print(f"Loading information from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Split into chunks
    text_chunks = chunk_text(content)
    print(f"Split into {len(text_chunks)} chunks")

    # Load embedding model
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = embeddings_model.encode(text_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"✅ Retriever ready with {len(text_chunks)} chunks indexed\n")

@app.on_event("startup")
async def startup_event():
    """Initialize the retriever on startup"""
    load_information()

@app.get("/search", response_model=List[SearchResult])
async def search(
    q: str = Query(..., description="Search query"),
    k: int = Query(4, description="Number of results to return", ge=1, le=10)
) -> List[SearchResult]:
    """
    Search for relevant text chunks based on query
    """
    if not q.strip():
        return []
    
    # Encode query
    query_embedding = embeddings_model.encode([q])
    query_embedding = np.array(query_embedding).astype('float32')
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, min(k, len(text_chunks)))
    
    # Prepare results
    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if idx < len(text_chunks):
            results.append(SearchResult(
                id=int(idx),
                score=float(dist),
                text=text_chunks[idx]
            ))
    
    print(f"🔍 Query: '{q[:50]}...' → Found {len(results)} results")
    return results

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chunks_indexed": len(text_chunks),
        "model": "all-MiniLM-L6-v2"
    }

@app.post("/reload")
async def reload_information():
    """Reload the information.txt file"""
    try:
        load_information()
        return {"status": "success", "chunks": len(text_chunks)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8700)