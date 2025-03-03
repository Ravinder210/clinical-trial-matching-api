
from fastapi import FastAPI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

app = FastAPI()

import subprocess

# Download spaCy model if not available
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Load model
nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_dim = 384
index = faiss.IndexFlatL2(embedding_dim)
vector_store = []

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def store_embedding(text):
    vector = np.array([get_embedding(text)], dtype=np.float32)
    index.add(vector)
    vector_store.append(text)
    return {"message": "Text stored successfully"}

def search_similar(query, top_k=3):
    query_vector = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(query_vector, top_k)
    return [vector_store[i] for i in indices[0]]

@app.post("/store/")
def store_text(text: str):
    return store_embedding(text)

@app.get("/search/")
def search_text(query: str):
    return {"matches": search_similar(query)}

@app.get("/")
def home():
    return {"message": "Clinical Trial Matching API Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
