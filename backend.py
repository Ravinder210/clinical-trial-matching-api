from fastapi import FastAPI
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import subprocess

# Download & load spaCy model
subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable unused components

# Use a smaller embedding model to reduce RAM usage
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# Create FAISS index
embedding_dim = 384  # MiniLM model dimension
index = faiss.IndexFlatL2(embedding_dim)
vector_store = []  # Store text metadata

app = FastAPI()

# Function to generate embeddings
def get_embedding(text):
    return embedding_model.encode(text, convert_to_numpy=True).tolist()

@app.get("/")
def home():
    return {"message": "Clinical Trial Matching API Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
