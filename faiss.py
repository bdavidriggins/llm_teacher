#pip install faiss-cpu

import faiss
import numpy as np
import json

# Load embeddings and metadata
embeddings = np.load('embeddings.npy')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Simple L2 distance

# Add embeddings to the index
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, 'faiss_index.bin')
