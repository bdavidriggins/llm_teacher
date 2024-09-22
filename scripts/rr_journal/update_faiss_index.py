import os
import json
import faiss
import numpy as np

# Paths for embeddings and FAISS index
embeddings_dir = '../../faiss/'
faiss_index_path = '../../faiss/faiss_index.bin'
faiss_metadata_path = '../../faiss/faiss_metadata.json'

def load_embeddings(file_path):
    """Load embeddings from a file."""
    return np.load(file_path)

def update_faiss_index(embeddings, index_path):
    """Load an existing FAISS index and add new embeddings to it."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    index = faiss.read_index(index_path)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Updated FAISS index with {len(embeddings)} new embeddings.")

def update_metadata(chunks, metadata_path, source):
    """Update the FAISS metadata with new chunk information."""
    if not os.path.exists(metadata_path):
        metadata = []
    else:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)

    current_length = len(metadata)
    new_metadata = []
    
    for i, chunk in enumerate(chunks):
        entry = {
            "id": current_length + i,
            "text": chunk,
            "source": source,
            "chunk": i + 1
        }
        new_metadata.append(entry)
    
    metadata.extend(new_metadata)
    
