# File: update_faiss_index.py
import os
import json
import faiss
import numpy as np

# Paths for embeddings and FAISS index
embeddings_file_path = '../../faiss/embeddings.npy'
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

def update_metadata(chunks, metadata_path, source="journal_roberts_rangers.txt"):
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
    
    with open(metadata_path, 'w') as file:
        json.dump(metadata, file, indent=2)
    print(f"Updated metadata with {len(new_metadata)} new entries.")

def main():
    # Load embeddings and chunks
    embeddings = load_embeddings(embeddings_file_path)
    
    # Load chunks for metadata update
    chunks_file_path = '../../sources/journal_roberts_rangers_chunks.txt'
    with open(chunks_file_path, 'r') as file:
        chunks = [chunk.strip() for chunk in file.read().split('\n\n') if chunk.strip()]
    
    # Update FAISS index
    update_faiss_index(embeddings, faiss_index_path)
    
    # Update metadata
    update_metadata(chunks, faiss_metadata_path)

if __name__ == '__main__':
    main()
