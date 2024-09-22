# File: generate_embeddings.py
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to the preprocessed chunks
chunks_file_path = '../../sources/journal_roberts_rangers_chunks.txt'
embeddings_output_path = '../../faiss/embeddings.npy'

def load_chunks(file_path):
    """Load text chunks from a file."""
    with open(file_path, 'r') as file:
        return [chunk.strip() for chunk in file.read().split('\n\n') if chunk.strip()]

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for the given text chunks."""
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype('float32')

def main():
    # Load the text chunks
    if not os.path.exists(chunks_file_path):
        print(f"Chunks file not found: {chunks_file_path}")
        return
    
    chunks = load_chunks(chunks_file_path)
    
    # Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks)
    print(f"Generated embeddings for {len(chunks)} chunks.")
    
    # Save embeddings to a file for later use
    np.save(embeddings_output_path, embeddings)
    print(f"Embeddings saved to {embeddings_output_path}")

if __name__ == '__main__':
    main()
