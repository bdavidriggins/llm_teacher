import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to the sources directory
sources_dir = '../../sources/'
# Embeddings output directory
embeddings_output_dir = '../../faiss/'

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
    # Process each chunks file in the sources directory
    for file in os.listdir(sources_dir):
        if file.endswith('_chunks.txt'):
            chunks_file_path = os.path.join(sources_dir, file)
            embeddings_file_path = os.path.join(embeddings_output_dir, f"{file}_embeddings.npy")
            
            # Load the text chunks
            chunks = load_chunks(chunks_file_path)
            if not chunks:
                print(f"No chunks found in {chunks_file_path}")
                continue
            
            # Generate embeddings for the chunks
            embeddings = generate_embeddings(chunks)
            print(f"Generated embeddings for {len(chunks)} chunks from {file}.")
            
            # Save embeddings to a file for later use
            np.save(embeddings_file_path, embeddings)
            print(f"Embeddings saved to {embeddings_file_path}")

if __name__ == '__main__':
    main()
