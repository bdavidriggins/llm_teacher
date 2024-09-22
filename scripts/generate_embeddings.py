from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm

def main():
    # Load chunked data
    with open('../data/chunked_wikipedia_articles.json', 'r') as f:
        chunked_data = json.load(f)
    
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective
    
    embeddings = []
    metadata = []
    
    for title, chunks in tqdm(chunked_data.items(), desc="Generating embeddings"):
        for idx, chunk in enumerate(chunks):
            embedding = model.encode(chunk)
            embeddings.append(embedding)
            metadata.append({
                'title': title,
                'chunk_index': idx,
                'text': chunk
            })
    
    embeddings = np.array(embeddings).astype('float32')
    
    # Save embeddings and metadata
    np.save('../embeddings/embeddings.npy', embeddings)
    with open('../embeddings/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Embeddings and metadata saved to embeddings/")

if __name__ == "__main__":
    main()
