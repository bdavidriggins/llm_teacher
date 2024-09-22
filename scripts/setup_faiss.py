import faiss
import numpy as np
import json

def main():
    # Load embeddings and metadata
    embeddings = np.load('../embeddings/embeddings.npy')
    with open('../embeddings/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Initialize FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Simple L2 distance
    print("FAISS index created with dimension:", dimension)
    
    # Add embeddings to the index
    index.add(embeddings)
    print(f"Number of vectors in the index: {index.ntotal}")
    
    # Save FAISS index
    faiss.write_index(index, '../faiss/faiss_index.bin')
    print("FAISS index saved to faiss/faiss_index.bin")
    
    # Save metadata (ensure alignment with FAISS index)
    with open('../faiss/faiss_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("FAISS metadata saved to faiss/faiss_metadata.json")

if __name__ == "__main__":
    main()
    
