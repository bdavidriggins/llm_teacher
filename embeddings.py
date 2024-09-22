from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Load chunked data
with open('chunked_wikipedia_articles.json', 'r') as f:
    chunked_data = json.load(f)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective

# Generate embeddings
embeddings = []
metadata = []

for title, chunks in chunked_data.items():
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
np.save('embeddings.npy', embeddings)
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)
