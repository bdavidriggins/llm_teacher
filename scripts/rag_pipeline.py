import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from reranker import Reranker
from generate_response import OllamaClient
from tqdm import tqdm

class RAGSystem:
    def __init__(self, faiss_index_path, faiss_metadata_path, embedding_model_name='all-MiniLM-L6-v2', reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', ollama_api_url='http://localhost:11434/api/generate'):
        # Load FAISS index
        self.index = faiss.read_index(faiss_index_path)
        print("FAISS index loaded.")
        
        # Load metadata
        with open(faiss_metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print("FAISS metadata loaded.")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        print(f"Embedding model '{embedding_model_name}' loaded.")
        
        # Initialize reranker
        self.reranker = Reranker(model_name=reranker_model_name)
        print(f"Reranker model '{reranker_model_name}' loaded.")
        
        # Initialize Ollama client
        self.ollama = OllamaClient(api_url=ollama_api_url)
        print(f"Ollama client initialized with API URL: {ollama_api_url}")
    
    def retrieve(self, query, top_k=25):
        #query_embedding = self.embedding_model.encode(query).astype('float32')
        query_embedding = self.embedding_model.encode(query, clean_up_tokenization_spaces=True).astype('float32')


        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        retrieved = [self.metadata[idx] for idx in indices[0]]
        print(f"Retrieved {len(retrieved)} documents for query: '{query}'")
        for doc in retrieved:
            print(f" - {doc['title'][:50]}")  # Assuming each doc has a 'title' field
        return retrieved
    
    def generate(self, prompt, context):
        return self.ollama.generate_response(prompt=prompt, context=context)

    def rag_pipeline(self, query, retrieval_k=25, rerank_k=5):
        # Step 1: Retrieve top_k documents
        retrieved_docs = self.retrieve(query, top_k=retrieval_k)
        
        # Step 2: Rerank the retrieved documents
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=rerank_k)
        
        # Step 3: Combine contexts
        contexts = "\n\n".join([doc['text'] for doc in reranked_docs])
        
        # Step 4: Generate response using Ollama
        answer = self.generate(query, contexts)
        
        return answer

if __name__ == "__main__":
    # Paths to FAISS index and metadata
    faiss_index_path = '../faiss/faiss_index.bin'
    faiss_metadata_path = '../faiss/faiss_metadata.json'
    
    # Initialize RAG system
    rag = RAGSystem(
        faiss_index_path=faiss_index_path,
        faiss_metadata_path=faiss_metadata_path,
        embedding_model_name='all-MiniLM-L6-v2',
        reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        ollama_api_url='http://localhost:11434/api/generate'  # Adjust if different
    )
    
    # Example queries
    queries = [
        "What caused the Treaty of Versailles?",
        "How did the Industrial Revolution impact European economies?",
        "What were the key battles of World War I?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        answer = rag.rag_pipeline(q)
        print(f"Answer: {answer}\n{'-'*50}")