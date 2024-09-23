import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from reranker import Reranker
from generate_response import OllamaClient
from tqdm import tqdm

class RAGSystem:
    def __init__(self, faiss_index_path, faiss_metadata_path, embedding_model_name='multi-qa-mpnet-base-dot-v1', reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', ollama_api_url='http://localhost:11434/api/generate'):
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
        query_embedding = self.embedding_model.encode(query, clean_up_tokenization_spaces=True).astype('float32')

        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        retrieved = [self.metadata[idx] for idx in indices[0]]
        print(f"Retrieved {len(retrieved)} documents for query: '{query}'")
        
        for doc in retrieved:
            # Check if 'title' exists in the document and print it, or print a placeholder
            title = doc.get('title', 'Untitled')
            print(f" - {title[:50]}")  # Print the first 50 characters of the title

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
        embedding_model_name='multi-qa-mpnet-base-dot-v1',
        reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        ollama_api_url='http://localhost:11434/api/generate'  # Adjust if different
    )
    
    #Roberts Rangers Specific for RAG
    queries = [
            "What challenges did Robert’s Rangers face during their missions? Give me a detailed long response in essay format."
        ]
    # queries = [
    #         "What was the role of Robert's Rangers during the conflict mentioned in the text?",
    #         "Who was the leader of the Rangers, and what were his key strategies?",
    #         "How did the tactics used by Robert’s Rangers differ from traditional military tactics of the time?",
    #         "What challenges did Robert’s Rangers face during their missions?",
    #         "Based on their missions, what can be inferred about the leadership style of Robert Rogers?",
    #         "Compare the leadership of Robert Rogers with other prominent military leaders mentioned in the text.",
    #         "What section of the text discusses the role of the Rangers in their first major engagement?"
    #     ]
    
    # # Example queries
    # queries = [
    #     "What caused the Treaty of Versailles? Tell me all you know as an essay",
    #     "How did the Industrial Revolution impact European economies? Tell me all you know as an essay",
    #     "What were the key battles of World War I? Tell me all you know as an essay"
    # ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        answer = rag.rag_pipeline(q)
        print(f"Answer: {answer}\n{'-'*50}")