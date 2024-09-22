Absolutely! Let's walk through a comprehensive, step-by-step guide to set up and run your **Retrieval-Augmented Generation (RAG)** system on your **GCP VM** with **Ollama** running in a Docker container. This guide will cover:

1. **Setting Up the VM Environment**
2. **Organizing Your Project Structure**
3. **Installing Necessary Dependencies**
4. **Preparing and Chunking Your Data**
5. **Generating Embeddings**
6. **Setting Up FAISS for Retrieval**
7. **Implementing the Reranker**
8. **Integrating with Ollama for Generation**
9. **Building and Running the RAG Pipeline**
10. **Testing and Evaluating the System**
11. **Optimizing and Monitoring**

Let's dive in!

---

## **1. Setting Up the VM Environment**

### **a. Connect to Your GCP VM**

1. **Access the VM via SSH:**

   ```bash
   gcloud compute ssh your-vm-instance-name --zone your-vm-zone
   ```

   Replace `your-vm-instance-name` and `your-vm-zone` with your actual VM details.

### **b. Update and Upgrade Packages**

2. **Update Package Lists:**

   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

### **c. Install Essential Tools**

3. **Install Git and Other Utilities:**

   ```bash
   sudo apt install -y git curl build-essential
   ```

---

## **2. Organizing Your Project Structure**

### **a. Create a Project Directory**

4. **Navigate to Your Home Directory and Create Project Folder:**

   ```bash
   cd ~
   mkdir rag_project
   cd rag_project
   ```

### **b. Set Up Directory Structure**

5. **Create Subdirectories:**

   ```bash
   mkdir data embeddings faiss models scripts logs
   ```

   - `data/`: For raw and processed data.
   - `embeddings/`: To store generated embeddings.
   - `faiss/`: To hold the FAISS index.
   - `models/`: For storing models if needed.
   - `scripts/`: For all your Python scripts.
   - `logs/`: To store logs for monitoring and debugging.

---

## **3. Installing Necessary Dependencies**

### **a. Install Python and Virtual Environment Tools**

6. **Install Python 3 and `pip`:**

   ```bash
   sudo apt install -y python3 python3-pip python3-venv
   ```

### **b. Create and Activate a Virtual Environment**

7. **Set Up Virtual Environment:**

   ```bash
   python3 -m venv rag_env
   source rag_env/bin/activate
   ```

   *You should see `(rag_env)` prefixed in your terminal now.*

### **c. Upgrade `pip`**

8. **Upgrade `pip` to the Latest Version:**

   ```bash
   pip install --upgrade pip
   ```

### **d. Install Required Python Packages**

9. **Create a `requirements.txt` File:**

   Create a file named `requirements.txt` inside the `rag_project` directory with the following content:

   ```txt
   wikipedia-api
   sentence-transformers
   faiss-cpu
   transformers
   torch
   requests
   scikit-learn
   numpy
   pandas
   tqdm
   ```

10. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### **e. Verify GPU Availability (Optional but Recommended)**

11. **Install `torch` with CUDA Support:**

    If you plan to utilize GPUs for embedding and reranking:

    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    *Ensure that your VM's GPU drivers are compatible with CUDA 11.8 as specified in your project plan.*

---

## **4. Preparing and Chunking Your Data**

### **a. Create Data Preparation Script**

12. **Navigate to the `scripts/` Directory:**

    ```bash
    cd scripts
    ```

13. **Create `prepare_data.py`:**

    ```bash
    nano prepare_data.py
    ```

    **Paste the Following Code:**

    ```python
    import wikipediaapi
    import json

    def get_wikipedia_articles(topics):
        wiki_wiki = wikipediaapi.Wikipedia('en')
        articles = {}
        for topic in topics:
            page = wiki_wiki.page(topic)
            if page.exists():
                articles[topic] = page.text
            else:
                print(f"Article for '{topic}' does not exist.")
        return articles

    if __name__ == "__main__":
        topics = [
            "Treaty of Versailles",
            "World War I",
            "Industrial Revolution",
            # Add more topics as needed
        ]
        articles = get_wikipedia_articles(topics)
        with open('../data/wikipedia_articles.json', 'w') as f:
            json.dump(articles, f, indent=2)
        print("Wikipedia articles saved to data/wikipedia_articles.json")
    ```

14. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **b. Run Data Preparation Script**

15. **Execute the Script:**

    ```bash
    python prepare_data.py
    ```

    *This will fetch the specified Wikipedia articles and save them to `data/wikipedia_articles.json`.*

### **c. Create Chunking Script**

16. **Create `chunk_text.py`:**

    ```bash
    nano chunk_text.py
    ```

    **Paste the Following Code:**

    ```python
    import re
    import json
    from tqdm import tqdm

    def chunk_text(text, max_length=512):
        paragraphs = re.split(r'\n{2,}', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_length:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                chunks.append(current_chunk)
                current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    if __name__ == "__main__":
        with open('../data/wikipedia_articles.json', 'r') as f:
            articles = json.load(f)
        
        chunked_data = {}
        for title, content in tqdm(articles.items(), desc="Chunking articles"):
            chunked_data[title] = chunk_text(content)
        
        with open('../data/chunked_wikipedia_articles.json', 'w') as f:
            json.dump(chunked_data, f, indent=2)
        print("Chunked data saved to data/chunked_wikipedia_articles.json")
    ```

17. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **d. Run Chunking Script**

18. **Execute the Script:**

    ```bash
    python chunk_text.py
    ```

    *This will split each article into manageable chunks and save them to `data/chunked_wikipedia_articles.json`.*

---

## **5. Generating Embeddings**

### **a. Create Embedding Generation Script**

19. **Create `generate_embeddings.py`:**

    ```bash
    nano generate_embeddings.py
    ```

    **Paste the Following Code:**

    ```python
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
    ```

20. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **b. Run Embedding Generation Script**

21. **Execute the Script:**

    ```bash
    python generate_embeddings.py
    ```

    *This will generate embeddings for each chunk and save them to `embeddings/embeddings.npy` and `embeddings/metadata.json`.*

---

## **6. Setting Up FAISS for Retrieval**

### **a. Create FAISS Setup Script**

22. **Create `setup_faiss.py`:**

    ```bash
    nano setup_faiss.py
    ```

    **Paste the Following Code:**

    ```python
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
    ```

23. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **b. Run FAISS Setup Script**

24. **Execute the Script:**

    ```bash
    python setup_faiss.py
    ```

    *This will create a FAISS index from your embeddings and save it along with the metadata.*

---

## **7. Implementing the Reranker**

### **a. Create Reranker Script**

25. **Create `reranker.py`:**

    ```bash
    nano reranker.py
    ```

    **Paste the Following Code:**

    ```python
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import json

    class Reranker:
        def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
        
        def rerank(self, query, candidates, top_k=5):
            inputs = [f"{query} [SEP] {candidate['text']}" for candidate in candidates]
            encoded = self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
                scores = outputs.logits.squeeze().cpu().numpy()
            
            # Attach scores to candidates
            for i, candidate in enumerate(candidates):
                candidate['score'] = scores[i]
            
            # Sort candidates by score
            sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
            return sorted_candidates[:top_k]
    ```

26. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **b. Install Additional Dependencies for Reranker**

27. **Ensure Transformers and Torch are Installed:**

    ```bash
    pip install transformers torch
    ```

---

## **8. Integrating with Ollama for Generation**

### **a. Understand Ollama's API**

Before proceeding, ensure you have access to Ollama's API documentation. Typically, Ollama provides an HTTP endpoint for generating text. For this guide, we'll assume it's accessible at `http://localhost:11434/generate`.

### **b. Create Generation Script**

28. **Create `generate_response.py`:**

    ```bash
    nano generate_response.py
    ```

    **Paste the Following Code:**

```python
import requests
import json

class OllamaClient:
    def __init__(self, api_url='http://localhost:11434/api/generate'):
        self.api_url = api_url
    
    def generate_response(self, prompt, context, max_tokens=512, temperature=0.7, stream=False):
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        payload = {
            "model": "llama3.1",
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        try:
            response = requests.post(self.api_url, json=payload, stream=stream)
            response.raise_for_status()

            if stream:
                result = ""
                for line in response.iter_lines():
                    if line:
                        result += json.loads(line.decode('utf-8')).get('response', '')
                return result
            else:
                return response.json().get('response', '')

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""
```

29. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **c. Test Ollama's API Connectivity**

30. **Create a Test Script `test_ollama.py`:**

    ```bash
    nano test_ollama.py
    ```

    **Paste the Following Code:**

    ```python
    from generate_response import OllamaClient

    def main():
        client = OllamaClient()
        prompt = "What caused the Treaty of Versailles?"
        context = "The Treaty of Versailles was primarily caused by the aftermath of World War I, where the Allied Powers sought to impose strict sanctions and reparations on Germany to prevent future aggression."
        response = client.generate_response(prompt, context)
        print("Generated Response:", response)

    if __name__ == "__main__":
        main()
    ```


curl http://localhost:11434/api/generate -d '{"model":"llama3.1", "prompt":"What is water made of?"}'



31. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

32. **Run the Test Script:**

    ```bash
    python test_ollama.py
    ```

    *Ensure that Ollama's Docker container is running and accessible at the specified `api_url`. You should see a generated response based on the prompt and context.*

---

## **9. Building and Running the RAG Pipeline**

### **a. Create RAG Pipeline Script**

33. **Create `rag_pipeline.py`:**

    ```bash
    nano rag_pipeline.py
    ```

    **Paste the Following Code:**

    ```python
    import faiss
    import numpy as np
    import json
    from sentence_transformers import SentenceTransformer
    from reranker import Reranker
    from generate_response import OllamaClient
    from tqdm import tqdm

    class RAGSystem:
        def __init__(self, faiss_index_path, faiss_metadata_path, embedding_model_name='all-MiniLM-L6-v2', reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2', ollama_api_url='http://localhost:11434/generate'):
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
            query_embedding = self.embedding_model.encode(query).astype('float32')
            distances, indices = self.index.search(np.array([query_embedding]), top_k)
            retrieved = [self.metadata[idx] for idx in indices[0]]
            return retrieved
        
        def generate(self, prompt, context):
            return self.ollama.generate_response(prompt, context)
        
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
            ollama_api_url='http://localhost:11434/generate'  # Adjust if different
        )
        
        # Example queries
        queries = [
            "What was the role of Robert's Rangers during the conflict mentioned in the text?",
            "Who was the leader of the Rangers, and what were his key strategies?",
            "How did the tactics used by Robert’s Rangers differ from traditional military tactics of the time?",
            "What challenges did Robert’s Rangers face during their missions?",
            "Based on their missions, what can be inferred about the leadership style of Robert Rogers?",
            "Compare the leadership of Robert Rogers with other prominent military leaders mentioned in the text.",
            "What section of the text discusses the role of the Rangers in their first major engagement?"
        ]
        
        for q in queries:
            print(f"\nQuery: {q}")
            answer = rag.rag_pipeline(q)
            print(f"Answer: {answer}\n{'-'*50}")
    ```

34. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **b. Run the RAG Pipeline**

35. **Execute the Pipeline:**

    ```bash
    python rag_pipeline.py
    ```

    *This script will process each query, retrieve relevant documents, rerank them, combine contexts, and generate responses using Ollama.*

---

## **10. Testing and Evaluating the System**

### **a. Create a Testing Script (Optional)**

36. **Create `test_rag.py`:**

    ```bash
    nano test_rag.py
    ```

    **Paste the Following Code:**

    ```python
    from rag_pipeline import RAGSystem

    def main():
        faiss_index_path = '../faiss/faiss_index.bin'
        faiss_metadata_path = '../faiss/faiss_metadata.json'
        
        rag = RAGSystem(
            faiss_index_path=faiss_index_path,
            faiss_metadata_path=faiss_metadata_path,
            embedding_model_name='all-MiniLM-L6-v2',
            reranker_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
            ollama_api_url='http://localhost:11434/generate'
        )
        
        queries = [
            "What were the main factors leading to the Treaty of Versailles?",
            "How did the Industrial Revolution impact European economies?",
            "What were the key battles of World War I?"
        ]
        
        for q in queries:
            print(f"\nQuery: {q}")
            answer = rag.rag_pipeline(q)
            print(f"Answer: {answer}\n{'-'*50}")

    if __name__ == "__main__":
        main()
    ```

37. **Save and Exit:**

    Press `Ctrl + O` to save and `Ctrl + X` to exit.

### **b. Execute the Testing Script**

38. **Run the Test Script:**

    ```bash
    python test_rag.py
    ```

    *Evaluate the generated answers for relevance, accuracy, and coherence.*

---

## **11. Optimizing and Monitoring**

### **a. Implement Logging**

39. **Modify `rag_pipeline.py` to Include Logging:**

   Update the `RAGSystem` class to include logging for better monitoring.

   ```python
   import logging

   # At the beginning of rag_pipeline.py, configure logging
   logging.basicConfig(
       filename='../logs/rag_pipeline.log',
       level=logging.INFO,
       format='%(asctime)s:%(levelname)s:%(message)s'
   )
   ```

   *Add `logging.info()` statements at key points in the `RAGSystem` methods to log events, errors, and statuses.*

### **b. Monitor System Resources**

40. **Use `htop` to Monitor CPU and Memory Usage:**

    ```bash
    sudo apt install -y htop
    htop
    ```

41. **Monitor Docker Container Resources:**

    ```bash
    docker stats
    ```

    *Ensure that Ollama's container is not overconsuming resources.*

### **c. Optimize Retrieval and Reranking Parameters**

42. **Adjust `top_k` Values:**

    Experiment with different values for `retrieval_k` and `rerank_k` in `rag_pipeline.py` to balance between speed and quality.

### **d. Implement Caching (Optional)**

43. **Add Caching for Frequently Accessed Queries:**

    Utilize Python's `functools.lru_cache` or implement a simple caching mechanism to store and retrieve responses for repeated queries.

### **e. Automate Pipeline Execution (Optional)**

44. **Create a Shell Script to Run the Pipeline:**

    ```bash
    nano run_rag.sh
    ```

    **Paste the Following Code:**

    ```bash
    #!/bin/bash

    source ~/rag_project/rag_env/bin/activate
    cd ~/rag_project/scripts

    python rag_pipeline.py >> ../logs/rag_pipeline.log 2>&1
    ```

45. **Make the Script Executable:**

    ```bash
    chmod +x run_rag.sh
    ```

46. **Execute the Script:**

    ```bash
    ./run_rag.sh
    ```

    *This will run the RAG pipeline and append logs to `logs/rag_pipeline.log`.*

---

## **Final Considerations**

1. **Security:**
   - Ensure that Ollama's API is secured, especially if exposed beyond `localhost`.
   - Implement authentication mechanisms if necessary.

2. **Scalability:**
   - As your data grows, consider using more advanced FAISS indexing methods (e.g., `IVFFlat`, `HNSW`) for scalability.
   - Monitor GPU usage and consider upgrading VM resources if needed.

3. **Documentation:**
   - Keep detailed documentation of your setup, scripts, and any modifications for future reference and team collaboration.

4. **Backup:**
   - Regularly back up your data, embeddings, and FAISS index to prevent data loss.

5. **Maintenance:**
   - Periodically update your models and dependencies to benefit from performance improvements and security patches.

---

## **Conclusion**

By following this step-by-step guide, you've set up a robust **Retrieval-Augmented Generation (RAG)** system on your **GCP VM** with **Ollama** handling the language model via a Docker container. This setup allows you to:

- **Retrieve** relevant historical documents using FAISS.
- **Rerank** the retrieved documents for enhanced relevance.
- **Generate** accurate and coherent responses using the LLaMA model via Ollama.

Feel free to iterate on each component to further enhance the system's capabilities, such as integrating more sophisticated rerankers, expanding your data sources, and fine-tuning the generation prompts for better contextual understanding.

**If you encounter any challenges or need further assistance with specific components, don't hesitate to reach out!**

---

*Happy Coding!*