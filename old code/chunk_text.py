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
