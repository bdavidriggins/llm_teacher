import re

def chunk_text(text, max_length=512):
    # Split text into paragraphs
    paragraphs = re.split(r'\n{2,}', text)
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_length:
            current_chunk += "\n\n" + para if current_chunk else para
        else:
            chunks.append(current_chunk)
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Apply chunking to all articles
import json

with open('wikipedia_articles.json', 'r') as f:
    articles = json.load(f)

chunked_data = {}
for title, content in articles.items():
    chunked_data[title] = chunk_text(content)

# Save chunked data
with open('chunked_wikipedia_articles.json', 'w') as f:
    json.dump(chunked_data, f)
