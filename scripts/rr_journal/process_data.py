# Define the relative path to the source file
text_file_path = '../../sources/journal_roberts_rangers.txt'

def preprocess_and_chunk(text, max_length=1000):
    paragraphs = text.split('\n\n')  # Split by paragraphs
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= max_length:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Read the file
with open(text_file_path, 'r') as file:
    text = file.read()

# Chunk the text
chunks = preprocess_and_chunk(text)
print(f"Total chunks created: {len(chunks)}")
