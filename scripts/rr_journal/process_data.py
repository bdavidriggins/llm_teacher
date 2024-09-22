import os

# Define the relative path to the source file
text_file_path = '../../sources/journal_roberts_rangers.txt'

def preprocess_and_chunk(text, max_length=1000):
    """
    Preprocesses and chunks the text into segments of up to `max_length` tokens.
    """
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

def main():
    # Check if the file exists
    if not os.path.exists(text_file_path):
        print(f"File not found: {text_file_path}")
        return
    
    # Read and preprocess the file
    with open(text_file_path, 'r') as file:
        text = file.read()
    
    # Chunk the text
    chunks = preprocess_and_chunk(text)
    print(f"Total chunks created: {len(chunks)}")

    # Save the chunks for later embedding processing
    with open('../../sources/journal_roberts_rangers_chunks.txt', 'w') as chunk_file:
        for chunk in chunks:
            chunk_file.write(chunk + '\n\n')

if __name__ == '__main__':
    main()