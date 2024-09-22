import os
import json

# Path to the sources directory
sources_dir = '../../sources/'
# Path to the processed files log
processed_files_log = '../../faiss/processed_files.json'

def load_processed_files():
    """Load the list of processed files."""
    if os.path.exists(processed_files_log):
        with open(processed_files_log, 'r') as file:
            return json.load(file)
    return []

def save_processed_files(processed_files):
    """Save the list of processed files."""
    with open(processed_files_log, 'w') as file:
        json.dump(processed_files, file, indent=2)

def preprocess_and_chunk(text, max_length=1000):
    """Preprocesses and chunks the text into segments of up to `max_length` tokens."""
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

def process_new_files():
    """Process newly added files in the sources directory."""
    processed_files = load_processed_files()
    
    # Get all text files in the sources directory
    files_to_process = [f for f in os.listdir(sources_dir) if f.endswith('.txt') and f not in processed_files]
    
    if not files_to_process:
        print("No new files to process.")
        return
    
    for filename in files_to_process:
        file_path = os.path.join(sources_dir, filename)
        with open(file_path, 'r') as file:
            text = file.read()
        
        # Chunk the text
        chunks = preprocess_and_chunk(text)
        print(f"Total chunks created from {filename}: {len(chunks)}")

        # Save the chunks for later embedding processing
        chunks_file_path = os.path.join(sources_dir, f"{filename}_chunks.txt")
        with open(chunks_file_path, 'w') as chunk_file:
            for chunk in chunks:
                chunk_file.write(chunk + '\n\n')
        
        # Mark this file as processed
        processed_files.append(filename)
    
    save_processed_files(processed_files)

if __name__ == '__main__':
    process_new_files()
