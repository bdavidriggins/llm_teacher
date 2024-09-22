import os

# Corrected relative path
text_file_path = '../../sources/journal_roberts_rangers.txt'
absolute_path = os.path.abspath(text_file_path)

print(f"Looking for file at: {absolute_path}")

# Attempt to open the file
with open(text_file_path, 'r') as file:
    text = file.read()
