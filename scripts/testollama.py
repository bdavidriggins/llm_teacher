import requests

def ask_ollama(question):
    url = "http://localhost:11434/ask"  # Replace with the correct Ollama API endpoint
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "prompt": question
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        return response.json().get('response')
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    question = "What is the capital of France?"
    answer = ask_ollama(question)
    print(f"Ollama says: {answer}")
