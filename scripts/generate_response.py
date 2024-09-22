import requests
import json

class OllamaClient:
    def __init__(self, api_url='http://localhost:11434/generate'):
        self.api_url = api_url
    
    def generate_response(self, prompt, context, max_tokens=512, temperature=0.7):
        full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
        payload = {
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return ""
