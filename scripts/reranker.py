from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-TinyBERT-L-6'):
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
