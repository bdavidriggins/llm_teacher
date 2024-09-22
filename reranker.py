from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load reranker model and tokenizer
reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # Example model
tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

def rerank(query, candidates, top_k=5):
    inputs = []
    for candidate in candidates:
        inputs.append((query, candidate['text']))
    encoded = tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        scores = reranker_model(**encoded).logits.squeeze()
    # Attach scores to candidates
    for i, candidate in enumerate(candidates):
        candidate['score'] = scores[i].item()
    # Sort candidates by score
    sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    return sorted_candidates[:top_k]
