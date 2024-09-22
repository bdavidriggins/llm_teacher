from flask import Flask, request, jsonify, render_template, Response
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time

# Initialize the Flask application
app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None

# Only load the model in the main process to avoid duplication
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "meta-llama/Llama-2-13b-hf"

    # Ensure the token is set (either via environment variable or directly)
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "your_access_token_here")

    # Load the tokenizer with the updated 'token' parameter
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Disable gradient computation for inference
    torch.set_grad_enabled(False)

    # Load the model without quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use float16 to reduce memory usage
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
        token=hf_token,  # Updated from use_auth_token to token
    )

    # Resize model embeddings if new tokens were added
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # Set the model to evaluation mode
    model.eval()

def generate_long_response_stream(prompt, model, tokenizer, max_new_tokens=2000, chunk_size=100):
    """
    Generator function to yield chunks of generated text.

    Args:
        prompt (str): The initial prompt.
        model: The language model.
        tokenizer: The tokenizer.
        max_new_tokens (int): Total number of tokens to generate.
        chunk_size (int): Number of tokens to generate per chunk.

    Yields:
        str: Chunks of generated text in SSE format.
    """
    generated_text = prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    total_generated = 0

    while total_generated < max_new_tokens:
        current_chunk_size = min(chunk_size, max_new_tokens - total_generated)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=current_chunk_size,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        new_tokens = outputs[0, input_ids.shape[-1]:]
        if new_tokens.size(0) == 0:
            break

        new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Remove any unwanted tags like [/INST]
        clean_text = new_text.replace("[/INST]", "").strip()
        
        yield f"data: {clean_text}\n\n"

        generated_text += clean_text

        input_ids = torch.cat([input_ids, new_tokens.unsqueeze(0)], dim=-1)

        if input_ids.size(-1) > model.config.max_position_embeddings:
            input_ids = input_ids[:, -model.config.max_position_embeddings:]

        total_generated += current_chunk_size

        if tokenizer.eos_token_id in new_tokens:
            break

        time.sleep(0.1)  # Optional: Simulate delay for better streaming effect

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    input_text = request.json.get("input_text", "")
    if not input_text:
        return jsonify({"response": "Error: No input provided."}), 400

    # Format the prompt according to LLaMA 2's expected format
    system_prompt = "You are a helpful assistant."
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{input_text}"

    try:
        return Response(
            generate_long_response_stream(prompt, model, tokenizer, max_new_tokens=2000, chunk_size=100),
            mimetype='text/event-stream'
        )
    except Exception as e:
        return jsonify({"response": f"Error during generation: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', use_reloader=False)
