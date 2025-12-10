import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./huggingface/Qwen3-0.6B" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100  

prompts = [
    "The meaning of life is",                               # Short
    "Explain the theory of relativity in simple terms:",    # Medium
    "Write a very long essay about the history of the Roman Empire, focusing on the political structure, military conquests, and eventual decline. Start by discussing the founding of Rome and then move to:", # Long
    "Hello"                                                 # Very Short
]

def print_metrics(start_time, total_generated_tokens, peak_mem, padding_tokens):
    total_time = time.time() - start_time
    tps = total_generated_tokens / total_time
    latency_per_token = (total_time / total_generated_tokens) * 1000
    
    print("\n" + "="*50)
    print(f"ANALYSIS: STATIC BATCHING (Batch Size: {len(prompts)})")
    print(f"="*50)
    print(f"Total Tokens Generated:     {total_generated_tokens}")
    print(f"Wasted 'Pad' Tokens:        {padding_tokens} (Memory wasted on zeros)")
    print(f"-"*50)
    print(f"Avg Latency per Token:      {latency_per_token:.2f} ms")
    print(f"Throughput:                 {tps:.2f} tokens/sec")
    print(f"Peak GPU Memory:            {peak_mem / 1024**3:.2f} GB")
    print(f"="*50 + "\n")

def run_static_batching():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    print("Tokenizing and padding batch...")
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_pad_tokens = (input_ids == tokenizer.pad_token_id).sum().item()
    print(f"Batch prepared. Input shape: {input_ids.shape}. Padding tokens: {num_pad_tokens}")

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"Starting Static Batch Inference for {MAX_NEW_TOKENS} steps...")
    start_time = time.time()
    
    past_key_values = None
    generated_tokens_count = 0

    with torch.no_grad():
        for i in range(MAX_NEW_TOKENS):
            if past_key_values is None:
                outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
            else:
                attention_mask = torch.cat([attention_mask, torch.ones((len(prompts), 1), device=DEVICE)], dim=-1)
                outputs = model(input_ids[:, -1:], attention_mask=attention_mask, use_cache=True, past_key_values=past_key_values)

            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            generated_tokens_count += len(prompts) # We generated 1 token for each sequence in batch

    peak_mem = torch.cuda.max_memory_allocated()
    
    # In static batching, 'Hello' keeps generating nonsense after it's done while waiting for the 'Essay' prompt.
    print_metrics(start_time, generated_tokens_count, peak_mem, num_pad_tokens)

    print("Example Output (Shortest Prompt):")
    print(tokenizer.decode(input_ids[3], skip_special_tokens=True))

if __name__ == "__main__":
    run_static_batching()
