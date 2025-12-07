import time
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity


MODEL_PATH = "./huggingface/Qwen3-0.6B" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = "The meaning of life is"
MAX_NEW_TOKENS = 1000

def print_metrics(start_time, first_token_time, total_tokens, peak_mem):
    total_time = time.time() - start_time
    # Latency to First Token
    ttft = (first_token_time - start_time) * 1000 
    itl = ((total_time - (first_token_time - start_time)) / (total_tokens - 1)) * 1000 if total_tokens > 1 else 0
    tps = total_tokens / total_time
    
    print("\n" + "="*40)
    print(f"PHASE 1 ANALYSIS: NAIVE INFERENCE")
    print(f"="*40)
    print(f"Input Prompt: '{PROMPT}'")
    print(f"Generated Tokens: {total_tokens}")
    print(f"-"*40)
    print(f"Avg Latency per Token:      {itl:.2f} ms")
    print(f"Throughput:                 {tps:.2f} tokens/sec")
    print(f"Peak GPU Memory:            {peak_mem / 1024**3:.2f} GB")
    print(f"="*40 + "\n")

def run_naive_inference():
    print(f"Loading model from {MODEL_PATH} to {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    inputs = tokenizer(PROMPT, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        model(input_ids)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Starting Inference...")
    start_time = time.time()
    first_token_time = None
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for i in range(MAX_NEW_TOKENS):
                with record_function("model_forward_step"):

                    outputs = model(input_ids, use_cache=False)
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if i == 0:
                    first_token_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated()
    print_metrics(start_time, first_token_time, MAX_NEW_TOKENS, peak_mem)
    
    prof.export_chrome_trace("phase1_trace.json")
    #print("Profiling trace saved to 'phase1_trace.json'.")
    
    # Decode output
    print("Output text:", tokenizer.decode(input_ids[0]))

if __name__ == "__main__":
    run_naive_inference()
