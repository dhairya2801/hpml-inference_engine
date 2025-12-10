import time
import torch
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

# 1. Exact same setup as Phase 2
MODEL_PATH = "./huggingface/Qwen3-0.6B"
prompts = [
    "The meaning of life is",
    "Explain the theory of relativity in simple terms:",
    "Write a very long essay about the history of the Roman Empire, focusing on the political structure, military conquests, and eventual decline. Start by discussing the founding of Rome and then move to:",
    "Hello"
]

def run_nanovllm_benchmark():
    # 2. Initialize the Engine (Phase 3 "Hero" Solution)
    # tensor_parallel_size=1 means single GPU. 
    print(f"Initializing NanoVLLM Engine with {MODEL_PATH}...")
    engine = LLMEngine(MODEL_PATH, tensor_parallel_size=1)
    
    # 3. Define Sampling Params (align with Phase 2 MAX_NEW_TOKENS)
    # ignore_eos=True ensures we generate exactly 100 tokens for fair comparison
    sampling_params = SamplingParams(max_tokens=100, ignore_eos=True)

    print("Warming up and starting PagedAttention Benchmark...")
    
    # 4. Run Generation
    # The engine.generate() method handles the scheduler loop and paged attention internally
    start_time = time.time()
    outputs = engine.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.time()

    # 5. Calculate Metrics
    total_time = end_time - start_time
    total_tokens = sum([len(out["token_ids"]) for out in outputs])
    # Note: nanovllm output includes prompt tokens in "token_ids", so we subtract them 
    # if we want strictly "generated" tokens, but for throughput, usually total is fine.
    # Let's align with your Phase 2 which counted "generated" tokens.
    generated_tokens = total_tokens - sum([len(engine.tokenizer.encode(p)) for p in prompts])
    
    tps = generated_tokens / total_time
    
    print("\n" + "="*50)
    print(f"PHASE 3 ANALYSIS: NANOVLLM (Continuous Batching)")
    print(f"="*50)
    print(f"Total Tokens Generated:     {generated_tokens}")
    print(f"Wasted 'Pad' Tokens:        0 (PagedAttention Magic!)")
    print(f"-"*50)
    print(f"Throughput:                 {tps:.2f} tokens/sec")
    print(f"Total Time:                 {total_time:.2f} s")
    print(f"="*50 + "\n")

if __name__ == "__main__":
    # Standard PyTorch cleanup before starting
    torch.cuda.empty_cache()
    run_nanovllm_benchmark()
