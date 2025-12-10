import time
import os
import torch
import gc
from huggingface_hub import snapshot_download
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import reset_context

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
LOCAL_MODEL_DIR = "./huggingface/Qwen2.5-14B-Instruct"

BATCH_SIZE = 64
PROMPT_LEN = 50 
GEN_LEN = 50

def download_model():
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"Model not found at {LOCAL_MODEL_DIR}. Downloading {MODEL_ID}")
        try:
            snapshot_download(
                repo_id=MODEL_ID,
                local_dir=LOCAL_MODEL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("Download complete.")
        except Exception as e:
            print(f"Download failed: {e}")
            exit(1)
    else:
        print(f"Model found at {LOCAL_MODEL_DIR}")

def run_experiment(tp_size):
    print(f"\n" + "="*50)
    print(f"   RUNNING EXPERIMENT WITH TP={tp_size}")
    print(f"="*50)
    
    try:
        # Force a fresh cleanup before loading
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Loading {MODEL_ID} on {tp_size} GPU(s)...")
        engine = LLMEngine(LOCAL_MODEL_DIR, tensor_parallel_size=tp_size)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\nTP={tp_size} crashed with OOM (Out of Memory)")
            return None
        else:
            raise e

    base_prompt = "Explain the history of the Roman Empire in detail."
    prompts = [base_prompt] * BATCH_SIZE
    sampling_params = SamplingParams(max_tokens=GEN_LEN, ignore_eos=True)

    print(f"Running Inference (Batch Size: {BATCH_SIZE}, Gen Len: {GEN_LEN})...")
    
    start_time = time.time()
    engine.generate(prompts, sampling_params, use_tqdm=True)
    end_time = time.time()
    
    del engine
    reset_context()
    gc.collect()
    torch.cuda.empty_cache()
    
    total_time = end_time - start_time
    total_tokens = BATCH_SIZE * GEN_LEN
    throughput = total_tokens / total_time
    latency = (total_time * 1000) / GEN_LEN # avg latency per token for the batch
    
    print(f"-"*50)
    print(f"RESULTS (TP={tp_size}):")
    print(f"Total Time:      {total_time:.2f} s")
    print(f"Throughput:      {throughput:.2f} tokens/sec")
    print(f"Batch Latency:   {latency:.2f} ms/token")
    print(f"-"*50)
    
    return throughput

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True, help="Tensor Parallel Size (1 or 2)")
    args = parser.parse_args()

    download_model()

    score = run_experiment(tp_size=args.tp)

    if score:
        print(f"SUCCESS: TP={args.tp} Throughput = {score:.2f} tok/s")
    else:
        print(f"FAILURE: TP={args.tp} crashed")
