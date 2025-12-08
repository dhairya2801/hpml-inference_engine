import time
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

MODEL_PATH = "./huggingface/Qwen3-0.6B"

# A long shared context
LONG_CONTEXT = "The history of computing began with the abacus... " * 50
QUESTION_1 = LONG_CONTEXT + "Question: What is the first device? Answer:"
QUESTION_2 = LONG_CONTEXT + "Question: Summarize the text. Answer:"

def run_prefix_caching_test():
    engine = LLMEngine(MODEL_PATH, tensor_parallel_size=1)
    sampling_params = SamplingParams(max_tokens=20, temperature=1e-6)

    print("\n=== REQUEST 1 (Cold Cache) ===")
    t0 = time.time()
    engine.generate([QUESTION_1], sampling_params, use_tqdm=False)
    t1 = time.time()
    print(f"Time Taken: {t1 - t0:.4f} seconds")

    print("\n=== REQUEST 2 (Warm Cache / Shared Prefix) ===")
    t0 = time.time()
    
    engine.generate([QUESTION_2], sampling_params, use_tqdm=False)
    t1 = time.time()
    print(f"Time Taken: {t1 - t0:.4f} seconds")
    

if __name__ == "__main__":
    run_prefix_caching_test()
