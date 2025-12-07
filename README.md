## Inference Optimization

In this project, we investigate the effect of various inference optimizations used in modern LLMs. All the experiments are performed using the Qwen3-0.6B model and on NVIDIA L4 GPU. 

Phase 1: Raw autoregressive inference without any optimizations
Phase 2: Addition of KV Cache to store previously generated token information, to avoid generating them on each iteration. This reduces the algorithmic complexity from O(n^2) to O(n).
