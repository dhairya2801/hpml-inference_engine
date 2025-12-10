## Inference Optimization

In this project, we investigate the effect of various inference optimizations used in modern LLMs. Mainly, we were able to isolate each individual optimizations by writing specific implementation code.

Models: Qwen2.5-14B, Qwen3-0.6B

Compute: 1x L4, 2x A100s

Phase 1 [Completed]: Raw autoregressive inference without any optimizations

Phase 2 [Completed]: Addition of KV Cache to store previously generated token information, to avoid generating them on each iteration. This reduces the algorithmic complexity from O(n^2) to O(n).

Phase 3 [Completed]: Using mulitple input prompts and handling them through Static batching

Phase 4 [Completed]: Using multiple input prompts and handling them through Continuous batching

Phase 5 [Completed]: Analyzed Prefix Caching to handle similar input prompts

Phase 6 [Completed]: Analyzed Tensor Parallelism for inference on more than 1 GPUs.


