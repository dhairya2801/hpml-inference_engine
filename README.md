# Efficient LLM Inference Engine (nano‑vLLM)

This repository contains a **from‑scratch, research‑oriented LLM inference engine** built to study and reproduce the core systems ideas behind modern high‑throughput LLM serving frameworks (e.g., vLLM). The project isolates and implements key inference optimizations in *pure Python*, profiles them, and empirically demonstrates why **LLM inference is fundamentally memory‑bound rather than compute‑bound**.

The work was developed as part of an HPML course project, with results presented in a final presentation and the full implementation hosted in this repository.

---

## 1. Project Description

Serving large language models efficiently is challenging due to:

* Rapidly growing **KV‑cache memory footprint** with long context lengths
* Severe inefficiencies from **static batching and padding**
* High **memory bandwidth pressure** during attention and decoding
* Communication overheads in multi‑GPU inference

In this project, we design a minimal inference engine that incrementally adds and evaluates the following optimizations:

* **KV Cache management** for autoregressive decoding
* **Paged Attention** to avoid KV‑cache fragmentation
* **Continuous Batching** to eliminate padding waste
* **Prefix Caching** to reuse shared prompt prefixes
* **Tensor Parallelism** to study scaling limits
* **CUDA Graphs** to reduce kernel launch overhead (under static‑shape constraints)

We validate each technique using controlled experiments on **Qwen‑0.6B** and **Qwen‑14B** models across different GPU configurations (L4 and A100), and quantify throughput, latency, memory usage, and wasted computation.

---

## 2. Project Milestones and Completion Status

| Milestone | Description                                | Status      |
| --------- | ------------------------------------------ | ----------- |
| Phase 1   | Baseline naive autoregressive inference    | ✅ Completed |
| Phase 2   | Static batching with KV cache              | ✅ Completed |
| Phase 3   | Continuous batching engine                 | ✅ Completed |
| Phase 4   | Paged Attention (block‑based KV cache)     | ✅ Completed |
| Phase 5   | Prefix caching with hash‑based block reuse | ✅ Completed |
| Phase 6   | Tensor Parallelism (TP=1 vs TP=2)          | ✅ Completed |
| Phase 7   | CUDA Graphs integration                    | ✅ Completed |
| Phase 8   | Profiling, benchmarking, and analysis      | ✅ Completed |

All milestones were implemented, profiled, and discussed in the final presentation. 

---

## 3. Repository and Code Structure

The repository is organized to mirror the logical components of an inference engine:

```
.
├── engine/
│   ├── model.py          # Model loading and forward pass wrappers
│   ├── kv_cache.py       # KV cache data structures and management
│   ├── paged_kv.py       # Block‑based (paged) KV cache implementation
│   ├── scheduler.py      # Continuous batching scheduler (prefill/decode queues)
│   ├── prefix_cache.py   # Hash‑based prefix caching logic
│   ├── tp.py             # Tensor parallel abstractions and NCCL integration
│   └── cuda_graphs.py    # CUDA Graph capture and replay logic
│
├── benchmarks/
│   ├── latency.py        # Latency and per‑token timing benchmarks
│   ├── throughput.py     # Throughput measurement scripts
│   └── memory.py         # GPU memory usage tracking
│
├── experiments/
│   ├── static_vs_continuous.py
│   ├── prefix_cache_eval.py
│   └── tp_scaling.py
│
├── plots/
│   ├── throughput_comparison.png
│   ├── prefix_cache_latency.png
│   └── tp_scaling.png
│
├── requirements.txt
└── README.md
```

The code is intentionally **modular and explicit**, prioritizing clarity and debuggability over production‑level abstraction. Each optimization can be enabled or disabled independently to allow clean ablation studies.

---

## 5. Results and Observations

### Throughput Improvements

| Configuration                   | Throughput (tok/s) | Improvement |
| ------------------------------- | ------------------ | ----------- |
| Naive inference                 | ~4.4               | 1×          |
| Static batching                 | ~82.3              | ~19×        |
| Continuous batching (nano‑vLLM) | ~176.1             | **~40×**    |

Continuous batching eliminates padding waste entirely and keeps the GPU busy by immediately scheduling new requests when sequences finish.

---

### Static vs Continuous Batching

| Metric           | Static     | Continuous      |
| ---------------- | ---------- | --------------- |
| Tokens generated | 400        | 344             |
| Padded tokens    | 100        | **0**           |
| Throughput       | 82.3 tok/s | **176.1 tok/s** |
| Peak GPU memory  | 1.2 GB     | 1.95 GB         |

**Key insight:** modest memory overhead is well worth the large throughput gains.

---

### Prefix Caching Impact

| Scenario              | Latency     |
| --------------------- | ----------- |
| Cold request          | ~2.19 s     |
| Warm (cached) request | **~0.30 s** |

Prefix caching enables reuse of pre‑computed KV blocks when prompts share prefixes, yielding up to **7× latency reduction** for repeated or overlapping prompts.

---

### Tensor Parallelism Scaling

| GPUs   | Throughput (tok/s) |
| ------ | ------------------ |
| TP = 1 | ~1331              |
| TP = 2 | ~1544              |

While throughput increases with tensor parallelism, scaling is **far from ideal** due to NCCL all‑reduce communication overhead dominating the reduced compute cost.

---

### Key Takeaways

* LLM inference is **memory‑bandwidth bound**, not compute‑bound
* KV cache management dominates both memory usage and performance
* Continuous batching provides the single largest real‑world speedup
* Prefix caching is critical for workloads with overlapping prompts
* Tensor parallelism has diminishing returns without careful communication optimization

These results align closely with observations from production‑grade LLM serving systems and validate the core architectural choices behind modern inference engines fileciteturn0file0.
