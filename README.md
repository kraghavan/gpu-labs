# gpu-labs

Scripts behind the GPU/inference experiments on
[kraghavan.ca](https://kraghavan.ca) — the "LLM Inference from First
Principles" series. Each subfolder maps to one post: hardware used, models
tested, what the scripts actually measure, and (where relevant) what broke
along the way and how it got fixed.

These are real scripts that ran on rented GPUs, not sanitized demos — if
something looks rough, it probably is, and the corresponding blog post
usually explains why. If you spot something that could be done better,
open an issue or a PR.

## Posts

- [`vllm-v030-fast-start-watermark/`](./vllm-v030-fast-start-watermark) —
  vLLM v0.30.0's Fast Start (weight-cache daemon + CUDA IPC restart) and
  text watermarking (Gumbel-max), measured on a Lambda Cloud A100 at two
  model sizes.
