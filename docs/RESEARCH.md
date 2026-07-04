# Psi Research Log

A running catalog of techniques I considered for Psi, aimed at one goal: the smallest model (params first, then bits) that still clears the TinyStories bar (see [EVAL.md](EVAL.md)). For each technique I note what it is, where it comes from, and a verdict for my regime, which is a sub-1M, dense, from-scratch transformer on a custom C++/Metal stack. A lot of famous techniques target large MoE, long context, or inference, and they don't transfer. The verdict column says why.

> Frontier check: I last swept the literature on 2026-06-18. My assistant training cutoff is Jan 2026, so anything newer I found via web search and marked _(post-cutoff)_. I re-sweep periodically and update this date.

---

## Incorporate: high value for sub-1M

| Technique | Source | What / why it fits | Status |
|---|---|---|---|
| Multi-head attention | standard | multiple attention subspaces at fixed proj params | DONE (`n_heads`, grad-checked) |
| Small-BPE tokenizer | (mine) | tiny vocab so the embedding doesn't eat the budget | DONE (`bpe.hpp`, 3.5 c/tok) |
| Tied / shared embeddings | [MobileLLM](https://arxiv.org/abs/2402.14905) | input=output embedding, big param saving | DONE |
| Block-wise weight sharing | [MobileLLM](https://arxiv.org/abs/2402.14905) | reuse a block across layer positions, so depth at near-zero param cost. The best capability-per-param lever I found. | TODO |
| Deep-and-thin | [MobileLLM](https://arxiv.org/abs/2402.14905) | more layers, smaller `d` beats wide-shallow at fixed params | config, TODO |
| SwiGLU MLP | [GLU variants](https://arxiv.org/abs/2002.05202), MobileLLM | better capability-per-param than GELU MLP | ops ready (`silu`); wire in |
| WSD LR scheduler | [MiniCPM](https://arxiv.org/abs/2404.06395) | warmup, stable, decay; beats flat/cosine, enables ckpt reuse | TODO |
| Small-model scaling law (D/N ≈ 192) | [MiniCPM](https://arxiv.org/abs/2404.06395) | tiny models need ~10× more tokens/param than Chinchilla, so my early runs were undertrained | budget guideline |
| Knowledge distillation (teacher→student) | [KD survey](https://arxiv.org/abs/2402.13116), [KD-SLM](https://arxiv.org/abs/2509.26497) | a weak tiny student gains a lot from a stronger teacher's full distribution | later |
| Data quality / synthetic curation | [phi / Textbooks](https://arxiv.org/abs/2306.11644), [phi-1.5](https://arxiv.org/abs/2309.05463) | curated "textbook-quality" data beats scale at small sizes (TinyStories is this lineage) | ongoing |

## The capability-per-bit track: ternary

My edge here: train dense, then compress to ternary {−1,0,+1} (~1.58-bit) on my own kernel ([GPU_KERNELS.md](GPU_KERNELS.md)). Recent literature sharpened the recipe.

| Technique | Source | Use |
|---|---|---|
| BitNet b1.58 (base recipe) | [Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) | weights to {−1,0,1} via abs-mean scale, shadow fp16 weights, STE through the quantizer, 8-bit activations (abs-max) |
| BitNet b1.58 Reloaded | [arXiv:2407.09527](https://arxiv.org/abs/2407.09527) | confirms ternary QAT works on small nets |
| "16-to-1.58" continual QAT _(post-cutoff)_ | [arXiv:2502.11895](https://arxiv.org/abs/2502.11895) | train fp16/fp32 first, then transition to ternary QAT. Nearly matches full precision (2 to 3 pt drop). This is exactly my reproduce-then-replace order, so it's validated. |
| Double hidden size for ternary _(post-cutoff)_ | [arXiv:2502.11895](https://arxiv.org/abs/2502.11895) | when going ternary, widen to recover capacity |
| TernaryLM, adaptive layer-wise scaling _(post-cutoff, Feb 2026)_ | [arXiv:2602.07374](https://arxiv.org/abs/2602.07374) | per-layer (not fixed per-tensor) scale, a quality upgrade over vanilla BitNet |
| HESTIA, Hessian-guided QAT _(post-cutoff; source unverified)_ | (flagged in a Jan-2026 result) | +4 to 5% over hard ternary QAT. Verify before relying on it. |

Ternary plan: find the smallest dense fp32 config that clears the bar, apply 16-to-1.58 QAT (with TernaryLM-style adaptive scaling, widen hidden), then measure capability-per-bit. A passing ~400K-param model at 1.58 bits comes out around 80 KB.

## Consider: medium value, later

| Technique | Source | Note |
|---|---|---|
| GQA / MQA | [GQA](https://arxiv.org/abs/2305.13245), [MQA](https://arxiv.org/abs/1911.02150) | share K/V across heads, fewer attn params |
| RoPE | [RoFormer](https://arxiv.org/abs/2104.09864) | drops learned pos-emb params and gives better positions |
| Multi-Token Prediction | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | predict next-N tokens for richer signal; heads dropped at inference |
| μP / hyperparameter transfer | [Tensor Programs V](https://arxiv.org/abs/2203.03466), MiniCPM | tune on a proxy, transfer; I'm already tiny so I can sweep directly |
| MLA, low-rank projections | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | the param-saving factorization only; KV-cache compression is for long-context inference, so skip that part |
| Self-distillation (best-of-N + filter) | [arXiv:2604.01193](https://arxiv.org/abs/2604.01193) | needs a cheap correctness signal (code has tests; stories don't) and a capable base model. Post-training polish only, with a grader/reward-model in the loop. |

## Reject for my regime, and why

| Technique | Why not |
|---|---|
| MoE / DeepSeekMoE | adds total params, and my metric is total bits, so it's the wrong direction. It buys capability-per-FLOP; I'm param-bound, not FLOP-bound. |
| FP8 training | M1 matrix units are precision-independent (I measured it, [GPU_KERNELS.md](GPU_KERNELS.md)); ternary is my more aggressive version. FP8 is a Hopper/GH200 lever. |
| Sparse attention (NSA / DSA) | for long context; TinyStories is short (~256 tokens). |
| SSM / Mamba | the linear-attention win shows at long context; big arch change for marginal gain here. |
| GRPO / RL (R1-style) | for reasoning/alignment; TinyStories is fluency, with no reward to optimize. |

---

## Sequencing: how the verdicts become work

1. First, and free: WSD scheduler, deep-and-thin configs, scaling-law-aware token budget.
2. Next arch pass (grad-checked): SwiGLU, then block-wise weight sharing, then RoPE / GQA.
3. Big levers: knowledge distillation, then ternary QAT (16-to-1.58 plus adaptive scaling) for the capability-per-bit record.
4. Post-training polish, much later: self-distillation / rejection sampling with a grader.
