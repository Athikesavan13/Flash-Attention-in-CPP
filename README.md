# Flash Attention — CPU Reference Implementation

A single-file C++ demo of the Flash Attention algorithm (Dao et al., NeurIPS 2022), written to be read, not to win benchmarks. All four core ideas are present and tested:

- **Online softmax** with running rescaling
- **Matrix tiling** over K and V
- **Causal masking** (autoregressive / decoder attention)
- **Multi-head attention** over a packed Q/K/V layout

---

## Build & run

```bash
g++ -O2 -std=c++17 flash_attention.cpp -o flash_attention
./flash_attention
```

No dependencies beyond the C++ standard library.

---

## Why Flash Attention exists — the IO argument

### The bottleneck is memory bandwidth, not FLOPs

Modern GPUs can do ~300 TFLOPS (fp16) but their HBM (high-bandwidth memory) peaks at ~2 TB/s. At N=4096, d=64:

| Step | FLOPs | Bytes read/written (HBM) |
|---|---|---|
| Compute S = QKᵀ | 2 N² d | — |
| Write S to HBM | — | N² × 4 = **64 MB** |
| Read S for softmax | — | 64 MB |
| Write P to HBM | — | 64 MB |
| Read P for PV | — | 64 MB |
| **Total naive** | O(N²d) | **O(N²) ≈ 256 MB** per head |

Flash Attention never writes the N×N matrix:

| Step | Bytes read/written (HBM) |
|---|---|
| Read Q, K, V in tiles | O(Nd) per pass |
| Write O | O(Nd) |
| **Total Flash** | **O(Nd) ≈ 2 MB** per head |

At N=4096 that is a **~128× reduction in HBM traffic**, which is why the paper reports 2–4× end-to-end speedup even though FLOPs are unchanged. The GPU is not limited by how fast it can multiply; it is limited by how fast data moves between HBM and the compute cores.

### The online softmax trick

Softmax(x)ᵢ = exp(xᵢ − max(x)) / Σⱼ exp(xⱼ − max(x))

Normally this requires two passes over the full row: one to find the max, one to compute exp and normalize. Flash Attention does it in one streaming pass by maintaining:

```
m   := running max of scores seen so far
l   := Σ exp(sⱼ − m)   [running unnormalized denominator]
acc := Σ exp(sⱼ − m) × Vⱼ  [running unnormalized output]
```

When a new tile arrives with tile-max `m_new`:

```
correction = exp(m - m_new)   # rescale everything relative to the new max
l   ←  l   × correction  +  Σ_tile exp(sⱼ − m_new)
acc ←  acc × correction  +  Σ_tile exp(sⱼ − m_new) × Vⱼ
m   ←  m_new
```

After the last tile: `O = acc / l`. This is mathematically identical to a standard two-pass softmax — the rescaling preserves exact equivalence via `exp(a − m_new) = exp(a − m_old) × exp(m_old − m_new)`.

---

## What's in the code

| Function | Description |
|---|---|
| `naive_attention` | Standard O(N²) attention — ground truth for correctness checks |
| `flash_attention` | Tiled + online softmax. Supports causal masking. |
| `multi_head_flash_attention` | Slices packed Q/K/V into H heads, runs Flash Attention per head |
| `benchmark` | Measures best-of-N wall-clock time for any `Matrix → Matrix` fn |

---

## Benchmark results (Apple M-class / x86, N up to 2048)

```
N       Naive (ms)    Flash (ms)    Speedup    Naive N×N memory
128       1.08          0.85          1.3×         0.06 MB
256       4.29          3.39          1.3×         0.25 MB
512      16.90         13.60          1.2×         1.00 MB
1024     67.97         53.09          1.3×         4.00 MB
2048    268.54        208.79          1.3×        16.00 MB
```

The modest CPU speedup (1.2–1.3×) comes from better cache locality. On a GPU the gap widens to **2–4× at N=4096+** because HBM bandwidth is the actual bottleneck, not arithmetic.

---

## What a GPU kernel adds

This implementation shows the algorithm; a production CUDA kernel adds:

1. `__shared__` memory tiles (SRAM) — the actual "flash" in Flash Attention
2. Warp-level reductions (`__shfl_xor_sync`) for the online max/sum
3. Thread-block parallelism: one block per (batch, head, query-tile) tuple
4. FP16/BF16 arithmetic with FP32 accumulation
5. FlashAttention-2: query-tile outer loop, better work partitioning

---

## References

- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022. https://arxiv.org/abs/2205.14135
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024. https://arxiv.org/abs/2307.08691
- Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018. https://arxiv.org/abs/1805.02867
