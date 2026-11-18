# Mixture of Experts (MoE): one layer's forward pass, kernel by kernel

[`moe-forward-unfused.html`](moe-forward-unfused.html) is a self-contained, interactive draft that draws every tensor one MoE layer touches in a forward pass, together with the interface of every kernel that reads or writes it. It assumes no fusion: each logical step is its own kernel launch, and a dashed line marks every launch boundary.

| Phase | Launches | Kernels |
|---|---|---|
| ① Routing | 1–4 | `router_gemm`, `softmax_rows`, `topk_rows`, `renormalize` (optional) |
| ② Permute | 5–8 | `expert_histogram`, `exclusive_scan`, `build_permutation`, `permute_rows` |
| ③ Expert FFNs | 9–11 | `grouped_gemm` (gate and up), `silu_and_mul`, `grouped_gemm` (down) |
| ④ Unpermute and combine | 12 | `unpermute_combine` |

One square is one element, drawn to scale with toy sizes: T = 6 tokens, H = 8, E = 4 experts, top-k = 2, I = 5. Color is the kind of data: green for the input, pink for parameters, blue for float activations and yellow for int32 routing metadata, with each value printed in its square. 1-D tensors are rounded strips, 2-D tensors are bracketed grids, and the 3-D expert weights are drawn as one slice per expert.

## Viewing

GitHub shows HTML files as source, so download the file (or clone the repo) and open it in a browser. The page loads its fonts from Google Fonts and otherwise works offline.

- Hover any square to trace its token through every launch, or a weight slice to trace its expert. Click to pin; Esc clears. The chips in the sticky bar do the same.
- Each launch shows what it reads, the kernel, what it writes, and a C-style signature. The table at the bottom lists every tensor with its shape, dtype, writer and readers.

## Scope and caveats

- One MoE layer, forward pass, one GPU, with dtypes typical for BF16 inference. Weights are drawn as [in, out] so the math reads X · W; checkpoints usually store [out, in].
- This is the unfused reference picture. The page ends with notes on what real implementations change: fusion in vLLM's Triton path, padding and tile-to-expert tables, masked layouts for decode, expert parallelism (DeepEP), shared experts, DeepSeek-V3 routing and FP8 scales.
- The routing probabilities in the toy example are made up.
