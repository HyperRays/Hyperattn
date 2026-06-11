## Engram (JAX)

`engram_jax/` is an optimized JAX implementation of the Engram architecture demo
(n-gram hash embeddings injected into the residual stream through a gated lookup and
a short causal conv). `engram_torch/reference.py` keeps the original torch math for
parity testing. Design notes:

- N-gram hashing is integer-only preprocessing, so it runs vectorized in host NumPy
  int64 (jax-metal has no int64) and only int32 hash ids cross to the device.
- The float path (`engram_jax.engram_forward`) is a single jittable pure function:
  the per-group key projections run as one stacked einsum, per-group RMSNorms use
  stacked weights, the multi-head embedding is one fused gather, and the grouped
  dilated Conv1d is unrolled into kernel_size shifted scaled adds (grouped conv does
  not lower well on Metal).

```bash
uv run python -m engram_jax.demo --length 2048           # offline, stub vocab
uv run python -m engram_jax.demo --tokenizer deepseek-ai/DeepSeek-V3
uv run python -m unittest tests.test_engram_jax_parity   # parity vs torch reference
```

## Sequence-Length Benchmarks

This project is pinned to Python 3.12 with `jax-metal`, `jax==0.4.34`, and `jaxlib==0.4.34` so the pure JAX backend can run on Apple Metal. On Apple Silicon, `uv run` should report `jax backend: METAL` for `--backend jax`.

Run synthetic token-length sweeps without downloading a dataset:

```bash
uv run python benchmarks/sequence_length.py \
  --backend torch \
  --device cuda \
  --dtype fp16 \
  --lengths 256,512,1024,2048,4096 \
  --batch-size 1 \
  --warmup 5 \
  --iters 20 \
  --measure both \
  --csv benchmark_results/sequence_length.csv \
  --json benchmark_results/sequence_length.json
```

The benchmark reports latency, tokens/sec, CUDA peak memory when available, and a fitted power law of the form:

```text
metric ~= coefficient * tokens^exponent
```

Use `--measure full` for the end-to-end LM including the vocabulary projection, and `--measure backbone` to focus on embeddings, local attention, span-hypergraph blocks, compressed memory, and final norm.

Compare against the side-by-side JAX implementation with:

```bash
uv run python benchmarks/sequence_length.py \
  --backend jax \
  --jax-attention windowed \
  --jax-span materialized \
  --compile \
  --lengths 256,512,1024,2048,4096 \
  --batch-size 1 \
  --warmup 5 \
  --iters 20 \
  --measure both
```

Use `--backend jax --compile` for the side-by-side pure JAX implementation. PyTorch remains available through `--backend torch` for reference comparisons.

The JAX local-attention block supports three implementations:

```bash
--jax-attention manual
--jax-attention windowed
--jax-attention chunked
```

`manual` uses the explicit score/mask/softmax path. `windowed` uses `jax.nn.dot_product_attention` with causal local-window settings. On Apple Metal with fp16 inputs, the windowed attention calculation is upcast to fp32 internally and cast back to fp16 to avoid a Metal compiler issue in the fp16 attention path. `chunked` splits the sequence into `local_window`-sized chunks where each query chunk attends to itself and the previous chunk, making the layer O(T * window) in time and memory instead of O(T^2); it is the fastest option at long sequence lengths and the only backend whose backward pass compiles on Apple Metal (`windowed` gradients hit an unsupported `dot_general` in the Metal compiler), so prefer it for `--mode forward_backward`.

The JAX span-hypergraph block also supports:

```bash
--jax-span materialized
--jax-span fused
--jax-span einsum_fused
```

`materialized` builds the concatenated span tensor and performs one projection. `fused` accumulates projected widths without materializing the concat tensor. `einsum_fused` stacks widths and contracts with the projection in one einsum. On current JAX Metal runs, `materialized` is the default because it benchmarks fastest.
