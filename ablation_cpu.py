"""Deterministic CPU ablation sweep for the 100M deep-Muon checkpoint.

Run with: JAX_PLATFORMS=cpu uv run python ablation_cpu.py
Logs every block to stdout and writes ablation_cpu_results.json incrementally.
"""
import functools
import json
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from datasets import load_dataset

import jax_model
from jax_model.config import EfficientHGConfig
from jax_model.analysis import _ablate_block, block_kind, _hca_null_mass
from jax_model.language_model import loss as jloss
from jax_model.rope import precompute_rope_cache

CKPT = sys.argv[1] if len(sys.argv) > 1 else "best_jax_span_hypergraph_lm.pkl"
OUT = os.path.splitext(os.path.basename(CKPT))[0] + "_ablation.json"
LENGTHS = (512, 1024, 2048)
AB = "chunked"
t0 = time.time()
jax.config.update('jax_platform_name', 'cpu')


def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


log(f"backend = {jax.default_backend()}  (expect cpu)")
ck = pickle.load(open(CKPT, "rb"))
params = jax.tree.map(jnp.asarray, ck["params"])
cfg = EfficientHGConfig(**ck["config"])
nblocks = len(params["blocks"])
log(f"loaded step {ck['step']} val {ck['val_loss']:.4f}; {nblocks} blocks")

tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
eos = tok.eos_token_id
buf = []
for row in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True):
    t = row.get("text")
    if isinstance(t, str) and t:
        buf.extend(tok.encode(t, add_special_tokens=False))
        buf.append(eos)
    if len(buf) >= max(LENGTHS) + 1:
        break
stream = np.asarray(buf[: max(LENGTHS) + 1], dtype=np.int32)  # one fixed text; nested prefixes
log("text ready")


@functools.partial(jax.jit, static_argnames=())
def loss_jit(p, x, y):
    return jloss(p, x, y, cfg, attention_backend=AB)


results = {"step": int(ck["step"]), "val": float(ck["val_loss"]), "lengths": {}}
for L in LENGTHS:
    idx = jnp.asarray(stream[:L][None])
    tgt = jnp.asarray(stream[1 : L + 1][None])
    tc = time.time()
    base = float(loss_jit(params, idx, tgt))
    log(f"=== L={L} ({L // cfg.compression_block} mem blocks) base loss {base:.4f}  (compile+forward {time.time()-tc:.1f}s) ===")
    deltas = []
    for i in range(nblocks):
        ti = time.time()
        d = float(loss_jit(_ablate_block(params, i), idx, tgt)) - base
        k = block_kind(params["blocks"][i])
        deltas.append([i, k, d])
        log(f"L={L} block {i:2d}/{nblocks-1} [{k:4s}] delta={d:+.4f}  ({time.time()-ti:.1f}s)")
    # HCA null-sink mass (clean per-block forward up to each HCA block)
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, L, dtype=params["token_embedding"]["weight"].dtype)
    null_mass = {}
    x = params["token_embedding"]["weight"][idx]
    from jax_model.layers import block_forward
    for i, b in enumerate(params["blocks"]):
        if "pool_score" in b:
            null_mass[i] = float(_hca_null_mass(b, x, cfg, cos, sin))
        x = block_forward(b, x, cfg, cos, sin, attention_backend=AB)
    log(f"L={L} HCA null-mass: {{ {', '.join(f'{i}:{m:.3f}' for i, m in null_mass.items())} }}")
    results["lengths"][str(L)] = {"base": base, "deltas": deltas, "null_mass": {str(i): m for i, m in null_mass.items()}}
    json.dump(results, open("ablation_cpu_results.json", "w"), indent=1)
    log(f"=== L={L} done, saved ===")

log("ALL DONE")
