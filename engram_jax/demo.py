"""Engram-in-a-mock-backbone forward demo, mirroring the torch reference __main__.

By default runs offline with random token ids and a stub compressed vocabulary.
Pass --tokenizer deepseek-ai/DeepSeek-V3 to hash real text through the real
compressed tokenizer (downloads from the HF hub on first use).

    uv run python -m engram_jax.demo
    uv run python -m engram_jax.demo --length 2048
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .config import BackBoneConfig, EngramConfig, layer_spec
from .hashing import NgramHashMapping
from .layers import engram_forward, init_engram_params


def build(engram_cfg, backbone_cfg, mapping, key):
    spec = layer_spec(engram_cfg, backbone_cfg)
    keys = jax.random.split(key, len(engram_cfg.layer_ids) + 2)
    params = {
        "token_embedding": jax.random.normal(keys[0], (backbone_cfg.vocab_size, backbone_cfg.hidden_size)) * 0.02,
        "lm_head": jax.random.normal(keys[1], (backbone_cfg.vocab_size, backbone_cfg.hidden_size)) * 0.02,
        "engram": {
            layer_id: init_engram_params(
                k,
                [x for heads in mapping.vocab_size_across_layers[layer_id] for x in heads],
                spec,
            )
            for layer_id, k in zip(engram_cfg.layer_ids, keys[2:])
        },
    }
    return params, spec


@partial(jax.jit, static_argnames=("spec", "num_layers", "hc_mult"))
def forward(params, input_ids, hash_ids, spec, num_layers, hc_mult):
    x = params["token_embedding"][input_ids]
    # mock hyper-connection: replicate the stream hc_mult times
    x = jnp.broadcast_to(x[:, :, None, :], (*x.shape[:2], hc_mult, x.shape[-1]))
    for layer_id in range(num_layers):
        if layer_id in params["engram"]:
            x = engram_forward(params["engram"][layer_id], x, hash_ids[layer_id], spec) + x
        # attention and MoE are identity mocks, matching the reference demo
    return x[:, :, 0, :] @ params["lm_head"].T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="", help="HF tokenizer path; empty = offline stub vocab")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if args.tokenizer:
        engram_cfg = EngramConfig(tokenizer_name_or_path=args.tokenizer)
        backbone_cfg = BackBoneConfig()
        mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        )
        text = "Only Alexander the Great could tame the horse Bucephalus."
        ids = mapping.compressed_tokenizer.tokenizer(text, return_tensors="np").input_ids
        input_ids = np.tile(ids, (args.batch_size, 1))
    else:
        # offline: stub compressed vocab + smaller hash tables, random ids
        engram_cfg = EngramConfig(engram_vocab_size=[50_000, 50_000])
        backbone_cfg = BackBoneConfig()
        mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed,
            tokenizer_vocab_size=backbone_cfg.vocab_size,
        )
        rng = np.random.default_rng(0)
        input_ids = rng.integers(0, backbone_cfg.vocab_size, size=(args.batch_size, args.length))

    params, spec = build(engram_cfg, backbone_cfg, mapping, jax.random.PRNGKey(0))

    t0 = time.perf_counter()
    hash_ids = {k: jnp.asarray(v) for k, v in mapping.hash(input_ids).items()}
    hash_ms = (time.perf_counter() - t0) * 1000

    idx = jnp.asarray(input_ids, dtype=jnp.int32)
    call = lambda: forward(params, idx, hash_ids, spec, backbone_cfg.num_layers, backbone_cfg.hc_mult)
    output = call().block_until_ready()  # compile + first run

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        call().block_until_ready()
        times.append(time.perf_counter() - start)

    print("✅ Forward Complete!")
    print(f"backend: {jax.default_backend()}")
    print(f"input_ids.shape={tuple(input_ids.shape)}  output.shape={tuple(output.shape)}")
    print(f"hashing (host, numpy): {hash_ms:.2f} ms")
    print(f"jitted forward: {min(times) * 1000:.2f} ms (best of {args.iters})")


if __name__ == "__main__":
    main()
