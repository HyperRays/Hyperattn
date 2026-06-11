import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import EfficientHGConfig, EfficientHypergraphLM, count_parameters as torch_count_parameters
from model.rope import precompute_rope_cache

try:
    import jax
    import jax.numpy as jnp

    import jax_model
except ImportError:
    jax = None
    jnp = None
    jax_model = None


def parse_lengths(value):
    lengths = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        lengths.append(int(item))
    if not lengths:
        raise argparse.ArgumentTypeError("provide at least one token length")
    if any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("token lengths must be positive")
    return lengths


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def block_jax_until_ready(value):
    if jax is None:
        return value
    return jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, value)


def autocast_context(device_type, dtype_name):
    if device_type != "cuda" or dtype_name == "fp32":
        return torch.amp.autocast(device_type=device_type, enabled=False)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=True)


def make_config(args, max_length):
    return EfficientHGConfig(
        vocab_size=args.vocab_size,
        block_size=max_length,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_local_attn_layers=args.n_local_attn_layers,
        n_span_layers=args.n_span_layers,
        n_compressed_memory_layers=args.n_compressed_memory_layers,
        span_widths=tuple(args.span_widths),
        local_window=args.local_window,
        compression_block=args.compression_block,
        dropout=0.0,
    )


def jax_dtype(dtype_name):
    if dtype_name == "fp16":
        return jnp.float16
    if dtype_name == "bf16":
        return jnp.bfloat16
    return jnp.float32


def cast_jax_floating(tree, dtype):
    return jax.tree.map(
        lambda x: x.astype(dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
        tree,
    )


def forward_backbone(model, idx):
    cfg = model.cfg
    _, T = idx.shape
    cos, sin = precompute_rope_cache(cfg.n_embd // cfg.n_head, T, idx.device)
    x = model.drop(model.token_embedding(idx))
    for block in model.blocks:
        x = block(x, cos, sin)
    return model.ln_f(x)


def run_once_torch(model, idx, targets, measure, do_backward):
    if measure == "full":
        _, loss = model(idx, targets if do_backward else None)
        output = loss if do_backward else None
    elif measure == "backbone":
        hidden = forward_backbone(model, idx)
        output = hidden.float().square().mean() if do_backward else None
    else:
        raise ValueError(f"unknown measure: {measure}")

    if do_backward:
        output.backward()


def measure_length_torch(model, args, length, device, measure):
    idx = torch.randint(args.vocab_size, (args.batch_size, length), device=device)
    targets = torch.randint(args.vocab_size, (args.batch_size, length), device=device)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    do_backward = args.mode == "forward_backward"

    model.train(do_backward)

    for _ in range(args.warmup):
        model.zero_grad(set_to_none=True)
        with autocast_context(device_type, args.dtype):
            run_once_torch(model, idx, targets, measure, do_backward)
        sync(device)

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    times = []
    for _ in range(args.iters):
        model.zero_grad(set_to_none=True)
        sync(device)
        start = time.perf_counter()
        with autocast_context(device_type, args.dtype):
            run_once_torch(model, idx, targets, measure, do_backward)
        sync(device)
        times.append(time.perf_counter() - start)

    if str(device).startswith("cuda"):
        max_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        max_memory_mb = None

    mean_s = sum(times) / len(times)
    std_s = 0.0
    if len(times) > 1:
        std_s = math.sqrt(sum((x - mean_s) ** 2 for x in times) / (len(times) - 1))

    tokens = args.batch_size * length
    return {
        "measure": measure,
        "mode": args.mode,
        "length": length,
        "batch_size": args.batch_size,
        "mean_ms": mean_s * 1000.0,
        "std_ms": std_s * 1000.0,
        "tokens_per_s": tokens / mean_s,
        "max_memory_mb": max_memory_mb,
    }


def run_once_jax(params, idx, targets, cfg, measure, do_backward, attention_backend, span_backend):
    if do_backward:
        if measure == "full":
            fn = lambda p: jax_model.loss(
                p,
                idx,
                targets,
                cfg,
                attention_backend=attention_backend,
                span_backend=span_backend,
            )
        elif measure == "backbone":
            fn = lambda p: jnp.mean(
                jnp.square(
                    jax_model.forward_backbone(
                        p,
                        idx,
                        cfg,
                        attention_backend=attention_backend,
                        span_backend=span_backend,
                    )
                )
            )
        else:
            raise ValueError(f"unknown measure: {measure}")
        return jax.value_and_grad(fn)(params)

    if measure == "full":
        return jax_model.forward(params, idx, cfg, attention_backend=attention_backend, span_backend=span_backend)
    if measure == "backbone":
        return jax_model.forward_backbone(params, idx, cfg, attention_backend=attention_backend, span_backend=span_backend)
    raise ValueError(f"unknown measure: {measure}")


def measure_length_jax(params, cfg, args, length, measure, key):
    idx_key, target_key = jax.random.split(key)
    idx = jax.random.randint(idx_key, (args.batch_size, length), minval=0, maxval=args.vocab_size, dtype=jnp.int32)
    targets = jax.random.randint(target_key, (args.batch_size, length), minval=0, maxval=args.vocab_size, dtype=jnp.int32)
    do_backward = args.mode == "forward_backward"

    def call(p, x, y):
        return run_once_jax(p, x, y, cfg, measure, do_backward, args.jax_attention, args.jax_span)

    if args.compile:
        call = jax.jit(call)

    for _ in range(args.warmup):
        block_jax_until_ready(call(params, idx, targets))

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        block_jax_until_ready(call(params, idx, targets))
        times.append(time.perf_counter() - start)

    mean_s = sum(times) / len(times)
    std_s = 0.0
    if len(times) > 1:
        std_s = math.sqrt(sum((x - mean_s) ** 2 for x in times) / (len(times) - 1))

    tokens = args.batch_size * length
    return {
        "measure": measure,
        "mode": args.mode,
        "length": length,
        "batch_size": args.batch_size,
        "mean_ms": mean_s * 1000.0,
        "std_ms": std_s * 1000.0,
        "tokens_per_s": tokens / mean_s,
        "max_memory_mb": None,
    }


def fit_power_law(rows, y_key):
    points = [(row["length"], row[y_key]) for row in rows if row.get(y_key) is not None and row[y_key] > 0]
    if len(points) < 2:
        return None

    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    denom = sum((x - x_bar) ** 2 for x in xs)
    if denom == 0:
        return None

    exponent = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / denom
    intercept = y_bar - exponent * x_bar
    predicted = [intercept + exponent * x for x in xs]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, predicted))
    ss_tot = sum((y - y_bar) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {
        "exponent": exponent,
        "coefficient": math.exp(intercept),
        "r2": r2,
    }


def fit_grouped_power_laws(rows, y_key):
    fits = {}
    groups = sorted({row["measure"] for row in rows})
    for measure in groups:
        fit = fit_power_law([row for row in rows if row["measure"] == measure], y_key)
        if fit is not None:
            fits[measure] = fit
    return fits


def print_table(rows):
    headers = ["measure", "mode", "length", "batch", "mean_ms", "std_ms", "tok/s", "max_mem_mb"]
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        mem = "n/a" if row["max_memory_mb"] is None else f"{row['max_memory_mb']:.1f}"
        print(
            f"{row['measure']} | "
            f"{row['mode']} | "
            f"{row['length']} | "
            f"{row['batch_size']} | "
            f"{row['mean_ms']:.3f} | "
            f"{row['std_ms']:.3f} | "
            f"{row['tokens_per_s']:.1f} | "
            f"{mem}"
        )


def write_outputs(rows, summary, args):
    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"rows": rows, "summary": summary}, f, indent=2)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Benchmark model speed and sequence-length scaling.")
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("128,256,512,1024,2048"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--mode", choices=["forward", "forward_backward"], default="forward")
    parser.add_argument("--measure", choices=["full", "backbone", "both"], default="both")
    parser.add_argument("--backend", choices=["torch", "jax"], default="torch")
    parser.add_argument("--jax-attention", choices=["manual", "windowed", "chunked"], default="windowed")
    parser.add_argument("--jax-span", choices=["materialized", "fused", "einsum_fused"], default="materialized")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-local-attn-layers", type=int, default=1)
    parser.add_argument("--n-span-layers", type=int, default=6)
    parser.add_argument("--n-compressed-memory-layers", type=int, default=1)
    parser.add_argument("--span-widths", type=parse_lengths, default=parse_lengths("2,4,8,16,32,64"))
    parser.add_argument("--local-window", type=int, default=256)
    parser.add_argument("--compression-block", type=int, default=64)

    parser.add_argument("--csv", default="")
    parser.add_argument("--json", default="")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    cfg = make_config(args, max(args.lengths))
    torch.manual_seed(1337)

    if args.backend == "torch":
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1337)
            print("GPU:", torch.cuda.get_device_name(device))
        model = EfficientHypergraphLM(cfg).to(device)
        model.eval()
        if args.compile:
            model = torch.compile(model)
        params = None
        print("device:", device)
        print(f"parameters: {torch_count_parameters(model)/1e6:.2f}M")
    else:
        if jax is None:
            raise ImportError("JAX backend requested, but jax/jax_model could not be imported.")
        torch_model = EfficientHypergraphLM(cfg).eval()
        params, cfg = jax_model.from_torch_model(torch_model)
        params = cast_jax_floating(params, jax_dtype(args.dtype))
        model = None
        print("jax devices:", jax.devices())
        print("jax backend:", jax.default_backend())
        print(f"parameters: {jax_model.count_parameters(params)/1e6:.2f}M")

    print("backend:", args.backend)
    if args.backend == "jax":
        print("jax attention:", args.jax_attention)
        print("jax span:", args.jax_span)
    print("dtype:", args.dtype)
    print("config:", asdict(cfg))

    rows = []
    measures = ["full", "backbone"] if args.measure == "both" else [args.measure]
    key = jax.random.PRNGKey(1337) if args.backend != "torch" else None
    for measure in measures:
        for length in args.lengths:
            if args.backend == "torch":
                rows.append(measure_length_torch(model, args, length, device, measure))
            else:
                key, subkey = jax.random.split(key)
                rows.append(measure_length_jax(params, cfg, args, length, measure, subkey))
            print_table(rows[-1:])

    latency_fits = fit_grouped_power_laws(rows, "mean_ms")
    memory_fits = fit_grouped_power_laws(rows, "max_memory_mb")
    summary = {
        "latency_power_law": latency_fits,
        "memory_power_law": memory_fits,
    }

    print()
    print_table(rows)
    for measure, latency_fit in latency_fits.items():
        print(
            f"\n{measure} latency scaling: "
            f"time_ms ~= {latency_fit['coefficient']:.4g} * tokens^{latency_fit['exponent']:.3f} "
            f"(R^2={latency_fit['r2']:.4f})"
        )
    for measure, memory_fit in memory_fits.items():
        print(
            f"{measure} memory scaling: "
            f"memory_mb ~= {memory_fit['coefficient']:.4g} * tokens^{memory_fit['exponent']:.3f} "
            f"(R^2={memory_fit['r2']:.4f})"
        )

    write_outputs(rows, summary, args)


if __name__ == "__main__":
    main()
