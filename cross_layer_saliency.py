import argparse
import csv
import html
import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

import jax_model2
from jax_model.ops import layer_norm, linear
from jax_model.rope import precompute_rope_cache
from jax_model2.layers import horizontal_block_forward


CASES = [
    {
        "name": "capital_france",
        "kind": "factual",
        "prompt": "The capital of France is",
        "true": " Paris",
        "false": " Berlin",
    },
    {
        "name": "gold_symbol",
        "kind": "factual",
        "prompt": "The chemical symbol for gold is",
        "true": " Au",
        "false": " Fe",
    },
    {
        "name": "hamlet_author",
        "kind": "factual",
        "prompt": "The author of Hamlet was",
        "true": " Shakespeare",
        "false": " Dickens",
    },
    {
        "name": "largest_planet",
        "kind": "factual",
        "prompt": "The largest planet in the solar system is",
        "true": " Jupiter",
        "false": " Mars",
    },
    {
        "name": "subject_verb_agreement",
        "kind": "syntax",
        "prompt": "The keys to the cabinet",
        "true": " are",
        "false": " is",
    },
    {
        "name": "determiner_agreement",
        "kind": "syntax",
        "prompt": "I saw many",
        "true": " birds",
        "false": " bird",
    },
    {
        "name": "anaphor_agreement",
        "kind": "syntax",
        "prompt": "The sisters blamed",
        "true": " themselves",
        "false": " himself",
    },
]


FILLER_PARAGRAPH = (
    "A researcher sorted notes from a small archive. The first folder contained "
    "weather reports, meeting minutes, descriptions of tools, and comments about "
    "school timetables. Another folder listed ordinary sentences for a language "
    "exercise, where each item had to be judged only from the final clause. The "
    "notes were deliberately mixed so that earlier details were mostly irrelevant. "
)


LONG_BRIDGES = {
    "factual": (
        "At the end of the archive there was a short factual quiz. The instruction "
        "said to answer the next item using general knowledge, not the unrelated "
        "notes above. The final item was: "
    ),
    "syntax": (
        "At the end of the archive there was a grammar completion task. The "
        "instruction said to choose the next word that makes the final sentence "
        "grammatical. The final sentence began: "
    ),
}


READ_SLOTS = ("emb", "mem1", "mem2", "mem3")
WRITE_SLOTS = ("mem1", "mem2", "mem3")


def sanitize_filename(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def select_cases(names):
    if names == "all":
        return CASES
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    known = {case["name"] for case in CASES}
    missing = wanted - known
    if missing:
        raise ValueError(f"unknown cases {sorted(missing)}; known cases: {sorted(known)}")
    return [case for case in CASES if case["name"] in wanted]


def expanded_case(case, *, context_style, filler_repeats, max_input_tokens, tokenizer=None):
    if context_style == "none" and filler_repeats <= 0:
        return dict(case)

    filler = FILLER_PARAGRAPH * max(0, filler_repeats)
    bridge = LONG_BRIDGES.get(case["kind"], LONG_BRIDGES["factual"]) if context_style == "long" else ""
    prompt = f"{filler}{bridge}{case['prompt']}"

    if tokenizer is not None and max_input_tokens is not None:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            prompt = tokenizer.decode(ids[-max_input_tokens:])

    out = dict(case)
    suffix = context_style
    if filler_repeats:
        suffix = f"{suffix}_r{filler_repeats}"
    out["name"] = f"{case['name']}_{suffix}"
    out["base_name"] = case["name"]
    out["prompt"] = prompt
    return out


def encode_one_token(tokenizer, text, case_name, field):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        decoded = [tokenizer.decode([i]) for i in ids]
        raise ValueError(
            f"case {case_name!r} {field}={text!r} must encode to exactly one token; "
            f"got {ids} / {decoded}. Use a leading-space GPT-style token."
        )
    return ids[0]


def token_label(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)])
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if text == " ":
        return "<space>"
    return text


def scaled_softmax(weights, scale, axis):
    scaled = weights * scale
    return scaled / jnp.maximum(jnp.sum(scaled, axis=axis, keepdims=True), 1e-12)


def margin_with_router_scales(
    params,
    cfg,
    layout,
    idx,
    true_id,
    false_id,
    read_scale,
    write_scale,
    *,
    attention_backend,
    span_backend,
):
    _, T = idx.shape
    cos, sin = precompute_rope_cache(
        cfg.n_embd // cfg.n_head,
        T,
        dtype=params["token_embedding"]["weight"].dtype,
    )
    x = params["token_embedding"]["weight"][idx]
    x0 = x
    B, _, C = x.shape
    mem = jnp.broadcast_to(params["mem_init"][None, :, None, :], (B, cfg.layer_memory_slots - 1, T, C))

    for i, (kind, block) in enumerate(zip(layout, params["blocks"])):
        bank = jnp.concatenate([x0[:, None], mem], axis=1)
        q = linear(layer_norm(x, block["route"]["ln"]), block["route"]["q_proj"])
        k = linear(layer_norm(bank, block["route"]["ln"]), block["route"]["k_proj"])
        v = linear(bank, block["route"]["v_proj"])
        scores = jnp.einsum("btc,bstc->bst", q, k) / math.sqrt(cfg.n_embd)
        read = jax.nn.softmax(scores.astype(jnp.float32), axis=1)
        read = scaled_softmax(read, read_scale[i][None, :, None], axis=1)
        routed = jnp.einsum("bst,bstc->btc", read.astype(x.dtype), v)
        routed = linear(routed, block["route"]["out_proj"])
        gate = jax.nn.sigmoid(block["route"]["gate"])
        x_in = gate * routed + (1.0 - gate) * x

        out = horizontal_block_forward(block, x_in, cfg, kind, cos, sin, attention_backend, span_backend)

        write_score = linear(out, block["route"]["write_score"])
        write = jax.nn.softmax(write_score.astype(jnp.float32), axis=-1)
        write = scaled_softmax(write, write_scale[i][None, None, :], axis=-1)
        w = jnp.moveaxis(write.astype(out.dtype), -1, 1)[..., None]
        mem = (1.0 - w) * mem + w * out[:, None]
        x = out

    h_last = layer_norm(x, params["ln_f"])[0, -1]
    token_embedding = params["token_embedding"]["weight"]
    return h_last @ (token_embedding[true_id] - token_embedding[false_id])


def saliency_for_case(params, cfg, layout, tokenizer, case, attention_backend, span_backend, *, use_jit):
    prompt_ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
    if not prompt_ids:
        raise ValueError(f"case {case['name']!r} has an empty prompt")
    prompt_ids = prompt_ids[-cfg.block_size :]
    true_id = encode_one_token(tokenizer, case["true"], case["name"], "true")
    false_id = encode_one_token(tokenizer, case["false"], case["name"], "false")
    idx = jnp.asarray(np.asarray(prompt_ids, dtype=np.int32)[None])
    read_scale = jnp.ones((len(layout), cfg.layer_memory_slots), dtype=jnp.float32)
    write_scale = jnp.ones((len(layout), cfg.layer_memory_slots - 1), dtype=jnp.float32)

    def metric(p, rs, ws):
        return margin_with_router_scales(
            p,
            cfg,
            layout,
            idx,
            true_id,
            false_id,
            rs,
            ws,
            attention_backend=attention_backend,
            span_backend=span_backend,
        )

    grad_fn = jax.value_and_grad(metric, argnums=(1, 2))
    if use_jit:
        grad_fn = jax.jit(grad_fn)
    value, (read_grad, write_grad) = grad_fn(params, read_scale, write_scale)
    return {
        "case": case,
        "token_ids": prompt_ids,
        "tokens": [token_label(tokenizer, token_id) for token_id in prompt_ids],
        "true_id": true_id,
        "false_id": false_id,
        "margin": float(np.asarray(value)),
        "read_grad": np.asarray(read_grad),
        "write_grad": np.asarray(write_grad),
    }


def kind_color(kind):
    return {
        "attn": "#4c78a8",
        "far_span": "#72b7b2",
        "mid_span": "#f58518",
        "local_span": "#54a24b",
    }.get(kind, "#999999")


def signed_color(value):
    value = max(-1.0, min(1.0, float(value)))
    if value < 0:
        t = -value
        a, b = (248, 249, 250), (49, 130, 189)
    else:
        t = value
        a, b = (248, 249, 250), (203, 67, 53)
    rgb = tuple(round(a[i] + t * (b[i] - a[i])) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def heat_color(value):
    value = max(0.0, min(1.0, float(value)))
    stops = [
        (0.00, (248, 249, 250)),
        (0.25, (198, 219, 239)),
        (0.55, (107, 174, 214)),
        (0.80, (253, 174, 97)),
        (1.00, (203, 67, 53)),
    ]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if value <= b:
            t = 0.0 if b == a else (value - a) / (b - a)
            rgb = tuple(round(ca[i] + t * (cb[i] - ca[i])) for i in range(3))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    r, g, b = stops[-1][1]
    return f"rgb({r},{g},{b})"


def percentile_scale(matrix, percentile):
    finite = np.asarray(matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    finite = np.abs(finite)
    finite = finite[finite > 0]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(finite, percentile))
    return value if value > 0 else float(finite.max() or 1.0)


def write_route_svg(path, title, matrix, row_labels, col_labels, *, signed, vmax_percentile=98.0, note=""):
    raw = np.asarray(matrix, dtype=np.float64)
    vmax = percentile_scale(raw, vmax_percentile)
    cell_w = 74
    cell_h = 18
    left = 116
    top = 74
    legend_w = 130
    width = left + len(col_labels) * cell_w + legend_w
    height = top + len(row_labels) * cell_h + 52
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;font-size:11px;fill:#17202a}"
        ".title{font-size:15px;font-weight:650}.small{font-size:10px;fill:#566573}"
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="12" y="24">{html.escape(title)}</text>',
    ]
    if note:
        parts.append(f'<text class="small" x="12" y="44">{html.escape(note)}</text>')
    for c, label in enumerate(col_labels):
        x = left + c * cell_w + cell_w / 2
        parts.append(f'<text class="small" x="{x:.1f}" y="{top - 12}" text-anchor="middle">{html.escape(label)}</text>')
    for r, label in enumerate(row_labels):
        y = top + r * cell_h
        kind = label.split(":", 1)[1] if ":" in label else ""
        parts.append(f'<rect x="8" y="{y + 3}" width="8" height="8" fill="{kind_color(kind)}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + cell_h - 5}" text-anchor="end">{html.escape(label)}</text>')
        for c in range(len(col_labels)):
            x = left + c * cell_w
            value = float(raw[r, c])
            scaled = value / vmax if vmax else value
            color = signed_color(scaled) if signed else heat_color(abs(scaled))
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{color}">'
                f'<title>{html.escape(label)} {html.escape(col_labels[c])}: {value:+.6g}</title></rect>'
            )
            parts.append(
                f'<text class="small" x="{x + cell_w / 2:.1f}" y="{y + cell_h - 5}" '
                f'text-anchor="middle">{value:+.2g}</text>'
            )
    lx = left + len(col_labels) * cell_w + 24
    parts.append(f'<text class="small" x="{lx}" y="{top - 8}">scale p{vmax_percentile:g}</text>')
    for i in range(80):
        t = i / 79
        if signed:
            v = 1.0 - 2.0 * t
            label_low, label_high = "+", "-"
            color = signed_color(v)
        else:
            color = heat_color(t)
            label_low, label_high = "high", "low"
        parts.append(f'<rect x="{lx}" y="{top + i}" width="18" height="1" fill="{color}"/>')
    parts.append(f'<text class="small" x="{lx + 26}" y="{top + 4}">{html.escape(label_high)}</text>')
    parts.append(f'<text class="small" x="{lx + 26}" y="{top + 82}">{html.escape(label_low)}</text>')
    parts.append("</svg>\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_summary_svg(path, title, results, layout):
    row_labels = [f"{i:02d}:{kind}" for i, kind in enumerate(layout)]
    case_names = [result["case"]["name"] for result in results]
    read_strength = np.stack([np.sum(np.abs(result["read_grad"]), axis=1) for result in results], axis=1)
    write_strength = np.stack([np.sum(np.abs(result["write_grad"]), axis=1) for result in results], axis=1)
    matrix = read_strength + write_strength
    write_route_svg(
        path,
        title,
        matrix,
        row_labels,
        case_names,
        signed=False,
        vmax_percentile=98.0,
        note="Absolute saliency summed over read and write slots. Rows are blocks; columns are cases.",
    )


def write_csv(path, results, layout):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "base_case",
                "kind",
                "margin",
                "route_kind",
                "block",
                "block_kind",
                "slot",
                "signed_grad",
                "abs_grad",
            ],
        )
        writer.writeheader()
        for result in results:
            case = result["case"]
            for i, block_kind in enumerate(layout):
                for slot, value in zip(READ_SLOTS, result["read_grad"][i]):
                    writer.writerow(
                        {
                            "case": case["name"],
                            "base_case": case.get("base_name", case["name"]),
                            "kind": case["kind"],
                            "margin": f"{result['margin']:.8f}",
                            "route_kind": "read",
                            "block": i,
                            "block_kind": block_kind,
                            "slot": slot,
                            "signed_grad": f"{float(value):.8g}",
                            "abs_grad": f"{abs(float(value)):.8g}",
                        }
                    )
                for slot, value in zip(WRITE_SLOTS, result["write_grad"][i]):
                    writer.writerow(
                        {
                            "case": case["name"],
                            "base_case": case.get("base_name", case["name"]),
                            "kind": case["kind"],
                            "margin": f"{result['margin']:.8f}",
                            "route_kind": "write",
                            "block": i,
                            "block_kind": block_kind,
                            "slot": slot,
                            "signed_grad": f"{float(value):.8g}",
                            "abs_grad": f"{abs(float(value)):.8g}",
                        }
                    )


def print_top_routes(result, layout, top_k):
    sites = []
    for i, kind in enumerate(layout):
        for slot, value in zip(READ_SLOTS, result["read_grad"][i]):
            sites.append((abs(float(value)), float(value), "read", i, kind, slot))
        for slot, value in zip(WRITE_SLOTS, result["write_grad"][i]):
            sites.append((abs(float(value)), float(value), "write", i, kind, slot))
    sites.sort(reverse=True, key=lambda x: x[0])
    for _, signed, route_kind, i, kind, slot in sites[:top_k]:
        print(f"    {signed:+10.4g}  {route_kind:5s} b{i:02d}:{kind:10s} {slot}")


def main():
    parser = argparse.ArgumentParser(description="Contrastive saliency over model2 cross-layer memory routes.")
    parser.add_argument("--checkpoint", default="latest_jax_model2_layer_routed_lm (1).pkl")
    parser.add_argument("--cases", default="all", help="comma-separated case names, or 'all'")
    parser.add_argument("--context-style", default="none", choices=["none", "long"])
    parser.add_argument("--filler-repeats", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--attention-backend", default="chunked", choices=["windowed", "chunked", "manual"])
    parser.add_argument("--span-backend", default="fused", choices=["fused", "materialized"])
    parser.add_argument("--out-prefix", default="outputs/cross_layer_saliency/latest1")
    parser.add_argument("--vmax-percentile", type=float, default=98.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--no-jit", action="store_true", help="Run without JIT compilation for slower but more transparent debugging.")
    args = parser.parse_args()

    ck = pickle.load(open(args.checkpoint, "rb"))
    cfg = jax_model2.LayerRoutedHGConfig(**ck["config"])
    if not cfg.use_memory_router:
        raise SystemExit("This saliency script is for jax_model2 checkpoints with use_memory_router=True")
    params = jax.tree.map(jnp.asarray, ck["params"])
    layout = jax_model2.resolve_block_layout(cfg)
    tokenizer = AutoTokenizer.from_pretrained(ck.get("tokenizer_name", "gpt2"), use_fast=True, local_files_only=True)

    print(
        f"checkpoint step={ck.get('step')} val={ck.get('val_loss')} "
        f"tokens={ck.get('total_tokens')} blocks={len(layout)}"
        ,
        flush=True,
    )
    print(
        f"attention_backend={args.attention_backend} span_backend={args.span_backend} "
        f"jit={not args.no_jit}",
        flush=True,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    row_labels = [f"{i:02d}:{kind}" for i, kind in enumerate(layout)]
    results = []
    for base_case in select_cases(args.cases):
        case = expanded_case(
            base_case,
            context_style=args.context_style,
            filler_repeats=args.filler_repeats,
            max_input_tokens=args.max_input_tokens,
            tokenizer=tokenizer,
        )
        print(f"\ncompiling/running {case['name']} ({case['kind']})...", flush=True)
        result = saliency_for_case(
            params,
            cfg,
            layout,
            tokenizer,
            case,
            args.attention_backend,
            args.span_backend,
            use_jit=not args.no_jit,
        )
        results.append(result)

        read_svg = out_prefix.with_name(f"{out_prefix.name}_{sanitize_filename(case['name'])}_read_signed.svg")
        write_svg = out_prefix.with_name(f"{out_prefix.name}_{sanitize_filename(case['name'])}_write_signed.svg")
        read_abs_svg = out_prefix.with_name(f"{out_prefix.name}_{sanitize_filename(case['name'])}_read_abs.svg")
        write_abs_svg = out_prefix.with_name(f"{out_prefix.name}_{sanitize_filename(case['name'])}_write_abs.svg")
        note = "Gradient of true-vs-false logit margin wrt route slot scale. Red helps; blue hurts."
        write_route_svg(
            read_svg,
            f"{case['name']} Read Route Saliency",
            result["read_grad"],
            row_labels,
            READ_SLOTS,
            signed=True,
            vmax_percentile=args.vmax_percentile,
            note=note,
        )
        write_route_svg(
            write_svg,
            f"{case['name']} Write Route Saliency",
            result["write_grad"],
            row_labels,
            WRITE_SLOTS,
            signed=True,
            vmax_percentile=args.vmax_percentile,
            note=note,
        )
        write_route_svg(
            read_abs_svg,
            f"{case['name']} Read Route Absolute Saliency",
            np.abs(result["read_grad"]),
            row_labels,
            READ_SLOTS,
            signed=False,
            vmax_percentile=args.vmax_percentile,
            note="Absolute route sensitivity, ignoring sign.",
        )
        write_route_svg(
            write_abs_svg,
            f"{case['name']} Write Route Absolute Saliency",
            np.abs(result["write_grad"]),
            row_labels,
            WRITE_SLOTS,
            signed=False,
            vmax_percentile=args.vmax_percentile,
            note="Absolute route sensitivity, ignoring sign.",
        )

        print(
            f"\n[{case['name']}] {case['kind']} margin={result['margin']:+.3f} "
            f"prompt_tokens={len(result['tokens'])}",
            flush=True,
        )
        print(f"  wrote {read_svg}", flush=True)
        print(f"  wrote {write_svg}", flush=True)
        print("  top routes:", flush=True)
        print_top_routes(result, layout, args.top_k)

    csv_path = out_prefix.with_suffix(".csv")
    npz_path = out_prefix.with_suffix(".npz")
    write_csv(csv_path, results, layout)
    np.savez(
        npz_path,
        layout=np.asarray(layout, dtype=object),
        cases=np.asarray([result["case"]["name"] for result in results], dtype=object),
        margins=np.asarray([result["margin"] for result in results], dtype=np.float32),
        read_grad=np.stack([result["read_grad"] for result in results]),
        write_grad=np.stack([result["write_grad"] for result in results]),
        read_slots=np.asarray(READ_SLOTS, dtype=object),
        write_slots=np.asarray(WRITE_SLOTS, dtype=object),
    )
    summary_svg = out_prefix.with_name(f"{out_prefix.name}_summary_abs.svg")
    write_summary_svg(summary_svg, "Cross-Layer Route Saliency Summary", results, layout)
    print(f"\nwrote {csv_path}", flush=True)
    print(f"wrote {npz_path}", flush=True)
    print(f"wrote {summary_svg}", flush=True)


if __name__ == "__main__":
    main()
