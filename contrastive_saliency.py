import argparse
import csv
import html
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from long_context_probe import load_torch_model
from model.blocks import (
    CausalCompressedMemoryAttentionBlock,
    CausalSpanHypergraphBlock,
    LocalAttentionBlock,
)


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


def expanded_case(case, *, context_style, filler_repeats, max_input_tokens, tokenizer=None):
    if context_style == "none" and filler_repeats <= 0:
        return dict(case)

    filler = FILLER_PARAGRAPH * max(0, filler_repeats)
    if context_style == "long":
        bridge = LONG_BRIDGES.get(case["kind"], LONG_BRIDGES["factual"])
    else:
        bridge = ""
    prompt = f"{filler}{bridge}{case['prompt']}"

    if tokenizer is not None and max_input_tokens is not None:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > max_input_tokens:
            ids = ids[-max_input_tokens:]
            prompt = tokenizer.decode(ids)

    out = dict(case)
    suffix = context_style
    if filler_repeats:
        suffix = f"{suffix}_r{filler_repeats}"
    out["name"] = f"{case['name']}_{suffix}"
    out["base_name"] = case["name"]
    out["prompt"] = prompt
    return out


def block_kind(block):
    if isinstance(block, LocalAttentionBlock):
        return "attn"
    if isinstance(block, CausalSpanHypergraphBlock):
        return "span"
    if isinstance(block, CausalCompressedMemoryAttentionBlock):
        return "hca"
    return block.__class__.__name__.lower()


def sanitize_filename(name):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def token_label(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)])
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    if text == " ":
        return "<space>"
    return text


def encode_one_token(tokenizer, text, case_name, field):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        decoded = [tokenizer.decode([i]) for i in ids]
        raise ValueError(
            f"case {case_name!r} {field}={text!r} must encode to exactly one token; "
            f"got {ids} / {decoded}. Use a leading-space GPT-style token."
        )
    return ids[0]


def register_activation_hooks(model, capture):
    handles = []

    def add_hook(name, module):
        def hook(_module, _inp, out):
            if torch.is_tensor(out) and out.requires_grad:
                out.retain_grad()
                capture.append((name, out))

        handles.append(module.register_forward_hook(hook))

    add_hook("emb", model.token_embedding)
    for i, block in enumerate(model.blocks):
        add_hook(f"{i:02d}:{block_kind(block)}", block)
    add_hook("ln_f", model.ln_f)
    return handles


def saliency_for_case(model, tokenizer, device, case, attribution):
    prompt_ids = tokenizer.encode(case["prompt"], add_special_tokens=False)
    if not prompt_ids:
        raise ValueError(f"case {case['name']!r} has an empty prompt")
    prompt_ids = prompt_ids[-model.cfg.block_size :]
    true_id = encode_one_token(tokenizer, case["true"], case["name"], "true")
    false_id = encode_one_token(tokenizer, case["false"], case["name"], "false")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    capture = []
    handles = register_activation_hooks(model, capture)
    try:
        model.zero_grad(set_to_none=True)
        logits, _ = model(ids)
        logp = F.log_softmax(logits[0, -1], dim=-1)
        metric = logp[true_id] - logp[false_id]
        metric.backward()
    finally:
        for handle in handles:
            handle.remove()

    token_labels = [token_label(tokenizer, i) for i in prompt_ids]
    rows = []
    for layer_idx, (name, act) in enumerate(capture):
        if act.grad is None:
            continue
        activation = act.detach()[0].float()
        grad = act.grad.detach()[0].float()
        grad_norm = grad.norm(dim=-1)
        grad_x_act = (grad * activation).abs().sum(dim=-1)
        signed_grad_x_act = (grad * activation).sum(dim=-1)
        if attribution == "grad_x_act":
            score = grad_x_act
        elif attribution == "signed_grad_x_act":
            score = signed_grad_x_act.abs()
        else:
            score = grad_norm
        rows.append(
            {
                "layer_idx": layer_idx,
                "layer": name,
                "score": score.cpu().numpy(),
                "grad_norm": grad_norm.cpu().numpy(),
                "grad_x_act": grad_x_act.cpu().numpy(),
                "signed_grad_x_act": signed_grad_x_act.cpu().numpy(),
            }
        )

    true_logp = float(logp[true_id].detach().cpu())
    false_logp = float(logp[false_id].detach().cpu())
    return {
        "case": case,
        "tokens": token_labels,
        "token_ids": prompt_ids,
        "true_id": true_id,
        "false_id": false_id,
        "true_logp": true_logp,
        "false_logp": false_logp,
        "margin": true_logp - false_logp,
        "layers": rows,
    }


def percentile_scale(matrix, percentile):
    finite = matrix[np.isfinite(matrix)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return 1.0
    value = float(np.percentile(finite, percentile))
    return value if value > 0 else float(finite.max() or 1.0)


def heat_color(value):
    value = max(0.0, min(1.0, float(value)))
    stops = [
        (0.00, (247, 248, 250)),
        (0.20, (198, 219, 239)),
        (0.45, (107, 174, 214)),
        (0.70, (253, 174, 97)),
        (1.00, (215, 48, 39)),
    ]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if value <= b:
            t = 0.0 if b == a else (value - a) / (b - a)
            rgb = tuple(round(ca[i] + t * (cb[i] - ca[i])) for i in range(3))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    r, g, b = stops[-1][1]
    return f"rgb({r},{g},{b})"


def write_svg(result, path, *, scale="log", vmax_percentile=99.0, max_tokens=None):
    tokens = result["tokens"]
    layers = result["layers"]
    if max_tokens is not None and len(tokens) > max_tokens:
        start = len(tokens) - max_tokens
        tokens = tokens[start:]
        scores = [row["score"][start:] for row in layers]
        offset = start
    else:
        scores = [row["score"] for row in layers]
        offset = 0

    matrix = np.stack(scores, axis=0) if scores else np.zeros((0, len(tokens)), dtype=np.float32)
    if scale == "log":
        denom = percentile_scale(matrix, vmax_percentile)
        plot = np.log1p(matrix / max(denom / 4.0, 1e-12))
        vmax = percentile_scale(plot, vmax_percentile)
    else:
        vmax = percentile_scale(matrix, vmax_percentile)
        plot = matrix
    plot = np.clip(plot / max(vmax, 1e-12), 0.0, 1.0)

    cell_w = 16
    cell_h = 14
    left = 138
    top = 58
    token_label_h = 92
    legend_w = 150
    width = left + max(1, len(tokens)) * cell_w + legend_w
    height = top + max(1, len(layers)) * cell_h + token_label_h
    title = (
        f"{result['case']['name']} | margin={result['margin']:+.3f} "
        f"| true={result['case']['true']!r} false={result['case']['false']!r}"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>"
        "text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;font-size:11px;fill:#17202a}"
        ".small{font-size:10px;fill:#566573}.title{font-size:14px;font-weight:600}"
        "</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="12" y="24">{html.escape(title)}</text>',
        f'<text class="small" x="12" y="42">Rows are activation sites; columns are prompt tokens. Color is {html.escape(scale)}-scaled saliency.</text>',
    ]

    for r, row in enumerate(layers):
        y = top + r * cell_h
        label = row["layer"]
        parts.append(f'<text x="{left - 8}" y="{y + cell_h - 3}" text-anchor="end">{html.escape(label)}</text>')
        for c in range(len(tokens)):
            x = left + c * cell_w
            value = plot[r, c] if plot.size else 0.0
            raw = float(scores[r][c]) if scores else 0.0
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                f'fill="{heat_color(value)}"><title>{html.escape(label)} token {c + offset}: '
                f'{html.escape(tokens[c])} score={raw:.6g}</title></rect>'
            )

    axis_y = top + len(layers) * cell_h + 6
    for c, tok in enumerate(tokens):
        x = left + c * cell_w + cell_w / 2
        parts.append(
            f'<text class="small" transform="translate({x:.1f},{axis_y + 4}) rotate(60)" '
            f'text-anchor="start">{html.escape(tok)}</text>'
        )

    lx = left + len(tokens) * cell_w + 28
    ly = top
    parts.append(f'<text class="small" x="{lx}" y="{ly - 8}">low</text>')
    for i in range(80):
        t = i / 79
        parts.append(
            f'<rect x="{lx}" y="{ly + i}" width="18" height="1" fill="{heat_color(t)}"/>'
        )
    parts.append(f'<text class="small" x="{lx}" y="{ly + 96}">high</text>')
    parts.append("</svg>\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "base_case",
                "kind",
                "margin",
                "true",
                "false",
                "layer_idx",
                "layer",
                "token_idx",
                "token",
                "token_id",
                "score",
                "grad_norm",
                "grad_x_act",
                "signed_grad_x_act",
            ],
        )
        writer.writeheader()
        for result in results:
            for row in result["layers"]:
                for token_idx, token in enumerate(result["tokens"]):
                    writer.writerow(
                        {
                            "case": result["case"]["name"],
                            "base_case": result["case"].get("base_name", result["case"]["name"]),
                            "kind": result["case"]["kind"],
                            "margin": f"{result['margin']:.8f}",
                            "true": result["case"]["true"],
                            "false": result["case"]["false"],
                            "layer_idx": row["layer_idx"],
                            "layer": row["layer"],
                            "token_idx": token_idx,
                            "token": token,
                            "token_id": result["token_ids"][token_idx],
                            "score": f"{float(row['score'][token_idx]):.8g}",
                            "grad_norm": f"{float(row['grad_norm'][token_idx]):.8g}",
                            "grad_x_act": f"{float(row['grad_x_act'][token_idx]):.8g}",
                            "signed_grad_x_act": f"{float(row['signed_grad_x_act'][token_idx]):.8g}",
                        }
                    )


def top_sites(result, k):
    sites = []
    for row in result["layers"]:
        for token_idx, value in enumerate(row["score"]):
            sites.append((float(value), row["layer"], token_idx, result["tokens"][token_idx]))
    sites.sort(reverse=True, key=lambda x: x[0])
    return sites[:k]


def layer_totals(result):
    totals = []
    for row in result["layers"]:
        totals.append((float(np.sum(row["score"])), row["layer"]))
    totals.sort(reverse=True)
    return totals


def select_cases(names):
    if names == "all":
        return CASES
    wanted = {name.strip() for name in names.split(",") if name.strip()}
    known = {case["name"] for case in CASES}
    missing = wanted - known
    if missing:
        raise ValueError(f"unknown cases {sorted(missing)}; known cases: {sorted(known)}")
    return [case for case in CASES if case["name"] in wanted]


def main():
    parser = argparse.ArgumentParser(description="Contrastive factual/syntactic saliency over model layers.")
    parser.add_argument("--checkpoint", default="best_jax_span_hypergraph_lm (7).pkl")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--cases", default="all", help="comma-separated case names, or 'all'")
    parser.add_argument("--attribution", default="grad_norm", choices=["grad_norm", "grad_x_act", "signed_grad_x_act"])
    parser.add_argument("--out-prefix", default="contrastive_saliency")
    parser.add_argument("--context-style", default="none", choices=["none", "long"])
    parser.add_argument("--filler-repeats", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--scale", default="log", choices=["log", "linear"])
    parser.add_argument("--vmax-percentile", type=float, default=99.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _cfg, tokenizer, ckpt = load_torch_model(args.checkpoint)
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(True)

    print(
        f"checkpoint step={ckpt.get('step')} val_loss={ckpt.get('val_loss')} "
        f"tokens={ckpt.get('total_tokens')} block_size={model.cfg.block_size}"
    )
    print(f"device={device} attribution={args.attribution}")
    if args.context_style != "none" or args.filler_repeats:
        print(
            f"context_style={args.context_style} filler_repeats={args.filler_repeats} "
            f"max_input_tokens={args.max_input_tokens}"
        )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for base_case in select_cases(args.cases):
        case = expanded_case(
            base_case,
            context_style=args.context_style,
            filler_repeats=args.filler_repeats,
            max_input_tokens=args.max_input_tokens,
            tokenizer=tokenizer,
        )
        result = saliency_for_case(model, tokenizer, device, case, args.attribution)
        results.append(result)
        svg_path = out_prefix.with_name(f"{out_prefix.name}_{sanitize_filename(case['name'])}.svg")
        write_svg(
            result,
            svg_path,
            scale=args.scale,
            vmax_percentile=args.vmax_percentile,
            max_tokens=args.max_tokens,
        )
        print(
            f"\n[{case['name']}] {case['kind']} margin={result['margin']:+.3f} "
            f"logp(true)={result['true_logp']:.3f} logp(false)={result['false_logp']:.3f}"
        )
        print(f"  prompt tokens: {len(result['tokens'])}")
        print(f"  wrote {svg_path}")
        print("  top sites:")
        for value, layer, token_idx, token in top_sites(result, args.top_k):
            print(f"    {value:10.4g}  {layer:8s} token {token_idx:2d} {token!r}")
        print("  strongest layers:")
        for value, layer in layer_totals(result)[: min(args.top_k, 8)]:
            print(f"    {value:10.4g}  {layer}")

    csv_path = out_prefix.with_suffix(".csv")
    write_csv(results, csv_path)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
