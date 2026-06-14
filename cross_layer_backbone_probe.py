import argparse
import csv
import html
import math
import pickle
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

import jax_model2
from jax_model.ops import layer_norm, linear
from jax_model.rope import precompute_rope_cache
from jax_model2.layers import horizontal_block_forward


DEFAULT_TEXT = (
    "The history of science is often described as a long conversation between observation "
    "and theory. Astronomers learned to predict the movements of planets by comparing "
    "careful measurements with mathematical models. Chemists built tables of elements by "
    "noticing that substances with similar reactions often shared hidden structure. "
    "Biologists connected anatomy, fossils, inheritance, and ecology into a theory of "
    "evolution. Scientific work also depends on memory. A result only becomes useful when "
    "it can be compared with earlier measurements, repeated by another group, or explained "
    "in terms of a broader pattern. Long documents therefore contain many dependencies: a "
    "name may be introduced in one paragraph and used again much later, a method may be "
    "described before its outcome is reported, and a definition may quietly govern the "
    "meaning of many later sentences. "
) * 8


ORIGIN_GROUPS = ("emb", "init", "attn", "far", "mid", "local")
ORIGIN_INDEX = {name: i for i, name in enumerate(ORIGIN_GROUPS)}


def block_group(kind):
    if kind == "attn":
        return "attn"
    if kind == "far_span":
        return "far"
    if kind == "mid_span":
        return "mid"
    if kind == "local_span":
        return "local"
    return kind


def load_text(tokenizer, text_arg, max_tokens):
    if text_arg:
        path = Path(text_arg)
        text = path.read_text(encoding="utf-8") if path.exists() else text_arg
    else:
        text = DEFAULT_TEXT
    ids = tokenizer.encode(text, add_special_tokens=False)
    while len(ids) < max_tokens:
        ids = ids + ids
    return np.asarray(ids[:max_tokens], dtype=np.int32)


def trace_router(params, cfg, layout, idx):
    @jax.jit
    def _trace(params, idx):
        _, T = idx.shape
        cos, sin = precompute_rope_cache(
            cfg.n_embd // cfg.n_head,
            T,
            dtype=params["token_embedding"]["weight"].dtype,
        )
        x = params["token_embedding"]["weight"][idx]
        x0 = x
        B, _, C = x.shape
        mem = jnp.broadcast_to(
            params["mem_init"][None, :, None, :],
            (B, cfg.layer_memory_slots - 1, T, C),
        )

        read_rows = []
        write_rows = []
        read_entropy = []
        write_entropy = []
        gates = []
        x_rms = []
        routed_rms = []
        prev_rms = []
        mem_rms = []
        read_buckets = []
        write_buckets = []

        n_buckets = 4
        bucket = jnp.minimum((jnp.arange(T) * n_buckets) // T, n_buckets - 1)
        bucket_mask = jax.nn.one_hot(bucket, n_buckets).astype(jnp.float32)  # [T, K]
        bucket_count = jnp.maximum(jnp.sum(bucket_mask, axis=0), 1.0)

        for kind, block in zip(layout, params["blocks"]):
            bank = jnp.concatenate([x0[:, None], mem], axis=1)
            q = linear(layer_norm(x, block["route"]["ln"]), block["route"]["q_proj"])
            k = linear(layer_norm(bank, block["route"]["ln"]), block["route"]["k_proj"])
            v = linear(bank, block["route"]["v_proj"])
            scores = jnp.einsum("btc,bstc->bst", q, k) / math.sqrt(cfg.n_embd)
            read = jax.nn.softmax(scores.astype(jnp.float32), axis=1)
            routed = jnp.einsum("bst,bstc->btc", read.astype(x.dtype), v)
            routed = linear(routed, block["route"]["out_proj"])
            gate = jax.nn.sigmoid(block["route"]["gate"])
            x_in = gate * routed + (1.0 - gate) * x
            out = horizontal_block_forward(block, x_in, cfg, kind, cos, sin, "chunked", "fused")

            write_score = linear(out, block["route"]["write_score"])
            write = jax.nn.softmax(write_score.astype(jnp.float32), axis=-1)  # [B, T, M]
            w = jnp.moveaxis(write.astype(out.dtype), -1, 1)[..., None]
            mem = (1.0 - w) * mem + w * out[:, None]

            read_rows.append(jnp.mean(read, axis=(0, 2)))
            write_rows.append(jnp.mean(write, axis=(0, 1)))
            read_entropy.append(-jnp.mean(jnp.sum(read * jnp.log(read + 1e-9), axis=1)) / jnp.log(read.shape[1]))
            write_entropy.append(-jnp.mean(jnp.sum(write * jnp.log(write + 1e-9), axis=-1)) / jnp.log(write.shape[-1]))
            gates.append(gate)
            x_rms.append(jnp.sqrt(jnp.mean(jnp.square(out))))
            routed_rms.append(jnp.sqrt(jnp.mean(jnp.square(routed))))
            prev_rms.append(jnp.sqrt(jnp.mean(jnp.square(x))))
            mem_rms.append(jnp.sqrt(jnp.mean(jnp.square(mem), axis=(0, 2, 3))))

            # [B,S,T] x [T,K] -> [S,K] -> [K,S].
            read_buckets.append((jnp.einsum("bst,tk->sk", read, bucket_mask) / (B * bucket_count)[None, :]).T)
            # [B,T,M] x [T,K] -> [M,K] -> [K,M].
            write_buckets.append((jnp.einsum("btm,tk->mk", write, bucket_mask) / (B * bucket_count)[None, :]).T)
            x = out

        return (
            jnp.stack(read_rows),
            jnp.stack(write_rows),
            jnp.stack(read_entropy),
            jnp.stack(write_entropy),
            jnp.stack(gates),
            jnp.stack(x_rms),
            jnp.stack(routed_rms),
            jnp.stack(prev_rms),
            jnp.stack(mem_rms),
            jnp.stack(read_buckets),
            jnp.stack(write_buckets),
        )

    return [np.asarray(x) for x in _trace(params, idx)]


def direct_origin_flow(layout, read, write):
    """Approximate direct layer-family flow through the streaming memory slots.

    Slot 0 is the embedding source forever. Streaming slots begin as learned init.
    After block i, slot s becomes a convex blend of its previous origin and the current
    block family according to the mean write probability for that slot. This ignores the
    horizontal block's internal residual algebra, but makes the layer-memory write/read
    pathway interpretable.
    """
    n_slots = read.shape[1]
    n_mem = write.shape[1]
    n_orig = len(ORIGIN_GROUPS)
    slot_origin = np.zeros((n_slots, n_orig), dtype=np.float64)
    slot_origin[0, ORIGIN_INDEX["emb"]] = 1.0
    slot_origin[1:, ORIGIN_INDEX["init"]] = 1.0

    rows = []
    slot_rows = []
    for i, kind in enumerate(layout):
        read_origin = read[i] @ slot_origin
        group = block_group(kind)
        cur = np.zeros((n_orig,), dtype=np.float64)
        cur[ORIGIN_INDEX[group]] = 1.0
        rows.append(read_origin)
        for s in range(n_mem):
            w = float(write[i, s])
            slot_origin[s + 1] = (1.0 - w) * slot_origin[s + 1] + w * cur
        slot_rows.append(slot_origin.copy())
    return np.stack(rows), np.stack(slot_rows)


def print_group_summary(layout, read, write, read_origin, rent, went):
    print("slots: read=[emb, mem1, mem2, mem3]  write=[mem1, mem2, mem3]")
    for kind in ("attn", "far_span", "mid_span", "local_span"):
        idxs = [i for i, k in enumerate(layout) if k == kind]
        if not idxs:
            continue
        r = read[idxs].mean(axis=0)
        w = write[idxs].mean(axis=0)
        o = read_origin[idxs].mean(axis=0)
        print(
            f"{kind:10s} n={len(idxs):2d} "
            f"read {r[0]:.3f} {r[1]:.3f} {r[2]:.3f} {r[3]:.3f} | "
            f"write {w[0]:.3f} {w[1]:.3f} {w[2]:.3f} | "
            f"origin " + " ".join(f"{name}:{o[j]:.3f}" for j, name in enumerate(ORIGIN_GROUPS)) + " | "
            f"H(read) {rent[idxs].mean():.3f} H(write) {went[idxs].mean():.3f}"
        )


def write_csv(path, layout, read, write, read_origin, rent, went, gates, x_rms, routed_rms, prev_rms, mem_rms):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "block",
            "kind",
            "gate",
            "read_emb",
            "read_mem1",
            "read_mem2",
            "read_mem3",
            "write_mem1",
            "write_mem2",
            "write_mem3",
            "read_entropy",
            "write_entropy",
            "prev_rms",
            "routed_rms",
            "x_rms",
            "mem1_rms",
            "mem2_rms",
            "mem3_rms",
        ] + [f"origin_{name}" for name in ORIGIN_GROUPS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, kind in enumerate(layout):
            row = {
                "block": i,
                "kind": kind,
                "gate": gates[i],
                "read_emb": read[i, 0],
                "read_mem1": read[i, 1],
                "read_mem2": read[i, 2],
                "read_mem3": read[i, 3],
                "write_mem1": write[i, 0],
                "write_mem2": write[i, 1],
                "write_mem3": write[i, 2],
                "read_entropy": rent[i],
                "write_entropy": went[i],
                "prev_rms": prev_rms[i],
                "routed_rms": routed_rms[i],
                "x_rms": x_rms[i],
                "mem1_rms": mem_rms[i, 0],
                "mem2_rms": mem_rms[i, 1],
                "mem3_rms": mem_rms[i, 2],
            }
            for j, name in enumerate(ORIGIN_GROUPS):
                row[f"origin_{name}"] = read_origin[i, j]
            writer.writerow({k: (f"{float(v):.8g}" if isinstance(v, (float, np.floating)) else v) for k, v in row.items()})


def heat_color(value):
    value = max(0.0, min(1.0, float(value)))
    stops = [
        (0.00, (248, 249, 250)),
        (0.20, (214, 234, 248)),
        (0.45, (93, 173, 226)),
        (0.70, (245, 176, 65)),
        (1.00, (203, 67, 53)),
    ]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if value <= b:
            t = (value - a) / (b - a) if b > a else 0.0
            rgb = tuple(round(ca[i] + t * (cb[i] - ca[i])) for i in range(3))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
    r, g, b = stops[-1][1]
    return f"rgb({r},{g},{b})"


def kind_color(kind):
    return {
        "attn": "#4c78a8",
        "far_span": "#72b7b2",
        "mid_span": "#f58518",
        "local_span": "#54a24b",
    }.get(kind, "#999999")


def write_heatmap_svg(path, title, matrix, row_labels, col_labels, *, vmax=1.0, note=""):
    cell_w = 58
    cell_h = 18
    left = 108
    top = 72
    width = left + len(col_labels) * cell_w + 38
    height = top + len(row_labels) * cell_h + 46
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
        parts.append(f'<text class="small" x="{x}" y="{top - 12}" text-anchor="middle">{html.escape(label)}</text>')

    for r, label in enumerate(row_labels):
        y = top + r * cell_h
        block_kind = label.split(":", 1)[1] if ":" in label else ""
        parts.append(f'<rect x="8" y="{y + 3}" width="8" height="8" fill="{kind_color(block_kind)}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + cell_h - 5}" text-anchor="end">{html.escape(label)}</text>')
        for c in range(len(col_labels)):
            x = left + c * cell_w
            raw = float(matrix[r, c])
            v = raw / vmax if vmax else raw
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{heat_color(v)}">'
                f'<title>{html.escape(label)} {html.escape(col_labels[c])}: {raw:.4f}</title></rect>'
            )
            parts.append(
                f'<text class="small" x="{x + cell_w / 2:.1f}" y="{y + cell_h - 5}" '
                f'text-anchor="middle">{raw:.2f}</text>'
            )
    parts.append("</svg>\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_stacked_bar_svg(path, title, matrix, row_labels, segment_labels, *, note=""):
    row_h = 20
    left = 116
    top = 70
    bar_w = 420
    legend_x = left + bar_w + 28
    width = legend_x + 130
    height = top + len(row_labels) * row_h + 46
    colors = ["#4c78a8", "#bab0ac", "#e45756", "#72b7b2", "#f58518", "#54a24b", "#b279a2"]
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

    for i, label in enumerate(segment_labels):
        y = top + i * 16
        parts.append(f'<rect x="{legend_x}" y="{y - 9}" width="10" height="10" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text class="small" x="{legend_x + 16}" y="{y}">{html.escape(label)}</text>')

    for r, label in enumerate(row_labels):
        y = top + r * row_h
        block_kind = label.split(":", 1)[1] if ":" in label else ""
        parts.append(f'<rect x="8" y="{y + 4}" width="8" height="8" fill="{kind_color(block_kind)}"/>')
        parts.append(f'<text x="{left - 8}" y="{y + row_h - 6}" text-anchor="end">{html.escape(label)}</text>')
        x = left
        for c, label_c in enumerate(segment_labels):
            w = max(0.0, float(matrix[r, c])) * bar_w
            parts.append(
                f'<rect x="{x:.1f}" y="{y + 2}" width="{w:.1f}" height="{row_h - 5}" '
                f'fill="{colors[c % len(colors)]}"><title>{html.escape(label)} {html.escape(label_c)}: '
                f'{float(matrix[r, c]):.4f}</title></rect>'
            )
            x += w
    parts.append("</svg>\n")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def write_position_bucket_svg(path, title, buckets, row_labels, slot_labels, *, note=""):
    # Flatten each block's 4 position buckets into rows like b10/q1.
    rows = []
    labels = []
    bucket_names = ["0-25%", "25-50%", "50-75%", "75-100%"]
    for i, label in enumerate(row_labels):
        for b, bucket in enumerate(bucket_names):
            rows.append(buckets[i, b])
            labels.append(f"{label}/{bucket}")
    write_heatmap_svg(path, title, np.asarray(rows), labels, slot_labels, vmax=1.0, note=note)


def write_visuals(out_prefix, layout, read, write, read_origin, slot_origin, read_buckets, write_buckets):
    row_labels = [f"{i:02d}:{kind}" for i, kind in enumerate(layout)]
    write_heatmap_svg(
        out_prefix.with_name(out_prefix.name + "_read_slots.svg"),
        "Cross-Layer Router Reads",
        read,
        row_labels,
        ["emb", "mem1", "mem2", "mem3"],
        note="Each row sums to 1. Shows which slot supplies each block input.",
    )
    write_heatmap_svg(
        out_prefix.with_name(out_prefix.name + "_write_slots.svg"),
        "Cross-Layer Router Writes",
        write,
        row_labels,
        ["mem1", "mem2", "mem3"],
        note="Each row sums to 1. Shows where each block output is folded into streaming memory.",
    )
    write_stacked_bar_svg(
        out_prefix.with_name(out_prefix.name + "_read_origins.svg"),
        "Estimated Direct Origin Of Read Information",
        read_origin,
        row_labels,
        ORIGIN_GROUPS,
        note="Approximate direct memory-flow attribution by block family.",
    )
    final_slot_labels = ["emb", "mem1", "mem2", "mem3"]
    write_stacked_bar_svg(
        out_prefix.with_name(out_prefix.name + "_final_slot_origins.svg"),
        "Final Memory Slot Origin Mix",
        slot_origin[-1],
        final_slot_labels,
        ORIGIN_GROUPS,
        note="Approximate direct-origin content remaining in each slot after the stack.",
    )
    selected = [0, 1, 5, 6, 7, 9, 11, 18, 24, 30, 34]
    selected = [i for i in selected if i < len(layout)]
    write_position_bucket_svg(
        out_prefix.with_name(out_prefix.name + "_selected_read_buckets.svg"),
        "Selected Blocks: Read Slots By Token Position",
        read_buckets[selected],
        [row_labels[i] for i in selected],
        ["emb", "mem1", "mem2", "mem3"],
        note="Rows are block/position-quarter; useful for seeing whether routing changes across sequence positions.",
    )
    write_position_bucket_svg(
        out_prefix.with_name(out_prefix.name + "_selected_write_buckets.svg"),
        "Selected Blocks: Write Slots By Token Position",
        write_buckets[selected],
        [row_labels[i] for i in selected],
        ["mem1", "mem2", "mem3"],
        note="Rows are block/position-quarter; shows position-dependent memory writes.",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="latest_jax_model2_layer_routed_lm (1).pkl")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--text", default=None, help="literal text or path to a text file")
    parser.add_argument("--out-prefix", default="outputs/cross_layer_backbone/model2_router")
    args = parser.parse_args()

    ck = pickle.load(open(args.checkpoint, "rb"))
    cfg = jax_model2.LayerRoutedHGConfig(**ck["config"])
    if not cfg.use_memory_router:
        raise SystemExit("This probe is for jax_model2 checkpoints with use_memory_router=True")
    params = jax.tree.map(jnp.asarray, ck["params"])
    layout = jax_model2.resolve_block_layout(cfg)
    tokenizer = AutoTokenizer.from_pretrained(ck.get("tokenizer_name", "gpt2"), use_fast=True, local_files_only=True)
    ids = load_text(tokenizer, args.text, args.tokens)
    idx = jnp.asarray(ids[None])

    (
        read,
        write,
        rent,
        went,
        gates,
        x_rms,
        routed_rms,
        prev_rms,
        mem_rms,
        read_buckets,
        write_buckets,
    ) = trace_router(params, cfg, layout, idx)
    read_origin, slot_origin = direct_origin_flow(layout, read, write)

    print(
        f"checkpoint step={ck.get('step')} tokens={ck.get('total_tokens'):,} "
        f"val={ck.get('val_loss'):.4f} text_tokens={len(ids)}"
    )
    print_group_summary(layout, read, write, read_origin, rent, went)

    print("\nmost memory-routed blocks (lowest embedding read):")
    for i in np.argsort(read[:, 0])[:8]:
        o = read_origin[i]
        print(
            f"b{i:02d} {layout[i]:10s} read_emb={read[i,0]:.3f} "
            f"read=[{read[i,0]:.3f},{read[i,1]:.3f},{read[i,2]:.3f},{read[i,3]:.3f}] "
            f"origin=" + " ".join(f"{name}:{o[j]:.2f}" for j, name in enumerate(ORIGIN_GROUPS))
        )

    print("\nfinal direct-origin mix in memory slots:")
    final_slots = slot_origin[-1]
    for s, slot in enumerate(("emb", "mem1", "mem2", "mem3")):
        print(f"{slot:4s} " + " ".join(f"{name}:{final_slots[s, j]:.3f}" for j, name in enumerate(ORIGIN_GROUPS)))

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    write_csv(csv_path, layout, read, write, read_origin, rent, went, gates, x_rms, routed_rms, prev_rms, mem_rms)
    npz_path = out_prefix.with_suffix(".npz")
    np.savez(
        npz_path,
        layout=np.asarray(layout, dtype=object),
        read=read,
        write=write,
        read_origin=read_origin,
        slot_origin=slot_origin,
        read_entropy=rent,
        write_entropy=went,
        gates=gates,
        x_rms=x_rms,
        routed_rms=routed_rms,
        prev_rms=prev_rms,
        mem_rms=mem_rms,
        read_buckets=read_buckets,
        write_buckets=write_buckets,
        origin_groups=np.asarray(ORIGIN_GROUPS, dtype=object),
    )
    write_visuals(out_prefix, layout, read, write, read_origin, slot_origin, read_buckets, write_buckets)
    print(f"\nwrote {csv_path}")
    print(f"wrote {npz_path}")
    print(f"wrote SVG visualizations with prefix {out_prefix}")


if __name__ == "__main__":
    main()
