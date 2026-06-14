import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from long_context_probe import load_torch_model
from model.rope import precompute_rope_cache


BLIMP_CONFIGS = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "animate_subject_trans",
    "causative",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "drop_argument",
    "ellipsis_n_bar_1",
    "ellipsis_n_bar_2",
    "existential_there_object_raising",
    "existential_there_quantifiers_1",
    "existential_there_quantifiers_2",
    "existential_there_subject_raising",
    "expletive_it_object_raising",
    "inchoative",
    "intransitive",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "passive_1",
    "passive_2",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "superlative_quantifiers_1",
    "superlative_quantifiers_2",
    "tough_vs_raising_1",
    "tough_vs_raising_2",
    "transitive",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
]


MINIMAL_PAIRS = [
    {
        "phenomenon": "subject_verb_agreement",
        "good": "The keys to the cabinet are rusty.",
        "bad": "The keys to the cabinet is rusty.",
    },
    {
        "phenomenon": "subject_verb_agreement",
        "good": "The book near the candles is old.",
        "bad": "The book near the candles are old.",
    },
    {
        "phenomenon": "subject_verb_agreement",
        "good": "The dogs that chased the cat were loud.",
        "bad": "The dogs that chased the cat was loud.",
    },
    {
        "phenomenon": "subject_verb_agreement",
        "good": "The teacher who helped the students was kind.",
        "bad": "The teacher who helped the students were kind.",
    },
    {
        "phenomenon": "determiner_noun_number",
        "good": "This apple is fresh.",
        "bad": "This apples is fresh.",
    },
    {
        "phenomenon": "determiner_noun_number",
        "good": "These apples are fresh.",
        "bad": "These apple are fresh.",
    },
    {
        "phenomenon": "determiner_noun_number",
        "good": "That river is wide.",
        "bad": "That rivers is wide.",
    },
    {
        "phenomenon": "determiner_noun_number",
        "good": "Those rivers are wide.",
        "bad": "Those river are wide.",
    },
    {
        "phenomenon": "reflexive_agreement",
        "good": "The girl saw herself in the mirror.",
        "bad": "The girl saw himself in the mirror.",
    },
    {
        "phenomenon": "reflexive_agreement",
        "good": "The boy blamed himself for the mistake.",
        "bad": "The boy blamed herself for the mistake.",
    },
    {
        "phenomenon": "reflexive_number",
        "good": "The children taught themselves a song.",
        "bad": "The children taught himself a song.",
    },
    {
        "phenomenon": "anaphor_binding",
        "good": "Alice said that Mary praised herself.",
        "bad": "Alice said that Mary praised himself.",
    },
    {
        "phenomenon": "negative_polarity",
        "good": "No student has ever visited the museum.",
        "bad": "The student has ever visited the museum.",
    },
    {
        "phenomenon": "negative_polarity",
        "good": "Nobody had any reason to complain.",
        "bad": "Somebody had any reason to complain.",
    },
    {
        "phenomenon": "argument_structure",
        "good": "The chef put the bowl on the table.",
        "bad": "The chef put the bowl.",
    },
    {
        "phenomenon": "argument_structure",
        "good": "The author gave the editor a draft.",
        "bad": "The author gave a draft.",
    },
    {
        "phenomenon": "word_order",
        "good": "The small bird quickly crossed the road.",
        "bad": "The small bird crossed quickly the road.",
    },
    {
        "phenomenon": "word_order",
        "good": "The committee approved the proposal yesterday.",
        "bad": "The committee the proposal approved yesterday.",
    },
    {
        "phenomenon": "wh_dependency",
        "good": "What did the student read yesterday?",
        "bad": "What did the student read the book yesterday?",
    },
    {
        "phenomenon": "wh_dependency",
        "good": "Who did the artist invite to dinner?",
        "bad": "Who did the artist invite the guest to dinner?",
    },
]


def load_blimp_pairs(configs, split="train", limit_per_config=None, seed=0):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("BLiMP loading requires `datasets`; install it or use the built-in suite.") from exc

    if configs == ["all"]:
        configs = BLIMP_CONFIGS

    pairs = []
    rng = random.Random(seed)
    for config in configs:
        ds = load_dataset("nyu-mll/blimp", config, split=split)
        rows = list(ds)
        if limit_per_config is not None and len(rows) > limit_per_config:
            rows = rng.sample(rows, limit_per_config)
        for row in rows:
            pairs.append(
                {
                    "phenomenon": row.get("linguistics_term") or row.get("field") or config,
                    "config": config,
                    "uid": row.get("UID"),
                    "good": row["sentence_good"],
                    "bad": row["sentence_bad"],
                }
            )
    return pairs


CONTEXT_CASES = [
    {
        "name": "topic_continuity_science",
        "prefix": (
            "Astronomers compared careful observations with mathematical predictions. "
            "Chemists organized elements by their reactions and weights. "
            "Biologists connected fossils, anatomy, and inheritance into a theory of evolution. "
        ),
        "target": "Scientific theories become stronger when new measurements explain old patterns.",
    },
    {
        "name": "entity_continuity",
        "prefix": (
            "Mira packed a brass compass before leaving the harbor. "
            "The road crossed three hills and ended near a quiet observatory. "
            "By sunset, Mira checked the compass again and wrote a note in her journal. "
        ),
        "target": "Mira used the compass to decide which path led back toward the harbor.",
    },
    {
        "name": "local_syntax",
        "prefix": "The keys to the old cabinet beside the window ",
        "target": "are lying on the wooden table.",
    },
]


def encode(tokenizer, text, device):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long, device=device)


@torch.no_grad()
def sequence_nll(model, tokenizer, text, device):
    ids = encode(tokenizer, text, device)
    if ids.numel() < 2:
        raise ValueError(f"text is too short after tokenization: {text!r}")
    ids = ids[-model.cfg.block_size :]
    x = ids[:-1].unsqueeze(0)
    y = ids[1:].unsqueeze(0)
    logits, _ = model(x)
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none")
    return {
        "sum_nll": float(losses.sum()),
        "mean_nll": float(losses.mean()),
        "tokens": int(losses.numel()),
    }


@torch.no_grad()
def batch_sequence_nll(
    model,
    tokenizer,
    texts,
    device,
    batch_size=32,
    progress_label=None,
    progress_every=20,
    length_bucket=True,
):
    rows = [None] * len(texts)
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    encoded_all = [tokenizer.encode(text, add_special_tokens=False)[-model.cfg.block_size :] for text in texts]
    if any(len(ids) < 2 for ids in encoded_all):
        raise ValueError("all minimal-pair texts must tokenize to at least two tokens")

    order = list(range(len(encoded_all)))
    if length_bucket:
        order.sort(key=lambda i: len(encoded_all[i]))

    t0 = time.perf_counter()
    n_batches = (len(order) + batch_size - 1) // batch_size
    for start in range(0, len(order), batch_size):
        chunk_indices = order[start : start + batch_size]
        encoded = [encoded_all[i] for i in chunk_indices]
        max_len = max(len(ids) for ids in encoded)
        x = torch.full((len(encoded), max_len - 1), pad_id, dtype=torch.long, device=device)
        y = torch.full((len(encoded), max_len - 1), -100, dtype=torch.long, device=device)
        for i, ids in enumerate(encoded):
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            n = ids_t.numel() - 1
            x[i, :n] = ids_t[:-1]
            y[i, :n] = ids_t[1:]
        logits, _ = model(x)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(y)
        mask = y.ne(-100)
        sums = (losses * mask).sum(dim=1)
        counts = mask.sum(dim=1)
        means = sums / counts.clamp_min(1)
        for original_idx, total, mean, count in zip(chunk_indices, sums, means, counts):
            rows[original_idx] = {"sum_nll": float(total), "mean_nll": float(mean), "tokens": int(count)}
        batch_idx = start // batch_size + 1
        if progress_label and (batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == n_batches):
            elapsed = time.perf_counter() - t0
            done = min(start + len(chunk_indices), len(texts))
            rate = done / max(elapsed, 1e-9)
            print(
                f"[{progress_label}] batch {batch_idx}/{n_batches} "
                f"examples {done}/{len(texts)} max_len {max_len} elapsed {elapsed:.1f}s rate {rate:.1f}/s",
                flush=True,
            )
    return rows


@torch.no_grad()
def target_nll(model, tokenizer, prefix, target, device):
    prefix_ids = encode(tokenizer, prefix, device)
    target_ids = encode(tokenizer, target, device)
    ids = torch.cat([prefix_ids, target_ids], dim=0)
    ids = ids[-model.cfg.block_size :]
    x = ids[:-1].unsqueeze(0)
    y = ids[1:].unsqueeze(0)
    logits, _ = model(x)
    target_len = min(target_ids.numel(), y.numel())
    losses = F.cross_entropy(logits[0, -target_len:], y[0, -target_len:], reduction="none")
    return float(losses.mean()), float(losses.sum()), int(target_len)


def is_span_block(block):
    return hasattr(block, "edge_proj")


def is_hca_block(block):
    return hasattr(block, "pool_score")


def is_attn_block(block):
    return hasattr(block, "attn")


@contextlib.contextmanager
def ablated(model, mode):
    saved_gates = []
    saved_attn_out = []
    if mode == "full":
        yield
        return

    for block in model.blocks:
        disable_span = mode in {"span_off", "span_hca_off", "all_mixers_off"} and is_span_block(block)
        disable_hca = mode in {"hca_off", "span_hca_off", "all_mixers_off"} and is_hca_block(block)
        disable_attn = mode in {"attn_off", "all_mixers_off"} and is_attn_block(block)
        if disable_span or disable_hca:
            saved_gates.append((block.gate, block.gate.detach().clone()))
            block.gate.data.fill_(-30.0)
        if disable_attn:
            saved_attn_out.append((block.attn.out.weight, block.attn.out.weight.detach().clone()))
            block.attn.out.weight.data.zero_()

    try:
        yield
    finally:
        for gate, value in saved_gates:
            gate.data.copy_(value)
        for weight, value in saved_attn_out:
            weight.data.copy_(value)


def score_minimal_pairs(
    model,
    tokenizer,
    device,
    modes,
    pairs,
    pair_score="mean",
    batch_size=32,
    print_pair_details_limit=200,
    progress_every=20,
    length_bucket=True,
):
    rows = []
    for mode in modes:
        with ablated(model, mode):
            correct = 0
            by_pheno = {}
            good_scores = batch_sequence_nll(
                model,
                tokenizer,
                [item["good"] for item in pairs],
                device,
                batch_size=batch_size,
                progress_label=f"{mode}:good",
                progress_every=progress_every,
                length_bucket=length_bucket,
            )
            bad_scores = batch_sequence_nll(
                model,
                tokenizer,
                [item["bad"] for item in pairs],
                device,
                batch_size=batch_size,
                progress_label=f"{mode}:bad",
                progress_every=progress_every,
                length_bucket=length_bucket,
            )
            for item, good, bad in zip(pairs, good_scores, bad_scores):
                score_key = "mean_nll" if pair_score == "mean" else "sum_nll"
                margin = bad[score_key] - good[score_key]
                ok = margin > 0
                correct += int(ok)
                stats = by_pheno.setdefault(item["phenomenon"], [0, 0])
                stats[0] += int(ok)
                stats[1] += 1
                rows.append(
                    {
                        "mode": mode,
                        "phenomenon": item["phenomenon"],
                        "config": item.get("config", "built_in"),
                        "uid": item.get("uid"),
                        "correct": ok,
                        "pair_score": pair_score,
                        "margin": margin,
                        "good_sum_nll": good["sum_nll"],
                        "bad_sum_nll": bad["sum_nll"],
                        "good_mean_nll": good["mean_nll"],
                        "bad_mean_nll": bad["mean_nll"],
                        "good_tokens": good["tokens"],
                        "bad_tokens": bad["tokens"],
                        "good": item["good"],
                        "bad": item["bad"],
                    }
                )
            print(f"\n== Minimal Pairs: {mode} ==")
            print(f"accuracy {correct}/{len(pairs)} = {correct / max(1, len(pairs)):.3f}")
            for pheno, (c, n) in sorted(by_pheno.items()):
                print(f"{pheno:24s} {c:2d}/{n:<2d} {c / n:.3f}")

    print("\n== Minimal Pairs By Config ==")
    for mode in modes:
        by_config = {}
        for row in rows:
            if row["mode"] != mode:
                continue
            stats = by_config.setdefault(row["config"], [0, 0])
            stats[0] += int(row["correct"])
            stats[1] += 1
        print(f"\n-- {mode} --")
        for config, (c, n) in sorted(by_config.items()):
            print(f"{config:52s} {c:4d}/{n:<4d} {c / n:.3f}")

    print("\n== Minimal Pair Margins ==")
    print(f"mode          phenomenon               ok   margin_{pair_score} good_nll bad_nll len")
    printable_rows = rows if print_pair_details_limit is None else rows[:print_pair_details_limit]
    for row in printable_rows:
        print(
            f"{row['mode']:13s} {row['phenomenon']:24s} "
            f"{str(row['correct']):5s} {row['margin']:+11.3f} "
            f"{row['good_mean_nll']:8.3f} {row['bad_mean_nll']:7.3f} "
            f"{row['good_tokens']:2d}/{row['bad_tokens']:<2d}"
        )
    if print_pair_details_limit is not None and len(rows) > print_pair_details_limit:
        print(f"... omitted {len(rows) - print_pair_details_limit} pair rows")
    return rows


def shuffle_words(text, seed):
    words = text.split()
    rng = random.Random(seed)
    rng.shuffle(words)
    return " ".join(words) + (" " if text.endswith(" ") else "")


def score_context_cases(model, tokenizer, device, modes):
    rows = []
    variants = {
        "full": lambda p: p,
        "drop_prefix": lambda p: "",
        "shuffle_prefix": lambda p: shuffle_words(p, 17),
        "half_prefix": lambda p: " ".join(p.split()[len(p.split()) // 2 :]) + " ",
    }
    for mode in modes:
        with ablated(model, mode):
            print(f"\n== Context Perturbations: {mode} ==")
            print("case                    variant          mean_nll  delta_vs_full")
            for case in CONTEXT_CASES:
                base = None
                values = {}
                for name, fn in variants.items():
                    mean, total, tokens = target_nll(model, tokenizer, fn(case["prefix"]), case["target"], device)
                    values[name] = mean
                    if name == "full":
                        base = mean
                for name, mean in values.items():
                    delta = mean - base
                    print(f"{case['name']:23s} {name:15s} {mean:8.3f} {delta:+13.3f}")
                    rows.append(
                        {
                            "mode": mode,
                            "case": case["name"],
                            "variant": name,
                            "mean_nll": mean,
                            "delta_vs_full": delta,
                        }
                    )
    return rows


@torch.no_grad()
def hidden_states(model, ids):
    T = ids.shape[1]
    cos, sin = precompute_rope_cache(model.cfg.n_embd // model.cfg.n_head, T, ids.device)
    x = model.drop(model.token_embedding(ids))
    states = [x.detach().cpu()]
    for block in model.blocks:
        x = block(x, cos, sin)
        states.append(x.detach().cpu())
    states.append(model.ln_f(x).detach().cpu())
    return states


def dump_hidden_states(model, tokenizer, device, path, texts):
    final_states = []
    mean_states = []
    lengths = []
    for text in texts:
        ids = encode(tokenizer, text, device)[-model.cfg.block_size :].unsqueeze(0)
        states = hidden_states(model, ids)
        lengths.append(int(ids.numel()))
        final_states.append(torch.stack([s[0, -1] for s in states]).numpy())
        mean_states.append(torch.stack([s[0].mean(dim=0) for s in states]).numpy())
    payload = {
        "final_states": np.stack(final_states),
        "mean_states": np.stack(mean_states),
        "lengths": np.asarray(lengths, dtype=np.int32),
        "texts": np.asarray(texts, dtype=object),
    }
    np.savez(path, **payload)
    print(f"\nwrote hidden-state dump: {path}")
    print(f"final_states shape: {payload['final_states'].shape}  mean_states shape: {payload['mean_states'].shape}")


def print_gate_summary(model):
    span_gates = []
    hca_gates = []
    for i, block in enumerate(model.blocks):
        if is_span_block(block):
            span_gates.append(float(torch.sigmoid(block.gate.detach())))
        elif is_hca_block(block):
            hca_gates.append((i, float(torch.sigmoid(block.gate.detach()))))
    if span_gates:
        arr = np.asarray(span_gates)
        print(f"span gates ({len(arr)}): min {arr.min():.3f} mean {arr.mean():.3f} max {arr.max():.3f}")
    if hca_gates:
        print("hca gates:", " ".join(f"b{i}:{v:.3f}" for i, v in hca_gates))


def main():
    parser = argparse.ArgumentParser(description="Linguistic probes for the span-hypergraph LM.")
    parser.add_argument("--checkpoint", default="best_jax_span_hypergraph_lm.pkl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--modes", default="full,span_off,hca_off,attn_off,all_mixers_off")
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--pair-score", choices=("mean", "sum"), default="mean")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--no-length-bucket", action="store_true")
    parser.add_argument("--print-pair-details-limit", type=int, default=200)
    parser.add_argument(
        "--blimp-configs",
        default=None,
        help="Comma-separated BLiMP configs, or 'all'. Omit to use the built-in tiny suite.",
    )
    parser.add_argument("--blimp-split", default="train")
    parser.add_argument("--blimp-limit-per-config", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--dump-hidden-states", default=None)
    parser.add_argument("--skip-context", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, cfg, tokenizer, ckpt = load_torch_model(args.checkpoint)
    model.to(device).eval()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(
        f"checkpoint step={ckpt.get('step')} val_loss={ckpt.get('val_loss'):.4f} "
        f"tokens={ckpt.get('total_tokens'):,} block_size={cfg.block_size}"
    )
    print_gate_summary(model)

    if args.blimp_configs:
        requested = [c.strip() for c in args.blimp_configs.split(",") if c.strip()]
        pairs = load_blimp_pairs(
            requested,
            split=args.blimp_split,
            limit_per_config=args.blimp_limit_per_config,
            seed=0,
        )
        if args.limit_pairs:
            pairs = pairs[: args.limit_pairs]
        print(f"loaded BLiMP pairs: {len(pairs)} from {args.blimp_configs}")
    else:
        pairs = MINIMAL_PAIRS[: args.limit_pairs] if args.limit_pairs else MINIMAL_PAIRS
        print(f"using built-in minimal pairs: {len(pairs)}")

    pair_rows = score_minimal_pairs(
        model,
        tokenizer,
        device,
        modes,
        pairs,
        args.pair_score,
        batch_size=args.batch_size,
        print_pair_details_limit=args.print_pair_details_limit,
        progress_every=args.progress_every,
        length_bucket=not args.no_length_bucket,
    )
    context_rows = [] if args.skip_context else score_context_cases(model, tokenizer, device, modes)

    if args.dump_hidden_states:
        texts = [p["good"] for p in pairs]
        texts += [case["prefix"] + case["target"] for case in CONTEXT_CASES]
        dump_hidden_states(model, tokenizer, device, args.dump_hidden_states, texts)

    if args.json_out:
        payload = {
            "checkpoint": args.checkpoint,
            "step": ckpt.get("step"),
            "val_loss": ckpt.get("val_loss"),
            "total_tokens": ckpt.get("total_tokens"),
            "minimal_pairs": pair_rows,
            "context_perturbations": context_rows,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote json: {args.json_out}")


if __name__ == "__main__":
    main()
