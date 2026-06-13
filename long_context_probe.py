import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import EfficientHGConfig, EfficientHypergraphLM


DEFAULT_TEXT = """
The history of science is often described as a long conversation between observation
and theory. Astronomers learned to predict the movements of planets by comparing
careful measurements with mathematical models. Chemists built tables of elements by
noticing that substances with similar reactions often shared hidden structure.
Biologists connected anatomy, fossils, inheritance, and ecology into a theory of
evolution. In each case, new instruments changed what could be seen, and new
language changed what could be asked.

Scientific work also depends on memory. A result only becomes useful when it can be
compared with earlier measurements, repeated by another group, or explained in terms
of a broader pattern. Long documents therefore contain many dependencies: a name may
be introduced in one paragraph and used again much later, a method may be described
before its outcome is reported, and a definition may quietly govern the meaning of
many later sentences.

This probe is not a benchmark of factual knowledge. It is only a controlled way to
ask whether an autoregressive language model assigns better probabilities when it is
allowed to read more of the preceding context. If longer context helps, the loss on
the same target suffix should decrease as more prefix tokens are included.
""" * 12


def as_tensor(x):
    return torch.from_numpy(np.asarray(x)).clone()


def load_torch_model(checkpoint_path):
    ckpt = pickle.load(open(checkpoint_path, "rb"))
    cfg = EfficientHGConfig(**ckpt["config"])
    params = ckpt["params"]
    model = EfficientHypergraphLM(cfg).eval()
    state = {
        "token_embedding.weight": as_tensor(params["token_embedding"]["weight"]),
        "ln_f.weight": as_tensor(params["ln_f"]["weight"]),
        "ln_f.bias": as_tensor(params["ln_f"]["bias"]),
    }
    state["lm_head.weight"] = state["token_embedding.weight"]

    for i, block in enumerate(params["blocks"]):
        prefix = f"blocks.{i}"
        for name in ("ln1", "ln2"):
            if name in block:
                state[f"{prefix}.{name}.weight"] = as_tensor(block[name]["weight"])
                state[f"{prefix}.{name}.bias"] = as_tensor(block[name]["bias"])

        if "attn" in block:
            state[f"{prefix}.attn.qkv.weight"] = as_tensor(block["attn"]["qkv"]["weight"])
            state[f"{prefix}.attn.out.weight"] = as_tensor(block["attn"]["out"]["weight"])
        elif "edge_proj" in block:
            for name in ("in_proj", "edge_proj", "out_proj"):
                state[f"{prefix}.{name}.weight"] = as_tensor(block[name]["weight"])
                state[f"{prefix}.{name}.bias"] = as_tensor(block[name]["bias"])
            state[f"{prefix}.gate"] = as_tensor(block["gate"])
        elif "pool_score" in block:
            for name in ("q_proj", "k_proj", "v_proj", "pool_score", "out"):
                state[f"{prefix}.{name}.weight"] = as_tensor(block[name]["weight"])
            state[f"{prefix}.gate"] = as_tensor(block["gate"])
            state[f"{prefix}.null_k"] = as_tensor(block["null_k"])
            state[f"{prefix}.null_v"] = as_tensor(block["null_v"])

        state[f"{prefix}.mlp.net.0.weight"] = as_tensor(block["mlp"]["fc"]["weight"])
        state[f"{prefix}.mlp.net.0.bias"] = as_tensor(block["mlp"]["fc"]["bias"])
        state[f"{prefix}.mlp.net.2.weight"] = as_tensor(block["mlp"]["proj"]["weight"])
        state[f"{prefix}.mlp.net.2.bias"] = as_tensor(block["mlp"]["proj"]["bias"])

    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("tokenizer_name", "gpt2"), use_fast=True, local_files_only=True)
    return model, cfg, tokenizer, ckpt


@torch.no_grad()
def target_suffix_loss(model, ids, context_len, target_len):
    segment = ids[-(context_len + target_len + 1) :]
    x = torch.tensor(segment[:-1], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(segment[1:], dtype=torch.long).unsqueeze(0)
    logits, _ = model(x)
    losses = F.cross_entropy(logits[0, -target_len:], y[0, -target_len:], reduction="none")
    return float(losses.mean()), float(torch.exp(losses.mean()))


def context_loss_sweep(model, tokenizer, lengths, target_len):
    ids = tokenizer.encode(DEFAULT_TEXT, add_special_tokens=False)
    need = max(lengths) + target_len + 1
    while len(ids) < need:
        ids = ids + ids
    print("\n== Fixed Suffix Loss vs Context Length ==")
    print(f"target_len={target_len} tokens; same final suffix scored each time")
    for length in lengths:
        loss, ppl = target_suffix_loss(model, ids, length, target_len)
        print(f"context {length:4d}: loss {loss:.4f}  ppl {ppl:.1f}")


@torch.no_grad()
def answer_logprob(model, tokenizer, prompt, answer):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long)
    logits, _ = model(ids[:, :-1])
    start = len(prompt_ids) - 1
    token_losses = []
    for j, token_id in enumerate(answer_ids):
        lp = F.log_softmax(logits[0, start + j], dim=-1)
        token_losses.append(float(-lp[token_id]))
    return -sum(token_losses), sum(token_losses) / max(1, len(token_losses))


def kv_retrieval_probe(model, tokenizer, filler_lengths):
    names = ["Mira", "Dax", "Lena", "Orin"]
    codes = [" 8472", " 1935", " 6208", " 4719"]
    filler_sentence = (
        " The archive contains ordinary notes about weather, gardens, tools, "
        "school schedules, and public meetings."
    )
    print("\n== Key-Value Retrieval Logprob ==")
    print("Scores are logprob(true code) - best logprob(decoy); higher is better.")
    for filler_len in filler_lengths:
        filler = (filler_sentence * math.ceil(filler_len / 20))[: filler_len * 5]
        facts = "\n".join(f"Name: {n} Code:{c}" for n, c in zip(names, codes))
        prompt = f"{facts}\n{filler}\nQuestion: Code for Mira?\nAnswer:"
        true_lp, true_nll = answer_logprob(model, tokenizer, prompt, codes[0])
        decoys = [answer_logprob(model, tokenizer, prompt, c)[0] for c in codes[1:]]
        margin = true_lp - max(decoys)
        print(f"filler approx {filler_len:4d} words: margin {margin:+.3f}  true_nll/token {true_nll:.3f}")


def fewshot_kv_retrieval_probe(model, tokenizer, filler_lengths):
    names = ["Mira", "Dax", "Lena", "Orin", "Sato", "Vera"]
    codes = [" 8472", " 1935", " 6208", " 4719", " 3064", " 9126"]
    filler_sentence = (
        " The archive contains ordinary notes about weather, gardens, tools, "
        "school schedules, and public meetings."
    )
    examples = "\n".join(
        f"Question: Code for {name}?\nAnswer:{code}" for name, code in zip(names[1:], codes[1:])
    )
    print("\n== Few-Shot Key-Value Retrieval Logprob ==")
    print("The target fact is before filler; other Q/A examples teach the format.")
    for filler_len in filler_lengths:
        filler = (filler_sentence * math.ceil(filler_len / 20))[: filler_len * 5]
        facts = "\n".join(f"Name: {n} Code:{c}" for n, c in zip(names, codes))
        prompt = f"{facts}\n{filler}\n{examples}\nQuestion: Code for Mira?\nAnswer:"
        true_lp, true_nll = answer_logprob(model, tokenizer, prompt, codes[0])
        decoys = [answer_logprob(model, tokenizer, prompt, c)[0] for c in codes[1:]]
        margin = true_lp - max(decoys)
        print(f"filler approx {filler_len:4d} words: margin {margin:+.3f}  true_nll/token {true_nll:.3f}")


def induction_probe(model, tokenizer, repeats):
    print("\n== Induction / Copy Probe ==")
    print("Pattern: rare phrase appears, then partial phrase recurs; score next token.")
    phrase = " zircon maple lantern"
    answer = " harbor"
    distractor = " velvet"
    filler = " ".join(["ordinary text about markets and rivers"] * repeats)
    prompt = f"{phrase}{answer}. {filler}. Later we saw{phrase}"
    true_lp, true_nll = answer_logprob(model, tokenizer, prompt, answer)
    decoy_lp, _ = answer_logprob(model, tokenizer, prompt, distractor)
    print(f"filler repeats {repeats:3d}: margin {true_lp - decoy_lp:+.3f}  true_nll/token {true_nll:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_jax_span_hypergraph_lm.pkl")
    parser.add_argument("--lengths", default="64,128,256,512,1024,1536")
    parser.add_argument("--target-len", type=int, default=96)
    args = parser.parse_args()

    model, cfg, tokenizer, ckpt = load_torch_model(args.checkpoint)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    lengths = [x for x in lengths if x + args.target_len + 1 <= cfg.block_size]
    print(f"checkpoint step={ckpt.get('step')} val_loss={ckpt.get('val_loss'):.4f} block_size={cfg.block_size}")

    context_loss_sweep(model, tokenizer, lengths, args.target_len)
    kv_retrieval_probe(model, tokenizer, [20, 80, 200, 500, 900])
    fewshot_kv_retrieval_probe(model, tokenizer, [20, 80, 200, 500, 900])
    induction_probe(model, tokenizer, 30)
    induction_probe(model, tokenizer, 120)


if __name__ == "__main__":
    main()
