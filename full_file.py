import os

import torch

from model import EfficientHGConfig, EfficientHypergraphLM, count_parameters, make_loaders, train_model


# -----------------------
# User-editable settings
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = 1337

# Dataset: FineWeb-Edu sample-10BT is much larger than Tiny Shakespeare.
# You can swap to e.g. dataset_name="roneneldan/TinyStories" for a smaller/debug dataset.
dataset_name = "HuggingFaceFW/fineweb-edu"
dataset_config = "sample-10BT"
dataset_split = "train"
text_field = "text"
tokenizer_name = "gpt2"
streaming = True
shuffle_buffer = 50_000
val_docs = 2_000

# Training shape. Start modestly. Increase block_size once the script works.
batch_size = 4
block_size = 2048
num_workers = 0

# Model size.
n_embd = 384
n_head = 6
n_local_attn_layers = 1
n_span_layers = 6
n_compressed_memory_layers = 1
span_widths = (2, 4, 8, 16, 32, 64)
local_window = 256
compression_block = 64

# Optimization.
max_iters = 10_000
eval_interval = 500
eval_iters = 50
learning_rate = 3e-4
min_lr_ratio = 0.1
warmup_iters = 200
weight_decay = 0.1
grad_clip = 1.0
dropout = 0.1
use_amp = True and device.startswith("cuda")
use_compile = False
save_best_checkpoint = True
checkpoint_path = "best_hca_span_hypergraph_lm.pt"

# Generation.
generate_tokens = 400
temperature = 0.8
top_k = 50
prompt = "The meaning of intelligence is"


def set_seed():
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        print("GPU:", torch.cuda.get_device_name(0))
    print("device:", device)


def build_config(vocab_size):
    return EfficientHGConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_local_attn_layers=n_local_attn_layers,
        n_span_layers=n_span_layers,
        n_compressed_memory_layers=n_compressed_memory_layers,
        span_widths=span_widths,
        local_window=local_window,
        compression_block=compression_block,
        dropout=dropout,
    )


def smoke_test(model, train_loader):
    xb, yb = next(iter(train_loader))
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = bool(use_amp and device_type == "cuda")

    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            logits, loss = model(xb, yb)
    print("x:", xb.shape, "logits:", logits.shape, "loss:", float(loss))


def generate_text(model, tokenizer):
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        state_model.load_state_dict(ckpt["model"])
        print("loaded checkpoint", checkpoint_path, "val_loss", ckpt.get("val_loss"), "step", ckpt.get("step"))

    ids = tokenizer.encode(prompt, add_special_tokens=False)
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=generate_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    print(tokenizer.decode(out))


def main():
    set_seed()
    train_loader, val_loader, tokenizer = make_loaders(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        tokenizer_name=tokenizer_name,
        text_field=text_field,
        block_size=block_size,
        batch_size=batch_size,
        streaming=streaming,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        val_docs=val_docs,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    vocab_size = len(tokenizer)
    print("vocab_size:", vocab_size)
    print("eos token:", tokenizer.eos_token, tokenizer.eos_token_id)

    cfg = build_config(vocab_size)
    model = EfficientHypergraphLM(cfg).to(device)
    print(model)
    print(f"parameters: {count_parameters(model)/1e6:.2f}M")

    if use_compile:
        print("Compiling model...")
        model = torch.compile(model)

    smoke_test(model, train_loader)
    train_model(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tokenizer_name=tokenizer_name,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
        learning_rate=learning_rate,
        min_lr_ratio=min_lr_ratio,
        warmup_iters=warmup_iters,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        use_amp=use_amp,
        save_best_checkpoint=save_best_checkpoint,
        checkpoint_path=checkpoint_path,
    )
    generate_text(model, tokenizer)


if __name__ == "__main__":
    main()
