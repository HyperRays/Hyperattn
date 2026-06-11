import math
import time

import torch
import torch.nn as nn


def get_lr(step, *, learning_rate, warmup_iters, max_iters, min_lr_ratio):
    if step < warmup_iters:
        return learning_rate * step / max(1, warmup_iters)
    if step > max_iters:
        return learning_rate * min_lr_ratio
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * min_lr_ratio + coeff * (learning_rate - learning_rate * min_lr_ratio)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters, *, device, use_amp):
    model.eval()
    losses = []
    it = iter(val_loader)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = bool(use_amp and device_type == "cuda")
    for _ in range(eval_iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(val_loader)
            xb, yb = next(it)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            _, loss = model(xb, yb)
        losses.append(loss.detach())
    model.train()
    return torch.stack(losses).mean().item()


def print_gates(model):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    for name, module in m.named_modules():
        if hasattr(module, "gate") and isinstance(module.gate, nn.Parameter):
            raw = float(module.gate.detach().cpu())
            sig = float(torch.sigmoid(module.gate.detach()).cpu())
            print(f"{name:45s} raw={raw:+.4f} sigmoid={sig:.4f}")


def train_model(
    *,
    model,
    cfg,
    train_loader,
    val_loader,
    device,
    tokenizer_name,
    max_iters,
    eval_interval,
    eval_iters,
    learning_rate,
    min_lr_ratio,
    warmup_iters,
    weight_decay,
    grad_clip,
    use_amp,
    save_best_checkpoint,
    checkpoint_path,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = bool(use_amp and device_type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_iter = cycle(train_loader)
    best_val = float("inf")
    loss_ema = None
    tokens_since_eval = 0
    total_tokens = 0
    t0 = time.time()

    model.train()
    for step in range(max_iters + 1):
        if step % eval_interval == 0 or step == max_iters:
            elapsed = time.time() - t0
            toks_per_sec = 0.0 if step == 0 else tokens_since_eval / max(elapsed, 1e-9)
            val_loss = estimate_loss(model, val_loader, eval_iters, device=device, use_amp=use_amp)
            train_loss_str = "nan" if loss_ema is None else f"{loss_ema:.4f}"
            print(
                f"step {step:6d} | "
                f"train_ema {train_loss_str} | "
                f"val {val_loss:.4f} | "
                f"lr {get_lr(step, learning_rate=learning_rate, warmup_iters=warmup_iters, max_iters=max_iters, min_lr_ratio=min_lr_ratio):.2e} | "
                f"tok/s {toks_per_sec:,.0f} | "
                f"tokens {total_tokens:,} | "
                f"elapsed {elapsed:.1f}s"
            )
            print_gates(model)

            t0 = time.time()
            tokens_since_eval = 0
            if save_best_checkpoint and val_loss < best_val:
                best_val = val_loss
                state_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(
                    {
                        "model": state_model.state_dict(),
                        "config": cfg.__dict__,
                        "step": step,
                        "val_loss": val_loss,
                        "train_loss_ema": loss_ema,
                        "total_tokens": total_tokens,
                        "tokenizer_name": tokenizer_name,
                    },
                    checkpoint_path,
                )
                print(f"saved best checkpoint to {checkpoint_path} with val={best_val:.4f}")

        if step == max_iters:
            break

        lr = get_lr(
            step,
            learning_rate=learning_rate,
            warmup_iters=warmup_iters,
            max_iters=max_iters,
            min_lr_ratio=min_lr_ratio,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = next(train_iter)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        tokens_this_step = xb.numel()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
            _, loss = model(xb, yb)

        if not torch.isfinite(loss):
            print(f"non-finite loss at step {step}: {loss.item()}")
            continue

        loss_value = loss.detach().float().item()
        loss_ema = loss_value if loss_ema is None else 0.99 * loss_ema + 0.01 * loss_value

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        tokens_since_eval += tokens_this_step
        total_tokens += tokens_this_step

    return {
        "best_val": best_val,
        "loss_ema": loss_ema,
        "total_tokens": total_tokens,
    }
