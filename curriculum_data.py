"""Curriculum data pipeline for the jax_model2 runner.

Trains on progressively harder corpora with hard cuts at step boundaries:

    1. grammar-heavy   -> Simple English Wikipedia   (wikimedia/wikipedia 20231101.simple)
    2. informational   -> full English Wikipedia      (wikimedia/wikipedia 20231101.en)
    3. information-dense -> full-text arXiv papers     (togethercomputer/RedPajama-Data-1T:arxiv)

Packing matches the runner's original StreamingPackedTokens: each document is tokenized,
terminated with an EOS id, the ids are concatenated, and the flat stream is sliced into
fixed ``block_size + 1`` windows -> ``x = window[:-1]``, ``y = window[1:]``. The trailing
partial window of each epoch is dropped (the stream reshuffles and repeats forever).

The HuggingFace imports live inside the functions that need them so the scheduling and
packing core can be imported and unit tested without ``datasets``/``transformers``.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# Phase scheduling (pure, network-free)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CurriculumPhase:
    label: str
    dataset_name: str
    dataset_config: Optional[str]
    steps: int
    text_field: str = "text"
    split: str = "train"
    trust_remote_code: bool = False


def phase_boundaries(phases: List[CurriculumPhase]) -> List[int]:
    """Cumulative end step (exclusive) of each phase: phase i is active for step < bounds[i]."""
    bounds: List[int] = []
    acc = 0
    for p in phases:
        acc += p.steps
        bounds.append(acc)
    return bounds


def phase_for_step(phases: List[CurriculumPhase], step: int) -> int:
    """Index of the active phase at a global training step (clamped to the last phase)."""
    for i, end in enumerate(phase_boundaries(phases)):
        if step < end:
            return i
    return len(phases) - 1


def default_curriculum(
    max_iters: int,
    fractions: Tuple[float, float, float] = (0.2, 0.4, 0.4),
    *,
    arxiv_dataset: str = "togethercomputer/RedPajama-Data-1T",
    arxiv_config: Optional[str] = "arxiv",
    arxiv_text_field: str = "text",
    arxiv_trust_remote_code: bool = True,
) -> List[CurriculumPhase]:
    """Three-stage curriculum sized as fractions of ``max_iters``.

    The last phase absorbs the rounding remainder so the phase steps sum exactly to
    ``max_iters``. Swap the arXiv source to ``ccdv/arxiv-summarization`` (config
    ``document``, field ``article``) if RedPajama streaming is unavailable.
    """
    f1, f2, _f3 = fractions
    s1 = round(f1 * max_iters)
    s2 = round(f2 * max_iters)
    s3 = max_iters - s1 - s2
    return [
        CurriculumPhase("simple-wiki", "wikimedia/wikipedia", "20231101.simple", s1, text_field="text"),
        CurriculumPhase("en-wiki", "wikimedia/wikipedia", "20231101.en", s2, text_field="text"),
        CurriculumPhase(
            "arxiv",
            arxiv_dataset,
            arxiv_config,
            s3,
            text_field=arxiv_text_field,
            trust_remote_code=arxiv_trust_remote_code,
        ),
    ]


# --------------------------------------------------------------------------------------
# Packing (pure, network-free)
# --------------------------------------------------------------------------------------
def pack_token_stream(
    token_lists: Iterable[List[int]],
    eos_id: int,
    block_size: int,
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Pack a stream of per-document token-id lists into ``(x, y)`` batches.

    Each document is terminated with ``eos_id`` and the ids are concatenated into a flat
    buffer, which is cut into contiguous ``block_size + 1`` windows. The trailing partial
    window is left unflushed (dropped when the caller restarts the document stream).
    """
    token_buffer: List[int] = []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for ids in token_lists:
        token_buffer.extend(ids)
        token_buffer.append(eos_id)
        while len(token_buffer) >= block_size + 1:
            chunk = np.asarray(token_buffer[: block_size + 1], dtype=np.int32)
            token_buffer = token_buffer[block_size + 1 :]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            if len(xs) == batch_size:
                yield np.stack(xs), np.stack(ys)
                xs, ys = [], []


# --------------------------------------------------------------------------------------
# HuggingFace streaming glue
# --------------------------------------------------------------------------------------
def _hf_document_tokens(phase, tokenizer, *, shuffle, seed, shuffle_buffer, skip_docs, take_docs):
    from datasets import load_dataset

    kwargs = dict(split=phase.split, streaming=True)
    if phase.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if phase.dataset_config is None:
        ds = load_dataset(phase.dataset_name, **kwargs)
    else:
        ds = load_dataset(phase.dataset_name, phase.dataset_config, **kwargs)

    # Holdout slicing happens before shuffling so train and validation never overlap.
    if skip_docs:
        ds = ds.skip(skip_docs)
    if take_docs is not None:
        ds = ds.take(take_docs)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    for row in ds:
        text = row.get(phase.text_field)
        if not isinstance(text, str) or len(text) == 0:
            continue
        yield tokenizer.encode(text, add_special_tokens=False)


def phase_batch_iter(
    phase,
    tokenizer,
    *,
    block_size,
    batch_size,
    seed,
    shuffle_buffer,
    shuffle,
    skip_docs=0,
    take_docs=None,
):
    """Infinite stream of packed ``(x, y)`` batches for a single phase/dataset."""
    eos_id = tokenizer.eos_token_id
    epoch = 0
    while True:
        docs = _hf_document_tokens(
            phase,
            tokenizer,
            shuffle=shuffle,
            seed=seed + epoch,
            shuffle_buffer=shuffle_buffer,
            skip_docs=skip_docs,
            take_docs=take_docs,
        )
        yielded_any = False
        for batch in pack_token_stream(docs, eos_id, block_size, batch_size):
            yielded_any = True
            yield batch
        if not yielded_any:
            raise RuntimeError(
                f"phase {phase.label!r}: {phase.dataset_name}[{phase.dataset_config}] yielded no "
                f"examples; check the name/config/text_field"
            )
        epoch += 1


class Prefetcher:
    """Run a batch iterator on a background thread so host data prep overlaps device compute.

    The Rust tokenizer and file IO release the GIL, so the next batch is built while the
    device runs the current step. ``close()`` lets the worker exit when a phase is retired.
    """

    def __init__(self, iterator, size=4):
        self._q: queue.Queue = queue.Queue(maxsize=size)
        self._stop = threading.Event()
        self._sentinel = object()
        self._thread = threading.Thread(target=self._worker, args=(iterator,), daemon=True)
        self._thread.start()

    def _put(self, item):
        # Block until there is room, but stay responsive to close().
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self, iterator):
        try:
            for item in iterator:
                if self._stop.is_set():
                    return
                if not self._put(item):
                    return
        except Exception as exc:  # surface producer errors to the consumer
            self._put(exc)
        else:
            self._put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        if item is self._sentinel:
            raise StopIteration
        return item

    def close(self):
        self._stop.set()


class CurriculumLoader:
    """Step-indexed batch source that hard-switches datasets at phase boundaries.

    The training loop's ``step`` stays authoritative: call ``batch(step)`` once per step.
    The active phase's stream is built lazily and the previous one is closed on transition,
    so only one HuggingFace stream runs at a time.
    """

    def __init__(
        self,
        *,
        phases: List[CurriculumPhase],
        tokenizer,
        block_size: int,
        batch_size: int,
        shuffle_buffer: int = 10_000,
        seed: int = 0,
        skip_docs: int = 0,
        prefetch: int = 4,
    ):
        self.phases = list(phases)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.skip_docs = skip_docs
        self.prefetch = prefetch
        self._active_idx: Optional[int] = None
        self._prefetcher: Optional[Prefetcher] = None

    def _start_phase(self, idx: int):
        if self._prefetcher is not None:
            self._prefetcher.close()
        phase = self.phases[idx]
        # Distinct per-phase seed so phases shuffle independently and reproducibly.
        it = phase_batch_iter(
            phase,
            self.tokenizer,
            block_size=self.block_size,
            batch_size=self.batch_size,
            seed=self.seed + 1009 * (idx + 1),
            shuffle_buffer=self.shuffle_buffer,
            shuffle=True,
            skip_docs=self.skip_docs,
            take_docs=None,
        )
        self._prefetcher = Prefetcher(it, size=self.prefetch)
        self._active_idx = idx

    def phase_index(self, step: int) -> int:
        return phase_for_step(self.phases, step)

    def phase_label(self, step: int) -> str:
        return self.phases[self.phase_index(step)].label

    def batch(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        idx = self.phase_index(step)
        if idx != self._active_idx:
            self._start_phase(idx)
        return next(self._prefetcher)

    def close(self):
        if self._prefetcher is not None:
            self._prefetcher.close()


def build_val_iter(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    text_field: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    shuffle_buffer: int,
    seed: int,
    val_docs: int,
    split: str = "train",
    trust_remote_code: bool = False,
    prefetch: int = 4,
) -> Prefetcher:
    """Fixed, neutral held-out validation iterator (outside the curriculum).

    Held-out so ``val_loss`` stays comparable across phases and measures generalization
    rather than in-domain fit. Loops the first ``val_docs`` documents forever, unshuffled.
    """
    phase = CurriculumPhase(
        "val",
        dataset_name,
        dataset_config,
        steps=0,
        text_field=text_field,
        split=split,
        trust_remote_code=trust_remote_code,
    )
    it = phase_batch_iter(
        phase,
        tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
        shuffle=False,
        skip_docs=0,
        take_docs=val_docs,
    )
    return Prefetcher(it, size=prefetch)
