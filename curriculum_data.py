"""Curriculum data pipeline for the jax_model2 runner.

Trains on progressively harder corpora with hard cuts at step boundaries:

    1. grammar-heavy   -> Simple English Wikipedia   (wikimedia/wikipedia 20231101.simple)
    2. informational   -> full English Wikipedia      (wikimedia/wikipedia 20231101.en)
    3. information-dense -> full-text arXiv papers     (RedPajama raw URL stream:arxiv)

Packing matches the runner's original StreamingPackedTokens: each document is tokenized,
terminated with an EOS id, the ids are concatenated, and the flat stream is sliced into
fixed ``block_size + 1`` windows -> ``x = window[:-1]``, ``y = window[1:]``. The trailing
partial window of each epoch is dropped (the stream reshuffles and repeats forever).

The HuggingFace imports live inside the functions that need them so the scheduling and
packing core can be imported and unit tested without ``datasets``/``transformers``.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import queue
import random
import shutil
import threading
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple
from urllib.request import Request, urlopen

import numpy as np


REDPAJAMA_SOURCE = "redpajama_urls"
REDPAJAMA_MANIFEST_URL = "https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt"
REDPAJAMA_DATA_DIR_ENV = "RED_PAJAMA_DATA_DIR"
REDPAJAMA_URLS_FILE_ENV = "RED_PAJAMA_URLS_FILE"
REDPAJAMA_MANIFEST_URL_ENV = "RED_PAJAMA_MANIFEST_URL"
REDPAJAMA_DOC_SHUFFLE_BUFFER_ENV = "RED_PAJAMA_DOC_SHUFFLE_BUFFER"
REDPAJAMA_TOKENIZE_BATCH_ENV = "RED_PAJAMA_TOKENIZE_BATCH"
REDPAJAMA_PREFETCH_SHARDS_ENV = "RED_PAJAMA_PREFETCH_SHARDS"
REDPAJAMA_BACKGROUND_SHARDS_ENV = "RED_PAJAMA_BACKGROUND_SHARDS"
REDPAJAMA_DOWNLOAD_WORKERS_ENV = "RED_PAJAMA_DOWNLOAD_WORKERS"
REDPAJAMA_DOWNLOAD_VERBOSE_ENV = "RED_PAJAMA_DOWNLOAD_VERBOSE"

_REDPAJAMA_DOWNLOADERS = {}


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
    source: str = "hf"


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
    arxiv_trust_remote_code: bool = False,
    arxiv_source: str = REDPAJAMA_SOURCE,
) -> List[CurriculumPhase]:
    """Three-stage curriculum sized as fractions of ``max_iters``.

    The last phase absorbs the rounding remainder so the phase steps sum exactly to
    ``max_iters``. RedPajama is streamed from Together's raw URL manifest by default,
    avoiding the HuggingFace dataset script that current ``datasets`` versions reject.
    Set ``arxiv_source="hf"`` with e.g. ``ccdv/arxiv-summarization`` if you want the
    script-free HuggingFace fallback instead.
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
            source=arxiv_source,
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
# Document streaming glue
# --------------------------------------------------------------------------------------
def _redpajama_manifest_lines() -> List[str]:
    """Load Together's RedPajama URL manifest from a local file or from the public URL."""
    urls_file = os.environ.get(REDPAJAMA_URLS_FILE_ENV)
    if urls_file:
        with open(urls_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    manifest_url = os.environ.get(REDPAJAMA_MANIFEST_URL_ENV, REDPAJAMA_MANIFEST_URL)
    req = Request(manifest_url, headers={"User-Agent": "Hyperattn/RedPajamaRawStream"})
    with urlopen(req, timeout=60) as response:
        text = response.read().decode("utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _redpajama_subset_urls(subset: Optional[str]) -> List[str]:
    subset = subset or "arxiv"
    needle = f"/{subset}/"
    urls = [url for url in _redpajama_manifest_lines() if needle in url]
    if not urls:
        raise RuntimeError(f"RedPajama manifest contained no URLs for subset {subset!r}")
    return urls


def _redpajama_cache_path(url: str) -> Optional[str]:
    """Map a public RedPajama URL to its intended path under RED_PAJAMA_DATA_DIR."""
    root = os.environ.get(REDPAJAMA_DATA_DIR_ENV)
    if not root:
        return None
    marker = "/redpajama-data-1T/v1.0.0/"
    rel = url.split(marker, 1)[1] if marker in url else url.rsplit("/", 1)[-1]
    return os.path.join(root, rel)


def _local_redpajama_path(url: str) -> Optional[str]:
    """Return the cached RedPajama shard path if present."""
    path = _redpajama_cache_path(url)
    return path if os.path.exists(path) else None


def _download_redpajama_shard(url: str, *, verbose: bool = False) -> Optional[str]:
    """Download one RedPajama shard into RED_PAJAMA_DATA_DIR using an atomic rename."""
    path = _redpajama_cache_path(url)
    if path is None:
        return None
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    if verbose:
        print(f"[redpajama] downloading {url} -> {path}", flush=True)
    req = Request(url, headers={"User-Agent": "Hyperattn/RedPajamaRawStream"})
    try:
        with urlopen(req, timeout=120) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if verbose:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[redpajama] cached {path} ({size_mb:.1f} MiB)", flush=True)
    return path


class _RedPajamaBackgroundDownloader:
    def __init__(self, urls: List[str], *, workers: int, verbose: bool):
        self.urls = list(urls)
        self.verbose = verbose
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._threads = []
        for url in self.urls:
            if _local_redpajama_path(url) is None:
                self._q.put(url)
        for _ in range(max(1, int(workers))):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            self._threads.append(thread)

    def _worker(self):
        while not self._stop.is_set():
            try:
                url = self._q.get(timeout=0.5)
            except queue.Empty:
                return
            try:
                _download_redpajama_shard(url, verbose=self.verbose)
            except Exception as exc:
                if self.verbose:
                    print(f"[redpajama] background download failed for {url}: {exc}", flush=True)
            finally:
                self._q.task_done()

    def stop(self):
        self._stop.set()


def _maybe_prepare_redpajama_cache(urls: List[str]) -> None:
    """Optionally predownload/cache RedPajama shards before and during streaming.

    Enabled only when RED_PAJAMA_DATA_DIR is set. Synchronous prefetch downloads the first
    RED_PAJAMA_PREFETCH_SHARDS URLs before yielding data. Background download then caches up
    to RED_PAJAMA_BACKGROUND_SHARDS URLs from this shuffled epoch order.
    """
    if not os.environ.get(REDPAJAMA_DATA_DIR_ENV):
        return
    prefetch_n = max(0, int(os.environ.get(REDPAJAMA_PREFETCH_SHARDS_ENV, "0")))
    background_n = max(0, int(os.environ.get(REDPAJAMA_BACKGROUND_SHARDS_ENV, "0")))
    workers = max(1, int(os.environ.get(REDPAJAMA_DOWNLOAD_WORKERS_ENV, "2")))
    verbose = os.environ.get(REDPAJAMA_DOWNLOAD_VERBOSE_ENV, "1") not in {"0", "false", "False"}
    if prefetch_n:
        for url in urls[:prefetch_n]:
            _download_redpajama_shard(url, verbose=verbose)
    if background_n:
        key = (os.environ.get(REDPAJAMA_DATA_DIR_ENV), tuple(urls[:background_n]))
        old = _REDPAJAMA_DOWNLOADERS.get(key)
        if old is None:
            _REDPAJAMA_DOWNLOADERS[key] = _RedPajamaBackgroundDownloader(
                urls[:background_n], workers=workers, verbose=verbose
            )


def _open_binary_url_or_file(url: str):
    local_path = _local_redpajama_path(url)
    if local_path is not None:
        return open(local_path, "rb")
    req = Request(url, headers={"User-Agent": "Hyperattn/RedPajamaRawStream"})
    return urlopen(req, timeout=120)


def _iter_jsonl_lines_from_url(url: str) -> Iterator[str]:
    raw = _open_binary_url_or_file(url)
    try:
        if url.endswith(".gz"):
            with gzip.GzipFile(fileobj=raw) as gz:
                wrapper = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
                yield from wrapper
        elif url.endswith(".zst") or url.endswith(".zstd"):
            try:
                import zstandard as zstd
            except ImportError as exc:
                raise RuntimeError(
                    "RedPajama shards are zstd-compressed; install `zstandard` or use "
                    "`uv add zstandard` before streaming raw RedPajama URLs."
                ) from exc
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw) as reader:
                wrapper = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield from wrapper
        else:
            wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            yield from wrapper
    finally:
        raw.close()


def _shuffle_buffered(items, *, buffer_size: int, seed: int):
    rng = random.Random(seed)
    buffer: List[List[int]] = []
    iterator = iter(items)
    for _ in range(max(0, buffer_size)):
        try:
            buffer.append(next(iterator))
        except StopIteration:
            break
    if not buffer:
        return
    for item in iterator:
        idx = rng.randrange(len(buffer))
        yield buffer[idx]
        buffer[idx] = item
    rng.shuffle(buffer)
    yield from buffer


def _batched_tokenize_texts(texts, tokenizer, *, batch_size: int):
    batch = []
    batch_size = max(1, int(batch_size))
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            try:
                for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                    yield ids
            except TypeError:
                for item in batch:
                    yield tokenizer.encode(item, add_special_tokens=False)
            batch = []
    if batch:
        try:
            for ids in tokenizer(batch, add_special_tokens=False)["input_ids"]:
                yield ids
        except TypeError:
            for item in batch:
                yield tokenizer.encode(item, add_special_tokens=False)


def _redpajama_document_tokens(phase, tokenizer, *, shuffle, seed, shuffle_buffer, skip_docs, take_docs):
    urls = _redpajama_subset_urls(phase.dataset_config)
    rng = random.Random(seed)
    if shuffle:
        urls = list(urls)
        rng.shuffle(urls)
    _maybe_prepare_redpajama_cache(urls)

    def docs():
        seen = 0
        yielded = 0
        for url in urls:
            for line in _iter_jsonl_lines_from_url(url):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row.get(phase.text_field)
                if not isinstance(text, str) or len(text) == 0:
                    continue
                if seen < skip_docs:
                    seen += 1
                    continue
                if take_docs is not None and yielded >= take_docs:
                    return
                seen += 1
                yielded += 1
                yield text

    texts = docs()
    if shuffle:
        doc_buffer = int(os.environ.get(REDPAJAMA_DOC_SHUFFLE_BUFFER_ENV, "256"))
        texts = _shuffle_buffered(texts, buffer_size=min(shuffle_buffer, doc_buffer), seed=seed)
    tokenize_batch = int(os.environ.get(REDPAJAMA_TOKENIZE_BATCH_ENV, "4"))
    yield from _batched_tokenize_texts(texts, tokenizer, batch_size=tokenize_batch)


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


def _document_tokens(phase, tokenizer, *, shuffle, seed, shuffle_buffer, skip_docs, take_docs):
    if phase.source == REDPAJAMA_SOURCE:
        return _redpajama_document_tokens(
            phase,
            tokenizer,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
            skip_docs=skip_docs,
            take_docs=take_docs,
        )
    if phase.source == "hf":
        return _hf_document_tokens(
            phase,
            tokenizer,
            shuffle=shuffle,
            seed=seed,
            shuffle_buffer=shuffle_buffer,
            skip_docs=skip_docs,
            take_docs=take_docs,
        )
    raise ValueError(f"unknown curriculum source {phase.source!r}")


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
        docs = _document_tokens(
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
