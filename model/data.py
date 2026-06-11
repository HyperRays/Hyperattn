from typing import Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError("Install dependencies with: pip install datasets transformers accelerate") from e


class StreamingPackedTokenDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        tokenizer_name: str,
        text_field: str,
        block_size: int,
        streaming: bool = True,
        shuffle: bool = True,
        shuffle_buffer: int = 10_000,
        seed: int = 0,
        skip_docs: int = 0,
        take_docs: Optional[int] = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.text_field = text_field
        self.block_size = block_size
        self.streaming = streaming
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.skip_docs = skip_docs
        self.take_docs = take_docs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        self.eos_id = self.tokenizer.eos_token_id

    def _make_stream(self):
        if self.dataset_config is None:
            ds = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
        else:
            ds = load_dataset(self.dataset_name, self.dataset_config, split=self.split, streaming=self.streaming)

        # Apply holdout slicing before shuffling so train and validation do not overlap.
        if self.skip_docs:
            ds = ds.skip(self.skip_docs)
        if self.take_docs is not None:
            ds = ds.take(self.take_docs)
        if self.shuffle:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        return ds

    def __iter__(self):
        while True:
            token_buffer = []
            yielded_any = False
            for row in self._make_stream():
                text = row.get(self.text_field, None)
                if not isinstance(text, str) or len(text) == 0:
                    continue
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                ids.append(self.eos_id)
                token_buffer.extend(ids)

                while len(token_buffer) >= self.block_size + 1:
                    chunk = token_buffer[: self.block_size + 1]
                    token_buffer = token_buffer[self.block_size + 1 :]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yielded_any = True
                    yield x, y

            if self.take_docs is not None:
                break
            if not yielded_any:
                raise RuntimeError("Dataset iterator yielded no examples. Check dataset config/text field.")


def make_loaders(
    *,
    dataset_name: str,
    dataset_config: Optional[str],
    dataset_split: str,
    tokenizer_name: str,
    text_field: str,
    block_size: int,
    batch_size: int,
    streaming: bool,
    shuffle_buffer: int,
    seed: int,
    val_docs: int,
    num_workers: int,
    pin_memory: bool = False,
):
    train_ds = StreamingPackedTokenDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        tokenizer_name=tokenizer_name,
        text_field=text_field,
        block_size=block_size,
        streaming=streaming,
        shuffle=True,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        skip_docs=val_docs,
        take_docs=None,
    )
    val_ds = StreamingPackedTokenDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=dataset_split,
        tokenizer_name=tokenizer_name,
        text_field=text_field,
        block_size=block_size,
        streaming=streaming,
        shuffle=False,
        seed=seed,
        skip_docs=0,
        take_docs=val_docs,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, train_ds.tokenizer
