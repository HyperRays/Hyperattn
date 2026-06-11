import numpy as np


class CompressedTokenizer:
    """Maps tokenizer ids onto a smaller id space where surface-equivalent tokens collide.

    Pure NumPy preprocessing; `transformers`/`tokenizers` are imported lazily so the
    rest of the package works offline.
    """

    def __init__(self, tokenizer_name_or_path):
        from tokenizers import Regex, normalizers
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            lookup[tid] = nid

        return lookup, len(new_tokens)

    def __call__(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        out = arr.copy()
        pos_mask = arr >= 0
        out[pos_mask] = self.lookup_table[arr[pos_mask]]
        return out
