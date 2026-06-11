import numpy as np
from sympy import isprime

from .tokenizer import CompressedTokenizer

_PRIME_1 = 10007


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Vectorized n-gram rolling hashes, matching the torch reference bit-for-bit.

    Hashing is integer-only, non-differentiable preprocessing, so it runs in NumPy
    int64 on the host (jax-metal has no int64) and only the final per-head ids
    (always < head vocab size < 2**31) cross to the device as int32.

    Pass `compressed_tokenizer=None` with an explicit `tokenizer_vocab_size` to use
    pre-compressed ids without instantiating a real tokenizer (e.g. in tests).
    """

    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_head_per_ngram,
        layer_ids,
        pad_id,
        seed,
        tokenizer_name_or_path=None,
        compressed_tokenizer=None,
        tokenizer_vocab_size=None,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_head_per_ngram = n_head_per_ngram
        self.layer_ids = list(layer_ids)
        self.pad_id = pad_id

        if compressed_tokenizer is None and tokenizer_name_or_path is not None:
            compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path)
        self.compressed_tokenizer = compressed_tokenizer
        if compressed_tokenizer is not None:
            self.tokenizer_vocab_size = len(compressed_tokenizer)
            if self.pad_id is not None:
                self.pad_id = int(compressed_tokenizer.lookup_table[self.pad_id])
        else:
            if tokenizer_vocab_size is None:
                raise ValueError("provide tokenizer_vocab_size when no compressed tokenizer is given")
            self.tokenizer_vocab_size = int(tokenizer_vocab_size)

        max_long = np.iinfo(np.int64).max
        m_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, m_max // 2)

        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            g = np.random.default_rng(int(seed + _PRIME_1 * int(layer_id)))
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()
        # (n_ngrams, n_head) int64 modulus matrix per layer, for one broadcasted `%`.
        self._layer_mods = {
            layer_id: np.asarray(sizes, dtype=np.int64)
            for layer_id, sizes in self.vocab_size_across_layers.items()
        }

    def _calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                heads = []
                search_start = self.vocab_size_per_ngram[ngram - 2] - 1
                for _ in range(self.n_head_per_ngram):
                    prime = find_next_prime(search_start, seen_primes)
                    seen_primes.add(prime)
                    heads.append(prime)
                    search_start = prime
                all_ngram_vocab_sizes.append(heads)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes(self, compressed_ids, layer_id):
        x = np.asarray(compressed_ids, dtype=np.int64)
        B, T = x.shape
        H = self.n_head_per_ngram

        # shifted[k, :, t] = x[:, t - k] (pad_id-filled), scaled by the layer multipliers
        shifted = np.empty((self.max_ngram_size, B, T), dtype=np.int64)
        shifted[0] = x
        for k in range(1, self.max_ngram_size):
            shifted[k, :, :k] = self.pad_id
            shifted[k, :, k:] = x[:, : T - k]
        scaled = shifted * self.layer_multipliers[layer_id][:, None, None]

        mods = self._layer_mods[layer_id]
        out = np.empty((B, T, (self.max_ngram_size - 1) * H), dtype=np.int32)
        mix = scaled[0]
        for n in range(2, self.max_ngram_size + 1):
            mix = np.bitwise_xor(mix, scaled[n - 1])
            out[:, :, (n - 2) * H : (n - 1) * H] = mix[..., None] % mods[n - 2]
        return out

    def hash(self, input_ids):
        if self.compressed_tokenizer is not None:
            input_ids = self.compressed_tokenizer(input_ids)
        return {layer_id: self._get_ngram_hashes(input_ids, layer_id) for layer_id in self.layer_ids}
