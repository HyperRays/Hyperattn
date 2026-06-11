import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

import engram_jax
from engram_torch.reference import Engram, MultiHeadEmbedding, ShortConv

RTOL = 1e-4
ATOL = 1e-5


def small_configs():
    engram_cfg = engram_jax.EngramConfig(
        engram_vocab_size=[97, 89],
        max_ngram_size=3,
        n_embed_per_ngram=8,
        n_head_per_ngram=2,
        layer_ids=[1, 5],
        pad_id=2,
        seed=0,
        kernel_size=4,
    )
    backbone_cfg = engram_jax.BackBoneConfig(hidden_size=12, hc_mult=3, vocab_size=50)
    return engram_cfg, backbone_cfg


def make_hash_mapping(engram_cfg, vocab_size=50):
    return engram_jax.NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=engram_cfg.layer_ids,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
        tokenizer_vocab_size=vocab_size,
    )


def reference_ngram_hashes(mapping, input_ids, layer_id):
    """Verbatim port of the original demo's loop-based _get_ngram_hashes."""
    x = np.asarray(input_ids, dtype=np.int64)
    B, T = x.shape
    multipliers = mapping.layer_multipliers[layer_id]

    def shift_k(k):
        if k == 0:
            return x
        return np.pad(x, ((0, 0), (k, 0)), mode="constant", constant_values=mapping.pad_id)[:, :T]

    base_shifts = [shift_k(k) for k in range(mapping.max_ngram_size)]
    all_hashes = []
    for n in range(2, mapping.max_ngram_size + 1):
        tokens = base_shifts[:n]
        mix = tokens[0] * multipliers[0]
        for k in range(1, n):
            mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
        head_vocab_sizes = mapping.vocab_size_across_layers[layer_id][n - 2]
        for j in range(mapping.n_head_per_ngram):
            all_hashes.append((mix % int(head_vocab_sizes[j])).astype(np.int64))
    return np.stack(all_hashes, axis=2)


class EngramHashingTest(unittest.TestCase):
    def test_vectorized_hash_matches_reference_loop(self):
        engram_cfg, _ = small_configs()
        mapping = make_hash_mapping(engram_cfg, vocab_size=1000)
        rng = np.random.default_rng(0)
        input_ids = rng.integers(0, 1000, size=(3, 17))
        hashes = mapping.hash(input_ids)
        for layer_id in engram_cfg.layer_ids:
            expected = reference_ngram_hashes(mapping, input_ids, layer_id)
            np.testing.assert_array_equal(hashes[layer_id], expected)
            self.assertEqual(hashes[layer_id].dtype, np.int32)

    def test_hash_short_sequences(self):
        engram_cfg, _ = small_configs()
        mapping = make_hash_mapping(engram_cfg)
        for T in (1, 2):
            input_ids = np.zeros((2, T), dtype=np.int64)
            for layer_id in engram_cfg.layer_ids:
                expected = reference_ngram_hashes(mapping, input_ids, layer_id)
                np.testing.assert_array_equal(mapping.hash(input_ids)[layer_id], expected)

    def test_head_vocab_sizes_are_distinct_primes(self):
        engram_cfg, _ = small_configs()
        mapping = make_hash_mapping(engram_cfg)
        sizes = [
            x
            for layer in mapping.vocab_size_across_layers.values()
            for heads in layer
            for x in heads
        ]
        self.assertEqual(len(sizes), len(set(sizes)))


class EngramParityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.engram_cfg, self.backbone_cfg = small_configs()
        self.mapping = make_hash_mapping(self.engram_cfg, vocab_size=self.backbone_cfg.vocab_size)
        self.layer_id = self.engram_cfg.layer_ids[0]
        self.torch_engram = Engram(self.layer_id, self.engram_cfg, self.backbone_cfg, self.mapping).eval()
        self.params, self.spec = engram_jax.from_torch_engram(self.torch_engram)

        rng = np.random.default_rng(1)
        self.input_ids = rng.integers(0, self.backbone_cfg.vocab_size, size=(2, 11))
        self.hidden = rng.standard_normal(
            (2, 11, self.backbone_cfg.hc_mult, self.backbone_cfg.hidden_size)
        ).astype("float32")
        self.hash_ids = jnp.asarray(self.mapping.hash(self.input_ids)[self.layer_id])

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL)

    def test_short_conv_parity(self):
        x = np.random.default_rng(2).standard_normal((2, 9, 3, 12)).astype("float32")
        with torch.no_grad():
            expected = self.torch_engram.short_conv(torch.from_numpy(x))
        actual = engram_jax.short_conv(self.params["short_conv"], jnp.asarray(x), self.spec)
        self.assert_close(actual, expected.numpy())

    def test_multi_head_embedding_parity(self):
        with torch.no_grad():
            expected = self.torch_engram.multi_head_embedding(torch.from_numpy(np.asarray(self.hash_ids)))
        actual = engram_jax.multi_head_embedding(self.params["embedding"], self.hash_ids)
        self.assert_close(actual, expected.flatten(start_dim=-2).numpy())

    def test_engram_forward_parity(self):
        with torch.no_grad():
            expected = self.torch_engram(torch.from_numpy(self.hidden), self.input_ids)
        actual = engram_jax.engram_forward(self.params, jnp.asarray(self.hidden), self.hash_ids, self.spec)
        self.assert_close(actual, expected.numpy())

    def test_engram_forward_jit_parity(self):
        jitted = jax.jit(engram_jax.engram_forward, static_argnames="spec")
        with torch.no_grad():
            expected = self.torch_engram(torch.from_numpy(self.hidden), self.input_ids)
        actual = jitted(self.params, jnp.asarray(self.hidden), self.hash_ids, self.spec)
        self.assert_close(actual, expected.numpy())

    def test_engram_grad_smoke(self):
        def loss(params, hidden):
            out = engram_jax.engram_forward(params, hidden, self.hash_ids, self.spec)
            return jnp.sum(jnp.square(out))

        grads = jax.grad(loss, allow_int=True)(self.params, jnp.asarray(self.hidden))
        for leaf in jax.tree.leaves(grads):
            if jnp.issubdtype(leaf.dtype, jnp.floating):
                self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

    def test_init_params_forward(self):
        head_sizes = [x for heads in self.mapping.vocab_size_across_layers[self.layer_id] for x in heads]
        params = engram_jax.init_engram_params(jax.random.PRNGKey(0), head_sizes, self.spec)
        out = engram_jax.engram_forward(params, jnp.asarray(self.hidden), self.hash_ids, self.spec)
        self.assertEqual(out.shape, self.hidden.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))


if __name__ == "__main__":
    unittest.main()
