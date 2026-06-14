import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jax_model2
from jax_model2.span import delayed_span_hypergraph


class JaxModel2Test(unittest.TestCase):
    def setUp(self):
        self.cfg = jax_model2.LayerRoutedHGConfig(
            vocab_size=53,
            block_size=10,
            n_embd=24,
            n_head=4,
            local_window=5,
            block_layout=("attn", "far_span", "attn", "mid_span", "local_span"),
            far_span_widths=(3, 5),
            far_span_lags=(3, 5),
            mid_span_widths=(3, 5),
            mid_span_lags=(1, 2),
            local_span_widths=(2, 4),
            local_span_lags=(0,),
        )
        self.params = jax_model2.init_params(jax.random.PRNGKey(123), self.cfg)
        self.idx = (jnp.arange(20, dtype=jnp.int32).reshape(2, 10) * 7) % self.cfg.vocab_size
        self.targets = (self.idx + 1) % self.cfg.vocab_size

    def test_delayed_span_means(self):
        h = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)
        specs = ((2, 0), (2, 2), (3, 4))
        actual = np.asarray(delayed_span_hypergraph(jnp.asarray(h), specs))
        expected = []
        for width, lag in specs:
            cols = []
            for t in range(h.shape[1]):
                end = max(0, t + 1 - lag)
                start = max(0, end - width)
                if end > start:
                    cols.append(h[:, start:end].mean(axis=1))
                else:
                    cols.append(np.zeros((1, h.shape[-1]), dtype=np.float32))
            expected.append(np.stack(cols, axis=1))
        np.testing.assert_allclose(actual, np.concatenate(expected, axis=-1), rtol=1e-6, atol=1e-6)

    def test_forward_loss_and_grad(self):
        logits = jax_model2.forward(
            self.params,
            self.idx,
            self.cfg,
            attention_backend="chunked",
            span_backend="fused",
            remat_blocks=True,
        )
        self.assertEqual(logits.shape, (2, 10, self.cfg.vocab_size))
        loss, grads = jax.value_and_grad(jax_model2.loss)(
            self.params,
            self.idx,
            self.targets,
            self.cfg,
            attention_backend="chunked",
            span_backend="fused",
            remat_blocks=True,
        )
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(jax.tree.structure(grads), jax.tree.structure(self.params))

    def test_scan_span_runs_parity(self):
        # A layout deep enough that, with a small source cap, the trailing local_span run is
        # fully capped (starts at block index >= max_sources) and therefore gets scanned.
        cfg = jax_model2.LayerRoutedHGConfig(
            vocab_size=53,
            block_size=12,
            n_embd=24,
            n_head=4,
            local_window=5,
            block_layout=("attn", "mid_span", "mid_span", "local_span", "local_span", "local_span", "local_span"),
            mid_span_widths=(3, 5),
            mid_span_lags=(1, 2),
            local_span_widths=(2, 4),
            local_span_lags=(0,),
            layer_attn_max_sources=3,  # local run starts at block 3 >= 3 -> scannable
        )
        params = jax_model2.init_params(jax.random.PRNGKey(7), cfg)
        idx = (jnp.arange(24, dtype=jnp.int32).reshape(2, 12) * 5) % cfg.vocab_size
        targets = (idx + 1) % cfg.vocab_size

        kw = dict(attention_backend="chunked", span_backend="fused", remat_blocks=True)
        logits_unrolled = jax_model2.forward(params, idx, cfg, scan_span_runs=False, **kw)
        logits_scanned = jax_model2.forward(params, idx, cfg, scan_span_runs=True, **kw)
        np.testing.assert_allclose(
            np.asarray(logits_scanned), np.asarray(logits_unrolled), rtol=1e-5, atol=1e-5
        )

        # jax-metal segfaults when differentiating through lax.scan; grad parity is verified on
        # CPU/TPU (where the real run lives), see [[jax-metal-lowering-pitfalls]].
        if jax.default_backend() == "METAL":
            self.skipTest("jax-metal cannot differentiate through lax.scan; grad parity checked on CPU/TPU")

        l0, g0 = jax.value_and_grad(jax_model2.loss)(params, idx, targets, cfg, scan_span_runs=False, **kw)
        l1, g1 = jax.value_and_grad(jax_model2.loss)(params, idx, targets, cfg, scan_span_runs=True, **kw)
        np.testing.assert_allclose(float(l1), float(l0), rtol=1e-5, atol=1e-5)
        for a, b in zip(jax.tree_util.tree_leaves(g0), jax.tree_util.tree_leaves(g1)):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-4, atol=1e-5)

    def _memory_cfg(self):
        return jax_model2.LayerRoutedHGConfig(
            vocab_size=53,
            block_size=12,
            n_embd=24,
            n_head=4,
            local_window=5,
            block_layout=("attn", "far_span", "far_span", "mid_span", "mid_span", "local_span", "local_span", "local_span"),
            far_span_widths=(3, 5),
            far_span_lags=(3, 5),
            mid_span_widths=(3, 5),
            mid_span_lags=(1, 2),
            local_span_widths=(2, 4),
            local_span_lags=(0,),
            use_memory_router=True,
            layer_memory_slots=4,
        )

    def test_memory_router_runs_and_grad(self):
        cfg = self._memory_cfg()
        params = jax_model2.init_params(jax.random.PRNGKey(3), cfg)
        self.assertEqual(params["mem_init"].shape, (cfg.layer_memory_slots - 1, cfg.n_embd))
        self.assertIn("write_score", params["blocks"][0]["route"])
        idx = (jnp.arange(24, dtype=jnp.int32).reshape(2, 12) * 5) % cfg.vocab_size
        targets = (idx + 1) % cfg.vocab_size
        kw = dict(attention_backend="chunked", span_backend="fused", remat_blocks=True)
        logits = jax_model2.forward(params, idx, cfg, scan_span_runs=True, **kw)
        self.assertEqual(logits.shape, (2, 12, cfg.vocab_size))
        if jax.default_backend() == "METAL":
            self.skipTest("jax-metal cannot differentiate through lax.scan; grad checked on CPU/TPU")
        loss, grads = jax.value_and_grad(jax_model2.loss)(params, idx, targets, cfg, scan_span_runs=True, **kw)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(jax.tree.structure(grads), jax.tree.structure(params))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(g))) for g in jax.tree_util.tree_leaves(grads)))

    def test_memory_router_scan_parity(self):
        # The scan is only a compile optimization, so memory-mode scan must match the unrolled
        # path exactly (validates the (x, mem) carry threading).
        cfg = self._memory_cfg()
        params = jax_model2.init_params(jax.random.PRNGKey(4), cfg)
        idx = (jnp.arange(24, dtype=jnp.int32).reshape(2, 12) * 7) % cfg.vocab_size
        kw = dict(attention_backend="chunked", span_backend="fused", remat_blocks=True)
        a = jax_model2.forward(params, idx, cfg, scan_span_runs=False, **kw)
        b = jax_model2.forward(params, idx, cfg, scan_span_runs=True, **kw)
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-5, atol=1e-5)

    def test_memory_router_token_causal(self):
        # Per-token router + causal blocks -> logits at position t depend only on tokens <= t.
        cfg = self._memory_cfg()
        params = jax_model2.init_params(jax.random.PRNGKey(5), cfg)
        full = (jnp.arange(24, dtype=jnp.int32).reshape(2, 12) * 3) % cfg.vocab_size
        kw = dict(attention_backend="chunked", span_backend="fused")
        ref = jax_model2.forward(params, full, cfg, **kw)
        for T in (1, 5, 8, 11):
            sub = jax_model2.forward(params, full[:, :T], cfg, **kw)
            np.testing.assert_allclose(np.asarray(sub), np.asarray(ref[:, :T]), rtol=1e-4, atol=1e-5)

    def test_layer_attention_weight_shapes(self):
        weights = jax_model2.layer_attention_weights(self.params, self.idx, self.cfg)
        self.assertEqual(len(weights), len(self.cfg.block_layout) - 1)
        for source_count, weight in enumerate(weights, start=2):
            self.assertEqual(weight.shape, (2, source_count, 10))
            sums = np.asarray(jnp.sum(weight, axis=1))
            np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
