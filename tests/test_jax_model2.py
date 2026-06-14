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

    def test_layer_attention_weight_shapes(self):
        weights = jax_model2.layer_attention_weights(self.params, self.idx, self.cfg)
        self.assertEqual(len(weights), len(self.cfg.block_layout) - 1)
        for source_count, weight in enumerate(weights, start=2):
            self.assertEqual(weight.shape, (2, source_count, 10))
            sums = np.asarray(jnp.sum(weight, axis=1))
            np.testing.assert_allclose(sums, np.ones_like(sums), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
