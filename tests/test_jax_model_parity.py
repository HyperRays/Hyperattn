import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

import jax_model
from jax_model.layers import block_forward
from jax_model.rope import apply_rope as jax_apply_rope
from jax_model.rope import precompute_rope_cache as jax_rope_cache
from jax_model.span import einsum_fused_span_edge_projection, fused_span_edge_projection, span_hypergraph
from model import EfficientHGConfig, EfficientHypergraphLM
from model.rope import apply_rope as torch_apply_rope
from model.rope import precompute_rope_cache as torch_rope_cache


RTOL = 1e-4
ATOL = 1e-5


class JaxModelParityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.cfg = EfficientHGConfig(
            vocab_size=37,
            block_size=8,
            n_embd=16,
            n_head=4,
            n_local_attn_layers=1,
            n_span_layers=1,
            n_compressed_memory_layers=1,
            span_widths=(2, 4),
            local_window=4,
            compression_block=4,
            dropout=0.0,
        )
        self.torch_model = EfficientHypergraphLM(self.cfg).eval()
        self.params, self.jax_cfg = jax_model.from_torch_model(self.torch_model)
        self.idx = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.block_size))
        self.x = torch.randn(2, self.cfg.block_size, self.cfg.n_embd)

    def assert_close(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=RTOL, atol=ATOL)

    def test_rope_cache_and_apply(self):
        head_dim = self.cfg.n_embd // self.cfg.n_head
        x = torch.randn(2, self.cfg.n_head, self.cfg.block_size, head_dim)
        t_cos, t_sin = torch_rope_cache(head_dim, self.cfg.block_size, x.device)
        j_cos, j_sin = jax_rope_cache(head_dim, self.cfg.block_size)

        self.assert_close(j_cos, t_cos.numpy())
        self.assert_close(j_sin, t_sin.numpy())
        self.assert_close(jax_apply_rope(jnp.asarray(x.numpy()), j_cos, j_sin), torch_apply_rope(x, t_cos, t_sin).numpy())

    def test_local_attention_block(self):
        head_dim = self.cfg.n_embd // self.cfg.n_head
        t_cos, t_sin = torch_rope_cache(head_dim, self.cfg.block_size, self.x.device)
        j_cos, j_sin = jax_rope_cache(head_dim, self.cfg.block_size)

        with torch.no_grad():
            expected = self.torch_model.blocks[0](self.x, t_cos, t_sin)
        actual = block_forward(
            self.params["blocks"][0],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
        )
        self.assert_close(actual, expected.numpy())
        actual_manual = block_forward(
            self.params["blocks"][0],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
            attention_backend="manual",
        )
        self.assert_close(actual_manual, expected.numpy())
        actual_chunked = block_forward(
            self.params["blocks"][0],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
            attention_backend="chunked",
        )
        self.assert_close(actual_chunked, expected.numpy())

    def test_chunked_attention_unaligned_length(self):
        T = 6  # not a multiple of local_window=4, exercises the padding path
        x = self.x[:, :T, :]
        head_dim = self.cfg.n_embd // self.cfg.n_head
        t_cos, t_sin = torch_rope_cache(head_dim, T, x.device)
        j_cos, j_sin = jax_rope_cache(head_dim, T)

        with torch.no_grad():
            expected = self.torch_model.blocks[0](x, t_cos, t_sin)
        actual = block_forward(
            self.params["blocks"][0],
            jnp.asarray(x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
            attention_backend="chunked",
        )
        self.assert_close(actual, expected.numpy())

    def test_span_hypergraph_means_and_block(self):
        h = jnp.asarray(np.random.default_rng(0).standard_normal((2, self.cfg.block_size, self.cfg.n_embd)).astype("float32"))
        expected_spans = []
        h_np = np.asarray(h)
        for width in self.cfg.span_widths:
            rows = []
            for t in range(self.cfg.block_size):
                start = max(0, t - width + 1)
                rows.append(h_np[:, start : t + 1, :].mean(axis=1))
            expected_spans.append(np.stack(rows, axis=1))
        self.assert_close(span_hypergraph(h, self.cfg.span_widths), np.concatenate(expected_spans, axis=-1))
        fused = fused_span_edge_projection(h, self.cfg.span_widths, self.params["blocks"][1]["edge_proj"])
        materialized = span_hypergraph(h, self.cfg.span_widths) @ self.params["blocks"][1]["edge_proj"]["weight"].T
        materialized = materialized + self.params["blocks"][1]["edge_proj"]["bias"]
        self.assert_close(fused, materialized)
        einsum_fused = einsum_fused_span_edge_projection(h, self.cfg.span_widths, self.params["blocks"][1]["edge_proj"])
        self.assert_close(einsum_fused, materialized)

        head_dim = self.cfg.n_embd // self.cfg.n_head
        t_cos, t_sin = torch_rope_cache(head_dim, self.cfg.block_size, self.x.device)
        j_cos, j_sin = jax_rope_cache(head_dim, self.cfg.block_size)
        with torch.no_grad():
            expected = self.torch_model.blocks[1](self.x, t_cos, t_sin)
        actual = block_forward(
            self.params["blocks"][1],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
        )
        self.assert_close(actual, expected.numpy())
        actual_materialized = block_forward(
            self.params["blocks"][1],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
            span_backend="materialized",
        )
        self.assert_close(actual_materialized, expected.numpy())
        actual_loop_fused = block_forward(
            self.params["blocks"][1],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
            span_backend="fused",
        )
        self.assert_close(actual_loop_fused, expected.numpy())

    def test_compressed_memory_block(self):
        head_dim = self.cfg.n_embd // self.cfg.n_head
        t_cos, t_sin = torch_rope_cache(head_dim, self.cfg.block_size, self.x.device)
        j_cos, j_sin = jax_rope_cache(head_dim, self.cfg.block_size)

        with torch.no_grad():
            expected = self.torch_model.blocks[2](self.x, t_cos, t_sin)
        actual = block_forward(
            self.params["blocks"][2],
            jnp.asarray(self.x.numpy()),
            self.jax_cfg,
            j_cos,
            j_sin,
        )
        self.assert_close(actual, expected.numpy())

    def test_full_model_logits(self):
        with torch.no_grad():
            expected, _ = self.torch_model(self.idx)
        actual = jax_model.forward(self.params, jnp.asarray(self.idx.numpy()), self.jax_cfg)
        self.assert_close(actual, expected.numpy())
        actual_manual = jax_model.forward(
            self.params,
            jnp.asarray(self.idx.numpy()),
            self.jax_cfg,
            attention_backend="manual",
        )
        self.assert_close(actual_manual, expected.numpy())
        actual_chunked = jax_model.forward(
            self.params,
            jnp.asarray(self.idx.numpy()),
            self.jax_cfg,
            attention_backend="chunked",
        )
        self.assert_close(actual_chunked, expected.numpy())
        actual_materialized = jax_model.forward(
            self.params,
            jnp.asarray(self.idx.numpy()),
            self.jax_cfg,
            span_backend="materialized",
        )
        self.assert_close(actual_materialized, expected.numpy())

    def test_loss_and_grad_smoke(self):
        targets = torch.randint(0, self.cfg.vocab_size, (2, self.cfg.block_size))
        with torch.no_grad():
            _, expected = self.torch_model(self.idx, targets)
        actual = jax_model.loss(self.params, jnp.asarray(self.idx.numpy()), jnp.asarray(targets.numpy()), self.jax_cfg)
        self.assert_close(actual, expected.numpy())

        grads = jax.grad(
            lambda p: jax_model.loss(p, jnp.asarray(self.idx.numpy()), jnp.asarray(targets.numpy()), self.jax_cfg)
        )(self.params)
        for leaf in jax.tree.leaves(grads):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

    def test_custom_block_layout_parity_and_ablation(self):
        layout = ("attn", "span", "span", "hca", "span", "span")
        cfg = EfficientHGConfig(
            vocab_size=37,
            block_size=8,
            n_embd=16,
            n_head=4,
            span_widths=(2, 4),
            local_window=4,
            compression_block=4,
            dropout=0.0,
            block_layout=layout,
        )
        torch_model = EfficientHypergraphLM(cfg).eval()
        self.assertEqual(tuple(torch_model.block_layout), layout)
        params, jax_cfg = jax_model.from_torch_model(torch_model)
        self.assertEqual(jax_cfg.block_layout, layout)
        self.assertEqual(len(params["blocks"]), len(layout))

        idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        with torch.no_grad():
            expected, _ = torch_model(idx)
        actual = jax_model.forward(params, jnp.asarray(idx.numpy()), jax_cfg, attention_backend="chunked")
        self.assert_close(actual, expected.numpy())

        # ablation hook returns one finite delta per block, with kinds matching the layout
        targets = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
        base, deltas = jax_model.block_ablation_deltas(
            params, jnp.asarray(idx.numpy()), jnp.asarray(targets.numpy()), jax_cfg, attention_backend="chunked"
        )
        self.assertEqual([k for _, k, _ in deltas], list(layout))
        self.assertTrue(all(np.isfinite(d) for _, _, d in deltas))

    def test_span_grad_smoke(self):
        for length, widths in [(5, (2, 3)), (8, (2, 4, 6))]:
            h = jnp.arange(2 * length * 3, dtype=jnp.float32).reshape(2, length, 3) / 10.0
            weights = jnp.linspace(-0.5, 0.5, num=2 * length * len(widths) * 3, dtype=jnp.float32).reshape(
                2, length, len(widths) * 3
            )

            grad = jax.grad(lambda x: jnp.sum(span_hypergraph(x, widths) * weights))(h)
            self.assertEqual(grad.shape, h.shape)
            self.assertTrue(jnp.all(jnp.isfinite(grad)))


if __name__ == "__main__":
    unittest.main()
