import unittest
import json
import os
import sys
import tempfile
import types
from unittest import mock

import numpy as np

from curriculum_data import (
    CurriculumLoader,
    CurriculumPhase,
    REDPAJAMA_SOURCE,
    REDPAJAMA_DATA_DIR_ENV,
    REDPAJAMA_URLS_FILE_ENV,
    default_curriculum,
    _hf_document_tokens,
    _redpajama_document_tokens,
    _redpajama_subset_urls,
    pack_token_stream,
    phase_boundaries,
    phase_for_step,
)


class FakeTokenizer:
    """Minimal stand-in for an HF tokenizer: maps each char to its ordinal."""

    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class PhaseScheduleTest(unittest.TestCase):
    def setUp(self):
        self.phases = [
            CurriculumPhase("a", "ds_a", None, steps=2),
            CurriculumPhase("b", "ds_b", None, steps=4),
            CurriculumPhase("c", "ds_c", None, steps=4),
        ]

    def test_boundaries(self):
        self.assertEqual(phase_boundaries(self.phases), [2, 6, 10])

    def test_phase_for_step_hard_cuts(self):
        got = [phase_for_step(self.phases, s) for s in range(12)]
        self.assertEqual(got, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])

    def test_clamps_past_end(self):
        self.assertEqual(phase_for_step(self.phases, 10_000), 2)

    def test_default_curriculum_sums_to_max_iters(self):
        phases = default_curriculum(10_000, fractions=(0.2, 0.4, 0.4))
        self.assertEqual(sum(p.steps for p in phases), 10_000)
        self.assertEqual([p.label for p in phases], ["simple-wiki", "en-wiki", "arxiv"])
        self.assertEqual(phases[0].dataset_config, "20231101.simple")
        self.assertEqual(phases[1].dataset_config, "20231101.en")
        self.assertEqual(phases[2].dataset_name, "togethercomputer/RedPajama-Data-1T")
        self.assertEqual(phases[2].dataset_config, "arxiv")
        self.assertEqual(phases[2].source, REDPAJAMA_SOURCE)

    def test_default_curriculum_remainder_goes_to_last(self):
        # 0.2/0.4/0.4 of 999 rounds to 200/400, so arxiv must absorb the remaining 399.
        phases = default_curriculum(999, fractions=(0.2, 0.4, 0.4))
        self.assertEqual([p.steps for p in phases], [200, 400, 399])


class PackingTest(unittest.TestCase):
    def test_packs_with_eos_and_drops_tail(self):
        # docs "abc","de" -> [97,98,99,0,100,101,0]; block_size+1=4 -> one window, tail dropped.
        batches = list(pack_token_stream([[97, 98, 99], [100, 101]], eos_id=0, block_size=3, batch_size=1))
        self.assertEqual(len(batches), 1)
        x, y = batches[0]
        self.assertEqual(x.shape, (1, 3))
        np.testing.assert_array_equal(x[0], [97, 98, 99])
        np.testing.assert_array_equal(y[0], [98, 99, 0])  # y is the window shifted by one

    def test_x_y_are_shifted_views(self):
        docs = [list(range(1, 50))]  # one long doc, eos=0 appended -> 50 tokens
        x, y = next(pack_token_stream(docs, eos_id=0, block_size=8, batch_size=1))
        np.testing.assert_array_equal(x[0, 1:], y[0, :-1])

    def test_batching_groups_windows(self):
        docs = [list(range(1, 200))]
        x, y = next(pack_token_stream(docs, eos_id=0, block_size=4, batch_size=3))
        self.assertEqual(x.shape, (3, 4))
        self.assertEqual(y.shape, (3, 4))


class HoldoutSplitTest(unittest.TestCase):
    def test_hf_stream_slices_before_shuffle(self):
        operations = []

        class FakeStream:
            def __init__(self, rows):
                self.rows = list(rows)

            def skip(self, n):
                operations.append(("skip", n))
                return FakeStream(self.rows[n:])

            def take(self, n):
                operations.append(("take", n))
                return FakeStream(self.rows[:n])

            def shuffle(self, buffer_size, seed):
                operations.append(("shuffle", buffer_size, seed))
                return self

            def __iter__(self):
                return iter(self.rows)

        def fake_load_dataset(*args, **kwargs):
            operations.append(("load_dataset", args, kwargs))
            return FakeStream([{"text": "a"}, {"text": "b"}, {"text": "c"}, {"text": "d"}])

        fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
        phase = CurriculumPhase("train", "fake", None, steps=1, text_field="text")
        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            docs = list(
                _hf_document_tokens(
                    phase,
                    FakeTokenizer(),
                    shuffle=True,
                    seed=7,
                    shuffle_buffer=11,
                    skip_docs=1,
                    take_docs=2,
                )
            )

        self.assertEqual(operations[1:], [("skip", 1), ("take", 2), ("shuffle", 11, 7)])
        self.assertEqual(docs, [[ord("b")], [ord("c")]])


class RedPajamaRawStreamTest(unittest.TestCase):
    def test_filters_manifest_by_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            urls_path = os.path.join(tmp, "urls.txt")
            with open(urls_path, "w", encoding="utf-8") as f:
                f.write("https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/a.jsonl\n")
                f.write("https://data.together.xyz/redpajama-data-1T/v1.0.0/github/g.jsonl\n")
                f.write("https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/b.jsonl\n")
            with mock.patch.dict(os.environ, {REDPAJAMA_URLS_FILE_ENV: urls_path}, clear=False):
                urls = _redpajama_subset_urls("arxiv")
        self.assertEqual(len(urls), 2)
        self.assertTrue(all("/arxiv/" in url for url in urls))

    def test_streams_local_jsonl_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            urls_path = os.path.join(tmp, "urls.txt")
            shard_dir = os.path.join(tmp, "arxiv")
            os.makedirs(shard_dir)
            shard_path = os.path.join(shard_dir, "sample.jsonl")
            url = "https://data.together.xyz/redpajama-data-1T/v1.0.0/arxiv/sample.jsonl"
            with open(urls_path, "w", encoding="utf-8") as f:
                f.write(url + "\n")
            with open(shard_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "alpha"}) + "\n")
                f.write(json.dumps({"text": ""}) + "\n")
                f.write(json.dumps({"text": "beta"}) + "\n")
            phase = CurriculumPhase(
                "arxiv",
                "togethercomputer/RedPajama-Data-1T",
                "arxiv",
                steps=1,
                source=REDPAJAMA_SOURCE,
            )
            env = {
                REDPAJAMA_URLS_FILE_ENV: urls_path,
                REDPAJAMA_DATA_DIR_ENV: tmp,
            }
            with mock.patch.dict(os.environ, env, clear=False):
                docs = list(
                    _redpajama_document_tokens(
                        phase,
                        FakeTokenizer(),
                        shuffle=False,
                        seed=0,
                        shuffle_buffer=10,
                        skip_docs=1,
                        take_docs=1,
                    )
                )
        self.assertEqual(docs, [[ord("b"), ord("e"), ord("t"), ord("a")]])


class CurriculumLoaderTest(unittest.TestCase):
    def test_lazy_switch_and_label(self):
        # Patch phase streams with synthetic batches so no network is touched.
        phases = [
            CurriculumPhase("a", "ds_a", None, steps=2),
            CurriculumPhase("b", "ds_b", None, steps=3),
        ]
        loader = CurriculumLoader(
            phases=phases, tokenizer=FakeTokenizer(), block_size=4, batch_size=1, seed=0, prefetch=2
        )

        started = []

        def fake_start(idx):
            started.append(idx)
            loader._active_idx = idx
            loader._prefetcher = iter(lambda: (np.full((1, 4), idx, np.int32),) * 2, None)

        loader._start_phase = fake_start

        self.assertEqual(loader.phase_label(0), "a")
        self.assertEqual(loader.phase_label(4), "b")

        seq = [loader.batch(s)[0][0, 0] for s in range(5)]
        # steps 0,1 -> phase 0; steps 2,3,4 -> phase 1; lazy start fires once per phase.
        self.assertEqual(seq, [0, 0, 1, 1, 1])
        self.assertEqual(started, [0, 1])


if __name__ == "__main__":
    unittest.main()
