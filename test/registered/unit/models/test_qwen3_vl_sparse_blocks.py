"""Sparse-block inputs reproduce the dense Qwen3-VL/3.5 geometry.

Keeping every block must give exactly the dense position embeddings, rotary
coordinates and M-RoPE positions; keeping some blocks must give the dense
values at those blocks. Runs the real methods on light stubs (no weights or
distributed init), on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.utils import RotaryPosMixin
from sglang.srt.multimodal.sparse_blocks import SparseFrame, sparse_patch_coordinates
from sglang.test.test_utils import CustomTestCase

GRIDS = [(16, 16), (8, 40), (64, 98), (2, 2)]  # patch rows, cols


class TestSparseVisionGeometry(CustomTestCase):
    def setUp(self):
        try:
            from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
        except Exception as e:  # heavy optional deps unavailable
            self.skipTest(f"cannot import Qwen3VLMoeVisionModel: {e}")
        self.model = Qwen3VLMoeVisionModel
        torch.manual_seed(0)
        self.stub = SimpleNamespace(
            num_grid_per_side=48,
            spatial_merge_size=2,
            pos_embed=nn.Embedding(2304, 64),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def test_all_blocks_match_dense_position_embeddings(self):
        for rows, cols in GRIDS:
            dense = self.model.fast_pos_embed_interpolate_from_list(
                self.stub, [[1, rows, cols]]
            )
            coords = sparse_patch_coordinates(
                torch.arange(rows * cols // 4), grid_cols=cols
            )
            sparse = self.model._sparse_pos_embed(self.stub, coords, rows, cols)
            self.assertTrue(torch.equal(dense, sparse), (rows, cols))

    def test_kept_blocks_match_dense_rows(self):
        rows, cols = 16, 24
        kept = torch.tensor([0, 5, 17, 95])
        dense = self.model.fast_pos_embed_interpolate_from_list(
            self.stub, [[1, rows, cols]]
        )
        coords = sparse_patch_coordinates(kept, grid_cols=cols)
        patch_rows = torch.cat([torch.arange(4 * b, 4 * b + 4) for b in kept.tolist()])
        sparse = self.model._sparse_pos_embed(self.stub, coords, rows, cols)
        self.assertTrue(torch.equal(dense[patch_rows], sparse))

    def test_all_blocks_match_dense_rotary_coordinates(self):
        for rows, cols in GRIDS:
            dense = RotaryPosMixin.rot_pos_ids(rows, cols, 2)
            sparse = sparse_patch_coordinates(
                torch.arange(rows * cols // 4), grid_cols=cols
            )
            self.assertTrue(torch.equal(dense, sparse), (rows, cols))


class TestSparseMRoPE(CustomTestCase):
    def setUp(self):
        try:
            from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
        except Exception as e:
            self.skipTest(f"cannot import QwenVLImageProcessor: {e}")
        self.processor = QwenVLImageProcessor
        self.stub = SimpleNamespace(
            model_type="qwen3_5",
            _spatial_merge_size=2,
            _as_grid_batch=QwenVLImageProcessor._as_grid_batch,
        )

    def _dense(self, input_len, layout):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                offsets=[(start, start + rows * cols // 4 - 1)],
                model_specific_data={"image_grid_thw": torch.tensor([[1, rows, cols]])},
            )
            for start, (rows, cols) in layout
        ]
        positions, delta = (
            self.processor._compute_image_only_mrope_positions_from_offsets(
                self.stub, input_len, items, torch.long, torch.device("cpu")
            )
        )
        return positions.squeeze(1), delta

    def _sparse(self, input_len, layout, kept=None):
        frames, runs = [], []
        for i, (start, (rows, cols)) in enumerate(layout):
            index = kept[i] if kept is not None else torch.arange(rows * cols // 4)
            frames.append(SparseFrame(None, (rows, cols), index, kept is None))
            runs.append((start, start + index.numel() - 1))
        return self.processor._sparse_mrope_positions(
            self.stub, input_len, frames, runs
        )

    def test_all_blocks_match_dense_positions(self):
        # text, image 8x12 patches (24 tokens), text, image 4x4 (4 tokens), text
        layout = [(3, (8, 12)), (30, (4, 4))]
        input_len = 30 + 4 + 5
        dense, dense_delta = self._dense(input_len, layout)
        sparse, sparse_delta = self._sparse(input_len, layout)
        self.assertTrue(torch.equal(dense, sparse))
        self.assertEqual(dense_delta.item(), sparse_delta.item())

    def test_kept_blocks_sit_at_dense_coordinates(self):
        rows, cols = 8, 12
        kept = torch.tensor([1, 7, 23])
        dense, _ = self._dense(3 + 24 + 2, [(3, (rows, cols))])
        sparse, delta = self._sparse(3 + 3 + 2, [(3, (rows, cols))], [kept])
        self.assertTrue(torch.equal(sparse[:, 3:6], dense[:, 3 + kept]))
        # Text after the image resumes where it would after the full image.
        self.assertTrue(torch.equal(sparse[:, 6:], dense[:, 27:]))
        self.assertEqual(delta.item(), sparse.max().item() + 1 - sparse.shape[1])


if __name__ == "__main__":
    unittest.main()
