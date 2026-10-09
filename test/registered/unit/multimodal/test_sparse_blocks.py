"""Unit tests for sparse-block selection, patch layout and per-chain state."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.multimodal.sparse_blocks import (
    SparseBlockConfig,
    SparseBlockEncoder,
    select_blocks,
    sparse_item_hash,
    sparse_patch_coordinates,
    sparse_patchify,
)
from sglang.test.test_utils import CustomTestCase


def _dense_patchify(pair):
    """Qwen2-VL's image processor layout for a [2, 3, H, W] temporal pair."""
    _, c, h, w = pair.shape
    patches = pair.reshape(1, 2, c, h // 32, 2, 16, w // 32, 2, 16)
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(-1, c * 2 * 16 * 16)


def _with_changed_blocks(image, blocks, cols):
    changed = image.clone()
    for index in blocks:
        row, col = divmod(index, cols)
        changed[:, row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32] = (
            1 - (changed[:, row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32])
        )
    return changed


class TestSelection(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.image = torch.rand(3, 128, 256) * 0.2  # 4 x 8 blocks

    def test_unchanged_pair_keeps_nothing(self):
        pair = torch.stack((self.image, self.image))
        final, selected = select_blocks(pair, SparseBlockConfig())
        self.assertEqual(final.shape, pair.shape)
        self.assertEqual(selected.numel(), 0)

    def test_few_changes_keep_exactly_those_blocks(self):
        changed = _with_changed_blocks(self.image, [3, 9, 30], cols=8)
        final, selected = select_blocks(
            torch.stack((self.image, changed)), SparseBlockConfig()
        )
        self.assertEqual(final.shape[-2:], (128, 256))
        self.assertEqual(selected.tolist(), [3, 9, 30])

    def test_widespread_change_shrinks_to_a_thumbnail(self):
        config = SparseBlockConfig(budgets=(4, 8, 16))
        changed = _with_changed_blocks(self.image, range(32), cols=8)
        final, selected = select_blocks(torch.stack((self.image, changed)), config)
        self.assertLess(final.shape[-1], 256)
        self.assertEqual(selected.numel(), final.shape[-2] * final.shape[-1] // 1024)

    def test_all_blocks_match_the_dense_layout(self):
        pair = torch.rand(2, 3, 96, 160)
        everything = torch.arange(3 * 5)
        self.assertTrue(
            torch.equal(sparse_patchify(pair, everything), _dense_patchify(pair))
        )

    def test_kept_blocks_are_rows_of_the_dense_layout(self):
        pair = torch.rand(2, 3, 96, 160)
        kept = torch.tensor([1, 7, 14])
        rows = torch.cat([torch.arange(4 * b, 4 * b + 4) for b in kept.tolist()])
        self.assertTrue(
            torch.equal(sparse_patchify(pair, kept), _dense_patchify(pair)[rows])
        )

    def test_patch_coordinates_follow_block_order(self):
        coords = sparse_patch_coordinates(torch.tensor([0, 6]), grid_cols=10)
        self.assertEqual(
            coords.tolist(),
            [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]],
        )

    def test_hash_depends_on_position(self):
        rows = torch.rand(4, 1536)
        self.assertNotEqual(
            sparse_item_hash(rows, (8, 8), torch.tensor([0])),
            sparse_item_hash(rows, (8, 8), torch.tensor([5])),
        )
        self.assertNotEqual(
            sparse_item_hash(rows, (8, 8), torch.tensor([0])),
            sparse_item_hash(rows, (8, 16), torch.tensor([0])),
        )


class TestEncoder(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder = SparseBlockEncoder(
            SparseBlockConfig(max_consecutive_p_frames=2), [0.5] * 3, [0.5] * 3
        )
        self.encoder.open("chain", self.encoder.config)
        self.image = torch.rand(3, 128, 256) * 0.2

    def _turn(self, *images):
        frames = self.encoder.encode("chain", list(images))
        self.encoder.commit("chain")
        return frames

    def test_first_image_is_a_full_i_frame(self):
        (frame,) = self._turn(self.image)
        self.assertTrue(frame.is_i_frame)
        self.assertEqual(frame.num_tokens, 32)
        self.assertEqual(frame.grid, (8, 16))
        # An I frame pairs the image with itself, normalized.
        pair = (torch.stack((self.image, self.image)) - 0.5) / 0.5
        self.assertTrue(torch.allclose(frame.pixel_rows, _dense_patchify(pair)))

    def test_p_frames_carry_only_changes_until_forced_i(self):
        self._turn(self.image)
        changed = _with_changed_blocks(self.image, [5], cols=8)
        first, second, third = self._turn(changed, changed, changed)
        self.assertEqual((first.is_i_frame, first.block_index.tolist()), (False, [5]))
        self.assertEqual((second.is_i_frame, second.num_tokens), (False, 0))
        self.assertIsNone(second.pixel_rows)
        # Two P frames in a row reach the limit: the next one is a full I frame.
        self.assertEqual((third.is_i_frame, third.num_tokens), (True, 32))

    def test_p_frame_pairing_options(self):
        changed = _with_changed_blocks(self.image, [5], cols=8)
        rows = torch.arange(20, 24)  # block 5's four patches
        for mode, pair in (
            ("previous", (self.image, changed)),
            ("current", (changed, changed)),
            ("reversed", (changed, self.image)),
        ):
            encoder = SparseBlockEncoder(
                SparseBlockConfig(p_frame_pair=mode), [0.5] * 3, [0.5] * 3
            )
            encoder.open("chain", encoder.config)
            encoder.encode("chain", [self.image])
            encoder.commit("chain")
            (frame,) = encoder.encode("chain", [changed])
            expected = _dense_patchify((torch.stack(pair) - 0.5) / 0.5)[rows]
            self.assertTrue(torch.allclose(frame.pixel_rows, expected), mode)

    def test_requested_i_frame_anchors_a_new_p_run(self):
        self._turn(self.image)
        changed = _with_changed_blocks(self.image, [5], cols=8)
        (p,) = self._turn(changed)  # one P frame into the run of 2
        self.assertFalse(p.is_i_frame)
        self.encoder.request_i_frame("chain")
        self.assertEqual(self._turn(), [])  # a turn without images keeps the request
        # The turn's last image is the forced one; earlier ones stay P frames.
        before, forced = self._turn(changed, changed)
        self.assertFalse(before.is_i_frame)
        self.assertEqual((forced.is_i_frame, forced.num_tokens), (True, 32))
        # The P run restarts at the forced frame, and the request is used once:
        # two P frames follow before the run limit forces the next I frame.
        first, second, limit = self._turn(changed, changed, changed)
        self.assertEqual((first.is_i_frame, second.is_i_frame, limit.is_i_frame), (False, False, True))

    def test_scene_change_and_new_size_are_i_frames(self):
        self._turn(self.image)
        (cut,) = self._turn(1 - self.image)
        self.assertTrue(cut.is_i_frame)
        (resized,) = self._turn(torch.rand(3, 64, 64))
        self.assertTrue(resized.is_i_frame)

    def test_uncommitted_turn_rolls_back(self):
        self._turn(self.image)
        changed = _with_changed_blocks(self.image, [5], cols=8)
        self.encoder.encode("chain", [changed])  # aborted: never committed
        (retry,) = self._turn(changed)
        self.assertEqual(retry.block_index.tolist(), [5])

    def test_request_config_overrides_server_defaults(self):
        self.assertIs(self.encoder.config_for(True), self.encoder.config)
        config = self.encoder.config_for({"p_frame_pair": "current"})
        self.assertEqual(config.p_frame_pair, "current")
        self.assertEqual(config.max_consecutive_p_frames, 2)  # server default kept
        with self.assertRaises(ValueError):
            self.encoder.config_for({"p_frame_pair": "sideways"})
        with self.assertRaises(ValueError):
            self.encoder.config_for({"no_such_field": 1})

    def test_closed_or_unknown_chain_is_not_open(self):
        self.assertTrue(self.encoder.is_open("chain"))
        self.encoder.close("chain")
        self.assertFalse(self.encoder.is_open("chain"))
        self.assertFalse(self.encoder.is_open(None))


if __name__ == "__main__":
    unittest.main()
