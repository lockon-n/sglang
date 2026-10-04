"""Sparse blocks: encode only the 32x32 RGB blocks that changed since the
previous image of a Responses streaming-session chain.

Each new image is resized to a 32-aligned grid and compared with the chain's
previous image. An I frame keeps every block (the image paired with itself, as
a native image). A P frame pairs (previous, current) and keeps only changed
blocks, shrinking the pair against a token budget when many blocks change. Each
kept block is 2x2 patches of 16x16 (temporal 2), i.e. one merged visual token,
and carries its block index so the ViT position embeddings and the LLM M-RoPE
use its real coordinates. Selection follows lmms-eval's sparse_blocks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Optional, Union

import torch
import torch.nn.functional as F

# model_specific_data keys of a sparse image item.
SPARSE_GRID_KEY = "sparse_block_grid"  # (patch rows, patch cols) of the kept grid
SPARSE_INDEX_KEY = "sparse_block_index"  # row-major 32x32 block indices, ascending

BLOCK = 32
PATCH = 16


@dataclasses.dataclass(frozen=True)
class SparseBlockConfig:
    rgb_mae_threshold: float = 0.17
    scan_resize_long_edge: int = 64
    max_consecutive_p_frames: int = 16
    block_metric: str = "max"
    block_threshold: float = 0.17
    thumbnail_ratio: float = 0.5
    budgets: tuple = (16, 32, 64)
    budget_boundaries: tuple = (128.0, 512.0)
    budget_alpha: float = 0.5
    max_resize_passes: int = 2
    # Temporal pair of a P block: "previous" encodes (previous, current) like a
    # video frame pair; "current" encodes (current, current) like an image;
    # "reversed" encodes (current, previous).
    p_frame_pair: str = "previous"

    def __post_init__(self):
        object.__setattr__(self, "budgets", tuple(int(b) for b in self.budgets))
        object.__setattr__(
            self, "budget_boundaries", tuple(float(b) for b in self.budget_boundaries)
        )
        if self.block_metric not in ("max", "mean"):
            raise ValueError("block_metric must be max or mean")
        if self.scan_resize_long_edge < 1 or self.max_consecutive_p_frames < -1:
            raise ValueError("Invalid scan size or consecutive-P limit")
        if not 0 <= self.thumbnail_ratio <= 1 or not 0 <= self.budget_alpha <= 1:
            raise ValueError("thumbnail_ratio and budget_alpha must lie in [0, 1]")
        if self.rgb_mae_threshold < 0 or self.block_threshold < 0:
            raise ValueError("RGB thresholds must be non-negative")
        if not self.budgets or len(self.budgets) != len(self.budget_boundaries) + 1:
            raise ValueError("Need one fewer budget boundary than budgets")
        for values in (self.budgets, self.budget_boundaries):
            if any(v <= 0 for v in values) or any(
                a >= b for a, b in zip(values, values[1:])
            ):
                raise ValueError(
                    "Budgets and boundaries must be positive and strictly increasing"
                )
        if self.max_resize_passes not in (1, 2):
            raise ValueError("max_resize_passes must be 1 or 2")
        if self.p_frame_pair not in ("previous", "current", "reversed"):
            raise ValueError("p_frame_pair must be previous, current or reversed")

    @classmethod
    def from_json(cls, text: Optional[str]) -> SparseBlockConfig:
        return cls(**json.loads(text)) if text else cls()


def aligned_size(height: int, width: int, scale: float) -> tuple[int, int]:
    """Nearest 32-aligned size for a uniform scale, never upscaling."""
    return tuple(
        max(BLOCK, min(d, round(d * min(scale, 1.0) / BLOCK) * BLOCK))
        for d in (height, width)
    )


def block_scores(pair: torch.Tensor, metric: str) -> torch.Tensor:
    """Per-block change between the two frames of a [2, 3, H, W] pair."""
    _, _, h, w = pair.shape
    diff = (pair[1] - pair[0]).abs().reshape(3, h // BLOCK, BLOCK, w // BLOCK, BLOCK)
    reduced = diff.amax(dim=(0, 2, 4)) if metric == "max" else diff.mean(dim=(0, 2, 4))
    return reduced.flatten()


def select_blocks(pair: torch.Tensor, config: SparseBlockConfig):
    """Changed blocks of a P pair, resized against the token budget.

    Returns the (possibly downscaled) pair and the ascending block indices into
    its grid. Few changed blocks are kept as-is; many changed blocks shrink the
    pair (locally, or as a whole-frame thumbnail past thumbnail_ratio).
    """
    h, w = pair.shape[-2:]
    scores = block_scores(pair, config.block_metric)
    selected = (scores > config.block_threshold).nonzero().flatten()
    n, k = scores.numel(), selected.numel()
    if k <= config.budgets[0]:
        return pair, selected
    ratio = k / n
    demand = n * ratio**config.budget_alpha
    budget = config.budgets[sum(demand > b for b in config.budget_boundaries)]
    thumbnail = ratio > config.thumbnail_ratio
    scale = min(1.0, math.sqrt(budget / (n if thumbnail else k)))
    target = aligned_size(h, w, scale)
    final = F.interpolate(pair, size=target, mode="nearest")
    if thumbnail:
        return final, torch.arange(
            target[0] * target[1] // BLOCK**2, device=pair.device
        )
    selected = (
        (block_scores(final, config.block_metric) > config.block_threshold)
        .nonzero()
        .flatten()
    )
    if selected.numel() > budget and config.max_resize_passes == 2:
        target = aligned_size(h, w, scale * math.sqrt(budget / selected.numel()))
        final = F.interpolate(pair, size=target, mode="nearest")
        selected = (
            (block_scores(final, config.block_metric) > config.block_threshold)
            .nonzero()
            .flatten()
        )
    return final, selected


def sparse_patchify(pair: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """Rows of the kept blocks in the ViT patch layout.

    Four patches per block in 2x2 row-major order, each flattened as
    (channel, time, 16, 16), which is what the patch embedding reads and what
    a dense Qwen image yields when every block is kept.
    """
    _, c, h, w = pair.shape
    blocks = pair.reshape(2, c, h // BLOCK, BLOCK, w // BLOCK, BLOCK).permute(
        2, 4, 0, 1, 3, 5
    )
    blocks = blocks[selected // (w // BLOCK), selected % (w // BLOCK)]
    return (
        blocks.reshape(-1, 2, c, 2, PATCH, 2, PATCH)
        .permute(0, 3, 5, 2, 1, 4, 6)
        .reshape(-1, c * 2 * PATCH**2)
    )


def is_scene_change(
    previous: torch.Tensor, current: torch.Tensor, config: SparseBlockConfig
) -> bool:
    """Mean absolute RGB change of a low-resolution scan above the threshold."""
    h, w = current.shape[-2:]
    scale = min(1.0, config.scan_resize_long_edge / max(h, w))
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    scan = F.interpolate(torch.stack((previous, current)), size=size, mode="nearest")
    return (scan[1] - scan[0]).abs().mean().item() > config.rgb_mae_threshold


@dataclasses.dataclass
class SparseFrame:
    """One processed image: rows for the ViT, or nothing if no block changed."""

    pixel_rows: Optional[torch.Tensor]  # [4 * blocks, 3 * 2 * 16 * 16]
    grid: tuple[int, int]  # patch rows, patch cols of the kept grid
    block_index: torch.Tensor  # ascending, into the (rows/2, cols/2) block grid
    is_i_frame: bool

    @property
    def num_tokens(self) -> int:
        return self.block_index.numel()


@dataclasses.dataclass
class _ChainFrames:
    config: SparseBlockConfig
    previous: Optional[torch.Tensor] = None  # resized RGB in [0, 1], [3, H, W]
    p_run: int = 0


class SparseBlockEncoder:
    """Per-chain previous images, staged per turn and committed on success.

    A chain is a streaming session the Responses layer opens with sparse blocks
    requested; it keeps that request's config. A turn's images are encoded
    against the committed state; the result is staged and only committed once
    the turn finishes, so an aborted turn rolls back with the session. All calls
    run on the tokenizer manager's event loop.
    """

    def __init__(self, config: SparseBlockConfig, image_mean, image_std):
        self.config = config  # server defaults; requests may override fields
        self._mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self._committed: dict[str, _ChainFrames] = {}
        self._staged: dict[str, _ChainFrames] = {}

    def config_for(self, request: Union[bool, dict]) -> SparseBlockConfig:
        """A request's ``sparse_blocks``: true for the defaults, or field overrides."""
        if request is True:
            return self.config
        if not isinstance(request, dict):
            raise ValueError("sparse_blocks must be true or an object of overrides")
        unknown = set(request) - {f.name for f in dataclasses.fields(SparseBlockConfig)}
        if unknown:
            raise ValueError(f"Unknown sparse_blocks fields: {sorted(unknown)}")
        return dataclasses.replace(self.config, **request)

    def open(self, session_id: str, config: SparseBlockConfig) -> None:
        self._committed[session_id] = _ChainFrames(config)
        self._staged.pop(session_id, None)

    def is_open(self, session_id: Optional[str]) -> bool:
        return session_id is not None and session_id in self._committed

    def commit(self, session_id: str) -> None:
        staged = self._staged.pop(session_id, None)
        if staged is not None and session_id in self._committed:
            self._committed[session_id] = staged

    def close(self, session_id: str) -> None:
        self._committed.pop(session_id, None)
        self._staged.pop(session_id, None)

    def encode(self, session_id: str, images: list[torch.Tensor]) -> list[SparseFrame]:
        """Encode a turn's resized images ([3, H, W] in [0, 1]) in order."""
        state = dataclasses.replace(self._committed[session_id])
        frames = [self._encode_one(state, image) for image in images]
        self._staged[session_id] = state
        return frames

    def _encode_one(self, state: _ChainFrames, current: torch.Tensor) -> SparseFrame:
        config, previous = state.config, state.previous
        is_i = (
            previous is None
            or previous.shape != current.shape
            or (
                config.max_consecutive_p_frames >= 0
                and state.p_run >= config.max_consecutive_p_frames
            )
            or is_scene_change(previous, current, config)
        )
        state.previous = current
        state.p_run = 0 if is_i else state.p_run + 1
        h, w = current.shape[-2:]
        if is_i:
            pair = torch.stack((current, current))
            selected = torch.arange(h * w // BLOCK**2, device=current.device)
        else:
            pair, selected = select_blocks(torch.stack((previous, current)), config)
            if config.p_frame_pair == "current":
                pair = pair[1:].expand(2, -1, -1, -1)
            elif config.p_frame_pair == "reversed":
                pair = pair.flip(0)
        grid = (pair.shape[-2] // PATCH, pair.shape[-1] // PATCH)
        if selected.numel() == 0:
            return SparseFrame(None, grid, selected.cpu(), is_i)
        mean, std = self._mean.to(pair.device), self._std.to(pair.device)
        rows = sparse_patchify((pair - mean) / std, selected)
        return SparseFrame(rows.float().cpu(), grid, selected.cpu(), is_i)


def sparse_item_hash(rows: torch.Tensor, grid, block_index: torch.Tensor) -> int:
    """Content identity of a sparse item: pixels plus where they sit.

    Equal pixels at other coordinates must not share a ViT embedding or a
    prefix-cache key, so the grid and block indices are part of the hash.
    """
    digest = hashlib.sha256()
    digest.update(rows.contiguous().view(torch.uint8).numpy().tobytes())
    digest.update(json.dumps(list(grid)).encode())
    digest.update(block_index.to(torch.int64).contiguous().numpy().tobytes())
    return int.from_bytes(digest.digest()[:8], "big", signed=True)


def sparse_patch_coordinates(block_index: torch.Tensor, grid_cols: int) -> torch.Tensor:
    """(row, col) of each kept patch in the patch grid, in row order of the ViT input."""
    blocks_per_row = grid_cols // 2
    origins = (
        torch.stack((block_index // blocks_per_row, block_index % blocks_per_row), -1)
        * 2
    )
    offsets = block_index.new_tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    return (origins[:, None, :] + offsets).reshape(-1, 2)
