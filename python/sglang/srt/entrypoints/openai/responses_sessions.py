"""Serve chained Responses turns from a streaming session.

Every ``previous_response_id`` turn re-renders the whole conversation, so the
tokenizer re-reads, re-preprocesses and re-ships every earlier image to the
scheduler even though their KV is already cached; that CPU work grows linearly
with the conversation and dominates TTFT for image-heavy chats. A chain here
owns one streaming session holding exactly the previous prompt plus the tokens
it generated. When the newly rendered prompt extends that text, only the
suffix and its new media are sent. Anything else (a branch off an older
response, a template that rewrote history, an aborted turn) moves the chain to
a fresh session holding the full prompt, which costs what a turn costs today.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Optional, Union

import msgspec

from sglang.srt.entrypoints.context import ConversationContext, SimpleContext
from sglang.srt.managers.io_struct import CloseSessionReqInput, OpenSessionReqInput
from sglang.srt.utils.common import ImageData

logger = logging.getLogger(__name__)

# Drop a chain a little before the scheduler times its session out, so a turn
# never lands on a session that is being closed under it.
_EXPIRY_MARGIN = 0.8


class _ChainHead(msgspec.Struct, frozen=True, kw_only=True):
    session_id: str
    # Previous prompt plus its generated tokens: exactly what the session holds.
    text: str
    # Identity of each image already inside the session, in prompt order.
    media_keys: tuple
    started_at: float
    # Sparse-block config the session was opened with, or None for dense.
    sparse_blocks: Any = None


class SessionTurn(msgspec.Struct, kw_only=True):
    session_id: str
    # What to send: the suffix past the session's text, or the full prompt.
    text: str
    image_data: Optional[list]
    modalities: Optional[list]
    full_prompt: str
    media_keys: tuple
    started_at: float
    output_ids: list = []
    sparse_blocks: Any = None


def _media_key(item: Any) -> Optional[tuple]:
    """Identity of one image input, or None when it cannot be keyed cheaply."""
    if isinstance(item, str):
        return (len(item), hash(item))
    if isinstance(item, ImageData) and item.preprocess_kwargs is None:
        # str caches its hash, and history urls are the same objects every turn.
        return (
            len(item.url),
            hash(item.url),
            item.detail,
            item.max_dynamic_patch,
            item.content_hash,
        )
    return None


class ResponsesSessionManager:
    def __init__(self, *, tokenizer_manager: Any, idle_timeout: float):
        self._tokenizer_manager = tokenizer_manager
        self._idle_timeout = idle_timeout
        self._heads: dict[str, _ChainHead] = {}
        self._close_tasks: set[asyncio.Task] = set()

    async def begin_turn(
        self,
        *,
        previous_response_id: Optional[str],
        prompt: str,
        image_data: Optional[list],
        modalities: Optional[list],
        sparse_blocks: Union[bool, dict, None] = None,
    ) -> Optional[SessionTurn]:
        """Plan one turn, or return None to serve it without a session."""
        sparse_config = self._sparse_config(sparse_blocks)
        now = time.monotonic()
        self._expire(now=now)
        images = list(image_data or [])
        media_keys = tuple(_media_key(item) for item in images)
        if None in media_keys or (
            modalities is not None and len(modalities) != len(images)
        ):
            return None

        head = self._heads.pop(previous_response_id, None)
        if head is not None and head.sparse_blocks == sparse_config:
            delta_turn = self._delta_turn(
                head=head,
                prompt=prompt,
                images=images,
                modalities=modalities,
                media_keys=media_keys,
                now=now,
            )
            if delta_turn is not None:
                return delta_turn
            logger.debug(
                "Responses chain %s diverged from its session; reopening",
                previous_response_id,
            )
        if head is not None:
            self._close(head.session_id)

        session_id = await self._open(prompt_len=len(prompt), sparse=sparse_config)
        if session_id is None:
            return None
        return SessionTurn(
            session_id=session_id,
            text=prompt,
            image_data=image_data,
            modalities=modalities,
            full_prompt=prompt,
            media_keys=media_keys,
            started_at=now,
            sparse_blocks=sparse_config,
        )

    async def track(
        self,
        *,
        turn: SessionTurn,
        response_id: str,
        generator: AsyncGenerator[ConversationContext, None],
        incremental_output: bool,
    ) -> AsyncGenerator[ConversationContext, None]:
        """Pass outputs through; on success make ``response_id`` the chain head."""
        finish_reason = None
        completed = False
        try:
            async for context in generator:
                if isinstance(context, SimpleContext) and context.last_output:
                    output = context.last_output
                    if incremental_output:
                        turn.output_ids.extend(output["output_ids"])
                    else:
                        turn.output_ids = output["output_ids"]
                    finish_reason = output["meta_info"]["finish_reason"]
                yield context
            completed = True
        finally:
            if (
                completed
                and isinstance(finish_reason, dict)
                and finish_reason["type"] != "abort"
            ):
                self._heads[response_id] = self._advance_head(turn)
                if (sparse_blocks := self._sparse_blocks()) is not None:
                    sparse_blocks.commit(turn.session_id)
            else:
                self._close(turn.session_id)

    def _delta_turn(
        self,
        *,
        head: _ChainHead,
        prompt: str,
        images: list,
        modalities: Optional[list],
        media_keys: tuple,
        now: float,
    ) -> Optional[SessionTurn]:
        known = len(head.media_keys)
        if not (
            len(prompt) > len(head.text)
            and prompt.startswith(head.text)
            and media_keys[:known] == head.media_keys
        ):
            return None
        return SessionTurn(
            session_id=head.session_id,
            text=prompt[len(head.text) :],
            image_data=images[known:] or None,
            modalities=modalities[known:] if modalities else modalities,
            full_prompt=prompt,
            media_keys=media_keys,
            started_at=now,
            sparse_blocks=head.sparse_blocks,
        )

    def _advance_head(self, turn: SessionTurn) -> _ChainHead:
        # The session appends every generated id, stop token included.
        generated = self._tokenizer_manager.tokenizer.decode(
            list(turn.output_ids), skip_special_tokens=False
        )
        return _ChainHead(
            session_id=turn.session_id,
            text=turn.full_prompt + generated,
            media_keys=turn.media_keys,
            started_at=turn.started_at,
            sparse_blocks=turn.sparse_blocks,
        )

    async def _open(self, *, prompt_len: int, sparse: Any = None) -> Optional[str]:
        try:
            session_id = await self._tokenizer_manager.open_session(
                OpenSessionReqInput(
                    capacity_of_str_len=prompt_len,
                    streaming=True,
                    timeout=self._idle_timeout,
                )
            )
        except Exception:
            logger.exception("Failed to open a streaming session for Responses")
            return None
        if session_id is not None and sparse is not None:
            self._sparse_blocks().open(session_id, sparse)
        return session_id

    def _sparse_config(self, request: Union[bool, dict, None]):
        """The chain config a request's ``sparse_blocks`` asks for, or None."""
        if request is None or request is False:
            return None
        sparse_blocks = self._sparse_blocks()
        if sparse_blocks is None:
            raise ValueError(
                "sparse_blocks needs a Qwen3.5 server started with "
                "--enable-responses-sparse-blocks."
            )
        return sparse_blocks.config_for(request)

    def request_i_frame(self, session_id: str) -> None:
        """Encode the session's next image as a whole I frame."""
        sparse_blocks = self._sparse_blocks()
        if sparse_blocks is not None and sparse_blocks.is_open(session_id):
            sparse_blocks.request_i_frame(session_id)

    def _sparse_blocks(self):
        """The chain image state of --enable-responses-sparse-blocks, if on."""
        mm_processor = getattr(self._tokenizer_manager, "mm_processor", None)
        return getattr(mm_processor, "sparse_blocks", None)

    def _expire(self, *, now: float) -> None:
        deadline = self._idle_timeout * _EXPIRY_MARGIN
        for response_id, head in list(self._heads.items()):
            if now - head.started_at > deadline:
                del self._heads[response_id]
                self._close(head.session_id)

    def _close(self, session_id: str) -> None:
        if (sparse_blocks := self._sparse_blocks()) is not None:
            sparse_blocks.close(session_id)
        task = asyncio.create_task(
            self._tokenizer_manager.close_session(
                CloseSessionReqInput(session_id=session_id)
            )
        )
        self._close_tasks.add(task)
        task.add_done_callback(self._close_tasks.discard)
