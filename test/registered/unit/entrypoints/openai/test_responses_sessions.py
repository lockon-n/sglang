import asyncio
import unittest

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.responses_sessions import ResponsesSessionManager
from sglang.srt.utils.common import ImageData
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Tokenizer:
    def decode(self, ids, skip_special_tokens):
        return "".join(chr(i) for i in ids)


class _TokenizerManager:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.opened = []
        self.closed = []

    async def open_session(self, obj):
        session_id = f"s{len(self.opened)}"
        self.opened.append(obj)
        return session_id

    async def close_session(self, obj):
        self.closed.append(obj.session_id)


async def _outputs(*outputs):
    for output in outputs:
        context = SimpleContext()
        context.append_output(output)
        yield context


def _finished(text, finish_type="stop"):
    return {
        "output_ids": [ord(c) for c in text],
        "meta_info": {"finish_reason": {"type": finish_type}},
    }


class ResponsesSessionManagerTestCase(CustomTestCase):
    def setUp(self):
        self.tm = _TokenizerManager()
        self.manager = ResponsesSessionManager(
            tokenizer_manager=self.tm, idle_timeout=300
        )

    def _run_turn(self, *, previous, response_id, prompt, images, output):
        async def run():
            turn = await self.manager.begin_turn(
                previous_response_id=previous,
                prompt=prompt,
                image_data=images,
                modalities=["image"] * len(images) if images else None,
            )
            async for _ in self.manager.track(
                turn=turn,
                response_id=response_id,
                generator=_outputs(output),
                incremental_output=False,
            ):
                pass
            await asyncio.sleep(0)
            return turn

        return asyncio.run(run())

    def test_follow_up_turn_sends_only_the_new_suffix_and_images(self):
        first = [ImageData(url="data:a")]
        turn = self._run_turn(
            previous=None,
            response_id="r1",
            prompt="P1<img>",
            images=first,
            output=_finished("A1"),
        )
        self.assertEqual((turn.session_id, turn.text), ("s0", "P1<img>"))

        both = first + [ImageData(url="data:b")]
        turn = self._run_turn(
            previous="r1",
            response_id="r2",
            prompt="P1<img>A1|P2<img>",
            images=both,
            output=_finished("A2"),
        )
        self.assertEqual(turn.session_id, "s0")
        self.assertEqual(turn.text, "|P2<img>")
        self.assertEqual([i.url for i in turn.image_data], ["data:b"])
        self.assertEqual(turn.modalities, ["image"])
        self.assertEqual(len(self.tm.opened), 1)

    def test_rewritten_history_moves_the_chain_to_a_new_session(self):
        self._run_turn(
            previous=None,
            response_id="r1",
            prompt="P1",
            images=[],
            output=_finished("A1"),
        )
        turn = self._run_turn(
            previous="r1",
            response_id="r2",
            prompt="P1-rewritten|P2",
            images=[],
            output=_finished("A2"),
        )
        self.assertEqual((turn.session_id, turn.text), ("s1", "P1-rewritten|P2"))
        self.assertEqual(self.tm.closed, ["s0"])

    def test_branching_off_a_consumed_response_opens_a_new_session(self):
        self._run_turn(
            previous=None,
            response_id="r1",
            prompt="P1",
            images=[],
            output=_finished("A1"),
        )
        self._run_turn(
            previous="r1",
            response_id="r2",
            prompt="P1A1|P2",
            images=[],
            output=_finished("A2"),
        )
        turn = self._run_turn(
            previous="r1",
            response_id="r3",
            prompt="P1A1|P3",
            images=[],
            output=_finished("A3"),
        )
        self.assertEqual((turn.session_id, turn.text), ("s1", "P1A1|P3"))

    def test_aborted_turn_closes_its_session_and_is_not_a_chain_head(self):
        self._run_turn(
            previous=None,
            response_id="r1",
            prompt="P1",
            images=[],
            output=_finished("A1", finish_type="abort"),
        )
        self.assertEqual(self.tm.closed, ["s0"])
        turn = self._run_turn(
            previous="r1",
            response_id="r2",
            prompt="P1A1|P2",
            images=[],
            output=_finished("A2"),
        )
        self.assertEqual((turn.session_id, turn.text), ("s1", "P1A1|P2"))

    def test_changed_earlier_image_is_not_served_from_the_session(self):
        self._run_turn(
            previous=None,
            response_id="r1",
            prompt="P1<img>",
            images=[ImageData(url="data:a")],
            output=_finished("A1"),
        )
        turn = self._run_turn(
            previous="r1",
            response_id="r2",
            prompt="P1<img>A1|P2",
            images=[ImageData(url="data:other")],
            output=_finished("A2"),
        )
        self.assertEqual(turn.session_id, "s1")
        self.assertEqual(turn.text, "P1<img>A1|P2")


if __name__ == "__main__":
    unittest.main()
