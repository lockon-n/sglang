"""A streaming session whose in-flight request left the scheduler unaccounted
for must still close: its deferred close would otherwise wait forever and the
session would hold its KV and mamba slot for good."""

from types import SimpleNamespace

from sglang.srt.session.session_controller import Session, SessionController
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeTreeCache:
    def __init__(self):
        self.released = []

    def release_radix_session(self, session_id):
        pass

    def release_session(self, session_id):
        self.released.append(session_id)


def _controller(where):
    cache = _FakeTreeCache()
    controller = SessionController(cache)
    controller.locate_req = lambda rid: where[0]
    session = Session(capacity_of_str_len=16, session_id="s", streaming=True, timeout=0.0)
    session._inflight, session._inflight_rid = True, "r1"
    controller.sessions["s"] = session
    return controller, session, cache


def test_close_waits_while_the_scheduler_holds_the_request():
    where = ["running"]
    controller, session, cache = _controller(where)
    controller._close("s")
    assert session.close_on_finish and "s" in controller.sessions
    for _ in range(3):
        controller._release_orphaned_sessions()
    assert "s" in controller.sessions and cache.released == []

    # Once the request finishes normally, the deferred close goes through.
    done = SimpleNamespace(rid="r1", origin_input_ids=[], origin_input_ids_unpadded=[],
                           full_untruncated_fill_ids=[], session=session, multimodal_inputs=None,
                           finished=lambda: True)
    session.finish_req(done)
    controller._close("s")
    assert "s" not in controller.sessions and cache.released == ["s"]


def test_orphaned_request_releases_after_two_reaps():
    where = [None]
    controller, session, cache = _controller(where)
    controller._close("s")
    assert session.close_on_finish
    controller._release_orphaned_sessions()  # one miss may be a transient
    assert "s" in controller.sessions
    controller._release_orphaned_sessions()
    assert "s" not in controller.sessions and cache.released == ["s"]


def test_a_request_seen_again_resets_the_count():
    where = [None]
    controller, session, cache = _controller(where)
    controller._close("s")
    controller._release_orphaned_sessions()
    where[0] = "waiting"
    controller._release_orphaned_sessions()
    where[0] = None
    controller._release_orphaned_sessions()
    assert "s" in controller.sessions
    controller._release_orphaned_sessions()
    assert "s" not in controller.sessions


def test_reap_releases_an_orphan_end_to_end():
    where = [None]
    controller, session, cache = _controller(where)
    controller._close("s")
    for t in range(1, 5):
        controller.maybe_reap(now=float(t) * 2)
    assert "s" not in controller.sessions and cache.released == ["s"]
