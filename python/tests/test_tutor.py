"""The tutor, with a fake Claude client: context assembly, streaming, errors."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from workbench_sidecar.__main__ import Server
from workbench_sidecar.tutor import (
    FALLBACK_BETA, MODEL, SYSTEM_PROMPT, Tutor, TutorError, TutorRequest,
    build_context, build_messages,
)

CONTENT = Path(__file__).resolve().parents[2] / "content"


class FakeStream:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn") -> None:
        self.chunks = chunks
        self.stop_reason = stop_reason

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    @property
    def text_stream(self):
        yield from self.chunks

    def get_final_message(self) -> Any:
        return SimpleNamespace(stop_reason=self.stop_reason, model=MODEL)


class FakeClient:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn") -> None:
        self.calls: list[dict[str, Any]] = []
        self.chunks = chunks
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self.chunks, self.stop_reason)


def make_tutor(client: FakeClient) -> Tutor:
    return Tutor(client_factory=lambda key: client)


def collect() -> tuple[list[dict], Any]:
    events: list[dict] = []
    return events, events.append


def test_request_parse_filters_history() -> None:
    r = TutorRequest.parse({
        "question": " why? ",
        "history": [{"role": "assistant", "text": "hi"}, {"role": "system", "text": "x"},
                    {"role": "user", "text": "q"}, {"role": "user", "text": ""}],
        "api_key": "k",
    })
    assert r.question == "why?"
    assert r.history == [{"role": "assistant", "text": "hi"}, {"role": "user", "text": "q"}]
    with pytest.raises(TutorError):
        TutorRequest.parse({"question": "  "})
    assert TutorRequest.parse({"tensor": {"shape": [2]}}).question == "Explain this tensor."


def test_messages_start_with_user_and_carry_context_last() -> None:
    r = TutorRequest(question="now?", history=[
        {"role": "assistant", "text": "greeting"},
        {"role": "user", "text": "first"}, {"role": "assistant", "text": "answer"}])
    messages = build_messages(r, "<context>c</context>")
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[-1]["content"] == [
        {"type": "text", "text": "<context>c</context>"}, {"type": "text", "text": "now?"}]


def test_streams_and_sets_api_parameters() -> None:
    client = FakeClient(["Look at ", "line 3."])
    events, emit = collect()
    status = make_tutor(client).ask(TutorRequest("help", [], api_key="k"), "ctx", emit)
    assert status == "ok"
    assert [e["text"] for e in events if e["type"] == "tutor_delta"] == ["Look at ", "line 3."]
    assert events[-1] == {"type": "tutor_done", "stop_reason": "end_turn", "model": MODEL}
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["system"][0]["text"] == SYSTEM_PROMPT
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in call and "temperature" not in call
    assert call["output_config"]["effort"] in ("low", "medium", "high")


def test_follow_up_replays_previous_turn_byte_for_byte() -> None:
    client = FakeClient(["A1"])
    tutor = make_tutor(client)
    tutor.ask(TutorRequest("Q1", [], api_key="k"), "<context>one</context>", lambda e: None)
    tutor.ask(TutorRequest("Q2", [{"role": "user", "text": "Q1"},
                                  {"role": "assistant", "text": "A1"}], api_key="k"),
              "<context>two</context>", lambda e: None)
    first, second = client.calls[0]["messages"], client.calls[1]["messages"]
    assert second[: len(first)] == first  # cached prefix stays identical
    assert second[1] == {"role": "assistant", "content": "A1"}
    # A different chat (the app cleared it) is rebuilt from plain text.
    tutor.ask(TutorRequest("Q3", [{"role": "user", "text": "other"},
                                  {"role": "assistant", "text": "A1"}], api_key="k"),
              "", lambda e: None)
    assert client.calls[2]["messages"][0] == {"role": "user", "content": "other"}


def test_refusal_is_reported() -> None:
    client = FakeClient([], stop_reason="refusal")
    events, emit = collect()
    assert make_tutor(client).ask(TutorRequest("q", [], api_key="k"), "", emit) == "error"
    assert events[-1]["stop_reason"] == "refusal" and events[-1]["note"]


def test_cancel_stops_the_stream() -> None:
    tutor = make_tutor(FakeClient(["a", "b", "c"]))
    events: list[dict] = []

    def emit(event: dict) -> None:
        events.append(event)
        tutor.cancel()

    assert tutor.ask(TutorRequest("q", [], api_key="k"), "", emit) == "aborted"
    assert [e["type"] for e in events] == ["tutor_delta", "tutor_done"]


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(TutorError, match="API key"):
        make_tutor(FakeClient([])).ask(TutorRequest("q", []), "", lambda e: None)


def test_api_errors_become_tutor_errors() -> None:
    import anthropic
    import httpx2

    def boom(**kwargs: Any) -> Any:
        raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))

    client = FakeClient([])
    client.beta.messages.stream = boom
    with pytest.raises(TutorError, match="network"):
        make_tutor(client).ask(TutorRequest("q", [], api_key="k"), "", lambda e: None)


def test_context_hides_explanation_until_passed() -> None:
    lesson = {
        "id": "1.1", "title": "T", "module_title": "M", "file": "m01/x.py",
        "concepts": ["slope"], "hook": "HOOK", "build": "BUILD", "explain": "SECRET",
        "predict": {"kind": "choice", "prompt": "which?", "options": ["a", "b"]},
        "progress": {"status": "in_progress", "attempts": 2,
                     "prediction": {"value": "1"}, "feedback": None},
    }
    ctx = build_context(lesson=lesson, code="def f(): pass", problem="AssertionError: 3 != 4",
                        mistakes=[{"lesson": "1.1", "test": "test_x", "message": "bad\nmore",
                                   "concepts": ["slope"]}],
                        tensor={"name": "w", "shape": [3]})
    assert "HOOK" in ctx and "BUILD" in ctx and "SECRET" not in ctx
    assert "Their prediction: 1 (b)" in ctx
    assert "def f(): pass" in ctx and "3 != 4" in ctx
    assert "test_x: bad [concepts: slope]" in ctx
    assert '"shape"' in ctx
    lesson["progress"]["status"] = "passed"
    assert "SECRET" in build_context(lesson=lesson, code=None, problem=None, mistakes=[], tensor=None)
    assert build_context(lesson=None, code=None, problem=None, mistakes=[], tensor=None) == ""


def test_server_assembles_context_from_workspace(tmp_path: Path) -> None:
    client = FakeClient(["ok"])
    out = io.StringIO()
    server = Server(str(tmp_path), out, str(CONTENT), tutor=make_tutor(client))
    server.lessons.open("1.1")
    server._note_problem("check_lesson", {
        "type": "test_report", "outcome": "failed",
        "tests": [{"nodeid": "test_slope.py::test_sign", "outcome": "failed",
                   "message": "expected positive slope"}]})
    events, emit = collect()
    assert server._ask({"question": "stuck", "lesson": "1.1", "api_key": "k"}, emit) == "ok"
    context = client.calls[0]["messages"][-1]["content"][0]["text"]
    lesson = server.lessons.catalog.lesson("1.1")
    assert f"Lesson 1.1: {lesson.title}" in context
    assert "expected positive slope" in context
    stub = (tmp_path / lesson.file).read_text()
    assert stub.strip()[:40] in context  # the learner's file, read from the workspace
    # Hidden tests and the reference solution are never sent.
    assert Path(lesson.tests_path).read_text().strip()[:60] not in context
    # A pass clears the remembered failure.
    server._note_problem("check_lesson", {"type": "test_report", "outcome": "passed", "tests": []})
    assert server.last_problem is None


def test_ask_tutor_runs_beside_the_kernel_queue(tmp_path: Path) -> None:
    out = io.StringIO()
    server = Server(str(tmp_path), out, str(CONTENT), tutor=make_tutor(FakeClient(["hi"])))
    server._tutor_worker.start()  # the kernel worker is never started here
    server.handle({"id": "t1", "op": "ask_tutor", "question": "q", "api_key": "k"})
    server._questions.put(None)
    server._tutor_worker.join(timeout=10)
    events = [json.loads(l) for l in out.getvalue().splitlines()]
    assert [e["type"] for e in events] == ["tutor_delta", "tutor_done", "done"]
    assert all(e["id"] == "t1" for e in events)


def test_errors_are_reported_as_tutor_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = io.StringIO()
    server = Server(str(tmp_path), out, str(CONTENT), tutor=make_tutor(FakeClient([])))
    server._tutor_worker.start()
    server.handle({"id": "t1", "op": "ask_tutor", "question": "q"})
    server._questions.put(None)
    server._tutor_worker.join(timeout=10)
    events = [json.loads(l) for l in out.getvalue().splitlines()]
    assert events[0]["ename"] == "TutorError"
    assert events[-1] == {"type": "done", "status": "error", "id": "t1"}
