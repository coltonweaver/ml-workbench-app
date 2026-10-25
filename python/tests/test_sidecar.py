"""End-to-end tests: launch the sidecar as the app does and speak JSON lines."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
import queue
from pathlib import Path

import pytest


class Sidecar:
    def __init__(self, workspace: Path) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "workbench_sidecar", "--workspace", str(workspace)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self.events: queue.Queue[dict] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.ready = self.next(timeout=60)
        self._n = 0

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.events.put(json.loads(line))

    def next(self, timeout: float = 30) -> dict:
        return self.events.get(timeout=timeout)

    def send(self, **request) -> str:
        self._n += 1
        rid = request.setdefault("id", f"r{self._n}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        return rid

    def request(self, timeout: float = 60, **request) -> list[dict]:
        """Send a request and collect its events up to and including ``done``."""
        rid = self.send(**request)
        out = []
        deadline = time.monotonic() + timeout
        while True:
            event = self.next(timeout=max(0.1, deadline - time.monotonic()))
            if event.get("id") != rid:
                continue
            out.append(event)
            if event["type"] == "done":
                return out

    def close(self) -> None:
        self.send(op="shutdown")
        self.proc.wait(timeout=30)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def sidecar(workspace: Path):
    sc = Sidecar(workspace)
    yield sc
    sc.close()


def stdout_of(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "stream" and e["name"] == "stdout")


def test_ready_reports_protocol_and_workspace(sidecar: Sidecar, workspace: Path) -> None:
    assert sidecar.ready["type"] == "ready"
    assert sidecar.ready["protocol"] == 1
    assert sidecar.ready["workspace"] == str(workspace)


def test_execute_streams_output_and_keeps_state(sidecar: Sidecar) -> None:
    events = sidecar.request(op="execute", code="x = 21\nprint('hi')")
    assert stdout_of(events) == "hi\n"
    assert events[-1] == {"type": "done", "id": events[-1]["id"], "status": "ok"}

    events = sidecar.request(op="execute", code="x * 2")
    result = [e for e in events if e["type"] == "execute_result"]
    assert result[0]["data"]["text/plain"] == "42"


def test_execute_error_is_reported(sidecar: Sidecar) -> None:
    events = sidecar.request(op="execute", code="1 / 0")
    errors = [e for e in events if e["type"] == "error"]
    assert errors[0]["ename"] == "ZeroDivisionError"
    assert events[-1]["status"] == "error"


def test_run_file_executes_workspace_file(sidecar: Sidecar, workspace: Path) -> None:
    (workspace / "hello.py").write_text("greeting = 'hello'\nprint(greeting)\n")
    events = sidecar.request(op="run_file", path="hello.py")
    assert stdout_of(events) == "hello\n"
    # Globals from the file are left in the kernel for inspection.
    events = sidecar.request(op="execute", code="print(greeting.upper())")
    assert stdout_of(events) == "HELLO\n"


def test_run_file_rejects_paths_outside_workspace(sidecar: Sidecar) -> None:
    events = sidecar.request(op="run_file", path="../escape.py")
    assert any(e["type"] == "error" and "outside the workspace" in e["evalue"] for e in events)
    assert events[-1]["status"] == "error"


def test_workspace_packages_are_importable(sidecar: Sidecar, workspace: Path) -> None:
    pkg = workspace / "m01"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "engine.py").write_text("def double(v):\n    return 2 * v\n")
    events = sidecar.request(op="execute", code="from m01.engine import double; print(double(4))")
    assert stdout_of(events) == "8\n"


def test_interrupt_stops_running_code(sidecar: Sidecar) -> None:
    rid = sidecar.send(op="execute", code="import time\nwhile True: time.sleep(0.05)")
    time.sleep(1.0)
    sidecar.send(op="interrupt")
    events = []
    while True:
        event = sidecar.next(timeout=30)
        if event.get("id") == rid:
            events.append(event)
            if event["type"] == "done":
                break
    assert any(e["type"] == "error" and e["ename"] == "KeyboardInterrupt" for e in events)
    # Kernel is still usable afterwards.
    assert stdout_of(sidecar.request(op="execute", code="print('alive')")) == "alive\n"


def test_run_tests_reports_pass_and_fail(sidecar: Sidecar, workspace: Path) -> None:
    (workspace / "mathy.py").write_text("def add(a, b):\n    return a - b  # bug\n")
    (workspace / "test_mathy.py").write_text(textwrap.dedent("""
        from mathy import add

        def test_add_zero():
            assert add(3, 0) == 3

        def test_add():
            assert add(2, 2) == 4, "add(2, 2) should be 4; check the operator"
    """))
    events = sidecar.request(op="run_tests", target="test_mathy.py")
    report = next(e for e in events if e["type"] == "test_report")
    assert report["outcome"] == "failed"
    assert (report["passed"], report["failed"]) == (1, 1)
    failing = next(t for t in report["tests"] if t["outcome"] == "failed")
    assert failing["nodeid"].endswith("test_add")
    assert "check the operator" in failing["message"]
    assert events[-1]["status"] == "error"

    (workspace / "mathy.py").write_text("def add(a, b):\n    return a + b\n")
    events = sidecar.request(op="run_tests", target="test_mathy.py")
    report = next(e for e in events if e["type"] == "test_report")
    assert report["outcome"] == "passed"
    assert events[-1]["status"] == "ok"


def test_run_tests_does_not_see_kernel_state(sidecar: Sidecar, workspace: Path) -> None:
    sidecar.request(op="execute", code="import builtins; builtins.leaked = 1")
    (workspace / "test_isolated.py").write_text(
        "import builtins\n\ndef test_clean():\n    assert not hasattr(builtins, 'leaked')\n"
    )
    events = sidecar.request(op="run_tests", target="test_isolated.py")
    report = next(e for e in events if e["type"] == "test_report")
    assert report["outcome"] == "passed"


def test_unknown_op_is_a_protocol_error(sidecar: Sidecar) -> None:
    sidecar.send(op="bogus", id="x1")
    event = sidecar.next()
    assert event["id"] == "x1"
    assert event["ename"] == "ProtocolError"
