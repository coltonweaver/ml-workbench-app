"""Entry point: ``python -m workbench_sidecar --workspace <dir>``.

Requests are read from stdin one JSON object per line. Long-running requests
(execute, run_file, run_tests, restart) are handled one at a time on a worker
thread, so ``interrupt`` and ``shutdown`` stay responsive while code runs.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import traceback
from typing import Any, TextIO

from . import PROTOCOL_VERSION
from .kernel import KernelSession
from .lessons import Catalog, ContentError, LessonService
from .testing import TestRun


class Server:
    def __init__(self, workspace: str, out: TextIO, content: str | None = None) -> None:
        self.workspace = os.path.abspath(workspace)
        self.kernel = KernelSession(self.workspace)
        self.lessons = LessonService(Catalog(content), self.workspace) if content else None
        self._out = out
        self._write_lock = threading.Lock()
        self._jobs: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._current_test: TestRun | None = None
        self._worker = threading.Thread(target=self._work, daemon=True)

    def send(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":"))
        with self._write_lock:
            self._out.write(line + "\n")
            self._out.flush()

    def start(self) -> None:
        self.kernel.start()
        self._worker.start()
        self.send({"type": "ready", "protocol": PROTOCOL_VERSION,
                   "python": sys.version.split()[0], "workspace": self.workspace})

    def handle(self, request: dict[str, Any]) -> bool:
        """Dispatch one request. Returns False when the server should exit."""
        op = request.get("op")
        if op == "interrupt":
            test = self._current_test
            if test is not None:
                test.cancel()
            else:
                self.kernel.interrupt()
        elif op == "shutdown":
            self._jobs.put(None)
            self._worker.join(timeout=5)
            self.kernel.shutdown()
            return False
        elif op in ("execute", "run_file", "run_tests", "restart", "check_lesson",
                    "list_lessons", "open_lesson", "submit_prediction"):
            self._jobs.put(request)
        else:
            self.send({"type": "error", "id": request.get("id"), "ename": "ProtocolError",
                       "evalue": f"unknown op {op!r}", "traceback": []})
        return True

    def _work(self) -> None:
        while (request := self._jobs.get()) is not None:
            rid = request.get("id")
            try:
                status = self._run(request)
            except ContentError as exc:
                self.send({"type": "error", "id": rid, "ename": "LessonError",
                           "evalue": str(exc), "traceback": []})
                status = "error"
            except Exception as exc:  # report and keep serving
                self.send({"type": "error", "id": rid, "ename": type(exc).__name__,
                           "evalue": str(exc), "traceback": traceback.format_exc().splitlines()})
                status = "error"
            self.send({"type": "done", "id": rid, "status": status})

    def _run(self, request: dict[str, Any]) -> str:
        rid = request.get("id")
        op = request["op"]

        def emit(event: dict[str, Any]) -> None:
            self.send({**event, "id": rid})

        if op == "execute":
            return self.kernel.execute(request["code"], emit)
        if op == "run_file":
            path = self._resolve(request["path"])
            # %run executes the file as __main__ in a fresh namespace, then
            # merges its globals into the kernel so the learner can poke at them.
            return self.kernel.execute(f"%run {json.dumps(path)}", emit)
        if op == "run_tests":
            report = self._test(self._resolve(request.get("target", ".")))
            emit({"type": "test_report", **report})
            return "ok" if report["outcome"] == "passed" else "error"
        if op in ("list_lessons", "open_lesson", "submit_prediction", "check_lesson"):
            return self._lesson_op(op, request, emit)
        if op == "restart":
            self.kernel.restart()
            return "ok"
        raise ValueError(f"unhandled op {op!r}")

    def _test(self, target: str) -> dict[str, Any]:
        run = TestRun(self.workspace, target)
        self._current_test = run
        try:
            return run.run()
        finally:
            self._current_test = None

    def _lesson_op(self, op: str, request: dict[str, Any], emit) -> str:  # noqa: ANN001
        if self.lessons is None:
            raise ContentError("no lesson content was given to the sidecar (--content)")
        if op == "list_lessons":
            self.lessons.catalog.reload()
            emit({"type": "lessons", "modules": self.lessons.summary()})
            return "ok"
        lesson_id = str(request["lesson"])
        if op == "open_lesson":
            emit({"type": "lesson", "lesson": self.lessons.open(lesson_id)})
            return "ok"
        if op == "submit_prediction":
            emit({"type": "lesson", "lesson": self.lessons.predict(lesson_id, request["value"])})
            return "ok"
        # check_lesson: hidden tests against the learner's file, then progress.
        lesson = self.lessons.catalog.lesson(lesson_id)
        report = self._test(lesson.tests_path)
        emit({"type": "test_report", **report})
        if report["outcome"] in ("cancelled", "timeout"):
            return "error"
        self.lessons.record_check(lesson_id, report)
        emit({"type": "lesson", "lesson": self.lessons.detail(lesson_id)})
        emit({"type": "lessons", "modules": self.lessons.summary()})
        return "ok" if report["outcome"] == "passed" else "error"

    def _resolve(self, path: str) -> str:
        full = os.path.abspath(os.path.join(self.workspace, path))
        if os.path.commonpath([full, self.workspace]) != self.workspace:
            raise ValueError(f"{path!r} is outside the workspace")
        return full


def serve(workspace: str, inp: TextIO, out: TextIO, content: str | None = None) -> None:
    server = Server(workspace, out, content)
    server.start()
    try:
        for line in inp:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                server.send({"type": "error", "id": None, "ename": "ProtocolError",
                             "evalue": f"bad JSON: {exc}", "traceback": []})
                continue
            if not server.handle(request):
                return
    finally:
        server.kernel.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(prog="workbench_sidecar")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--content", help="lesson content folder")
    args = parser.parse_args()
    os.makedirs(args.workspace, exist_ok=True)
    # Keep our protocol stream clean: anything stray goes to stderr.
    protocol_out = sys.stdout
    sys.stdout = sys.stderr
    serve(args.workspace, sys.stdin, protocol_out, args.content)


if __name__ == "__main__":
    main()
