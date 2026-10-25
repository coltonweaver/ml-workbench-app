"""Run pytest out of process and collect a structured report.

Tests run in a fresh interpreter, not in the learner's kernel, so a passing
test never depends on state left over from an earlier Run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
from typing import Any

PLUGIN = "workbench_sidecar.pytest_plugin"


class TestRun:
    def __init__(self, workspace: str, target: str, timeout: float = 300) -> None:
        self.workspace = workspace
        self.target = target
        self.timeout = timeout
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self.cancelled = False

    def run(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "report.json")
            env = dict(os.environ)
            env["WORKBENCH_REPORT"] = report_path
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (self.workspace, env.get("PYTHONPATH")) if p
            )
            cmd = [sys.executable, "-m", "pytest", "-p", PLUGIN, "-q",
                   "--no-header", "-p", "no:cacheprovider", self.target]
            with self._lock:
                if self.cancelled:
                    return _cancelled()
                self._proc = subprocess.Popen(
                    cmd, cwd=self.workspace, env=env, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
            try:
                output, _ = self._proc.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                output, _ = self._proc.communicate()
                return {"outcome": "timeout", "exit_code": None, "tests": [],
                        "passed": 0, "failed": 0, "output": output}
            if self.cancelled:
                return _cancelled()
            tests: list[dict[str, Any]] = []
            if os.path.exists(report_path):
                with open(report_path) as f:
                    tests = json.load(f)
            passed = sum(t["outcome"] == "passed" for t in tests)
            failed = sum(t["outcome"] in ("failed", "error") for t in tests)
            code = self._proc.returncode
            # pytest exit codes: 0 all passed, 1 some failed, 5 none collected.
            outcome = {0: "passed", 1: "failed", 5: "no_tests"}.get(code, "error")
            return {"outcome": outcome, "exit_code": code, "tests": tests,
                    "passed": passed, "failed": failed, "output": output}

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()


def _cancelled() -> dict[str, Any]:
    return {"outcome": "cancelled", "exit_code": None, "tests": [],
            "passed": 0, "failed": 0, "output": ""}
