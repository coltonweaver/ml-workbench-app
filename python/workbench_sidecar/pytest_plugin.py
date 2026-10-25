"""pytest plugin loaded with ``-p``; writes one JSON record per test.

Failure messages are the short form (the assertion text or exception line),
since lesson tests are written to put the useful explanation in the message.
"""

from __future__ import annotations

import json
import os
from typing import Any

_results: list[dict[str, Any]] = []


def pytest_runtest_logreport(report):  # noqa: ANN001 - pytest hook signature
    # Record the call phase, plus setup/teardown only when they didn't pass
    # (a failing fixture is an "error", a skip marker shows up in setup).
    if report.when == "call":
        outcome = report.outcome
    elif report.skipped:
        outcome = "skipped"
    elif report.failed:
        outcome = "error"
    else:
        return
    _results.append({
        "nodeid": report.nodeid,
        "outcome": outcome,
        "duration": round(report.duration, 4),
        "message": _message(report) if not report.passed else None,
    })


def pytest_sessionfinish(session, exitstatus):  # noqa: ANN001
    path = os.environ.get("WORKBENCH_REPORT")
    if path:
        with open(path, "w") as f:
            json.dump(_results, f)


def _message(report) -> str | None:  # noqa: ANN001
    crash = getattr(report.longrepr, "reprcrash", None)
    if crash is not None:
        return crash.message
    if report.longrepr is not None:
        return str(report.longrepr).strip().splitlines()[-1]
    return None
