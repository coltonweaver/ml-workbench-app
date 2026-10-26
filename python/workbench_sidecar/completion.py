"""Editor completions and call signatures, from Jedi's static analysis.

This never touches the kernel, so completions keep working while code runs.
Jedi resolves imports against the sidecar's own environment, which is the
kernel's, and the workspace root, so ``numpy``, ``torch``, ``workbench`` and
the learner's ``m01`` packages all complete.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Callable

import jedi

# Details (signature, docstring) are the slow part, so only the top of the
# list gets them. The rest still show with their name and kind.
MAX_COMPLETIONS = 80
DETAILED = 12
DOC_CHARS = 600


def _first_paragraph(doc: str) -> str | None:
    doc = doc.strip()
    if not doc:
        return None
    para = doc.split("\n\n", 1)[0].strip()
    return para[:DOC_CHARS]


def _signature(name: jedi.api.classes.BaseName) -> str | None:
    try:
        sigs = name.get_signatures()
    except Exception:
        return None
    return sigs[0].to_string() if sigs else None


def complete(code: str, line: int, column: int, path: str | None, workspace: str) -> dict[str, Any]:
    """Completions at ``line`` (1-based) and ``column`` (0-based, in code points).

    Returns ``{completions: [{name, complete, type, signature, doc}],
    signatures: [{name, params, index, doc}]}``.
    """
    project = jedi.Project(workspace, added_sys_path=[workspace])
    script = jedi.Script(code, path=path, project=project)
    lines = code.split("\n")
    line = max(1, min(line, len(lines)))
    column = max(0, min(column, len(lines[line - 1])))

    prefix_start = column
    text = lines[line - 1]
    while prefix_start > 0 and (text[prefix_start - 1].isalnum() or text[prefix_start - 1] == "_"):
        prefix_start -= 1
    typed = text[prefix_start:column]

    out: list[dict[str, Any]] = []
    try:
        found = script.complete(line, column)
    except Exception:
        found = []
    for c in found:
        # Private names only when asked for: `_` typed, or nothing else matches.
        if c.name.startswith("_") and not typed.startswith("_"):
            continue
        out.append({"name": c.name, "complete": c.complete or "", "type": c.type,
                    "signature": None, "doc": None})
        if len(out) >= MAX_COMPLETIONS:
            break
    by_name = {c.name: c for c in found}
    for item in out[:DETAILED]:
        c = by_name[item["name"]]
        if c.type in ("function", "class"):
            item["signature"] = _signature(c)
        try:
            item["doc"] = _first_paragraph(c.docstring(raw=True))
        except Exception:
            pass

    signatures: list[dict[str, Any]] = []
    try:
        for sig in script.get_signatures(line, column)[:1]:
            try:
                doc = _first_paragraph(sig.docstring(raw=True))
            except Exception:
                doc = None
            signatures.append({
                "name": sig.name,
                "params": [p.to_string() for p in sig.params],
                "index": sig.index,
                "doc": doc,
            })
    except Exception:
        pass
    return {"completions": out, "signatures": signatures}


class Completer:
    """Answers completion requests on its own thread, newest first.

    Typing sends a request per keystroke; one still waiting when a newer one
    arrives is answered empty rather than computed, since the editor has
    already moved on.
    """

    def __init__(self, workspace: str, send: Callable[[dict[str, Any]], None]) -> None:
        self.workspace = workspace
        self._send = send
        self._cond = threading.Condition()
        self._pending: dict[str, Any] | None = None
        self._stopped = False
        self._thread = threading.Thread(target=self._work, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify()

    def submit(self, request: dict[str, Any]) -> None:
        with self._cond:
            stale, self._pending = self._pending, request
            self._cond.notify()
        if stale is not None:
            self._reply(stale, {"completions": [], "signatures": []}, "aborted")

    def _work(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                request, self._pending = self._pending, None
            try:
                path = request.get("path")
                if path:
                    path = os.path.abspath(os.path.join(self.workspace, path))
                result = complete(request.get("code", ""), int(request.get("line", 1)),
                                  int(request.get("column", 0)), path, self.workspace)
                status = "ok"
            except Exception:
                result, status = {"completions": [], "signatures": []}, "error"
            self._reply(request, result, status)

    def _reply(self, request: dict[str, Any], result: dict[str, Any], status: str) -> None:
        rid = request.get("id")
        self._send({"type": "completions", "id": rid, **result})
        self._send({"type": "done", "id": rid, "status": status})
