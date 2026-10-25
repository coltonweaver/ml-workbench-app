"""A thin wrapper over jupyter_client that turns IOPub traffic into flat events."""

from __future__ import annotations

import queue
from collections.abc import Callable
from typing import Any

from jupyter_client.manager import KernelManager

Event = dict[str, Any]
Emit = Callable[[Event], None]


class KernelSession:
    """One IPython kernel whose working directory is the learner's workspace."""

    def __init__(self, workspace: str, kernel_name: str = "python3") -> None:
        self.workspace = workspace
        self.kernel_name = kernel_name
        self._km: KernelManager | None = None
        self._kc = None

    def start(self, timeout: float = 60) -> None:
        self._km = KernelManager(kernel_name=self.kernel_name)
        self._km.start_kernel(cwd=self.workspace)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=timeout)
        # Learner packages live at the workspace root; make them importable
        # regardless of how the kernel resolved its cwd.
        self.execute(
            f"import sys as _s; _s.path.insert(0, {self.workspace!r}) "
            f"if {self.workspace!r} not in _s.path else None; del _s",
            emit=lambda _e: None,
        )

    def execute(self, code: str, emit: Emit, timeout: float | None = None) -> str:
        """Run ``code`` and stream its outputs through ``emit``.

        Returns the final status: ``"ok"``, ``"error"`` or ``"aborted"``.
        """
        assert self._kc is not None, "kernel not started"
        msg_id = self._kc.execute(code, store_history=True, allow_stdin=False)
        status = "ok"
        while True:
            try:
                msg = self._kc.get_iopub_msg(timeout=timeout)
            except queue.Empty:
                emit({"type": "error", "ename": "TimeoutError",
                      "evalue": "kernel produced no output before the timeout",
                      "traceback": []})
                return "error"
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind = msg["msg_type"]
            content = msg["content"]
            if kind == "stream":
                emit({"type": "stream", "name": content["name"], "text": content["text"]})
            elif kind in ("execute_result", "display_data", "update_display_data"):
                event = {"type": kind, "data": _text_data(content["data"])}
                display_id = content.get("transient", {}).get("display_id")
                if display_id:
                    event["display_id"] = display_id
                emit(event)
            elif kind == "error":
                status = "error"
                emit({"type": "error", "ename": content["ename"],
                      "evalue": content["evalue"], "traceback": content["traceback"]})
            elif kind == "status" and content["execution_state"] == "idle":
                break
        reply = self._kc.get_shell_msg(timeout=5)
        if reply["content"].get("status") == "aborted":
            status = "aborted"
        return status

    def interrupt(self) -> None:
        if self._km is not None:
            self._km.interrupt_kernel()

    def restart(self) -> None:
        assert self._km is not None and self._kc is not None
        self._km.restart_kernel(now=True)
        self._kc.wait_for_ready(timeout=60)

    def shutdown(self) -> None:
        if self._kc is not None:
            self._kc.stop_channels()
        if self._km is not None:
            self._km.shutdown_kernel(now=True)
        self._kc = None
        self._km = None


def _text_data(data: dict[str, Any]) -> dict[str, Any]:
    """Keep the MIME types the app can render: plain formats plus the
    instrument payloads published by the ``workbench`` helper."""
    keep = ("text/plain", "text/markdown", "image/png", "application/json")
    return {k: v for k, v in data.items()
            if k in keep or k.startswith("application/vnd.workbench.")}
