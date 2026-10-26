"""Editor completions: Jedi over the editor's text, no kernel involved."""

from __future__ import annotations

import queue
from pathlib import Path

from workbench_sidecar.completion import Completer, complete


def names(result: dict) -> list[str]:
    return [c["name"] for c in result["completions"]]


def test_completes_attributes_of_a_module(tmp_path: Path) -> None:
    code = "import math\nmath.sq"
    result = complete(code, 2, len("math.sq"), None, str(tmp_path))
    assert "sqrt" in names(result)
    sqrt = next(c for c in result["completions"] if c["name"] == "sqrt")
    assert sqrt["complete"] == "rt"
    assert sqrt["type"] == "function"
    assert sqrt["signature"] and "sqrt" in sqrt["signature"]


def test_completes_names_defined_in_the_file(tmp_path: Path) -> None:
    code = "def slope(f, x, h=1e-5):\n    return 0\n\nslo"
    result = complete(code, 4, 3, None, str(tmp_path))
    assert names(result)[0] == "slope"


def test_hides_private_names_unless_asked(tmp_path: Path) -> None:
    code = "class A:\n    def _hidden(self): ...\n    def shown(self): ...\n\nA()."
    assert "_hidden" not in names(complete(code, 5, 4, None, str(tmp_path)))
    code_private = code + "_"
    assert "_hidden" in names(complete(code_private, 5, 5, None, str(tmp_path)))


def test_completes_workspace_packages(tmp_path: Path) -> None:
    (tmp_path / "m01").mkdir()
    (tmp_path / "m01" / "__init__.py").write_text("")
    (tmp_path / "m01" / "engine.py").write_text("class Value:\n    pass\n")
    code = "from m01.engine import Va"
    path = str(tmp_path / "scratch.py")
    assert "Value" in names(complete(code, 1, len(code), path, str(tmp_path)))


def test_signature_inside_a_call(tmp_path: Path) -> None:
    code = "def slope(f, x, h=1e-5):\n    return 0\n\nslope(abs, "
    result = complete(code, 4, len("slope(abs, "), None, str(tmp_path))
    [sig] = result["signatures"]
    assert sig["name"] == "slope"
    assert sig["index"] == 1
    assert sig["params"][0] == "f"


def test_completer_answers_newest_and_drops_stale(tmp_path: Path) -> None:
    events: queue.Queue[dict] = queue.Queue()

    def send(event: dict) -> None:
        events.put(event)

    completer = Completer(str(tmp_path), send)
    # Queue two before the worker starts: the first is answered empty.
    completer.submit({"id": "c1", "code": "import math\nmath.", "line": 2, "column": 5})
    completer.submit({"id": "c2", "code": "import math\nmath.", "line": 2, "column": 5})
    completer.start()
    got = [events.get(timeout=30) for _ in range(4)]
    completer.stop()
    first = [e for e in got if e["id"] == "c1"]
    second = [e for e in got if e["id"] == "c2"]
    assert first[0]["completions"] == [] and first[1] == {"type": "done", "id": "c1", "status": "aborted"}
    assert any(c["name"] == "sqrt" for c in second[0]["completions"])
    assert second[1]["status"] == "ok"
