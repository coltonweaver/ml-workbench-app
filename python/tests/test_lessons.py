"""The lesson flow: open, predict, check, progress.json and commits."""

from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from workbench_sidecar.lessons import Catalog, ContentError, LessonService, load_lesson
from workbench_sidecar.testing import TestRun

CONTENT = Path(__file__).resolve().parents[2] / "content"
SOLUTIONS = CONTENT / "m01-backprop" / "solutions"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=ws, check=True)
    return ws


@pytest.fixture
def service(workspace: Path) -> LessonService:
    return LessonService(Catalog(str(CONTENT)), str(workspace))


def check(service: LessonService, lesson_id: str) -> dict:
    lesson = service.catalog.lesson(lesson_id)
    report = TestRun(service.workspace, lesson.tests_path).run()
    service.record_check(lesson_id, report)
    return report


def test_summary_lists_module_one(service: LessonService) -> None:
    modules = service.summary()
    assert modules[0]["id"] == "m01"
    assert [l["id"] for l in modules[0]["lessons"]][:3] == ["1.1", "1.2", "1.3"]
    assert {l["status"] for l in modules[0]["lessons"]} == {"new"}


def test_open_copies_stub_and_makes_package(service: LessonService, workspace: Path) -> None:
    detail = service.open("1.1")
    assert (workspace / "m01" / "slope.py").read_text() == (
        CONTENT / "m01-backprop" / "stubs" / "slope.py"
    ).read_text()
    assert (workspace / "m01" / "__init__.py").exists()
    assert detail["progress"]["status"] == "in_progress"
    assert detail["explain"] is None and detail["stretch"] is None, "explain is hidden until passing"
    assert detail["hook"].startswith("You run a service")


def test_open_never_overwrites_learner_code(service: LessonService, workspace: Path) -> None:
    (workspace / "m01").mkdir()
    (workspace / "m01" / "slope.py").write_text("# mine\n")
    service.open("1.1")
    assert (workspace / "m01" / "slope.py").read_text() == "# mine\n"


def test_prediction_validation(service: LessonService) -> None:
    service.open("1.1")
    with pytest.raises(ContentError, match="isn't a number"):
        service.predict("1.1", "about three")
    detail = service.predict("1.1", "3.2")
    assert detail["progress"]["prediction"]["value"] == "3.2"
    with pytest.raises(ContentError, match="already recorded"):
        service.predict("1.1", "3")

    service.open("1.2")
    with pytest.raises(ContentError, match="isn't one of the options"):
        service.predict("1.2", "9")
    service.predict("1.2", "2")


def test_failed_check_logs_mistakes(service: LessonService, workspace: Path) -> None:
    service.open("1.1")
    report = check(service, "1.1")
    assert report["outcome"] == "failed"
    progress = json.loads((workspace / "progress.json").read_text())
    assert progress["lessons"]["1.1"]["attempts"] == 1
    assert progress["lessons"]["1.1"]["status"] == "in_progress"
    mistakes = progress["mistakes"]
    assert mistakes and all(m["lesson"] == "1.1" for m in mistakes)
    assert mistakes[0]["concepts"] == ["derivative", "finite-differences"]
    assert "NotImplementedError" in (mistakes[0]["message"] or "")


def test_pass_reveals_explain_grades_prediction_and_commits(
    service: LessonService, workspace: Path
) -> None:
    service.open("1.1")
    service.predict("1.1", "3")
    shutil.copy(SOLUTIONS / "slope.py", workspace / "m01" / "slope.py")
    report = check(service, "1.1")
    assert report["outcome"] == "passed", report["output"]

    detail = service.detail("1.1")
    assert detail["progress"]["status"] == "passed"
    assert "derivative" in detail["explain"]
    assert detail["progress"]["feedback"] == {"correct": True, "yours": "3", "answer": "3"}

    log = subprocess.run(["git", "log", "--format=%s"], cwd=workspace,
                         capture_output=True, text=True, check=True).stdout
    assert "Pass lesson 1.1: Which way is downhill?" in log
    tracked = subprocess.run(["git", "ls-files"], cwd=workspace,
                             capture_output=True, text=True, check=True).stdout.split()
    assert "m01/slope.py" in tracked and "progress.json" in tracked

    # A second pass is not a new pass: no extra commit.
    check(service, "1.1")
    log2 = subprocess.run(["git", "log", "--format=%s"], cwd=workspace,
                          capture_output=True, text=True, check=True).stdout
    assert log2.count("Pass lesson 1.1") == 1


def test_choice_feedback_shows_option_text(service: LessonService, workspace: Path) -> None:
    for lesson_id, name in (("1.1", "slope.py"), ("1.2", "descent.py")):
        service.open(lesson_id)
        shutil.copy(SOLUTIONS / name, workspace / "m01" / name)
    service.predict("1.2", "0")
    check(service, "1.2")
    feedback = service.detail("1.2")["progress"]["feedback"]
    assert feedback["correct"] is False
    assert feedback["yours"] == "w has settled at 7"
    assert feedback["answer"].startswith("w swings back and forth")


def test_bad_front_matter_is_reported(tmp_path: Path) -> None:
    (tmp_path / "lesson.md").write_text(textwrap.dedent("""\
        ---
        id: "9.9"
        title: Broken
        file: m09/x.py
        tests: test_x.py
        predict: {kind: number, prompt: "?"}
        ---
        ## Hook
        h
        ## Build
        b
        ## Explain
        e
    """))
    with pytest.raises(ContentError, match="need an answer"):
        load_lesson(str(tmp_path / "lesson.md"), {"id": "m09", "title": "t"}, str(tmp_path))
