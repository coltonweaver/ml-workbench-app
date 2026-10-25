"""Checks every lesson in content/ against its own hidden tests.

For each lesson: the reference solution passes, the starting stub fails, and
every planted bug ("mutant") listed in the lesson's front matter fails. That
last one is the spec's acceptance bar: the hidden tests catch a deliberately
wrong implementation.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from workbench_sidecar.lessons import Catalog, Lesson
from workbench_sidecar.testing import TestRun

CONTENT = Path(__file__).resolve().parents[2] / "content"
CATALOG = Catalog(str(CONTENT))
LESSONS = [lesson for module in CATALOG.modules for lesson in module.lessons]


def module_dir(lesson: Lesson) -> Path:
    return Path(next(m.directory for m in CATALOG.modules if m.id == lesson.module_id))


def workspace_with_solutions(tmp_path: Path, lesson: Lesson) -> Path:
    """A workspace where the learner has finished every lesson up to the end
    of this lesson's module. Later modules import earlier ones (m03 builds on
    m02's tensor engine), so earlier modules' solutions are there too."""
    ws = tmp_path / "ws"
    this = next(m for m in CATALOG.modules if m.id == lesson.module_id)
    for module in CATALOG.modules:
        if module.number > this.number or not module.lessons:
            continue
        package = ws / Path(module.lessons[0].file).parent
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
        for solution in (Path(module.directory) / "solutions").glob("*.py"):
            shutil.copy(solution, package / solution.name)
    return ws


def run(ws: Path, lesson: Lesson) -> dict:
    return TestRun(str(ws), lesson.tests_path).run()


def test_content_loaded():
    assert LESSONS, "no lessons found"
    ids = [l.id for l in LESSONS]
    key = lambda i: [int(p) for p in i.split(".")]  # noqa: E731
    assert ids == sorted(ids, key=key), f"lessons should be in id order: {ids}"


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda l: l.id)
def test_solution_passes(tmp_path: Path, lesson: Lesson) -> None:
    assert lesson.solution, f"lesson {lesson.id} has no solutions/{os.path.basename(lesson.file)}"
    report = run(workspace_with_solutions(tmp_path, lesson), lesson)
    assert report["outcome"] == "passed", report["output"]


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda l: l.id)
def test_stub_fails(tmp_path: Path, lesson: Lesson) -> None:
    stub = Path(lesson.stub) if lesson.stub else module_dir(lesson) / "stubs" / Path(lesson.file).name
    assert stub.exists(), f"no stub for lesson {lesson.id}"
    ws = workspace_with_solutions(tmp_path, lesson)
    shutil.copy(stub, ws / lesson.file)
    report = run(ws, lesson)
    assert report["outcome"] == "failed", f"the stub passes lesson {lesson.id}'s tests:\n{report['output']}"


MUTANTS = [(lesson, m) for lesson in LESSONS for m in lesson.mutants]


@pytest.mark.parametrize("lesson,mutant", MUTANTS, ids=lambda x: getattr(x, "id", None) or x["name"])
def test_mutant_is_caught(tmp_path: Path, lesson: Lesson, mutant: dict) -> None:
    ws = workspace_with_solutions(tmp_path, lesson)
    target = ws / mutant["file"]
    old, new = mutant["replace"]
    source = target.read_text()
    assert old in source, f"mutant {mutant['name']!r}: {old!r} not found in the solution"
    target.write_text(source.replace(old, new))
    report = run(ws, lesson)
    assert report["outcome"] == "failed", (
        f"lesson {lesson.id}'s tests don't catch the bug {mutant['name']!r}"
    )


def test_every_lesson_has_a_mutant():
    missing = [l.id for l in LESSONS if not l.mutants]
    assert not missing, f"lessons without a planted bug: {missing}"
