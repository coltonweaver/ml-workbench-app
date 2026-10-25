"""Lesson content and the learner's progress through it.

Content lives in a folder of modules (see ``content/README.md``)::

    content/
      m01-backprop/
        module.yaml               id, number, title, summary
        stubs/engine.py           starting code copied into the workspace
        solutions/engine.py       reference implementation (content tests only)
        01-slope/
          lesson.md               YAML front matter + Markdown sections
          test_slope.py           hidden tests, run against the workspace

Progress lives in the workspace as ``progress.json`` so it travels with the
learner's code and is diffable in git.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

SECTIONS = ("hook", "build", "explain", "stretch")
PREDICTION_KINDS = ("choice", "number", "text")


class ContentError(ValueError):
    pass


@dataclass
class Prediction:
    kind: str
    prompt: str
    options: list[str] | None = None
    answer: Any = None
    tolerance: float = 0.05

    def grade(self, value: str) -> bool | None:
        """True/False for gradable kinds, None for free text."""
        if self.kind == "choice":
            return str(value).strip() == str(self.answer)
        if self.kind == "number":
            try:
                v = float(value)
            except ValueError:
                return False
            a = float(self.answer)
            return abs(v - a) <= (self.tolerance * abs(a) if a else self.tolerance)
        return None

    def display(self, value: Any) -> str:
        if self.kind == "choice" and self.options is not None:
            try:
                return self.options[int(value)]
            except (ValueError, IndexError):
                return str(value)
        return str(value)


@dataclass
class Warmup:
    problem: Prediction
    explain: str | None = None


def _prediction(p: dict[str, Any], path: str, where: str,
                kinds: tuple[str, ...] = PREDICTION_KINDS) -> Prediction:
    if p.get("kind") not in kinds:
        raise ContentError(f"{path}: {where}.kind must be one of {kinds}")
    if "prompt" not in p:
        raise ContentError(f"{path}: {where} needs a prompt")
    if p["kind"] == "choice" and not p.get("options"):
        raise ContentError(f"{path}: choice predictions need options")
    if p["kind"] != "text" and "answer" not in p:
        raise ContentError(f"{path}: {p['kind']} predictions need an answer")
    return Prediction(
        kind=p["kind"],
        prompt=p["prompt"],
        options=p.get("options"),
        answer=p.get("answer"),
        tolerance=float(p.get("tolerance", 0.05)),
    )


@dataclass
class Lesson:
    id: str
    title: str
    module_id: str
    module_title: str
    directory: str
    file: str
    stub: str | None
    solution: str | None
    tests: str
    concepts: list[str]
    predict: Prediction
    sections: dict[str, str]
    mutants: list[dict[str, Any]] = field(default_factory=list)
    # Short review problems for the warm-up queue (see review.py).
    warmups: list[Warmup] = field(default_factory=list)

    @property
    def tests_path(self) -> str:
        return os.path.join(self.directory, self.tests)


@dataclass
class Module:
    id: str
    number: int
    title: str
    summary: str
    directory: str
    lessons: list[Lesson]


def _split_front_matter(text: str, path: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", text, re.S)
    if not match:
        raise ContentError(f"{path}: missing YAML front matter")
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, match.group(2)


def _split_sections(body: str, path: str) -> dict[str, str]:
    """Split on ``## Hook`` / ``## Build`` / ``## Explain`` / ``## Stretch``."""
    sections: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in body.splitlines():
        heading = re.match(r"^## +(\w+)\s*$", line)
        if heading and heading.group(1).lower() in SECTIONS:
            if current:
                sections[current] = "\n".join(lines).strip()
            current, lines = heading.group(1).lower(), []
        elif current:
            lines.append(line)
    if current:
        sections[current] = "\n".join(lines).strip()
    missing = [s for s in ("hook", "build", "explain") if s not in sections]
    if missing:
        raise ContentError(f"{path}: missing sections {missing}")
    return sections


def load_lesson(path: str, module: dict[str, Any], module_dir: str) -> Lesson:
    with open(path) as f:
        meta, body = _split_front_matter(f.read(), path)
    for key in ("id", "title", "file", "tests", "predict"):
        if key not in meta:
            raise ContentError(f"{path}: front matter needs {key!r}")
    predict = _prediction(meta["predict"], path, "predict")
    warmups = [
        Warmup(_prediction(w, path, f"warmups[{i}]", ("choice", "number")), w.get("explain"))
        for i, w in enumerate(meta.get("warmups", []))
    ]
    lesson_dir = os.path.dirname(path)
    file_base = os.path.basename(meta["file"])
    stub = meta.get("stub")
    solution = os.path.join(module_dir, "solutions", file_base)
    lesson = Lesson(
        id=str(meta["id"]),
        title=meta["title"],
        module_id=module["id"],
        module_title=module["title"],
        directory=lesson_dir,
        file=meta["file"],
        stub=os.path.join(module_dir, stub) if stub else None,
        solution=solution if os.path.exists(solution) else None,
        tests=meta["tests"],
        concepts=list(meta.get("concepts", [])),
        predict=predict,
        sections=_split_sections(body, path),
        mutants=list(meta.get("mutants", [])),
        warmups=warmups,
    )
    if not os.path.exists(lesson.tests_path):
        raise ContentError(f"{path}: tests file {lesson.tests} not found")
    if lesson.stub and not os.path.exists(lesson.stub):
        raise ContentError(f"{path}: stub {stub} not found")
    return lesson


class Catalog:
    def __init__(self, content_dir: str) -> None:
        self.content_dir = os.path.abspath(content_dir)
        self.modules: list[Module] = []
        self._by_id: dict[str, Lesson] = {}
        self.reload()

    def reload(self) -> None:
        modules = []
        for entry in sorted(os.listdir(self.content_dir)):
            module_dir = os.path.join(self.content_dir, entry)
            meta_path = os.path.join(module_dir, "module.yaml")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            lessons = []
            for sub in sorted(os.listdir(module_dir)):
                lesson_md = os.path.join(module_dir, sub, "lesson.md")
                if os.path.isfile(lesson_md):
                    lessons.append(load_lesson(lesson_md, meta, module_dir))
            modules.append(Module(
                id=meta["id"], number=int(meta["number"]), title=meta["title"],
                summary=meta.get("summary", ""), directory=module_dir, lessons=lessons,
            ))
        modules.sort(key=lambda m: m.number)
        by_id = {}
        for m in modules:
            for lesson in m.lessons:
                if lesson.id in by_id:
                    raise ContentError(f"duplicate lesson id {lesson.id}")
                by_id[lesson.id] = lesson
        self.modules, self._by_id = modules, by_id

    def lesson(self, lesson_id: str) -> Lesson:
        try:
            return self._by_id[lesson_id]
        except KeyError:
            raise ContentError(f"no lesson {lesson_id!r}") from None


class Progress:
    """``progress.json`` in the workspace."""

    def __init__(self, workspace: str) -> None:
        self.path = os.path.join(workspace, "progress.json")
        self.data: dict[str, Any] = {"lessons": {}, "mistakes": []}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
            self.data.setdefault("lessons", {})
            self.data.setdefault("mistakes", [])

    def lesson(self, lesson_id: str) -> dict[str, Any]:
        return self.data["lessons"].setdefault(
            lesson_id, {"status": "new", "attempts": 0, "prediction": None}
        )

    def status(self, lesson_id: str) -> str:
        return self.data["lessons"].get(lesson_id, {}).get("status", "new")

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, self.path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class LessonService:
    def __init__(self, catalog: Catalog, workspace: str) -> None:
        self.catalog = catalog
        self.workspace = workspace

    def _progress(self) -> Progress:
        # Re-read every time: the learner may edit or reset it by hand.
        return Progress(self.workspace)

    def summary(self) -> list[dict[str, Any]]:
        progress = self._progress()
        return [
            {
                "id": m.id, "number": m.number, "title": m.title, "summary": m.summary,
                "lessons": [
                    {"id": l.id, "title": l.title, "status": progress.status(l.id)}
                    for l in m.lessons
                ],
            }
            for m in self.catalog.modules
        ]

    def open(self, lesson_id: str) -> dict[str, Any]:
        lesson = self.catalog.lesson(lesson_id)
        target = os.path.join(self.workspace, lesson.file)
        if lesson.stub and not os.path.exists(target):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # Every folder on the way becomes a package so `m01.engine` imports.
            pkg = os.path.dirname(target)
            while os.path.abspath(pkg) != os.path.abspath(self.workspace):
                init = os.path.join(pkg, "__init__.py")
                if not os.path.exists(init):
                    open(init, "w").close()
                pkg = os.path.dirname(pkg)
            shutil.copyfile(lesson.stub, target)
        progress = self._progress()
        entry = progress.lesson(lesson_id)
        if entry["status"] == "new":
            entry["status"] = "in_progress"
            entry["opened_at"] = _now()
            progress.save()
        return self.detail(lesson_id)

    def detail(self, lesson_id: str) -> dict[str, Any]:
        lesson = self.catalog.lesson(lesson_id)
        entry = self._progress().lesson(lesson_id)
        passed = entry["status"] == "passed"
        prediction = entry.get("prediction")
        feedback = None
        if passed and prediction is not None:
            feedback = {
                "correct": lesson.predict.grade(prediction["value"]),
                "yours": lesson.predict.display(prediction["value"]),
                "answer": (lesson.predict.display(lesson.predict.answer)
                           if lesson.predict.kind != "text" else None),
            }
        return {
            "id": lesson.id,
            "title": lesson.title,
            "module_id": lesson.module_id,
            "module_title": lesson.module_title,
            "file": lesson.file,
            "concepts": lesson.concepts,
            "hook": lesson.sections["hook"],
            "build": lesson.sections["build"],
            # Revealed only after passing: problem first, name last.
            "explain": lesson.sections["explain"] if passed else None,
            "stretch": lesson.sections.get("stretch") if passed else None,
            "predict": {
                "kind": lesson.predict.kind,
                "prompt": lesson.predict.prompt,
                "options": lesson.predict.options,
            },
            "progress": {
                "status": entry["status"],
                "attempts": entry.get("attempts", 0),
                "prediction": prediction,
                "feedback": feedback,
            },
        }

    def predict(self, lesson_id: str, value: str,
                confidence: float | None = None) -> dict[str, Any]:
        """Record a prediction. ``confidence`` is the learner's stated chance
        of being right, for the calibration score."""
        lesson = self.catalog.lesson(lesson_id)
        value = str(value).strip()
        if not value:
            raise ContentError("prediction is empty")
        if lesson.predict.kind == "number":
            try:
                float(value)
            except ValueError:
                raise ContentError(f"{value!r} isn't a number") from None
        if lesson.predict.kind == "choice":
            if not value.isdigit() or int(value) >= len(lesson.predict.options or []):
                raise ContentError(f"{value!r} isn't one of the options")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                raise ContentError(f"confidence {confidence!r} isn't a number") from None
            if not 0 < confidence <= 1:
                raise ContentError("confidence must be in (0, 1]")
        progress = self._progress()
        entry = progress.lesson(lesson_id)
        if entry.get("prediction") is not None:
            raise ContentError("a prediction is already recorded for this lesson")
        entry["prediction"] = {"value": value, "submitted_at": _now()}
        if confidence is not None and lesson.predict.kind != "text":
            entry["prediction"]["confidence"] = confidence
        progress.save()
        return self.detail(lesson_id)

    def mistakes(self) -> list[dict[str, Any]]:
        return list(self._progress().data.get("mistakes", []))

    def record_check(self, lesson_id: str, report: dict[str, Any]) -> bool:
        """Update progress after a hidden-test run. Returns True on a new pass."""
        lesson = self.catalog.lesson(lesson_id)
        progress = self._progress()
        entry = progress.lesson(lesson_id)
        entry["attempts"] = entry.get("attempts", 0) + 1
        newly_passed = False
        if report["outcome"] == "passed":
            if entry["status"] != "passed":
                newly_passed = True
                entry["status"] = "passed"
                entry["passed_at"] = _now()
        else:
            for test in report["tests"]:
                if test["outcome"] in ("failed", "error"):
                    progress.data["mistakes"].append({
                        "lesson": lesson_id,
                        "concepts": lesson.concepts,
                        "test": test["nodeid"].split("::")[-1],
                        "message": test.get("message"),
                        # Which check run: one run can fail several tests.
                        "attempt": entry["attempts"],
                        "at": _now(),
                    })
        progress.save()
        if newly_passed:
            commit(self.workspace, f"Pass lesson {lesson.id}: {lesson.title}")
        return newly_passed


def commit(workspace: str, message: str) -> bool:
    """Commit everything in the workspace. Best effort: returns False if git
    isn't there or there was nothing to commit."""
    if not os.path.isdir(os.path.join(workspace, ".git")):
        return False
    identity: list[str] = []
    probe = subprocess.run(["git", "config", "user.email"], cwd=workspace,
                           capture_output=True, text=True)
    if not probe.stdout.strip():
        identity = ["-c", "user.name=ML Workbench", "-c", "user.email=workbench@localhost"]
    try:
        subprocess.run(["git", "add", "-A"], cwd=workspace, check=True, capture_output=True)
        result = subprocess.run(["git", *identity, "commit", "-q", "-m", message],
                                cwd=workspace, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return result.returncode == 0
