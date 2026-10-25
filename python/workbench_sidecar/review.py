"""The review queue and the prediction score, both computed from
``progress.json``.

Review queue: a concept the learner has failed checks on at least twice
(two separate check runs, not two tests in one run) comes back as a short
warm-up problem at the start of a later session. Mistakes made in the
current session wait for the next one. Answering the warm-up right clears
the concept until two new mistakes; answering it wrong keeps it queued and
the next warm-up for it is a different problem where there is one.

Problems come from the lessons that teach the concept: the authored
``warmups`` in a lesson's front matter first, then the lesson's own
prediction question. Only lessons the learner has opened are used, so a
warm-up never gives away a lesson they haven't reached.

Prediction score: every graded prediction whose lesson is passed (the
answer is only revealed then). With the stated confidence, it's scored for
calibration: Brier score, and hit rate per confidence level against the
confidence itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .lessons import Catalog, ContentError, Lesson, Prediction, Progress, _now

# Repeated mistakes: this many separate failed check runs.
REPEATED = 2
# At most this many warm-ups at the start of a session.
MAX_WARMUPS = 3
# The confidence levels the app offers when locking in a prediction.
CONFIDENCE_LEVELS = (0.5, 0.7, 0.9)


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None


def now() -> datetime:
    return datetime.now().astimezone()


def _attempt_key(mistake: dict[str, Any]) -> tuple[str, Any]:
    # Older progress files have no attempt number; the timestamp is the run.
    return (mistake.get("lesson", ""), mistake.get("attempt", mistake.get("at")))


def concept_mistakes(progress: Progress, before: datetime | None) -> dict[str, dict[str, Any]]:
    """Per concept: failed check runs since the concept was last cleared by
    a right warm-up, counting only runs before ``before``."""
    reviews = progress.data.get("reviews", {})
    runs: dict[str, dict[tuple[str, Any], dict[str, Any]]] = {}
    for m in progress.data.get("mistakes", []):
        at = parse_time(m.get("at"))
        if at is None or (before is not None and at >= before):
            continue
        for concept in m.get("concepts", []):
            cleared = parse_time(reviews.get(concept, {}).get("cleared_at"))
            if cleared is not None and at <= cleared:
                continue
            runs.setdefault(concept, {}).setdefault(_attempt_key(m), m)
    out = {}
    for concept, by_run in runs.items():
        latest = max(by_run.values(), key=lambda m: parse_time(m["at"]))
        out[concept] = {
            "runs": len(by_run),
            "last_at": latest["at"],
            "last_message": latest.get("message"),
            "lessons": sorted({m["lesson"] for m in by_run.values()}),
        }
    return out


class ReviewService:
    def __init__(self, catalog: Catalog, workspace: str,
                 session_started: datetime | None = None) -> None:
        self.catalog = catalog
        self.workspace = workspace
        self.session_started = session_started or now()

    def _progress(self) -> Progress:
        return Progress(self.workspace)

    # Problems

    def _problems(self, concept: str, progress: Progress) -> list[tuple[str, Lesson, Prediction, str | None]]:
        """(id, lesson, problem, explain) for every warm-up that can review
        ``concept``, in curriculum order."""
        out = []
        for module in self.catalog.modules:
            for lesson in module.lessons:
                if concept not in lesson.concepts or progress.status(lesson.id) == "new":
                    continue
                for i, w in enumerate(lesson.warmups):
                    out.append((f"{concept}|{lesson.id}|{i}", lesson, w.problem, w.explain))
                if lesson.predict.kind != "text":
                    out.append((f"{concept}|{lesson.id}|p", lesson, lesson.predict, None))
        return out

    def _find(self, warmup_id: str, progress: Progress) -> tuple[str, Lesson, Prediction, str | None]:
        concept = warmup_id.split("|", 1)[0]
        for problem in self._problems(concept, progress):
            if problem[0] == warmup_id:
                return problem
        raise ContentError(f"no warm-up {warmup_id!r}")

    def _card(self, problem: tuple[str, Lesson, Prediction, str | None],
              info: dict[str, Any]) -> dict[str, Any]:
        wid, lesson, p, _ = problem
        concept = wid.split("|", 1)[0]
        return {
            "id": wid,
            "concept": concept,
            "lesson": lesson.id,
            "lesson_title": lesson.title,
            # A lesson's own prediction needs its hook for context.
            "context": lesson.sections["hook"] if wid.endswith("|p") else None,
            "kind": p.kind,
            "prompt": p.prompt,
            "options": p.options,
            "mistakes": info["runs"],
            "last_message": info["last_message"],
        }

    # Queue

    def queue(self, progress: Progress | None = None) -> list[dict[str, Any]]:
        progress = progress or self._progress()
        due = [
            (concept, info) for concept, info in
            concept_mistakes(progress, before=self.session_started).items()
            if info["runs"] >= REPEATED
        ]
        # Most-missed first, then most recent.
        due.sort(key=lambda ci: (-ci[1]["runs"], -parse_time(ci[1]["last_at"]).timestamp()))
        reviews = progress.data.get("reviews", {})
        cards = []
        for concept, info in due:
            problems = self._problems(concept, progress)
            if not problems:
                continue
            history = reviews.get(concept, {}).get("history", [])
            # Rotate so a wrong answer is followed by a different problem.
            cards.append(self._card(problems[len(history) % len(problems)], info))
            if len(cards) == MAX_WARMUPS:
                break
        return cards

    def answer(self, warmup_id: str, value: str) -> dict[str, Any]:
        progress = self._progress()
        wid, lesson, problem, explain = self._find(str(warmup_id), progress)
        value = str(value).strip()
        if not value:
            raise ContentError("answer is empty")
        correct = bool(problem.grade(value))
        concept = wid.split("|", 1)[0]
        review = progress.data.setdefault("reviews", {}).setdefault(concept, {"history": []})
        at = _now()
        review["history"].append({"warmup": wid, "value": value, "correct": correct, "at": at})
        if correct:
            review["cleared_at"] = at
        progress.save()
        return {
            "id": wid,
            "concept": concept,
            "correct": correct,
            "yours": problem.display(value),
            "answer": problem.display(problem.answer),
            "explain": explain,
            "lesson": lesson.id,
            "lesson_title": lesson.title,
        }

    # Prediction score

    def score(self, progress: Progress | None = None) -> dict[str, Any]:
        progress = progress or self._progress()
        graded = []
        for module in self.catalog.modules:
            for lesson in module.lessons:
                entry = progress.data["lessons"].get(lesson.id)
                if not entry or entry.get("status") != "passed":
                    continue
                prediction = entry.get("prediction")
                if not prediction or lesson.predict.kind == "text":
                    continue
                graded.append({
                    "lesson": lesson.id,
                    "correct": bool(lesson.predict.grade(prediction["value"])),
                    "confidence": prediction.get("confidence"),
                    "at": prediction.get("submitted_at"),
                })
        graded.sort(key=lambda g: parse_time(g["at"]) or datetime.min.astimezone())
        return calibration(graded)


def calibration(graded: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a chronological list of ``{lesson, correct, confidence, at}``."""
    n = len(graded)
    right = sum(g["correct"] for g in graded)
    rated = [g for g in graded if g.get("confidence") is not None]
    buckets = []
    for level in CONFIDENCE_LEVELS:
        # Nearest offered level, so a hand-edited 0.75 still lands somewhere.
        members = [g for g in rated
                   if min(CONFIDENCE_LEVELS, key=lambda c: abs(c - g["confidence"])) == level]
        if members:
            buckets.append({
                "confidence": level,
                "count": len(members),
                "hit_rate": sum(g["correct"] for g in members) / len(members),
            })
    brier = (sum((g["confidence"] - g["correct"]) ** 2 for g in rated) / len(rated)
             if rated else None)
    # Positive: more sure than right.
    overconfidence = (sum(g["confidence"] for g in rated) / len(rated)
                      - sum(g["correct"] for g in rated) / len(rated)) if rated else None
    running, hits = [], 0
    for i, g in enumerate(graded, 1):
        hits += g["correct"]
        running.append({"lesson": g["lesson"], "correct": g["correct"],
                        "confidence": g.get("confidence"), "hit_rate": hits / i})
    return {
        "count": n,
        "correct": right,
        "hit_rate": right / n if n else None,
        "rated": len(rated),
        "brier": brier,
        "overconfidence": overconfidence,
        "buckets": buckets,
        "history": running,
    }
