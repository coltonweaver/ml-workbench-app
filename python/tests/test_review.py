"""The review queue (warm-ups for concepts with repeated mistakes) and the
prediction score (calibration)."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from workbench_sidecar.lessons import Catalog, ContentError, Progress
from workbench_sidecar.review import ReviewService, calibration, now

CONTENT = Path(__file__).resolve().parents[2] / "content"
CATALOG = Catalog(str(CONTENT))


def stamp(dt) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_progress(ws: Path, lessons: dict, mistakes: list, reviews: dict | None = None) -> None:
    data = {"lessons": lessons, "mistakes": mistakes}
    if reviews is not None:
        data["reviews"] = reviews
    (ws / "progress.json").write_text(json.dumps(data))


def mistake(lesson: str, attempt: int, at, test: str = "test_x") -> dict:
    return {"lesson": lesson, "concepts": CATALOG.lesson(lesson).concepts, "test": test,
            "message": f"{test} failed", "attempt": attempt, "at": stamp(at)}


def opened(*ids: str, status: str = "in_progress") -> dict:
    return {i: {"status": status, "attempts": 3, "prediction": None} for i in ids}


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    return tmp_path


def test_one_run_with_many_failures_is_not_repeated(ws: Path) -> None:
    yesterday = now() - timedelta(days=1)
    write_progress(ws, opened("1.1"), [
        mistake("1.1", 1, yesterday, "test_a"),
        mistake("1.1", 1, yesterday, "test_b"),
        mistake("1.1", 1, yesterday, "test_c"),
    ])
    assert ReviewService(CATALOG, str(ws)).queue() == []


def test_repeated_mistakes_come_back_next_session(ws: Path) -> None:
    yesterday = now() - timedelta(days=1)
    write_progress(ws, opened("1.1"), [mistake("1.1", 1, yesterday), mistake("1.1", 2, yesterday)])
    cards = ReviewService(CATALOG, str(ws)).queue()
    assert {c["concept"] for c in cards} == {"derivative", "finite-differences"}
    card = cards[0]
    assert card["lesson"] == "1.1" and card["mistakes"] == 2
    assert card["kind"] in ("number", "choice") and card["prompt"]
    assert card["last_message"] == "test_x failed"


def test_mistakes_this_session_wait_for_the_next(ws: Path) -> None:
    started = now() - timedelta(minutes=5)
    later = now()
    write_progress(ws, opened("1.1"), [mistake("1.1", 1, later), mistake("1.1", 2, later)])
    assert ReviewService(CATALOG, str(ws), session_started=started).queue() == []
    assert ReviewService(CATALOG, str(ws), session_started=now() + timedelta(seconds=2)).queue()


def test_queue_is_capped_and_most_missed_first(ws: Path) -> None:
    t = now() - timedelta(days=2)
    lessons = opened("1.1", "1.2", "1.3", "1.4")
    mistakes = [mistake(l, a, t) for l in ("1.2", "1.3", "1.4") for a in (1, 2)]
    mistakes += [mistake("1.1", a, t) for a in (1, 2, 3, 4)]
    write_progress(ws, lessons, mistakes)
    cards = ReviewService(CATALOG, str(ws)).queue()
    assert len(cards) == 3
    assert cards[0]["mistakes"] == 4 and cards[0]["lesson"] == "1.1"


def test_unopened_lessons_are_never_used(ws: Path) -> None:
    # "derivative" is also taught in later lessons the learner hasn't opened.
    t = now() - timedelta(days=1)
    write_progress(ws, opened("1.1"), [mistake("1.1", 1, t), mistake("1.1", 2, t)])
    service = ReviewService(CATALOG, str(ws))
    problems = service._problems("derivative", Progress(str(ws)))
    assert problems and {lesson.id for _, lesson, _, _ in problems} == {"1.1"}


def test_right_answer_clears_until_new_mistakes(ws: Path) -> None:
    t = now() - timedelta(days=1)
    write_progress(ws, opened("1.1"), [mistake("1.1", 1, t), mistake("1.1", 2, t)])
    service = ReviewService(CATALOG, str(ws))
    card = next(c for c in service.queue() if c["concept"] == "derivative")
    _, lesson, problem, _ = service._find(card["id"], Progress(str(ws)))
    result = service.answer(card["id"], str(problem.answer))
    assert result["correct"] and result["concept"] == "derivative"
    assert "derivative" not in {c["concept"] for c in service.queue()}
    # Two new failed runs bring it back (in a later session).
    data = json.loads((ws / "progress.json").read_text())
    later = now() + timedelta(seconds=2)
    data["mistakes"] += [mistake("1.1", 3, later), mistake("1.1", 4, later)]
    (ws / "progress.json").write_text(json.dumps(data))
    again = ReviewService(CATALOG, str(ws), session_started=later + timedelta(seconds=1))
    assert "derivative" in {c["concept"] for c in again.queue()}


def test_wrong_answer_keeps_it_and_rotates_the_problem(ws: Path) -> None:
    t = now() - timedelta(days=1)
    write_progress(ws, opened("1.1", "1.3", "1.6"),
                   [mistake("1.1", 1, t), mistake("1.1", 2, t)])
    service = ReviewService(CATALOG, str(ws))
    problems = service._problems("derivative", Progress(str(ws)))
    assert len(problems) >= 2, "lessons 1.1, 1.3 and 1.6 all teach derivatives"
    first = next(c for c in service.queue() if c["concept"] == "derivative")
    result = service.answer(first["id"], "123456")
    assert not result["correct"] and result["answer"]
    second = next(c for c in service.queue() if c["concept"] == "derivative")
    assert second["id"] != first["id"]
    reviews = json.loads((ws / "progress.json").read_text())["reviews"]
    assert [h["correct"] for h in reviews["derivative"]["history"]] == [False]


def test_answer_validation(ws: Path) -> None:
    write_progress(ws, opened("1.1"), [])
    service = ReviewService(CATALOG, str(ws))
    with pytest.raises(ContentError, match="no warm-up"):
        service.answer("derivative|9.9|p", "1")
    with pytest.raises(ContentError, match="empty"):
        service.answer("derivative|1.1|p", " ")


def test_every_concept_has_a_warmup() -> None:
    concepts = {c for m in CATALOG.modules for l in m.lessons for c in l.concepts}
    gradable = {c for m in CATALOG.modules for l in m.lessons for c in l.concepts
                if l.warmups or l.predict.kind != "text"}
    assert concepts == gradable


def test_authored_warmups_are_answerable() -> None:
    authored = [(l, w) for m in CATALOG.modules for l in m.lessons for w in l.warmups]
    assert len(authored) >= 16
    for lesson, w in authored:
        assert w.problem.grade(str(w.problem.answer)), f"{lesson.id}: its own answer grades wrong"
        if w.problem.kind == "choice":
            assert 0 <= int(w.problem.answer) < len(w.problem.options)
        assert w.explain, f"{lesson.id}: warm-ups explain the answer"


# Prediction score


def test_score_counts_only_passed_graded_predictions(ws: Path) -> None:
    t = stamp(now())
    lessons = {
        "1.1": {"status": "passed", "prediction": {"value": "3", "submitted_at": t, "confidence": 0.9}},
        "1.2": {"status": "passed", "prediction": {"value": "0", "submitted_at": t, "confidence": 0.9}},
        "1.3": {"status": "in_progress", "prediction": {"value": "1", "submitted_at": t}},
    }
    write_progress(ws, lessons, [])
    score = ReviewService(CATALOG, str(ws)).score()
    right = CATALOG.lesson("1.2").predict.grade("0")
    assert score["count"] == 2, "1.3 isn't passed, so its answer isn't revealed yet"
    assert score["correct"] == 1 + right
    assert score["rated"] == 2


def test_calibration_math() -> None:
    graded = [
        {"lesson": "a", "correct": True, "confidence": 0.9},
        {"lesson": "b", "correct": False, "confidence": 0.9},
        {"lesson": "c", "correct": True, "confidence": 0.5},
        {"lesson": "d", "correct": True, "confidence": None},
    ]
    s = calibration(graded)
    assert s["count"] == 4 and s["correct"] == 3 and s["hit_rate"] == 0.75
    assert s["rated"] == 3
    assert s["brier"] == pytest.approx((0.01 + 0.81 + 0.25) / 3)
    assert s["overconfidence"] == pytest.approx((0.9 + 0.9 + 0.5) / 3 - 2 / 3)
    assert s["buckets"] == [
        {"confidence": 0.5, "count": 1, "hit_rate": 1.0},
        {"confidence": 0.9, "count": 2, "hit_rate": 0.5},
    ]
    assert [h["hit_rate"] for h in s["history"]] == [1.0, 0.5, pytest.approx(2 / 3), 0.75]


def test_empty_score() -> None:
    s = calibration([])
    assert s["count"] == 0 and s["hit_rate"] is None and s["brier"] is None and s["buckets"] == []


def test_prediction_confidence_is_stored(tmp_path: Path) -> None:
    from workbench_sidecar.lessons import LessonService
    service = LessonService(CATALOG, str(tmp_path))
    service.open("1.1")
    with pytest.raises(ContentError, match="confidence"):
        service.predict("1.1", "3", confidence=1.5)
    detail = service.predict("1.1", "3", confidence=0.7)
    assert detail["progress"]["prediction"]["confidence"] == 0.7
