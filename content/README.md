# Lesson content

Lessons are data: the app reads this folder at runtime, so adding or editing a
lesson needs no rebuild. The sidecar loads it (`python/workbench_sidecar/lessons.py`)
and `python/tests/test_content.py` checks every lesson on each CI run.

## Layout

```
m01-backprop/
  module.yaml              id, number, title, summary
  stubs/<file>.py          starting code, copied into the learner's workspace
  solutions/<file>.py      reference implementation (used by content tests only)
  01-slope/
    lesson.md              front matter + sections
    test_slope.py          hidden tests, run against the learner's workspace
```

Modules build on each other: `m03` imports the tensor engine from `m02`.
The content tests give each lesson a workspace holding the solutions of its
module and every earlier one.

A stub is copied only when the learner's file doesn't exist yet, so several
lessons can share one file (lessons 1.4 to 1.6 all build `m01/engine.py`).
Solution and stub files are matched to a lesson by the file name in its
`file:` field.

## lesson.md

```yaml
---
id: "1.1"                    # module.lesson, unique, in order
title: Which way is downhill?
concepts: [derivative]       # tags for the mistake log and review queue
file: m01/slope.py           # workspace file the learner edits
stub: stubs/slope.py         # optional, relative to the module folder
tests: test_slope.py         # hidden tests, relative to the lesson folder
predict:
  kind: number               # number | choice | text
  prompt: ...
  answer: 3                  # number: the value; choice: option index; text: none
  tolerance: 0.1             # number only, relative (absolute when answer is 0)
  options: [...]             # choice only
mutants:                     # planted bugs the hidden tests must catch
  - name: divides by h instead of 2h
    file: m01/slope.py
    replace: ["(2 * h)", "h"]    # applied to the solution
warmups:                     # optional review problems, number or choice
  - kind: number
    prompt: ...
    answer: 6
    tolerance: 0.02
    explain: ...             # shown after answering
---

## Hook
A concrete problem with numbers. Shown first.

## Build
What to write, with a snippet to try in the kernel.

## Explain
The theory, the proper name and the original paper. Shown only after passing.

## Stretch
Optional harder variant. Shown only after passing.

```

## Review queue

A concept with failed checks in two or more separate runs comes back as a
warm-up at the start of a later session (at most three at a time). The
problem is one of the `warmups` of a lesson that teaches the concept and
that the learner has opened, or failing that the lesson's own prediction,
rotating after a wrong answer. A right answer clears the concept until two
new failed runs. See `python/workbench_sidecar/review.py`.

## Hidden tests

Plain pytest, importing the learner's code (`from m01.slope import slope`).
They run in a fresh Python process with the workspace on `sys.path`. Put the
explanation in the assertion message: it is what the learner sees.

Every lesson needs at least one mutant. CI checks that the solution passes,
the stub fails, and every mutant fails.
