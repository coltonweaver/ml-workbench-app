# Sidecar protocol (v1)

The app launches the sidecar with

```
uv run --quiet --project python python -m workbench_sidecar --workspace <dir> --content <content-dir>
```

and exchanges one JSON object per line over its stdin and stdout. Anything else
the sidecar prints goes to stderr, which the app shows if the process dies.

The sidecar owns one IPython kernel (via `jupyter_client`) whose working
directory is the workspace, with the workspace root on `sys.path` so learner
packages such as `m01.engine` import directly.

## Requests (app → sidecar)

Every request has an `id` chosen by the app; every event about that request
echoes it.

| op          | fields   | what it does |
|-------------|----------|--------------|
| `execute`   | `code`   | Runs code in the kernel. State persists between requests. |
| `run_file`  | `path`   | `%run`s a workspace-relative file in the kernel; its globals stay in the kernel afterwards. |
| `run_tests` | `target` | Runs pytest on a workspace-relative file or folder **in a fresh interpreter**, not the kernel. |
| `restart`   |          | Restarts the kernel, clearing its state. |
| `interrupt` |          | Interrupts the running test process if there is one, else the kernel. Handled immediately. |
| `shutdown`  |          | Stops the kernel and exits. |
| `list_lessons` |       | Re-reads the content folder; answers with `lessons`. |
| `open_lesson` | `lesson` | Copies the lesson's stub into the workspace if its file doesn't exist, marks it in progress; answers with `lesson`. |
| `submit_prediction` | `lesson`, `value`, `confidence` | Records the prediction (once per lesson). `value` is a string: the number, the option index, or free text. `confidence` (optional, in (0, 1]) is the learner's stated chance of being right, used for the prediction score. Answers with `lesson`. |
| `get_review` | | Answers with `review`. |
| `answer_warmup` | `warmup`, `value` | Grades a warm-up from the review queue and records it in `progress.json`. Answers with `review`, whose `result` is set. |
| `ask_tutor` | `question`, `history`, `api_key`, `lesson`, `code`, `tensor` | Asks the tutor. `history` is the chat so far as `[{role, text}]`; `code` is the editor's text for the open lesson's file (read from disk when absent); `tensor` is a summary from an instrument's `explain` list. `api_key` may be null, then `ANTHROPIC_API_KEY` is used. Answers with `tutor_delta` events and `tutor_done`. |
| `cancel_tutor` | | Stops the answer that's streaming. Handled immediately. |
| `complete` | `code`, `path`, `line`, `column` | Editor completions at a position in `code` (the editor's unsaved text; `path` is the workspace-relative file it belongs to, for resolving relative imports). `line` is 1-based, `column` 0-based in code points. Static analysis with Jedi, never the kernel, so it answers while code runs. Answers with `completions`. A request still waiting when a newer one arrives is answered empty with `done` status `aborted`. |
| `check_lesson` | `lesson` | Runs the lesson's hidden tests against the workspace, updates `progress.json`, commits the workspace on a first pass. Answers with `test_report`, `lesson`, `lessons`, `review`. |

Everything except `interrupt`, `shutdown`, `ask_tutor`, `cancel_tutor` and
`complete` is queued and handled one at a time. Tutor questions have their own queue, so
they're answered while code runs. The sidecar remembers the last `error` or
failing `test_report` it sent and gives it to the tutor; a passing report
clears it. Tutor problems (no key, a rejected key, no network) come back as
an `error` event with `ename` `TutorError`. Paths that resolve outside the workspace are refused. Lesson errors (an
unknown lesson, an invalid prediction) come back as an `error` event with
`ename` `LessonError`.

## Instruments

The `workbench` package (in `python/workbench`) is importable in the kernel.
`show_graph`, `show_landscape`, `show_slice`, `show_training`,
`show_embeddings`, `show_attention` and `show_matrix` publish IPython displays
with the MIME types `application/vnd.workbench.<kind>+json`, where kind is
`graph`, `landscape`, `training`, `embedding`, `attention` or `matrix`. An
attention viewer made from a function carries a `callback` id; to re-run it
on a new prompt the app sends `execute` with
`import workbench as _wb; _wb.rerun("<callback>", "<prompt>")`, and the
viewer updates in place. The sidecar forwards those keys
untouched; the app draws them in the Instruments tab and applies
`update_display_data` to the widget with the same `display_id`. Non-finite
numbers are sent as null.

Every payload also has an `explain` list of tensor summaries, made by
`workbench.tensor_summary`: `{name, shape, count, non_finite, stats: {min,
max, mean, std, zeros}, values (16 or fewer entries only), provenance: {file,
line, function, code}}` plus instrument-specific notes. "Explain this tensor"
sends one of them as `ask_tutor`'s `tensor`.

## Events (sidecar → app)

| type             | fields |
|------------------|--------|
| `ready`          | `protocol`, `python`, `workspace`. Sent once, after the kernel is up. |
| `stream`         | `name` (`stdout`/`stderr`), `text` |
| `execute_result` | `data` (MIME bundle: `text/plain`, `text/markdown`, `image/png`, `application/json`) |
| `display_data`   | `data`, as above; `display_id` when the display can be updated |
| `update_display_data` | `data` and `display_id`: replace that display's content in place |
| `error`          | `ename`, `evalue`, `traceback` (list of lines, may contain ANSI colour codes) |
| `test_report`    | `outcome` (`passed`, `failed`, `no_tests`, `timeout`, `cancelled`, `error`), `exit_code`, `passed`, `failed`, `tests` (list of `{nodeid, outcome, duration, message}`), `output` (pytest's console text) |
| `lessons`        | `modules`: list of `{id, number, title, summary, lessons: [{id, title, status}]}`; status is `new`, `in_progress` or `passed` |
| `lesson`         | `lesson`: `{id, title, module_id, module_title, file, concepts, hook, build, explain, stretch, predict: {kind, prompt, options}, progress: {status, attempts, prediction, feedback}}`. `explain`, `stretch` and `feedback` are null until the lesson is passed. |
| `review`         | `review`: `{warmups, score, result}`. `warmups`: up to 3 of `{id, concept, lesson, lesson_title, context, kind, prompt, options, mistakes, last_message}` for concepts with failed checks in two or more runs before this session. `score`: `{count, correct, hit_rate, rated, brier, overconfidence, buckets: [{confidence, count, hit_rate}], history: [{lesson, correct, confidence, hit_rate}]}` over passed lessons' graded predictions. `result`: after `answer_warmup`, `{id, concept, correct, yours, answer, explain, lesson, lesson_title}`, else null. |
| `tutor_delta`    | `text`: the next piece of the tutor's answer |
| `tutor_done`     | `stop_reason` (`end_turn`, `max_tokens`, `refusal`, `cancelled`), `model`, `note` (why it stopped, for refusals) |
| `completions`    | `completions`: list of `{name, complete, type, signature, doc}`, where `complete` is the text to insert after what's typed and `type` is Jedi's kind (`function`, `class`, `module`, `instance`, `keyword`, `statement`, `param`, `property`, `path`); `signature` and `doc` are filled for the first dozen only. `signatures`: the call the cursor is inside, as `[{name, params, index, doc}]` (at most one). |
| `done`           | `status` (`ok`, `error`, `aborted`). Always the last event for a request. |

Example:

```
→ {"id":"r1","op":"run_file","path":"scratch.py"}
← {"type":"stream","name":"stdout","text":"w = [1. 2. 3.] sum = 6.0\n","id":"r1"}
← {"type":"done","id":"r1","status":"ok"}
```
