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
| `submit_prediction` | `lesson`, `value` | Records the prediction (once per lesson). `value` is a string: the number, the option index, or free text. Answers with `lesson`. |
| `check_lesson` | `lesson` | Runs the lesson's hidden tests against the workspace, updates `progress.json`, commits the workspace on a first pass. Answers with `test_report`, `lesson`, `lessons`. |

Everything except `interrupt` and `shutdown` is queued and handled one at a
time. Paths that resolve outside the workspace are refused. Lesson errors (an
unknown lesson, an invalid prediction) come back as an `error` event with
`ename` `LessonError`.

## Events (sidecar → app)

| type             | fields |
|------------------|--------|
| `ready`          | `protocol`, `python`, `workspace`. Sent once, after the kernel is up. |
| `stream`         | `name` (`stdout`/`stderr`), `text` |
| `execute_result` | `data` (MIME bundle: `text/plain`, `text/markdown`, `image/png`, `application/json`) |
| `display_data`   | `data`, as above |
| `error`          | `ename`, `evalue`, `traceback` (list of lines, may contain ANSI colour codes) |
| `test_report`    | `outcome` (`passed`, `failed`, `no_tests`, `timeout`, `cancelled`, `error`), `exit_code`, `passed`, `failed`, `tests` (list of `{nodeid, outcome, duration, message}`), `output` (pytest's console text) |
| `lessons`        | `modules`: list of `{id, number, title, summary, lessons: [{id, title, status}]}`; status is `new`, `in_progress` or `passed` |
| `lesson`         | `lesson`: `{id, title, module_id, module_title, file, concepts, hook, build, explain, stretch, predict: {kind, prompt, options}, progress: {status, attempts, prediction, feedback}}`. `explain`, `stretch` and `feedback` are null until the lesson is passed. |
| `done`           | `status` (`ok`, `error`, `aborted`). Always the last event for a request. |

Example:

```
→ {"id":"r1","op":"run_file","path":"scratch.py"}
← {"type":"stream","name":"stdout","text":"w = [1. 2. 3.] sum = 6.0\n","id":"r1"}
← {"type":"done","id":"r1","status":"ok"}
```
