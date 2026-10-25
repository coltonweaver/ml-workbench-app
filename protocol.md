# Sidecar protocol (v1)

The app launches the sidecar with

```
uv run --quiet --project python python -m workbench_sidecar --workspace <dir>
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

`execute`, `run_file`, `run_tests` and `restart` are queued and handled one at
a time. Paths that resolve outside the workspace are refused.

## Events (sidecar → app)

| type             | fields |
|------------------|--------|
| `ready`          | `protocol`, `python`, `workspace`. Sent once, after the kernel is up. |
| `stream`         | `name` (`stdout`/`stderr`), `text` |
| `execute_result` | `data` (MIME bundle: `text/plain`, `text/markdown`, `image/png`, `application/json`) |
| `display_data`   | `data`, as above |
| `error`          | `ename`, `evalue`, `traceback` (list of lines, may contain ANSI colour codes) |
| `test_report`    | `outcome` (`passed`, `failed`, `no_tests`, `timeout`, `cancelled`, `error`), `exit_code`, `passed`, `failed`, `tests` (list of `{nodeid, outcome, duration, message}`), `output` (pytest's console text) |
| `done`           | `status` (`ok`, `error`, `aborted`). Always the last event for a request. |

Example:

```
→ {"id":"r1","op":"run_file","path":"scratch.py"}
← {"type":"stream","name":"stdout","text":"w = [1. 2. 3.] sum = 6.0\n","id":"r1"}
← {"type":"done","id":"r1","status":"ok"}
```
