# ML Workbench

A macOS app for learning how neural networks and LLMs work by building every
piece yourself: an autograd engine, a language model, a small GPT, then
interpretability, post-training, serving and training at scale. Every concept
is learned by writing the code for it, and every visualization reads the
tensors your own code produced.

The full spec lives in the "Hands-on ML tutorial app: spec" doc.

## Status

Milestone 1 of the build plan: the app shell, the kernel bridge, a
file-backed editor, and Run and Test buttons. Lessons arrive in milestone 2.

## Requirements

- macOS 15 on Apple silicon, Xcode 16 (Swift 6 toolchain)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`). The app finds it on
  `PATH`, in `~/.local/bin`, `~/.cargo/bin`, `/opt/homebrew/bin` or
  `/usr/local/bin`, or wherever `WORKBENCH_UV` points.

## Run it

```sh
swift run MLWorkbench
```

The first launch has uv build the Python environment in `python/.venv`, which
takes a minute. Your workspace is created at `~/ml-workbench` (override with
`WORKBENCH_WORKSPACE`) as a git repo with a `scratch.py` and a test for it.

| Shortcut | Action |
|----------|--------|
| ⌘S | Save |
| ⌘R | Run the open file in the kernel |
| ⌘U | Run the tests for the open file |
| ⌘. | Interrupt |
| ⇧⌘0 | Restart the kernel |

Test picks `tests/test_<name>.py`, `<dir>/tests/test_<name>.py` or
`<dir>/test_<name>.py` for the open file, in that order, and falls back to the
whole `tests/` folder. A test file runs itself. Tests always run in a fresh
Python process, so they never depend on what you ran in the kernel.

You can also open the workspace in any editor; the app reloads the open file
when it comes back to the foreground, as long as you have no unsaved edits.

## Layout

```
Package.swift
Sources/
  WorkbenchCore/    sidecar client, protocol types, workspace files (no UI)
  MLWorkbench/      SwiftUI app
Tests/
  WorkbenchCoreTests/
python/             the sidecar: runs the Jupyter kernel and pytest
  workbench_sidecar/
  tests/
protocol.md         the JSON-lines protocol between the two
```

### Why a Python sidecar

The spec has Swift talking the Jupyter messaging protocol directly. That
protocol runs over ZeroMQ, which would mean a native dependency on the Swift
side. Instead a small Python process owns the kernel through `jupyter_client`
and speaks line-delimited JSON over stdio (see [protocol.md](protocol.md)).
The Swift side stays dependency-free, and the bridge is tested end to end in
Python. The tensor bridge for the widgets will extend this same channel.

## Tests

```sh
cd python && uv run pytest          # sidecar, end to end against a real kernel
swift test                          # Swift core (unit)
WORKBENCH_INTEGRATION=1 swift test  # plus Swift → sidecar → kernel
```
