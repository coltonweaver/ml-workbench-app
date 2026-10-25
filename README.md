# ML Workbench

A macOS app for learning how neural networks and LLMs work by building every
piece yourself: an autograd engine, a language model, a small GPT, then
interpretability, post-training, serving and training at scale. Every concept
is learned by writing the code for it, and every visualization reads the
tensors your own code produced.

The full spec lives in the "Hands-on ML tutorial app: spec" doc.

## Status

All eight modules of the build plan are in: the app shell, kernel bridge,
file-backed editor and Run/Test buttons, the lesson flow, and module 1
(gradient descent through a working autograd engine and MLP) onward.

Milestone 3 adds the first instruments, which draw what your own code
computed: a computation graph you can step through (values forward, gradients
backward) and 1D/2D loss landscapes with the optimizer's path drawn live and a
draggable probe. From the kernel:

```python
from workbench import show_graph, show_landscape, show_slice
```

Milestone 4 adds the tutor (⇧⌘T), a side panel on the Claude API. Each
question goes with the open lesson, your file as it is in the editor, your
last error or test failure, your prediction and your mistake history, and the
tutor is told to coach: ask back, give the smallest hint that unblocks, and
only write the full solution once you've asked for it twice. "Explain" on any
instrument sends that tensor's shape, statistics and the line of your code
that produced it. Add your API key in Settings (⌘,); it's kept in your login
Keychain. The tutor uses the latest Opus model with server-side fallbacks
turned on, so a request its safety classifiers decline is re-run on
Anthropic's recommended fallback model instead of failing.

Module 2 (language modeling basics) builds a tensor autograd engine on
NumPy, bigram and MLP language models on 32,000 names, and comes with the
training dashboard (loss and validation curves, gradient norms per layer,
learning rate, tokens/sec), the embedding explorer (PCA of any embedding
table, colored by token class) and a heatmap. The attention viewer is ready
for module 3. Bundled data: `from workbench.data import names, shakespeare`.

Module 3 builds self-attention, multi-head attention, layer norm, the MLP
block and the residual stream on that engine, assembles a GPT, trains it on
tiny Shakespeare (or any text file: `workbench.data.corpus(path)`), then
rebuilds it in PyTorch and checks the two agree to about 1e-15 before
training a bigger one on the Apple GPU (MPS).

```python
from workbench import show_training, show_embeddings, show_attention, show_matrix, grad_norms
```

Module 4 writes convolution by hand (checked against PyTorch), Sobel edges
and pooling, trains a small CNN on a generated shapes dataset
(`workbench.data.shapes`), then looks inside it: feature visualization by
gradient ascent on the input, top-activating dataset examples, and reading
and ablating a circuit between two layers. `show_images` draws image grids
(grayscale or RGB) for all of it.

Module 5 opens up your GPT: a forward pass with named hook points to read
and patch, the logit lens and direct logit attribution, activation
patching, induction heads (trained and found), toy models of
superposition, a sparse autoencoder on your own residual stream, steering
vectors, attribution graphs, and a capstone that critiques an
interpretability claim with experiments. Two instruments come with it: a
feature browser (pick a feature or neuron; its top contexts are shaded by
activation, with the tokens it promotes and suppresses) and the attribution
view (component × token heatmap plus the pruned attribution graph).

```python
from workbench import feature_card, show_features, show_attribution
```

Module 6 is post-training and scaling: a prompt/response format with a loss
mask, supervised fine-tuning (a small GPT learns to reverse names it has
never seen), the DPO loss and DPO training against a frozen reference, then
size sweeps of your own GPTs fit to a power law with an irreducible floor,
and Chinchilla's compute-optimal allocation and isoflop method. `show_plot`
draws lines and points with log axes; `show_timeline` draws lanes of spans
and is used by modules 7 and 8.

Module 7 is serving: why decoding is memory-bound (a roofline model), a KV
cache on your GPT, int8 and grouped int4 quantization, static versus
continuous batching in a discrete-event simulation, a paged KV block
allocator, speculative decoding (exact rejection sampling and a greedy
version with a real draft model), and a serving simulator that combines
continuous batching, paged memory and preemption, drawn with
`show_timeline` (request lanes, KV memory and batch size over time).

Module 8 is training at scale, run for real across processes on your Mac:
naive and ring all-reduce, data parallelism that matches a single-process
run to the last bit, Megatron-style tensor parallelism of the MLP, GPipe
schedules and the pipeline bubble, fp16 underflow and loss scaling, and
checkpointing, crashes, stragglers and backup workers. `workbench.dist.launch(fn, world_size)`
starts the ranks (gloo on localhost) and returns what each one returned.

Lessons live in [`content/`](content/README.md) as Markdown with YAML front
matter. Each lesson goes problem, prediction, build, hidden-test check, then
the explanation. Progress is kept in `progress.json` in your workspace, and
the workspace is committed each time you pass a lesson.

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
| ⌘K | Check the open lesson against its hidden tests |
| ⌘. | Interrupt |
| ⇧⌘T | Show or hide the tutor |
| ⌘, | Settings (tutor API key) |
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
content/            lessons, stubs, reference solutions, hidden tests
python/             the sidecar: runs the Jupyter kernel, pytest and the lesson flow
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

CI runs the Python tests on Linux and builds and tests the app on a macOS
runner.
