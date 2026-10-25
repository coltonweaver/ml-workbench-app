"""Sidecar process that sits between the Swift app and the learner's Python.

The app launches ``python -m workbench_sidecar --workspace <dir>`` and talks to
it with one JSON object per line on stdin/stdout. See ``protocol.md`` at the repo
root for the message shapes.
"""

PROTOCOL_VERSION = 1
