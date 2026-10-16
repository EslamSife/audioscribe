# Contributing

Thanks for helping. Small, focused pull requests are easiest to review.

## Setup

```sh
uv sync --all-extras --dev
```

`--all-extras` includes the `speakers` extra (PyTorch, large). Without it, the pyannote
contract tests are skipped, the same way they are skipped in CI.

## Before you open a pull request

```sh
uv run ruff check .
uv run ruff format .
uv run pytest
```

CI runs the same checks on Linux and macOS, with Python 3.11 to 3.14.

## Tests that touch vendor libraries

AssemblyAI, Anthropic, pyannote and faster-whisper change their APIs from time to time. When a
test replaces one of their classes with a fake, build the fake with `unittest.mock.create_autospec`
(or check `inspect.signature` against the real library). A plain `MagicMock` accepts any call, so
it hides the exact breakages these tests exist to catch.

## Adding a transcription or notes provider

Write one adapter file in `src/transcribe/providers/` (or `src/transcribe/notes/`) that matches
the protocol in `base.py`, register it in `factory.py`, and keep the vendor's types inside that
file.
