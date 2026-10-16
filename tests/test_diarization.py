import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from transcribe.diarization import diarize_transcript
from transcribe.transcription.model import Segment, Transcript

TRANSCRIPT = Transcript(
    text="Hi. Hello.",
    segments=(Segment("Hi.", 0, 1000), Segment("Hello.", 1500, 2500)),
    warnings=("earlier warning",),
)


class FakeAnnotation:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label=False):
        for start, end, label in self.turns:
            yield SimpleNamespace(start=start, end=end), None, label


def fake_pyannote(monkeypatch, *, output=None, error=None):
    """Install a stand-in pyannote.audio whose Pipeline has pyannote 4's signature."""

    class Pipeline:
        @classmethod
        def from_pretrained(cls, checkpoint, revision=None, token=None):
            assert token == "hf-token"
            return cls()

        def __call__(self, path, **kwargs):
            if error:
                raise error
            return output

    audio = ModuleType("pyannote.audio")
    audio.Pipeline = Pipeline
    root = ModuleType("pyannote")
    root.audio = audio
    monkeypatch.setitem(sys.modules, "pyannote", root)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    monkeypatch.setenv("HF_TOKEN", "hf-token")


def test_reads_the_exclusive_annotation_of_pyannote_4(monkeypatch):
    output = SimpleNamespace(
        exclusive_speaker_diarization=FakeAnnotation([(0.0, 1.2, "S0"), (1.4, 2.6, "S1")]),
        speaker_diarization=FakeAnnotation([]),
    )
    fake_pyannote(monkeypatch, output=output)

    labeled = diarize_transcript(Path("a.wav"), TRANSCRIPT)

    assert [s.speaker for s in labeled.segments] == ["Speaker 1", "Speaker 2"]
    assert labeled.warnings == ("earlier warning",)


def test_failed_diarization_keeps_the_transcript_with_a_warning(monkeypatch):
    fake_pyannote(monkeypatch, error=RuntimeError("out of memory"))

    result = diarize_transcript(Path("a.wav"), TRANSCRIPT)

    assert result.segments == TRANSCRIPT.segments
    assert "out of memory" in result.warnings[-1]


def test_installed_pyannote_matches_what_the_code_calls():
    """Contract check against the real library (skipped when the 'speakers' extra is absent)."""
    import dataclasses
    import inspect

    pytest.importorskip("pyannote.audio")
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

    assert "token" in inspect.signature(Pipeline.from_pretrained).parameters
    fields = {f.name for f in dataclasses.fields(DiarizeOutput)}
    assert "exclusive_speaker_diarization" in fields
