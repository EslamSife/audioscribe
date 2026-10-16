from types import SimpleNamespace

from transcribe.providers.assemblyai import to_transcript


def vendor_result(**overrides):
    base = dict(
        text="Hi there. Hello.",
        utterances=None,
        words=None,
        audio_duration=3,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_utterances_become_speaker_segments():
    result = vendor_result(
        utterances=[
            SimpleNamespace(text="Hi there.", start=0, end=1200, speaker="A"),
            SimpleNamespace(text="Hello.", start=1500, end=2400, speaker="B"),
        ]
    )
    transcript = to_transcript(result)
    assert transcript.text == "Hi there. Hello."
    assert transcript.audio_duration_s == 3
    assert [s.speaker for s in transcript.segments] == ["Speaker A", "Speaker B"]
    assert transcript.segments[0].start_ms == 0
    assert transcript.segments[1].end_ms == 2400


def test_words_are_grouped_when_no_utterances():
    result = vendor_result(
        words=[
            SimpleNamespace(text="Hi", start=0, end=400),
            SimpleNamespace(text="there.", start=500, end=1200),
            SimpleNamespace(text="Hello.", start=1500, end=2400),
        ]
    )
    transcript = to_transcript(result)
    assert [s.text for s in transcript.segments] == ["Hi there.", "Hello."]
    assert all(s.speaker is None for s in transcript.segments)


def test_no_timing_data_gives_empty_segments():
    transcript = to_transcript(vendor_result(text="just text"))
    assert transcript.text == "just text"
    assert transcript.segments == ()


def test_chapters_and_summary_are_mapped():
    result = vendor_result(
        summary="Executive summary bullet points",
        language_code="es",
        confidence=0.98,
        chapters=[
            SimpleNamespace(
                headline="Introduction",
                summary="Discussing project kick-off",
                start=0,
                end=5000,
                gist="Kick-off",
            ),
            SimpleNamespace(
                headline="Next Steps",
                summary="Planning the release",
                start=5000,
                end=10000,
                gist="Release",
            ),
        ],
    )
    transcript = to_transcript(result)
    assert transcript.summary == "Executive summary bullet points"
    assert transcript.language_code == "es"
    assert transcript.confidence == 0.98
    assert len(transcript.chapters) == 2
    assert transcript.chapters[0].headline == "Introduction"
    assert transcript.chapters[0].start_ms == 0
    assert transcript.chapters[0].end_ms == 5000
    assert transcript.chapters[0].gist == "Kick-off"
    assert transcript.chapters[1].headline == "Next Steps"
