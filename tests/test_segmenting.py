from transcribe.transcription.segmenting import group_words


def test_empty_input_gives_no_segments():
    assert group_words([]) == ()


def test_sentence_end_flushes_a_segment():
    words = [("Hello", 0, 400), ("world.", 500, 900), ("Next", 1000, 1400)]
    segments = group_words(words)
    assert [s.text for s in segments] == ["Hello world.", "Next"]
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 900


def test_long_text_is_split_before_exceeding_char_limit():
    words = [(f"word{i:02d}", i * 300, i * 300 + 250) for i in range(30)]
    segments = group_words(words)
    assert len(segments) > 1
    assert all(len(s.text) <= 84 for s in segments)


def test_long_duration_is_split():
    words = [("one", 0, 500), ("two", 8000, 8500), ("three", 9000, 9500)]
    segments = group_words(words)
    assert segments[0].text == "one"
    assert segments[1].text == "two three"
