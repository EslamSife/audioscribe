from transcribe.audio.phrases import PhraseSplitter

SPEECH = b"S"
QUIET = b"."


def feed_all(splitter, pattern: str):
    """Feed chunks from a pattern like '...SSSS.....' and collect finished phrases."""
    phrases = []
    for ch in pattern:
        chunk = SPEECH if ch == "S" else QUIET
        phrase = splitter.feed(chunk, is_speech=ch == "S")
        if phrase is not None:
            phrases.append(phrase)
    return phrases


def test_silence_alone_makes_no_phrase():
    splitter = PhraseSplitter()
    assert feed_all(splitter, "." * 20) == []
    assert splitter.flush() is None
    assert splitter.now_ms == 2000


def test_a_pause_ends_the_phrase_and_keeps_audio_before_the_speech():
    splitter = PhraseSplitter()
    (phrase,) = feed_all(splitter, "....." + "SSSS" + ".....")
    # 3 chunks of pre-roll + 4 speech + 5 pause, starting 300ms before the first word
    assert phrase.audio == b"..." + b"SSSS" + b"....."
    assert phrase.start_ms == 200
    assert phrase.end_ms == 1400
    assert not splitter.in_speech


def test_a_short_pause_is_not_enough_for_a_short_phrase():
    splitter = PhraseSplitter()
    assert feed_all(splitter, "SSSS...SSSS") == []
    assert splitter.in_speech


def test_a_long_phrase_ends_on_a_short_pause():
    splitter = PhraseSplitter()
    (phrase,) = feed_all(splitter, "S" * 50 + "...")
    assert len(phrase.audio) == 53


def test_hard_cap_commits_without_any_pause():
    splitter = PhraseSplitter()
    phrases = feed_all(splitter, "S" * 150)
    assert [len(p.audio) for p in phrases] == [100]
    assert splitter.chunk_count == 50


def test_cut_finishes_the_sentence_and_keeps_the_tail():
    splitter = PhraseSplitter()
    feed_all(splitter, "SSSSSSSS")
    head = splitter.cut(5)
    assert (head.start_ms, head.end_ms) == (0, 500)
    assert splitter.chunk_count == 3
    rest = splitter.flush()
    assert (rest.start_ms, rest.end_ms) == (500, 800)


def test_cut_always_leaves_the_newest_chunk():
    splitter = PhraseSplitter()
    feed_all(splitter, "SSS")
    head = splitter.cut(99)
    assert len(head.audio) == 2
    assert splitter.chunk_count == 1
    assert splitter.cut(1) is None


def test_new_chunks_counts_audio_since_the_last_preview():
    splitter = PhraseSplitter()
    feed_all(splitter, "SSSS")
    assert splitter.new_chunks == 3  # the first speech chunk starts the phrase
    splitter.new_chunks = 0
    feed_all(splitter, "SS")
    assert splitter.new_chunks == 2
