from .model import Segment

MAX_CHARS = 84
MAX_DURATION_MS = 7_000
SENTENCE_ENDINGS = (".", "!", "?")


def group_words(words: list[tuple[str, int, int]]) -> tuple[Segment, ...]:
    """Group (text, start_ms, end_ms) words into subtitle-sized segments."""
    segments: list[Segment] = []
    current: list[tuple[str, int, int]] = []

    def flush() -> None:
        if current:
            segments.append(
                Segment(
                    text=" ".join(w[0] for w in current),
                    start_ms=current[0][1],
                    end_ms=current[-1][2],
                )
            )
            current.clear()

    for word in words:
        text, _, end_ms = word
        if current:
            too_long = _char_count(current) + 1 + len(text) > MAX_CHARS
            too_slow = end_ms - current[0][1] > MAX_DURATION_MS
            if too_long or too_slow:
                flush()
        current.append(word)
        if text.endswith(SENTENCE_ENDINGS):
            flush()
    flush()
    return tuple(segments)


def _char_count(words: list[tuple[str, int, int]]) -> int:
    return sum(len(w[0]) for w in words) + max(0, len(words) - 1)
