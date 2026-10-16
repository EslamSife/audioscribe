"""Cut a live stream of short audio chunks into phrases at natural pauses.

Pure logic with no audio devices, models or threads, so it is easy to test:
feed it chunks plus a speech/silence flag, and it hands back finished phrases.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Phrase:
    audio: bytes
    start_ms: int
    end_ms: int


class PhraseSplitter:
    """Group chunks into phrases.

    A phrase starts at the first speech chunk (plus a little audio before it,
    so word onsets aren't clipped) and ends at a pause. Long phrases end on a
    shorter pause, and a hard cap commits even without any pause.
    """

    def __init__(
        self,
        *,
        chunk_ms: int = 100,
        end_silence_chunks: int = 5,  # a 500ms pause always ends the phrase
        soft_commit_chunks: int = 50,  # once the phrase reaches 5s of speech...
        soft_silence_chunks: int = 3,  # ...a 300ms micro-pause is enough to end it
        max_phrase_chunks: int = 100,  # hard cap: commit after 10s even with no pause
        preroll_chunks: int = 3,  # keep 300ms before speech starts
    ) -> None:
        self.chunk_ms = chunk_ms
        self.end_silence_chunks = end_silence_chunks
        self.soft_commit_chunks = soft_commit_chunks
        self.soft_silence_chunks = soft_silence_chunks
        self.max_phrase_chunks = max_phrase_chunks
        self.preroll_chunks = preroll_chunks

        # Audio time at the end of the last chunk fed, i.e. how much audio was heard
        self.now_ms = 0
        # Chunks added to the current phrase since it started or was last previewed
        self.new_chunks = 0
        self._phrase: list[bytes] = []
        self._preroll: list[bytes] = []
        self._start_ms = 0
        self._silence_run = 0
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def chunk_count(self) -> int:
        return len(self._phrase)

    @property
    def audio(self) -> bytes:
        """The current, unfinished phrase."""
        return b"".join(self._phrase)

    def feed(self, chunk: bytes, is_speech: bool) -> Phrase | None:
        """Add one chunk; return a phrase when this chunk finished one."""
        self.now_ms += self.chunk_ms

        if not self._in_speech:
            if not is_speech:
                self._preroll.append(chunk)
                if len(self._preroll) > self.preroll_chunks:
                    self._preroll.pop(0)
                return None
            self._in_speech = True
            self._phrase = [*self._preroll, chunk]
            self._preroll = []
            self._start_ms = max(0, self.now_ms - len(self._phrase) * self.chunk_ms)
            self._silence_run = 0
            self.new_chunks = 0
            return None

        self._phrase.append(chunk)
        self.new_chunks += 1
        self._silence_run = 0 if is_speech else self._silence_run + 1

        if (
            self._silence_run >= self.end_silence_chunks
            or len(self._phrase) >= self.max_phrase_chunks
            or (
                len(self._phrase) >= self.soft_commit_chunks
                and self._silence_run >= self.soft_silence_chunks
            )
        ):
            return self.flush()
        return None

    def cut(self, chunks: int) -> Phrase | None:
        """Finish the first `chunks` chunks as a phrase (a sentence ended there); keep the rest.

        At least the newest chunk always stays, since the speaker is still talking.
        """
        chunks = min(chunks, len(self._phrase) - 1)
        if chunks <= 0:
            return None
        head = Phrase(
            audio=b"".join(self._phrase[:chunks]),
            start_ms=self._start_ms,
            end_ms=self._start_ms + chunks * self.chunk_ms,
        )
        self._phrase = self._phrase[chunks:]
        self._start_ms += chunks * self.chunk_ms
        return head

    def flush(self) -> Phrase | None:
        """Finish whatever is buffered (e.g. when the recording stops)."""
        buf, start_ms = self._phrase, self._start_ms
        self._phrase = []
        self._in_speech = False
        self._silence_run = 0
        self.new_chunks = 0
        if not buf:
            return None
        return Phrase(
            audio=b"".join(buf), start_ms=start_ms, end_ms=start_ms + len(buf) * self.chunk_ms
        )
