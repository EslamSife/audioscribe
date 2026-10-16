import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

from rich.console import Console
from rich.table import Table

from transcribe import config
from transcribe.audio.phrases import Phrase, PhraseSplitter
from transcribe.transcription.model import (
    Segment,
    Transcript,
    TranscriptionError,
    TranscriptionOptions,
)

console = Console(stderr=True)

CHUNK_MS = 100

# Called with every finished line while recording, e.g. to print and save it
OnTurn = Callable[[Segment], None]


def list_audio_devices() -> list[dict[str, Any]]:
    """Enumerate all available audio input devices (microphones & loopbacks)."""
    try:
        import sounddevice as sd
    except ImportError:
        return []

    devices = sd.query_devices()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else 0
    return [
        {
            "id": idx,
            "name": dev.get("name", f"Device {idx}"),
            "channels": dev.get("max_input_channels", 1),
            "default_samplerate": int(dev.get("default_samplerate", 16000)),
            "is_default": idx == default_in,
        }
        for idx, dev in enumerate(devices)
        if dev.get("max_input_channels", 0) > 0
    ]


def resolve_device_id(device_query: str | int | None) -> int | None:
    """Resolve device ID from string query (index or name substring) or integer."""
    if device_query is None:
        return None

    if isinstance(device_query, int):
        return device_query

    query_str = str(device_query).strip()
    if query_str.isdigit():
        return int(query_str)

    devices = list_audio_devices()
    for dev in devices:
        if query_str.lower() in dev["name"].lower():
            return dev["id"]

    raise TranscriptionError(
        f"Audio device matching '{device_query}' not found.\n"
        "Run 'transcribe --list-devices' to see all available audio input devices."
    )


class SpeechDetector:
    """Decides whether a short audio chunk contains human speech.

    Prefers the Silero VAD model bundled with faster-whisper (it can tell
    speech apart from music and noise). Falls back to a simple volume
    threshold when the model isn't available.
    """

    def __init__(self, silence_threshold_db: float = -45.0) -> None:
        self.silence_threshold_db = silence_threshold_db
        self._vad_ready = False
        try:
            import numpy as np  # noqa: F401
            from faster_whisper.vad import VadOptions, get_speech_timestamps  # noqa: F401

            self._np = np
            self._get_speech_timestamps = get_speech_timestamps
            self._vad_options = VadOptions(
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
            )
            self._vad_ready = True
        except ImportError:
            pass

    def is_speech(self, chunk: bytes) -> bool:
        from transcribe.audio.metrics import calculate_dbfs

        # A chunk far below the threshold is silence either way; this also
        # keeps pure digital silence (all zeros) away from the VAD model.
        dbfs = calculate_dbfs(chunk)
        if dbfs < self.silence_threshold_db:
            return False
        if not self._vad_ready:
            return True
        try:
            audio = (
                self._np.frombuffer(chunk, dtype=self._np.int16).astype(self._np.float32) / 32768.0
            )
            return bool(self._get_speech_timestamps(audio, self._vad_options))
        except Exception:
            self._vad_ready = False
            return True


def mix_chunks(chunk_a: bytes, chunk_b: bytes) -> bytes:
    """Mix two 16-bit PCM chunks into one by summing samples (with clipping)."""
    import numpy as np

    a = np.frombuffer(chunk_a, dtype=np.int16).astype(np.int32)
    b = np.frombuffer(chunk_b, dtype=np.int16).astype(np.int32)
    if len(a) != len(b):
        size = max(len(a), len(b))
        a = np.pad(a, (0, size - len(a)))
        b = np.pad(b, (0, size - len(b)))
    return np.clip(a + b, -32768, 32767).astype(np.int16).tobytes()


def pair_chunks(
    queue_a: "queue.Queue[bytes | None]",
    queue_b: "queue.Queue[bytes | None]",
    silence: bytes,
    stopped: Callable[[], bool],
    *,
    b_wait_s: float = 0.2,
    b_stall_after: int = 5,
    max_b_backlog: int = 2,
) -> Generator[bytes, None, None]:
    """Mix device A's chunks with device B's, with A setting the pace.

    Ends when A's queue gets None (its reader stopped). B is optional: when
    it is late, silence takes its place. After a few misses in a row B counts
    as stalled and A stops waiting for it, so a dead second device (e.g.
    headphones running out of battery) never slows down the main one.
    """
    b_missed = 0
    while not stopped():
        try:
            chunk_a = queue_a.get(timeout=0.5)
        except queue.Empty:
            continue
        if chunk_a is None:
            return

        try:
            if b_missed < b_stall_after:
                chunk_b = queue_b.get(timeout=b_wait_s)
            else:
                chunk_b = queue_b.get_nowait()
        except queue.Empty:
            chunk_b = None
        if chunk_b is None:
            b_missed += 1
            chunk_b = silence
        else:
            b_missed = 0

        # Two devices have two clocks that drift apart; drop B's surplus so it stays in step
        while queue_b.qsize() > max_b_backlog:
            try:
                queue_b.get_nowait()
            except queue.Empty:
                break

        yield mix_chunks(chunk_a, chunk_b)


def record_stream(
    stream: Iterator[bytes], wav_path: Path, sample_rate: int
) -> Generator[bytes, None, None]:
    """Pass audio chunks through unchanged while saving them to a 16-bit mono WAV file.

    The live text is only a fast draft; keeping the audio lets a bigger model
    transcribe the whole meeting again afterwards. `wave` rewrites the header
    after every write, and we flush each chunk to disk, so the file stays
    playable with all audio so far even if the process dies mid-meeting.
    """
    import wave

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wav_path.open("wb") as f, wave.open(f, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in stream:
            wf.writeframes(chunk)
            f.flush()
            yield chunk


def print_audio_devices_table() -> None:
    """Render a formatted table of all audio input devices to the terminal."""
    devices = list_audio_devices()
    if not devices:
        console.print(
            "[yellow]No audio driver installed. Run 'uv add sounddevice' to list devices.[/yellow]"
        )
        return

    table = Table(title="Available Audio Input Devices (Microphone & System Audio)")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Device Name", style="white")
    table.add_column("Channels", justify="right", style="magenta")
    table.add_column("Sample Rate", justify="right", style="green")
    table.add_column("Default", justify="center", style="bold yellow")

    for dev in devices:
        default_mark = "★ Default" if dev["is_default"] else ""
        table.add_row(
            str(dev["id"]),
            dev["name"],
            str(dev["channels"]),
            f"{dev['default_samplerate']} Hz",
            default_mark,
        )

    console.print(table)
    console.print(
        "\n[dim]Tip: To capture computer/browser sound, select a virtual device (e.g. BlackHole, Loopback, Stereo Mix).[/dim]\n"
    )


def _clock(ms: int) -> str:
    mins, secs = divmod(ms // 1000, 60)
    return f"{mins:02d}:{secs:02d}"


def _print_line(segment: Segment) -> None:
    speaker = f"{segment.speaker}: " if segment.speaker else ""
    console.print(f"[dim]{_clock(segment.start_ms)}[/dim] [bold]{speaker}{segment.text}[/bold]")


class LiveTranscriptionSession:
    """Manages a live microphone or audio stream transcription session."""

    def __init__(
        self,
        api_key: str | None = None,
        sample_rate: int = 16000,
        language: str | None = None,
        device: str | int | None = None,
        mix_device: str | int | None = None,
        require_api_key: bool = True,
    ) -> None:
        self.api_key = (api_key or config.load_api_key()) if require_api_key else (api_key or "")
        self.sample_rate = sample_rate
        self.language = language
        self.device = device
        self.device_id = resolve_device_id(device)
        self.mix_device = mix_device
        self.mix_device_id = resolve_device_id(mix_device)
        self.segments: list[Segment] = []
        # Set to ask the capture threads to close their audio streams and
        # exit; exiting the process while a thread is inside CoreAudio/
        # PortAudio C code crashes Python on shutdown.
        self._stop_capture = threading.Event()

    @property
    def _chunk_size(self) -> int:
        return int(self.sample_rate * CHUNK_MS / 1000)

    def _read_device(self, device_id: int | None) -> Generator[bytes, None, None]:
        """Yield 100ms chunks of 16-bit mono PCM from one input device until asked to stop."""
        try:
            import sounddevice as sd
        except ImportError:
            raise TranscriptionError(
                "Live microphone transcription requires 'sounddevice'.\n"
                "Install it with:\n"
                "  uv add sounddevice"
            )

        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self._chunk_size,
                channels=1,
                dtype="int16",
                device=device_id,
            ) as stream:
                while not self._stop_capture.is_set():
                    data, _ = stream.read(self._chunk_size)
                    if data:
                        yield bytes(data)
        except sd.PortAudioError as error:
            raise TranscriptionError(f"Audio device error: {error}") from error

    def get_mixed_stream(self) -> Generator[bytes, None, None]:
        """Meeting mode: merge two input devices (e.g. BlackHole + microphone) into one mono stream.

        Each device is read on its own thread and the chunks are paired up
        by `pair_chunks`. If the main device fails, the stream stops with
        its error; if the second one fails, recording goes on without it.
        """
        silence = b"\x00\x00" * self._chunk_size
        stop = threading.Event()
        main_error: list[TranscriptionError] = []

        def reader(device_id: int | None, q: queue.Queue, is_main: bool) -> None:
            try:
                for chunk in self._read_device(device_id):
                    if stop.is_set():
                        break
                    q.put(chunk)
            except TranscriptionError as error:
                if is_main:
                    main_error.append(error)
                else:
                    console.print(
                        f"\n[yellow]Warning: second device '{self.mix_device}' stopped ({error}). "
                        f"Recording goes on with '{self.device}' only.[/yellow]"
                    )
            finally:
                q.put(None)

        queue_a: queue.Queue[bytes | None] = queue.Queue()
        queue_b: queue.Queue[bytes | None] = queue.Queue()
        threads = [
            threading.Thread(target=reader, args=(self.device_id, queue_a, True), daemon=True),
            threading.Thread(target=reader, args=(self.mix_device_id, queue_b, False), daemon=True),
        ]
        for thread in threads:
            thread.start()

        try:
            yield from pair_chunks(queue_a, queue_b, silence, self._stop_capture.is_set)
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=2.0)
        if main_error:
            raise main_error[0]

    def get_microphone_stream(self) -> Generator[bytes, None, None]:
        """Generator yielding 16-bit PCM mono audio chunks from system microphone or selected device."""
        if self.mix_device_id is not None:
            yield from self.get_mixed_stream()
        else:
            yield from self._read_device(self.device_id)

    def _devices_label(self) -> str:
        if self.mix_device is not None:
            return f" (Devices: {self.device} + {self.mix_device})"
        return f" (Device: {self.device})" if self.device is not None else ""

    def run(
        self,
        audio_stream: Iterator[bytes] | None = None,
        on_turn: OnTurn | None = None,
        record_to: Path | None = None,
        options: TranscriptionOptions | None = None,
    ) -> Transcript:
        """Stream live audio to AssemblyAI and return the final Transcript."""
        import assemblyai.streaming.v3 as v3

        from transcribe.audio.metrics import calculate_dbfs, calculate_rms_and_peak

        opts = options or TranscriptionOptions(language=self.language)
        params = v3.StreamingParameters(
            sample_rate=self.sample_rate,
            encoding=v3.Encoding.pcm_s16le,
            language_code=None if opts.auto_detect_language else opts.language,
            language_detection=True if opts.auto_detect_language else None,
            speaker_labels=True if opts.speakers else None,
            keyterms_prompt=list(opts.boost_words) or None,
        )

        stream_source = audio_stream or self.get_microphone_stream()
        if record_to is not None:
            stream_source = record_stream(stream_source, record_to, self.sample_rate)
        client = v3.StreamingClient(api_key=self.api_key)

        self.segments.clear()

        def handle_turn(client_instance: Any, event: Any) -> None:
            text = (getattr(event, "transcript", "") or "").strip()
            if not text:
                return

            # A turn has no times of its own; they come from its first and last word
            words = getattr(event, "words", None) or []
            start_ms = words[0].start if words else 0
            end_ms = words[-1].end if words else start_ms

            if not getattr(event, "end_of_turn", True):
                # Instant word-by-word live update on the same line
                sys.stderr.write(f"\r\033[K{_clock(start_ms)} {text}")
                sys.stderr.flush()
                return

            # Sentence completed: commit line cleanly
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            label = getattr(event, "speaker_label", None)
            segment = Segment(
                text=text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=f"Speaker {label}" if label else None,
            )
            self.segments.append(segment)
            (on_turn or _print_line)(segment)

        def handle_error(client_instance: Any, error: Any) -> None:
            console.print(f"[bold red]Live Streaming Error:[/bold red] {error}")

        client.on(v3.RealTimeEvents.Turn, handle_turn)
        client.on(v3.RealTimeEvents.Error, handle_error)

        console.print(
            f"[bold cyan]🔴 Live Transcription Started{self._devices_label()}[/bold cyan] (Press [bold yellow]Ctrl+C[/bold yellow] to finish)..."
        )

        silent_chunk_count = 0
        clipping_chunk_count = 0
        warned_low_volume = False
        warned_clipping = False
        streamed_bytes = 0

        try:
            client.connect(params)
            for chunk in stream_source:
                streamed_bytes += len(chunk)
                # 1. Analyze audio quality of the incoming chunk
                _, peak = calculate_rms_and_peak(chunk)
                dbfs = calculate_dbfs(chunk)

                if dbfs < -48.0:
                    silent_chunk_count += 1
                else:
                    silent_chunk_count = max(0, silent_chunk_count - 1)

                if peak >= 32700:
                    clipping_chunk_count += 1
                else:
                    clipping_chunk_count = max(0, clipping_chunk_count - 1)

                # Warn if silent / extremely quiet for ~3 seconds (30 consecutive 100ms chunks)
                if silent_chunk_count >= 30 and not warned_low_volume and not self.segments:
                    console.print(
                        f"[bold yellow]⚠️ Warning:[/bold yellow] Audio input level is extremely low ({dbfs:.1f} dBFS).\n"
                        "[dim]Check that your microphone is unmuted and the input volume is raised in System Settings.[/dim]"
                    )
                    warned_low_volume = True

                # Warn if clipping / distorted
                if clipping_chunk_count >= 10 and not warned_clipping:
                    console.print(
                        "[bold red]⚠️ Warning:[/bold red] Audio is clipping / distorting. Please lower your microphone input level."
                    )
                    warned_clipping = True

                # 2. Stream chunk to AssemblyAI
                client.stream(chunk)

        except KeyboardInterrupt:
            console.print("\n[dim]Stopping live stream...[/dim]")
        except TranscriptionError as error:
            # The device failed mid-session: keep what was heard so far
            if not streamed_bytes:
                raise
            console.print(f"\n[bold red]Audio capture stopped:[/bold red] {error}")
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
            # Closes the WAV recorder (if any) right away instead of at garbage collection
            close = getattr(stream_source, "close", None)
            if close is not None:
                close()

        return Transcript(
            text=" ".join(s.text for s in self.segments),
            segments=tuple(self.segments),
            audio_duration_s=streamed_bytes // (2 * self.sample_rate),
            language_code=self.language,
        )

    def run_offline(
        self,
        provider: Any,
        options: TranscriptionOptions | None = None,
        audio_stream: Iterator[bytes] | None = None,
        on_turn: OnTurn | None = None,
        preview_provider: Any = None,
        record_to: Path | None = None,
    ) -> Transcript:
        """Transcribe live audio phrase-by-phrase with low latency.

        Dual-model design (same approach as RealtimeSTT): a fast preview model
        redraws the gray in-progress line and finds sentence boundaries, while
        the accurate main model transcribes committed lines on a background
        worker thread, so the audio loop never blocks. Audio capture also runs
        on its own thread, so no sound is lost while models are busy.
        """
        import tempfile
        import wave
        from dataclasses import replace

        from transcribe.audio.metrics import calculate_dbfs

        preview_engine = preview_provider or provider

        # Load the Whisper models BEFORE recording starts. On first use this
        # downloads them, which can take minutes; doing it mid-session
        # would freeze the stream and look like a hang.
        with console.status(
            "[bold cyan]Loading Whisper models (first use downloads them, this can take a few minutes)...[/bold cyan]",
            spinner="dots",
        ):
            for engine in {id(provider): provider, id(preview_engine): preview_engine}.values():
                if hasattr(engine, "warm_up"):
                    engine.warm_up()

        # Keep macOS from throttling us (App Nap) or sleeping while we record:
        # a hidden terminal window otherwise slows transcription below real
        # time and the live output falls behind. caffeinate exits with us (-w).
        caffeinate = None
        if sys.platform == "darwin":
            import os
            import subprocess

            try:
                caffeinate = subprocess.Popen(
                    ["caffeinate", "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                pass

        stream_source = audio_stream or self.get_microphone_stream()
        if record_to is not None:
            stream_source = record_stream(stream_source, record_to, self.sample_rate)
        console.print(
            f"[bold cyan]🔴 Live Transcription Started{self._devices_label()}[/bold cyan] (Press [bold yellow]Ctrl+C[/bold yellow] to finish)..."
        )

        silence_threshold_db = -45.0
        preview_min_chunks = 2  # try to refresh the preview every ~200ms (RealtimeSTT's pace)...
        preview_cost_chunks = 2  # ...slowed down adaptively to 2x what a pass costs on this machine

        all_segments: list[Segment] = []
        detector = SpeechDetector(silence_threshold_db=silence_threshold_db)
        splitter = PhraseSplitter(chunk_ms=CHUNK_MS)
        base_options = options or TranscriptionOptions(language=self.language)

        # Background reader: drains the audio source continuously so chunks
        # aren't dropped while the (slow) transcription call is running.
        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        reader_error: list[Exception] = []

        def _reader() -> None:
            try:
                for chunk in stream_source:
                    audio_queue.put(chunk)
            except Exception as error:
                reader_error.append(error)
            finally:
                audio_queue.put(None)

        self._stop_capture.clear()
        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        io_lock = threading.Lock()
        # Set once the session is over; a worker still busy after that (when
        # the user skipped the wait) must not add lines to a finished transcript
        finished = threading.Event()

        def transcribe_phrase(
            raw_bytes: bytes, engine: Any, beam_size: int | None = None
        ) -> Transcript | None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_wav_path = Path(tmp.name)
            try:
                with wave.open(str(tmp_wav_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(raw_bytes)
                # Give Whisper the previous sentence as context so this phrase
                # is not transcribed blind
                context = all_segments[-1].text if all_segments else None
                opts = replace(base_options, context_prompt=context, beam_size=beam_size)
                return engine.transcribe(tmp_wav_path, options=opts)
            except Exception:
                return None
            finally:
                try:
                    tmp_wav_path.unlink()
                except OSError:
                    pass

        def emit_line(segment: Segment) -> None:
            with io_lock:
                if finished.is_set():
                    return
                all_segments.append(segment)
                sys.stderr.write("\r\033[K")
                (on_turn or _print_line)(segment)

        # Committed lines are transcribed by the accurate main model on this
        # worker thread, in order, so the audio loop never waits for it.
        commit_queue: queue.Queue[Phrase | None] = queue.Queue()

        def _commit_worker() -> None:
            while True:
                phrase = commit_queue.get()
                if phrase is None or finished.is_set():
                    return
                if calculate_dbfs(phrase.audio) < silence_threshold_db:
                    continue
                result = transcribe_phrase(phrase.audio, provider)
                text = (result.text or "").strip() if result else ""
                if text:
                    emit_line(Segment(text=text, start_ms=phrase.start_ms, end_ms=phrase.end_ms))

        commit_worker = threading.Thread(target=_commit_worker, daemon=True)
        commit_worker.start()

        def show_preview(text: str) -> None:
            if not text:
                return
            # Fit one terminal line, always showing the END of the sentence
            # (that's where the speaker currently is). Rendered dim so it's
            # visually distinct from committed lines.
            width = max(20, shutil.get_terminal_size().columns)
            budget = width - 7  # room for the leading "  … " plus a safety margin
            line = text if len(text) <= budget else "…" + text[-(budget - 1) :]
            with io_lock:
                sys.stderr.write(f"\r\033[K\033[2m  … {line}\033[0m")
                sys.stderr.flush()

        def clear_preview() -> None:
            with io_lock:
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()

        warned_lagging = False
        try:
            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    break

                # More than ~8s of unprocessed audio piled up: the model is
                # slower than real time on this machine.
                if not warned_lagging and audio_queue.qsize() > 80:
                    console.print(
                        "\n[bold yellow]⚠️ Transcription is falling behind the audio.[/bold yellow] "
                        "[dim]This model is too slow for live mode on this machine; try a smaller one (e.g. --model base).[/dim]"
                    )
                    warned_lagging = True

                # Don't clear the preview line when a phrase is committed: it
                # stays on screen until the worker's final line replaces it in
                # one motion, so the reader never sees the text vanish.
                phrase = splitter.feed(chunk, detector.is_speech(chunk))
                if phrase is not None:
                    commit_queue.put(phrase)
                    continue

                if (
                    not splitter.in_speech
                    or splitter.new_chunks < max(preview_min_chunks, preview_cost_chunks)
                    or audio_queue.qsize() > 5
                ):
                    # Only analyze when we're (nearly) keeping up with the
                    # audio stream; a small backlog is fine.
                    continue

                # The fast preview model redraws the gray line and looks for
                # sentence boundaries; the accurate model never runs here.
                splitter.new_chunks = 0
                pass_started = time.time()
                result = transcribe_phrase(splitter.audio, preview_engine, beam_size=3)
                preview_cost_chunks = max(
                    preview_min_chunks, int((time.time() - pass_started) * 20)
                )
                if result is None:
                    continue
                if len(result.segments) >= 2:
                    # The preview model split the buffer into sentences. The
                    # last one may still be mid-sentence, but everything
                    # before it is finished: hand it to the main model now
                    # instead of waiting for a pause, and keep only the
                    # unfinished tail in the buffer.
                    head = splitter.cut(result.segments[-2].end_ms // CHUNK_MS)
                    if head is not None:
                        commit_queue.put(head)
                        show_preview(result.segments[-1].text.strip())
                        continue
                show_preview((result.text or "").strip())

        except KeyboardInterrupt:
            clear_preview()
            console.print("\n[dim]Stopping live transcription...[/dim]")

        # Close the audio stream cleanly and wait for the capture thread;
        # exiting while it's inside audio driver C code crashes Python.
        self._stop_capture.set()
        reader_thread.join(timeout=3.0)

        if reader_error:
            if not splitter.now_ms:
                raise TranscriptionError(f"Could not record audio: {reader_error[0]}")
            console.print(f"\n[bold red]Audio capture stopped:[/bold red] {reader_error[0]}")

        # Transcribe whatever was still being spoken when we stopped, and let
        # the worker finish pending lines. A second Ctrl+C skips the wait.
        try:
            last = splitter.flush()
            if last is not None or not commit_queue.empty():
                console.print(
                    "[dim]Finishing the last phrases (press Ctrl+C again to skip)...[/dim]"
                )
            if last is not None:
                commit_queue.put(last)
            commit_queue.put(None)
            commit_worker.join()
        except KeyboardInterrupt:
            console.print("\n[dim]Remaining phrases skipped.[/dim]")

        with io_lock:
            finished.set()
            segments = tuple(all_segments)

        if caffeinate is not None:
            caffeinate.terminate()

        return Transcript(
            text=" ".join(s.text for s in segments),
            segments=segments,
            audio_duration_s=splitter.now_ms // 1000,
            language_code=self.language,
        )
