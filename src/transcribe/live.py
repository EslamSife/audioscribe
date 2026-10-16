import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

from rich.console import Console
from rich.table import Table

from transcribe import config
from transcribe.transcription.model import Segment, Transcript, TranscriptionError

console = Console(stderr=True)


def list_audio_devices() -> list[dict[str, Any]]:
    """Enumerate all available audio input devices (microphones & loopbacks)."""
    # 1. Try sounddevice
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices: list[dict[str, Any]] = []
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else 0

        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                input_devices.append(
                    {
                        "id": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": int(dev.get("default_samplerate", 16000)),
                        "is_default": idx == default_in,
                    }
                )
        return input_devices
    except ImportError:
        pass

    # 2. Try PyAudio
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        input_devices = []
        default_idx = None
        try:
            default_idx = pa.get_default_input_device_info().get("index")
        except Exception:
            pass

        for idx in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(idx)
            if dev.get("maxInputChannels", 0) > 0:
                input_devices.append(
                    {
                        "id": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "channels": dev.get("maxInputChannels", 1),
                        "default_samplerate": int(dev.get("defaultSampleRate", 16000)),
                        "is_default": idx == default_idx,
                    }
                )
        pa.terminate()
        return input_devices
    except ImportError:
        pass

    return []


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
        "Run 'transcribe live --list-devices' to see all available audio input devices."
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
        self._start_time = 0.0
        # Set to ask the capture threads to close their audio streams and
        # exit; exiting the process while a thread is inside CoreAudio/
        # PortAudio C code crashes Python on shutdown.
        import threading

        self._stop_capture = threading.Event()

    def get_mixed_stream(self) -> Generator[bytes, None, None]:
        """Meeting mode: merge two input devices (e.g. BlackHole + microphone) into one mono stream.

        Each device is read on its own thread; the generator pairs up 100ms
        chunks and sums them. If one device stalls, silence fills its side so
        the other keeps flowing.
        """
        import queue
        import threading

        try:
            import sounddevice as sd
        except ImportError:
            raise TranscriptionError(
                "Meeting mode (--mix-device) requires 'sounddevice'.\n"
                "Install it with:\n"
                "  uv add sounddevice"
            )

        chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
        silence = b"\x00\x00" * chunk_size
        stop = threading.Event()

        def reader(device_id: int | None, q: queue.Queue) -> None:
            try:
                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=chunk_size,
                    channels=1,
                    dtype="int16",
                    device=device_id,
                ) as stream:
                    while not stop.is_set() and not self._stop_capture.is_set():
                        data, _ = stream.read(chunk_size)
                        q.put(bytes(data))
            except Exception:
                pass

        queue_a: queue.Queue[bytes] = queue.Queue()
        queue_b: queue.Queue[bytes] = queue.Queue()
        thread_a = threading.Thread(target=reader, args=(self.device_id, queue_a), daemon=True)
        thread_b = threading.Thread(target=reader, args=(self.mix_device_id, queue_b), daemon=True)
        thread_a.start()
        thread_b.start()

        try:
            while not self._stop_capture.is_set():
                try:
                    chunk_a = queue_a.get(timeout=0.5)
                except queue.Empty:
                    continue
                try:
                    chunk_b = queue_b.get(timeout=0.5)
                except queue.Empty:
                    chunk_b = silence
                yield mix_chunks(chunk_a, chunk_b)
        finally:
            stop.set()
            thread_a.join(timeout=2.0)
            thread_b.join(timeout=2.0)

    def get_microphone_stream(self) -> Generator[bytes, None, None]:
        """Generator yielding 16-bit PCM mono audio chunks from system microphone or selected device."""
        if self.mix_device_id is not None:
            yield from self.get_mixed_stream()
            return

        # 1. Try sounddevice
        try:
            import sounddevice as sd

            chunk_size = int(self.sample_rate * 0.1)  # 100ms chunks
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=chunk_size,
                channels=1,
                dtype="int16",
                device=self.device_id,
            ) as stream:
                while not self._stop_capture.is_set():
                    data, overflowed = stream.read(chunk_size)
                    if data:
                        yield bytes(data)
            return
        except ImportError:
            pass

        # 2. Try PyAudio
        try:
            import pyaudio

            pa = pyaudio.PyAudio()
            chunk_size = int(self.sample_rate * 0.1)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_id,
                frames_per_buffer=chunk_size,
            )
            try:
                while not self._stop_capture.is_set():
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    if data:
                        yield data
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            return
        except ImportError:
            pass

        raise TranscriptionError(
            "Live microphone transcription requires 'sounddevice' or 'pyaudio'.\n"
            "To enable microphone capture, install sounddevice:\n"
            "  uv add sounddevice\n"
            "or:\n"
            "  pip install sounddevice"
        )

    def run(
        self,
        audio_stream: Generator[bytes, None, None] | None = None,
        on_turn: Callable[[str, int, int], None] | None = None,
        record_to: Path | None = None,
    ) -> Transcript:
        """Run the live streaming session with audio quality monitoring and return the final Transcript."""
        import assemblyai.streaming.v3 as v3

        from transcribe.audio.metrics import calculate_dbfs, calculate_rms_and_peak

        stream_source = audio_stream or self.get_microphone_stream()
        if record_to is not None:
            stream_source = record_stream(stream_source, record_to, self.sample_rate)
        client = v3.StreamingClient(api_key=self.api_key)

        self.segments.clear()
        self._start_time = time.time()

        @client.on(v3.RealTimeEvents.Turn)
        def handle_turn(client_instance: Any, event: Any) -> None:
            text = (getattr(event, "transcript", "") or getattr(event, "text", "")).strip()
            if not text:
                return

            start_ms = getattr(event, "start", 0) or 0
            end_ms = getattr(event, "end", 0) or 0
            end_of_turn = getattr(event, "end_of_turn", True)

            mins, secs = divmod(start_ms // 1000, 60)
            if not end_of_turn:
                # Instant word-by-word live update on the same line
                sys.stderr.write(f"\r\033[K[dim]{mins:02d}:{secs:02d}[/dim] {text}")
                sys.stderr.flush()
            else:
                # Sentence completed: commit line cleanly
                sys.stderr.write("\r\033[K")
                sys.stderr.flush()
                self.segments.append(
                    Segment(
                        text=text,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                )
                if on_turn:
                    on_turn(text, start_ms, end_ms)
                else:
                    console.print(f"[dim]{mins:02d}:{secs:02d}[/dim] [bold]{text}[/bold]")

        @client.on(v3.RealTimeEvents.Error)
        def handle_error(client_instance: Any, error: Any) -> None:
            console.print(f"[bold red]Live Streaming Error:[/bold red] {error}")

        dev_desc = f" (Device: {self.device})" if self.device is not None else ""
        console.print(
            f"[bold cyan]🔴 Live Transcription Started{dev_desc}[/bold cyan] (Press [bold yellow]Ctrl+C[/bold yellow] to finish)..."
        )

        silent_chunk_count = 0
        clipping_chunk_count = 0
        warned_low_volume = False
        warned_clipping = False

        try:
            client.connect()
            for chunk in stream_source:
                # 1. Analyze audio quality of the incoming chunk
                rms, peak = calculate_rms_and_peak(chunk)
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
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
            # Closes the WAV recorder (if any) right away instead of at garbage collection
            close = getattr(stream_source, "close", None)
            if close is not None:
                close()

        total_duration = int(time.time() - self._start_time)
        full_text = " ".join(s.text for s in self.segments)

        return Transcript(
            text=full_text,
            segments=tuple(self.segments),
            audio_duration_s=total_duration,
            language_code=self.language or "en",
        )

    def run_offline(
        self,
        provider: Any,
        options: Any = None,
        audio_stream: Generator[bytes, None, None] | None = None,
        on_turn: Callable[[str, int, int], None] | None = None,
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
        import queue
        import tempfile
        import threading
        import wave
        from dataclasses import replace

        from transcribe.audio.metrics import calculate_dbfs
        from transcribe.transcription.model import TranscriptionOptions

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
        dev_desc = f" (Device: {self.device})" if self.device is not None else ""
        if self.mix_device is not None:
            dev_desc = f" (Devices: {self.device} + {self.mix_device})"
        console.print(
            f"[bold cyan]🔴 Live Transcription Started{dev_desc}[/bold cyan] (Press [bold yellow]Ctrl+C[/bold yellow] to finish)..."
        )

        chunk_duration_ms = 100  # 100ms per chunk
        silence_threshold_db = -45.0
        end_silence_chunks = 5  # a 500ms pause always ends the phrase
        soft_commit_chunks = 50  # once the phrase reaches 5s of speech...
        soft_silence_chunks = 3  # ...a 300ms micro-pause is enough to end it
        max_phrase_chunks = 100  # hard cap: commit after 10s even with no pause
        preview_min_chunks = 2  # try to refresh the preview every ~200ms (RealtimeSTT's pace)...
        preview_cost_chunks = 2  # ...slowed down adaptively to 2x what a pass costs on this machine
        preroll_chunks = 3  # keep 300ms before speech starts so word onsets aren't clipped

        all_segments: list[Segment] = []
        session_start_time = time.time()
        detector = SpeechDetector(silence_threshold_db=silence_threshold_db)
        base_options = options or TranscriptionOptions(language=self.language)

        # Background reader: drains the audio source continuously so chunks
        # aren't dropped while the (slow) transcription call is running.
        audio_queue: queue.Queue[bytes | None] = queue.Queue()

        def _reader() -> None:
            try:
                for chunk in stream_source:
                    audio_queue.put(chunk)
            except Exception:
                pass
            finally:
                audio_queue.put(None)

        self._stop_capture.clear()
        reader_thread = threading.Thread(target=_reader, daemon=True)
        reader_thread.start()

        io_lock = threading.Lock()

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

        def emit_line(text: str, start_ms: int, end_ms: int) -> None:
            all_segments.append(Segment(text=text, start_ms=start_ms, end_ms=end_ms))
            with io_lock:
                sys.stderr.write("\r\033[K")
                if on_turn:
                    on_turn(text, start_ms, end_ms)
                else:
                    mins, secs = divmod(start_ms // 1000, 60)
                    console.print(f"[dim]{mins:02d}:{secs:02d}[/dim] [bold]{text}[/bold]")

        # Committed lines are transcribed by the accurate main model on this
        # worker thread, in order, so the audio loop never waits for it.
        commit_queue: queue.Queue[tuple[bytes, int, int] | None] = queue.Queue()

        def _commit_worker() -> None:
            while True:
                item = commit_queue.get()
                if item is None:
                    return
                raw_bytes, start_ms, end_ms = item
                if calculate_dbfs(raw_bytes) < silence_threshold_db:
                    continue
                result = transcribe_phrase(raw_bytes, provider)
                text = (result.text or "").strip() if result else ""
                if text:
                    emit_line(text, start_ms, end_ms)

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

        phrase: list[bytes] = []
        preroll: list[bytes] = []
        in_speech = False
        silence_run = 0
        chunks_since_preview = 0
        phrase_start_ms = 0
        current_ms = 0

        def commit_phrase() -> None:
            nonlocal phrase, in_speech, silence_run, chunks_since_preview
            buf, start_ms = phrase, phrase_start_ms
            phrase = []
            in_speech = False
            silence_run = 0
            chunks_since_preview = 0
            if not buf:
                return
            # Don't clear the preview line here: it stays on screen until the
            # worker's final line replaces it in one motion, so the reader
            # never sees the text vanish while the accurate model works.
            commit_queue.put((b"".join(buf), start_ms, start_ms + len(buf) * chunk_duration_ms))

        warned_lagging = False
        try:
            while True:
                chunk = audio_queue.get()
                if chunk is None:
                    break
                current_ms += chunk_duration_ms

                # More than ~8s of unprocessed audio piled up: the model is
                # slower than real time on this machine.
                if not warned_lagging and audio_queue.qsize() > 80:
                    console.print(
                        "\n[bold yellow]⚠️ Transcription is falling behind the audio.[/bold yellow] "
                        "[dim]This model is too slow for live mode on this machine; try a smaller one (e.g. --model base).[/dim]"
                    )
                    warned_lagging = True

                is_silent = not detector.is_speech(chunk)

                if not in_speech:
                    if is_silent:
                        preroll.append(chunk)
                        if len(preroll) > preroll_chunks:
                            preroll.pop(0)
                        continue
                    in_speech = True
                    phrase = preroll + [chunk]
                    preroll = []
                    phrase_start_ms = max(0, current_ms - len(phrase) * chunk_duration_ms)
                    silence_run = 0
                    chunks_since_preview = 0
                    continue

                phrase.append(chunk)
                chunks_since_preview += 1
                silence_run = silence_run + 1 if is_silent else 0

                if (
                    silence_run >= end_silence_chunks
                    or len(phrase) >= max_phrase_chunks
                    or (len(phrase) >= soft_commit_chunks and silence_run >= soft_silence_chunks)
                ):
                    commit_phrase()
                elif (
                    chunks_since_preview >= max(preview_min_chunks, preview_cost_chunks)
                    and audio_queue.qsize() <= 5
                ):
                    # Only analyze when we're (nearly) keeping up with the
                    # audio stream; a small backlog is fine.
                    # The fast preview model redraws the gray line and looks for
                    # sentence boundaries; the accurate model never runs here.
                    chunks_since_preview = 0
                    pass_started = time.time()
                    result = transcribe_phrase(b"".join(phrase), preview_engine, beam_size=3)
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
                        cut = min(len(phrase) - 1, result.segments[-2].end_ms // chunk_duration_ms)
                        if cut > 0:
                            commit_queue.put(
                                (
                                    b"".join(phrase[:cut]),
                                    phrase_start_ms,
                                    phrase_start_ms + cut * chunk_duration_ms,
                                )
                            )
                            phrase = phrase[cut:]
                            phrase_start_ms += cut * chunk_duration_ms
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

        # Transcribe whatever was still being spoken when we stopped, and let
        # the worker finish pending lines. A second Ctrl+C skips the wait.
        try:
            if phrase or not commit_queue.empty():
                console.print(
                    "[dim]Finishing the last phrases (press Ctrl+C again to skip)...[/dim]"
                )
            commit_phrase()
            commit_queue.put(None)
            commit_worker.join()
        except KeyboardInterrupt:
            console.print("\n[dim]Remaining phrases skipped.[/dim]")

        if caffeinate is not None:
            caffeinate.terminate()

        total_duration = int(time.time() - session_start_time)
        full_text = " ".join(s.text for s in all_segments)

        return Transcript(
            text=full_text,
            segments=tuple(all_segments),
            audio_duration_s=total_duration,
            language_code=self.language or "en",
        )
