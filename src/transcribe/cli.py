from dataclasses import replace
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from transcribe import config, output
from transcribe.audio.preprocess import optimized_audio_source
from transcribe.audio.url import is_url, url_audio_source
from transcribe.live import LiveTranscriptionSession, print_audio_devices_table
from transcribe.providers.factory import PROVIDERS, get_provider
from transcribe.transcription.model import (
    Segment,
    Transcript,
    TranscriptionError,
    TranscriptionOptions,
)
from transcribe.transcription.service import transcribe_file, validate_audio_file

app = typer.Typer(
    add_completion=False,
    help="Transcribe audio/video files, web URLs (YouTube, etc.), or live microphone to text, markdown, subtitles, or JSON.",
)
console = Console(stderr=True)

LOCAL_PROVIDERS = ("whisper", "local", "offline")


def is_local(provider: str) -> bool:
    return provider.strip().lower() in LOCAL_PROVIDERS


def write_notes_file(transcript, target: Path, notes_dir: Path | None, notes_provider) -> None:
    """Turn the transcript into a structured note with the configured LLM.

    The note goes into `notes_dir` (e.g. an Obsidian inbox) when given, otherwise next to the transcript.
    """
    from transcribe.notes import document, generate_notes

    with console.status(
        f"[bold cyan]Sending the transcript to {notes_provider.label} for notes...[/bold cyan]",
        spinner="dots",
    ):
        notes = generate_notes(transcript, title=target.stem, provider=notes_provider)
    for warning in notes.warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    today = date.today()
    if notes_dir is not None:
        name = document.filename(notes.text, today=today, fallback=target.stem)
        notes_target = output.unique_path(notes_dir / name, sep="-")
    else:
        notes_target = output.unique_path(target.with_name(f"{target.stem}_notes.md"))
    content = document.render(notes.text, sources=[str(target.resolve())], today=today)
    notes_target.write_text(content, encoding="utf-8")
    typer.echo(f"Notes written to {notes_target}")


def print_warnings(transcript: Transcript) -> None:
    for warning in transcript.warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


def warn_ignored_options(provider: str, options: TranscriptionOptions, *, live: bool) -> None:
    """Say which options this provider can't use, instead of dropping them silently."""
    ignored = []
    if is_local(provider) or live:
        if options.summarize:
            ignored.append("--summarize")
        if options.chapters:
            ignored.append("--chapters")
    if ignored:
        where = "live mode" if live else "local Whisper"
        verb, it = ("works", "it") if len(ignored) == 1 else ("work", "them")
        console.print(
            f"[yellow]Note: {' and '.join(ignored)} only {verb} when transcribing a file with "
            f"AssemblyAI, so {where} ignores {it}.[/yellow]"
        )


def run_final_pass(
    audio_path: Path,
    provider: str,
    model: str,
    options: TranscriptionOptions,
    *,
    compress: bool,
) -> Transcript | None:
    """Transcribe the saved live recording again with a bigger model.

    Returns None when the pass fails, is skipped with Ctrl+C, or hears nothing,
    so the caller keeps the live transcript and the meeting is never lost.
    """
    try:
        engine = get_provider(provider, model=model)
        with console.status(
            f"[bold cyan]Final pass: transcribing the full recording with '{model}' "
            "(this can take a few minutes, Ctrl+C keeps the live transcript)...[/bold cyan]",
            spinner="dots",
        ):
            with optimized_audio_source(audio_path, compress=compress) as source:
                final = transcribe_file(source, engine, options=options)
    except KeyboardInterrupt:
        console.print("[yellow]Final pass skipped, keeping the live transcript.[/yellow]")
        return None
    except (config.ConfigError, TranscriptionError, OSError) as error:
        console.print(f"[yellow]Final pass failed, keeping the live transcript: {error}[/yellow]")
        return None
    if not final.text.strip():
        console.print("[yellow]Final pass found no speech, keeping the live transcript.[/yellow]")
        return None
    print_warnings(final)
    return final


def resolve_target_path(
    stem_or_file: str | Path,
    out: Path | None,
    fmt: str,
    is_batch: bool,
) -> Path:
    if out is None:
        if isinstance(stem_or_file, Path):
            return stem_or_file.with_suffix(f".{fmt}")
        return Path(f"{stem_or_file}.{fmt}")

    if is_batch or out.is_dir() or str(out).endswith(("/", "\\")):
        stem = stem_or_file.stem if isinstance(stem_or_file, Path) else stem_or_file
        return out / f"{stem}.{fmt}"
    return out


def save_api_key(key: str | None) -> None:
    # Asking hides the key and keeps it out of the shell history
    key = key or typer.prompt("AssemblyAI API key", hide_input=True)
    saved_path = config.save_api_key(key)
    typer.echo(f"API key successfully saved to {saved_path}")


def show_config() -> None:
    typer.echo(f"Config path: {config.get_config_path()}")
    try:
        key = config.load_api_key()
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        typer.echo(f"API key: configured ({masked})")
    except config.ConfigError:
        typer.echo("API key: not configured")


def handle_config_command(args: list[str]) -> None:
    """`transcribe config set-key [KEY]` or `transcribe config show`."""
    if args and args[0] in ("set-key", "set"):
        save_api_key(args[1] if len(args) > 1 else None)
        return
    if args and args[0] in ("show", "path"):
        show_config()
        return
    typer.echo("Usage: transcribe config set-key [YOUR_API_KEY] | transcribe config show", err=True)
    raise typer.Exit(2)


def run_doctor() -> None:
    from rich.table import Table

    from transcribe.doctor import FAIL, OK, WARN, run_checks

    style = {OK: "[green]ok[/green]", WARN: "[yellow]warn[/yellow]", FAIL: "[red]fail[/red]"}
    checks = run_checks()
    table = Table.grid(padding=(0, 2))
    table.add_column(no_wrap=True)
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    for check in checks:
        fix = f"\n[dim]Fix: {check.fix}[/dim]" if check.fix and check.status != OK else ""
        table.add_row(style[check.status], check.name, check.detail + fix)
    console.print(table)
    failed = [c for c in checks if c.status == FAIL]
    if failed:
        console.print(f"\n[red]{len(failed)} problem(s) will break a run. Fix them first.[/red]")
        raise typer.Exit(1)
    console.print(
        "\n[green]Nothing broken.[/green] Warnings only matter for the features they name."
    )


def run_live(
    *,
    provider: str,
    model: str | None,
    fmt: str,
    options: TranscriptionOptions,
    device: str | None,
    mix_device: str | None,
    out: Path | None,
    force: bool,
    stdout: bool,
    save_audio: bool,
    final_model: str | None,
    compress: bool,
    notes_provider,
    notes_dir: Path | None,
) -> None:
    if final_model and (stdout or not save_audio):
        typer.echo(
            "Error: --final-model needs the saved audio, so it can't be used with --stdout or --no-save-audio.",
            err=True,
        )
        raise typer.Exit(2)

    is_offline = is_local(provider)
    warn_ignored_options(provider, options, live=True)
    # The final pass works on the full recording, so it can still label speakers
    final_options = options
    if is_offline and options.speakers:
        if not final_model:
            console.print(
                "[yellow]Note: speaker labels are not supported in local live mode; ignoring --speakers. "
                "Use --final-model to add them after the recording.[/yellow]"
            )
        options = replace(options, speakers=False)

    session = LiveTranscriptionSession(
        language=options.language,
        device=device,
        mix_device=mix_device,
        require_api_key=not is_offline,
    )

    # A live recording can't be repeated, so protect it twice:
    # pick a free filename up front (never fail on a taken name), and
    # append every finished line to it immediately so a crash mid-session
    # loses nothing. The final formatted render replaces the file at the end.
    target: Path | None = None
    audio_path: Path | None = None
    on_turn = None
    if not stdout:
        target = out or Path(f"live_session.{fmt}")
        if not force:
            target = output.unique_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if save_audio:
            audio_path = target.with_suffix(".wav")
            if not force:
                audio_path = output.unique_path(audio_path)

        def on_turn(segment: Segment) -> None:
            mins, secs = divmod(segment.start_ms // 1000, 60)
            speaker = f"{segment.speaker}: " if segment.speaker else ""
            line = f"{mins:02d}:{secs:02d} {speaker}{segment.text}"
            console.print(f"[dim]{mins:02d}:{secs:02d}[/dim] [bold]{speaker}{segment.text}[/bold]")
            with target.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    if is_offline:
        engine = get_provider(provider, model=model)
        # Fast 'tiny' model for the live preview line and sentence
        # detection; the main model only transcribes committed lines.
        preview_engine = None
        if (model or "").strip() not in ("", "tiny", "tiny.en"):
            preview_engine = get_provider(provider, model="tiny")
        transcript = session.run_offline(
            engine,
            options=options,
            on_turn=on_turn,
            preview_provider=preview_engine,
            record_to=audio_path,
        )
    else:
        transcript = session.run(on_turn=on_turn, record_to=audio_path, options=options)

    if audio_path is not None and audio_path.exists():
        typer.echo(f"Audio saved to {audio_path}")
        if final_model:
            final = run_final_pass(
                audio_path,
                provider,
                final_model,
                final_options,
                compress=compress and not is_offline,
            )
            if final is not None:
                transcript = final

    if stdout:
        typer.echo(output.render(transcript, fmt, title="live_recording"), nl=False)
        return
    output.write(transcript, fmt, target, title="live_recording", force=True)
    typer.echo(f"Live transcript written to {target}")
    if notes_provider is not None:
        write_notes_file(transcript, target, notes_dir, notes_provider)


def save_transcript(
    transcript: Transcript,
    *,
    fmt: str,
    title: str,
    target: Path,
    force: bool,
    notes_provider,
    notes_dir: Path | None,
) -> None:
    output.write(transcript, fmt, target, title=title, force=force)
    if transcript.audio_duration_s is not None:
        typer.echo(f"Audio length: {transcript.audio_duration_s / 60:.1f} min")
    typer.echo(f"Transcript written to {target}")
    if notes_provider is not None:
        write_notes_file(transcript, target, notes_dir, notes_provider)


def transcribe_items(
    items: list[str],
    *,
    engine,
    options: TranscriptionOptions,
    fmt: str,
    out: Path | None,
    force: bool,
    stdout: bool,
    compress: bool,
    notes_provider,
    notes_dir: Path | None,
) -> None:
    is_batch = len(items) > 1
    for item in items:
        try:
            if is_url(item):
                with console.status(
                    f"[bold cyan]Downloading audio from URL: {item}...[/bold cyan]", spinner="dots"
                ):
                    with url_audio_source(item) as (audio_path, title):
                        with optimized_audio_source(audio_path, compress=compress) as audio:
                            transcript = transcribe_file(audio, engine, options=options)
                source: str | Path = title
            else:
                file = Path(item)
                title = file.stem
                with console.status(
                    f"[bold cyan]Transcribing {file.name}...[/bold cyan]", spinner="dots"
                ):
                    with optimized_audio_source(file, compress=compress) as audio:
                        transcript = transcribe_file(audio, engine, options=options)
                source = file

            print_warnings(transcript)
            if stdout:
                typer.echo(output.render(transcript, fmt, title=title), nl=False)
                continue
            save_transcript(
                transcript,
                fmt=fmt,
                title=title,
                target=resolve_target_path(source, out, fmt, is_batch=is_batch),
                force=force,
                notes_provider=notes_provider,
                notes_dir=notes_dir,
            )
        except (TranscriptionError, FileExistsError, OSError) as error:
            typer.echo(f"Error ({item}): {error}", err=True)
            raise typer.Exit(1)


@app.command()
def main(
    files: list[str] = typer.Argument(
        None,
        help="Audio/video files or URLs (YouTube, etc.) to transcribe, 'live' for microphone, 'doctor' to check your setup, or 'config'. A file really named 'live', 'doctor' or 'config' needs a path, e.g. ./live",
    ),
    provider: str = typer.Option(
        "assemblyai",
        "--provider",
        "-p",
        help=f"Transcription engine: {', '.join(PROVIDERS)}",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Model size/variant (e.g. 'tiny', 'base', 'small', 'medium', 'large-v3', 'turbo' for Whisper)",
    ),
    fmt: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help=f"Output format: {', '.join(sorted(output.FORMATS))}",
    ),
    speakers: bool = typer.Option(
        False,
        "--speakers",
        "-s",
        help="Label who is speaking (speaker diarization)",
    ),
    speakers_expected: int | None = typer.Option(
        None,
        "--speakers-expected",
        help="Expected number of speakers (optional hint for diarization)",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code (e.g. 'en', 'es', 'fr', 'de', 'ar')",
    ),
    auto_detect_language: bool = typer.Option(
        False,
        "--auto-detect-language",
        "-d",
        help="Automatically detect the spoken language",
    ),
    summarize: bool = typer.Option(
        False,
        "--summarize",
        help="Generate an executive summary of the transcript (AssemblyAI files only)",
    ),
    chapters: bool = typer.Option(
        False,
        "--chapters",
        help="Automatically generate topic chapters with summaries (AssemblyAI files only)",
    ),
    boost: str | None = typer.Option(
        None,
        "--boost",
        help="Comma-separated words or phrases to boost recognition accuracy",
    ),
    compress: bool = typer.Option(
        True,
        "--compress/--no-compress",
        help="Extract or downsample audio with ffmpeg before uploading large files/video",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output file path (single file) or directory (multiple files)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing output files",
    ),
    stdout: bool = typer.Option(
        False,
        "--stdout",
        help="Output transcript to stdout instead of saving to a file",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        "-D",
        help="Audio input device ID or name substring for live mode (e.g. 'BlackHole', 'Stereo Mix')",
    ),
    mix_device: str | None = typer.Option(
        None,
        "--mix-device",
        help="Second input device to merge with --device in live mode (e.g. capture BlackHole + your microphone for full meeting audio)",
    ),
    save_audio: bool = typer.Option(
        True,
        "--save-audio/--no-save-audio",
        help="Live mode: save the recorded audio as a .wav file next to the transcript",
    ),
    final_model: str | None = typer.Option(
        None,
        "--final-model",
        help="Live mode: after you stop, transcribe the saved audio again with this bigger, more accurate model (e.g. 'small', 'turbo'). The result replaces the live transcript",
    ),
    notes: bool = typer.Option(
        False,
        "--notes",
        help="After transcribing, write a structured Markdown note (summary, key points, action items, quotes) with an LLM. Claude by default; set NOTES_PROVIDER for openai, gemini or ollama",
    ),
    notes_dir: Path | None = typer.Option(
        None,
        "--notes-dir",
        envvar="NOTES_DIR",
        help="Folder for --notes, e.g. your Obsidian inbox. Notes get vault-style names like 'release-planning-2026-10-16.md'. Default: next to the transcript",
    ),
    list_devices: bool = typer.Option(
        False,
        "--list-devices",
        help="List all available audio input devices (microphones & loopbacks)",
    ),
    set_key: str | None = typer.Option(
        None,
        "--set-key",
        help="Save AssemblyAI API key to configuration file (safer: 'transcribe config set-key' asks for it)",
    ),
    show_config_flag: bool = typer.Option(
        False,
        "--show-config",
        help="Display current configuration file path and API key status",
    ),
) -> None:
    """Transcribe audio files, YouTube/web URLs, or live microphone input."""
    if list_devices:
        print_audio_devices_table()
        return
    if set_key:
        save_api_key(set_key)
        return
    if show_config_flag:
        show_config()
        return

    if not files:
        typer.echo(
            "Error: Please provide audio files, URLs, or 'live'. Use --help for usage.", err=True
        )
        raise typer.Exit(2)

    first_arg = files[0].strip().lower()
    if first_arg == "config":
        handle_config_command(files[1:])
        return
    if first_arg == "doctor":
        run_doctor()
        return

    if fmt not in output.FORMATS:
        typer.echo(
            f"Error: unknown format '{fmt}'. Use one of: {', '.join(sorted(output.FORMATS))}",
            err=True,
        )
        raise typer.Exit(2)

    # Check the notes setup now, not after a meeting has been recorded
    notes_provider = None
    if notes:
        if notes_dir is not None:
            notes_dir = notes_dir.expanduser()
            if not notes_dir.is_dir():
                typer.echo(f"Error: notes folder not found: {notes_dir}", err=True)
                raise typer.Exit(2)
        from transcribe.notes import get_notes_provider

        try:
            notes_provider = get_notes_provider()
        except TranscriptionError as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(2)

    boost_words = tuple(w.strip() for w in boost.split(",") if w.strip()) if boost else ()
    options = TranscriptionOptions(
        speakers=speakers or (speakers_expected is not None),
        speakers_expected=speakers_expected,
        language=language,
        auto_detect_language=auto_detect_language,
        summarize=summarize,
        chapters=chapters,
        boost_words=boost_words,
    )

    try:
        if first_arg == "live":
            run_live(
                provider=provider,
                model=model,
                fmt=fmt,
                options=options,
                device=device,
                mix_device=mix_device,
                out=out,
                force=force,
                stdout=stdout,
                save_audio=save_audio,
                final_model=final_model,
                compress=compress,
                notes_provider=notes_provider,
                notes_dir=notes_dir,
            )
            return

        for item in files:
            if not is_url(item):
                validate_audio_file(Path(item))
        engine = get_provider(provider, model=model)
        warn_ignored_options(provider, options, live=False)
    except (config.ConfigError, TranscriptionError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)

    transcribe_items(
        files,
        engine=engine,
        options=options,
        fmt=fmt,
        out=out,
        force=force,
        stdout=stdout,
        compress=compress,
        notes_provider=notes_provider,
        notes_dir=notes_dir,
    )
