from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console

from transcribe import config, output
from transcribe.audio.preprocess import optimized_audio_source
from transcribe.audio.url import is_url, url_audio_source
from transcribe.live import LiveTranscriptionSession, print_audio_devices_table
from transcribe.providers.factory import PROVIDERS, get_provider
from transcribe.transcription.model import TranscriptionError, TranscriptionOptions
from transcribe.transcription.service import transcribe_file, validate_audio_file

app = typer.Typer(
    add_completion=False,
    help="Transcribe audio/video files, web URLs (YouTube, etc.), or live microphone to text, markdown, subtitles, or JSON.",
)
console = Console(stderr=True)


def write_notes_file(transcript, target: Path) -> None:
    """Turn the transcript into a structured note with Claude and save it next to the transcript."""
    from transcribe.notes import generate_notes

    with console.status("[bold cyan]Creating notes with Claude...[/bold cyan]", spinner="dots"):
        note = generate_notes(transcript, title=target.stem)
    notes_target = output.unique_path(target.with_name(f"{target.stem}_notes.md"))
    notes_target.write_text(note, encoding="utf-8")
    typer.echo(f"Notes written to {notes_target}")


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


@app.command()
def main(
    files: list[str] = typer.Argument(
        None,
        help="Audio/video files or URLs (YouTube, etc.) to transcribe, 'live' for microphone, or 'config'",
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
        help="Generate an executive summary of the transcript (AssemblyAI)",
    ),
    chapters: bool = typer.Option(
        False,
        "--chapters",
        help="Automatically generate topic chapters with summaries (AssemblyAI)",
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
    notes: bool = typer.Option(
        False,
        "--notes",
        help="After transcribing, use Claude to write a structured Markdown note (summary, key points, action items, quotes)",
    ),
    list_devices: bool = typer.Option(
        False,
        "--list-devices",
        help="List all available audio input devices (microphones & loopbacks)",
    ),
    set_key: str | None = typer.Option(
        None,
        "--set-key",
        help="Save AssemblyAI API key to configuration file",
    ),
    show_config: bool = typer.Option(
        False,
        "--show-config",
        help="Display current configuration file path and API key status",
    ),
) -> None:
    """Transcribe audio files, YouTube/web URLs, or live microphone input."""
    # 1. Handle device listing
    if list_devices:
        print_audio_devices_table()
        return

    # 2. Handle configuration helpers
    if set_key:
        saved_path = config.save_api_key(set_key)
        typer.echo(f"API key successfully saved to {saved_path}")
        return

    if show_config:
        cfg_path = config.get_config_path()
        typer.echo(f"Config path: {cfg_path}")
        try:
            key = config.load_api_key()
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
            typer.echo(f"API key: configured ({masked})")
        except config.ConfigError:
            typer.echo("API key: not configured")
        return

    if not files:
        typer.echo("Error: Please provide audio files, URLs, or 'live'. Use --help for usage.", err=True)
        raise typer.Exit(2)

    # 3. Handle 'transcribe config ...' syntax
    first_arg = files[0].strip().lower()
    if first_arg == "config":
        args = files[1:]
        if len(args) >= 2 and args[0] in ("set-key", "set"):
            saved_path = config.save_api_key(args[1])
            typer.echo(f"API key successfully saved to {saved_path}")
            return
        if args and args[0] in ("show", "path"):
            cfg_path = config.get_config_path()
            typer.echo(f"Config path: {cfg_path}")
            try:
                key = config.load_api_key()
                masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
                typer.echo(f"API key: configured ({masked})")
            except config.ConfigError:
                typer.echo("API key: not configured")
            return
        typer.echo("Usage: transcribe config set-key <YOUR_API_KEY> | transcribe config show", err=True)
        raise typer.Exit(2)

    # 4. Validate format
    if fmt not in output.FORMATS:
        typer.echo(f"Error: unknown format '{fmt}'. Use one of: {', '.join(sorted(output.FORMATS))}", err=True)
        raise typer.Exit(2)

    # 5. Parse options
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

    # 6. Handle 'transcribe live' syntax
    if first_arg == "live":
        try:
            is_offline = provider.strip().lower() in ("whisper", "local", "offline")
            if is_offline and options.speakers:
                console.print("[yellow]Note: speaker labels are not supported in local live mode; ignoring --speakers.[/yellow]")
                options = replace(options, speakers=False)

            session = LiveTranscriptionSession(
                language=language,
                device=device,
                mix_device=mix_device,
                require_api_key=not is_offline,
            )

            # A live recording can't be repeated, so protect it twice:
            # pick a free filename up front (never fail on a taken name), and
            # append every finished line to it immediately so a crash mid-session
            # loses nothing. The final formatted render replaces the file at the end.
            target: Path | None = None
            on_turn = None
            if not stdout:
                target = out or Path(f"live_session.{fmt}")
                if not force:
                    target = output.unique_path(target)
                target.parent.mkdir(parents=True, exist_ok=True)

                def on_turn(text: str, start_ms: int, end_ms: int) -> None:
                    mins, secs = divmod(start_ms // 1000, 60)
                    console.print(f"[dim]{mins:02d}:{secs:02d}[/dim] [bold]{text}[/bold]")
                    with target.open("a", encoding="utf-8") as f:
                        f.write(f"{mins:02d}:{secs:02d} {text}\n")

            if is_offline:
                engine = get_provider(provider, model=model)
                # Fast 'tiny' model for the live preview line and sentence
                # detection; the main model only transcribes committed lines.
                preview_engine = None
                if (model or "").strip() not in ("", "tiny", "tiny.en"):
                    preview_engine = get_provider(provider, model="tiny")
                transcript = session.run_offline(
                    engine, options=options, on_turn=on_turn, preview_provider=preview_engine
                )
            else:
                transcript = session.run(on_turn=on_turn)

            if stdout:
                rendered = output.render(transcript, fmt, title="live_recording")
                typer.echo(rendered, nl=False)
            else:
                output.write(transcript, fmt, target, title="live_recording", force=True)
                typer.echo(f"Live transcript written to {target}")
                if notes:
                    write_notes_file(transcript, target)
            return
        except (config.ConfigError, TranscriptionError) as error:
            typer.echo(f"Error: {error}", err=True)
            raise typer.Exit(1)

    options = TranscriptionOptions(
        speakers=speakers or (speakers_expected is not None),
        speakers_expected=speakers_expected,
        language=language,
        auto_detect_language=auto_detect_language,
        summarize=summarize,
        chapters=chapters,
        boost_words=boost_words,
    )

    # 7. Validate local files and load provider
    try:
        for item in files:
            if not is_url(item):
                validate_audio_file(Path(item))
        engine = get_provider(provider, model=model)
    except (config.ConfigError, TranscriptionError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1)

    # 8. Process files and URLs
    is_batch = len(files) > 1
    for item in files:
        try:
            if is_url(item):
                # Download audio track from web URL
                with console.status(f"[bold cyan]Downloading audio from URL: {item}...[/bold cyan]", spinner="dots"):
                    with url_audio_source(item) as (audio_path, title):
                        with optimized_audio_source(audio_path, compress=compress) as final_audio:
                            transcript = transcribe_file(final_audio, engine, options=options)
                            if stdout:
                                rendered = output.render(transcript, fmt, title=title)
                                typer.echo(rendered, nl=False)
                            else:
                                target = resolve_target_path(title, out, fmt, is_batch=is_batch)
                                output.write(transcript, fmt, target, title=title, force=force)
                                if transcript.audio_duration_s is not None:
                                    minutes = transcript.audio_duration_s / 60
                                    typer.echo(f"Audio length: {minutes:.1f} min")
                                typer.echo(f"Transcript written to {target}")
                                if notes:
                                    write_notes_file(transcript, target)
            else:
                file = Path(item)
                with optimized_audio_source(file, compress=compress) as audio_path:
                    if stdout:
                        transcript = transcribe_file(audio_path, engine, options=options)
                        rendered = output.render(transcript, fmt, title=file.stem)
                        typer.echo(rendered, nl=False)
                    else:
                        with console.status(f"[bold cyan]Transcribing {file.name}...[/bold cyan]", spinner="dots"):
                            transcript = transcribe_file(audio_path, engine, options=options)
                        target = resolve_target_path(file, out, fmt, is_batch=is_batch)
                        output.write(transcript, fmt, target, title=file.stem, force=force)

                        if transcript.audio_duration_s is not None:
                            minutes = transcript.audio_duration_s / 60
                            typer.echo(f"Audio length: {minutes:.1f} min")
                        typer.echo(f"Transcript written to {target}")
                        if notes:
                            write_notes_file(transcript, target)
        except (TranscriptionError, FileExistsError, OSError) as error:
            typer.echo(f"Error ({item}): {error}", err=True)
            raise typer.Exit(1)
