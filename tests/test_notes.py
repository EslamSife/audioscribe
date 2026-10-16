from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from transcribe.notes import Completion, generate_notes, get_notes_provider
from transcribe.notes.claude import ClaudeNotesProvider
from transcribe.notes.factory import GEMINI_BASE_URL, NO_KEY_NEEDED, OLLAMA_BASE_URL
from transcribe.notes.openai_compatible import OpenAICompatibleNotesProvider
from transcribe.transcription.model import Segment, Transcript, TranscriptionError

TRANSCRIPT = Transcript(
    text="We ship on Friday.",
    segments=(Segment("We ship on Friday.", 0, 2000, speaker="Speaker 1"),),
)
NOTES_ENV = ("NOTES_PROVIDER", "NOTES_MODEL", "NOTES_BASE_URL", "NOTES_API_KEY")
KEY_ENV = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")


class FakeProvider:
    def __init__(self, completion: Completion) -> None:
        self.completion = completion
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.prompts.append((system, user))
        return self.completion


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate tests from the developer's real keys and .env file."""
    for name in NOTES_ENV + KEY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("transcribe.notes.factory.load_dotenv", lambda: None)
    return monkeypatch


# --- service ---


def test_generate_notes_sends_timestamped_transcript():
    provider = FakeProvider(Completion(text="# Release\n", input_tokens=10_000))

    notes = generate_notes(TRANSCRIPT, title="standup", provider=provider)

    assert notes.text == "# Release\n"
    assert notes.warnings == ()
    _, user = provider.prompts[0]
    assert "Recording name: standup" in user
    assert "[00:00] Speaker 1: We ship on Friday." in user


def test_generate_notes_warns_when_model_read_much_less_than_sent():
    long = Transcript(text="word " * 20_000)  # ~25k tokens sent
    provider = FakeProvider(Completion(text="# Note", input_tokens=4096))

    notes = generate_notes(long, title="meeting", provider=provider)

    assert len(notes.warnings) == 1
    assert "dropped the start" in notes.warnings[0]


def test_generate_notes_warns_when_note_is_cut_off():
    provider = FakeProvider(Completion(text="# Note", input_tokens=None, cut_off=True))

    notes = generate_notes(TRANSCRIPT, title="t", provider=provider)

    assert notes.warnings == (
        "The model reached its output limit, so the end of the note may be missing.",
    )


def test_generate_notes_rejects_empty_transcript_and_empty_note():
    with pytest.raises(TranscriptionError, match="empty"):
        generate_notes(Transcript(text="  "), title="t", provider=FakeProvider(Completion("x")))
    with pytest.raises(TranscriptionError, match="empty note"):
        generate_notes(TRANSCRIPT, title="t", provider=FakeProvider(Completion("  ")))


# --- factory ---


def test_factory_defaults_to_claude_opus(clean_env):
    provider = get_notes_provider()
    assert isinstance(provider, ClaudeNotesProvider)
    assert provider.model == "claude-opus-5-5"


def test_factory_ollama_needs_no_key(clean_env):
    clean_env.setenv("NOTES_PROVIDER", "ollama")
    clean_env.setenv("NOTES_MODEL", "my-local-model")

    provider = get_notes_provider()

    assert isinstance(provider, OpenAICompatibleNotesProvider)
    assert provider.base_url == OLLAMA_BASE_URL
    assert provider._api_key == NO_KEY_NEEDED


def test_factory_gemini_uses_gemini_key_and_url(clean_env):
    clean_env.setenv("NOTES_PROVIDER", "Gemini")
    clean_env.setenv("NOTES_MODEL", "some-gemini-model")
    clean_env.setenv("GEMINI_API_KEY", "g-key")

    provider = get_notes_provider()

    assert provider.base_url == GEMINI_BASE_URL
    assert provider._api_key == "g-key"


def test_factory_openai_without_key_fails_but_custom_server_does_not(clean_env):
    clean_env.setenv("NOTES_PROVIDER", "openai")
    clean_env.setenv("NOTES_MODEL", "some-model")
    with pytest.raises(TranscriptionError, match="OPENAI_API_KEY is not set"):
        get_notes_provider()

    clean_env.setenv("NOTES_BASE_URL", "http://localhost:1234/v1")  # e.g. LM Studio
    assert get_notes_provider()._api_key == NO_KEY_NEEDED


def test_factory_errors_are_clear(clean_env):
    clean_env.setenv("NOTES_PROVIDER", "ollama")
    with pytest.raises(TranscriptionError, match="NOTES_MODEL is not set"):
        get_notes_provider()

    clean_env.setenv("NOTES_PROVIDER", "nope")
    with pytest.raises(TranscriptionError, match="Unknown NOTES_PROVIDER 'nope'"):
        get_notes_provider()


# --- adapters ---


def test_openai_adapter_maps_response():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="# Note"), finish_reason="length")
        ],
        usage=SimpleNamespace(prompt_tokens=1234),
    )
    with patch("openai.OpenAI") as client_cls:
        client_cls.return_value.chat.completions.create.return_value = response
        provider = OpenAICompatibleNotesProvider("m", api_key="k", base_url=OLLAMA_BASE_URL)
        result = provider.complete("sys", "user")

    client_cls.assert_called_once_with(api_key="k", base_url=OLLAMA_BASE_URL)
    assert result == Completion(text="# Note", input_tokens=1234, cut_off=True)


def test_openai_adapter_explains_connection_error():
    import openai

    with patch("openai.OpenAI") as client_cls:
        client_cls.return_value.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )
        provider = OpenAICompatibleNotesProvider("m", api_key="k", base_url=OLLAMA_BASE_URL)
        with pytest.raises(TranscriptionError, match="ollama serve"):
            provider.complete("sys", "user")


def _claude_message(stop_reason: str, text: str = "# Note"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=500),
    )


def _patch_claude_stream(message=None, error=None):
    client = MagicMock()
    stream_ctx = client.beta.messages.stream.return_value
    if error:
        client.beta.messages.stream.side_effect = error
    else:
        stream_ctx.__enter__.return_value.get_final_message.return_value = message
    return patch("anthropic.Anthropic", return_value=client), client


def test_claude_adapter_maps_response_and_enables_fallback():
    patcher, client = _patch_claude_stream(_claude_message("end_turn"))
    with patcher:
        result = ClaudeNotesProvider().complete("sys", "user")

    assert result == Completion(text="# Note", input_tokens=500, cut_off=False)
    kwargs = client.beta.messages.stream.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["fallbacks"] == "default"


def test_claude_adapter_skips_fallback_for_other_models():
    patcher, client = _patch_claude_stream(_claude_message("end_turn"))
    with patcher:
        ClaudeNotesProvider(model="claude-haiku-4-5").complete("sys", "user")

    assert "fallbacks" not in client.beta.messages.stream.call_args.kwargs


def test_claude_adapter_refusal_raises():
    patcher, _ = _patch_claude_stream(_claude_message("refusal", text=""))
    with patcher, pytest.raises(TranscriptionError, match="declined"):
        ClaudeNotesProvider().complete("sys", "user")


def test_claude_adapter_missing_credentials_is_explained_but_other_type_errors_are_not():
    no_auth = TypeError("Could not resolve authentication method. Expected one of api_key ...")
    patcher, _ = _patch_claude_stream(error=no_auth)
    with patcher, pytest.raises(TranscriptionError, match="ANTHROPIC_API_KEY"):
        ClaudeNotesProvider().complete("sys", "user")

    patcher, _ = _patch_claude_stream(error=TypeError("unexpected keyword argument 'x'"))
    with patcher, pytest.raises(TypeError, match="unexpected keyword"):
        ClaudeNotesProvider().complete("sys", "user")


def test_claude_missing_key_is_detected_with_the_real_sdk(monkeypatch, tmp_path):
    """Runs the real SDK with no credentials anywhere, so a changed SDK error fails this test."""
    import os

    for name in list(os.environ):
        if name.startswith("ANTHROPIC_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    # Should anything still find a key, the request goes nowhere instead of to the real API
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")

    with pytest.raises(TranscriptionError, match="API key is missing"):
        ClaudeNotesProvider().complete("system", "user")


def test_providers_say_where_the_transcript_goes():
    assert ClaudeNotesProvider(model="m").label == "Claude (m)"
    assert OpenAICompatibleNotesProvider("qwen", "k", OLLAMA_BASE_URL).label == (
        f"qwen at {OLLAMA_BASE_URL}"
    )


# --- templates ---


@pytest.mark.parametrize(
    ("template", "sections"),
    [
        ("general", ["## Summary", "## Key Points", "## Action Items", "## Notable Quotes"]),
        ("standup", ["## Updates", "## Blockers", "- [ ]"]),
        ("design-review", ["## Options Discussed", "## Decisions", "## Draft ADR"]),
        ("incident", ["## Timeline", "## Root Cause", "## Follow-ups", "blameless"]),
    ],
)
def test_each_template_asks_for_its_sections_and_keeps_the_shared_rules(template, sections):
    provider = FakeProvider(Completion(text="# Note\n", input_tokens=10_000))

    generate_notes(TRANSCRIPT, title="call", provider=provider, template=template)

    system, _ = provider.prompts[0]
    assert "Never invent facts" in system
    for section in sections:
        assert section in system


def test_unknown_template_is_refused_with_the_valid_names():
    with pytest.raises(TranscriptionError, match="standup, design-review, incident"):
        generate_notes(
            TRANSCRIPT, title="call", provider=FakeProvider(Completion(text="x")), template="retro"
        )
