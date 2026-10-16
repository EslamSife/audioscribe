"""Choose the notes provider from environment variables (a .env file works too).

NOTES_PROVIDER  claude (default), openai, gemini or ollama
NOTES_MODEL     model name; required for every provider except claude
NOTES_BASE_URL  server address, to override the preset (e.g. LM Studio, OpenRouter)
NOTES_API_KEY   key for that server; falls back to ANTHROPIC_/OPENAI_/GEMINI_API_KEY
"""

import os

from dotenv import load_dotenv

from transcribe.transcription.model import TranscriptionError

from .base import NotesProvider
from .claude import DEFAULT_MODEL, ClaudeNotesProvider
from .openai_compatible import OpenAICompatibleNotesProvider

PROVIDERS = ("claude", "openai", "gemini", "ollama")

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
# Local servers don't check the key, but the OpenAI client refuses to start without one
NO_KEY_NEEDED = "not-needed"


def get_notes_provider() -> NotesProvider:
    load_dotenv()
    name = (_env("NOTES_PROVIDER") or "claude").lower()
    model = _env("NOTES_MODEL")
    base_url = _env("NOTES_BASE_URL")
    api_key = _env("NOTES_API_KEY")

    if name in ("claude", "anthropic"):
        return ClaudeNotesProvider(model=model or DEFAULT_MODEL, api_key=api_key)
    if name not in PROVIDERS:
        raise TranscriptionError(
            f"Unknown NOTES_PROVIDER '{name}'. Supported: {', '.join(PROVIDERS)}"
        )
    if not model:
        raise TranscriptionError(
            f"NOTES_MODEL is not set. Choose a {name} model, e.g.:  export NOTES_MODEL=<model-name>"
        )

    if name == "gemini":
        base_url = base_url or GEMINI_BASE_URL
        api_key = api_key or _env("GEMINI_API_KEY")
        if not api_key:
            raise TranscriptionError(
                _missing_key("GEMINI_API_KEY", "https://aistudio.google.com/apikey")
            )
    elif name == "ollama":
        base_url = base_url or OLLAMA_BASE_URL
        api_key = api_key or NO_KEY_NEEDED
    else:
        api_key = api_key or _env("OPENAI_API_KEY")
        if not api_key:
            if base_url is None:
                raise TranscriptionError(
                    _missing_key("OPENAI_API_KEY", "https://platform.openai.com/api-keys")
                )
            api_key = NO_KEY_NEEDED

    return OpenAICompatibleNotesProvider(model=model, api_key=api_key, base_url=base_url)


def _env(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def _missing_key(env_name: str, url: str) -> str:
    return f"{env_name} is not set.\nGet a key at {url} and run:  export {env_name}=your_key"
