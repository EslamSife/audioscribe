"""Turn a finished transcript into a structured Markdown note with an LLM (Claude, GPT, Gemini or a local model)."""

from .base import Completion, NotesProvider
from .factory import PROVIDERS, get_notes_provider
from .service import Notes, generate_notes

__all__ = [
    "Completion",
    "Notes",
    "NotesProvider",
    "PROVIDERS",
    "generate_notes",
    "get_notes_provider",
]
