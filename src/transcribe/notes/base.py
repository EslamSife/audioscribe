from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    # Prompt tokens the model reports it actually read; None if the provider doesn't say
    input_tokens: int | None = None
    # True when the model stopped because it reached its output limit
    cut_off: bool = False


class NotesProvider(Protocol):
    """Any LLM that can answer one system + user prompt. Adapters raise TranscriptionError."""

    def complete(self, system: str, user: str) -> Completion: ...
