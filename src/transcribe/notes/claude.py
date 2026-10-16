from transcribe.transcription.model import TranscriptionError

from .base import Completion

DEFAULT_MODEL = "claude-opus-5-5"

# Models that accept the server-side fallback: if a safety check declines the
# request, the API retries it on another model inside the same call.
_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class ClaudeNotesProvider:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.label = f"Claude ({model})"
        # None lets the SDK find credentials itself (ANTHROPIC_API_KEY or an `ant auth login` profile)
        self._api_key = api_key

    def complete(self, system: str, user: str) -> Completion:
        try:
            import anthropic
        except ImportError:
            raise TranscriptionError(
                "Notes with Claude need the 'anthropic' package.\nInstall it with:\n  uv add anthropic"
            )

        extra: dict = {}
        if self.model in _FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        client = anthropic.Anthropic(api_key=self._api_key)
        try:
            with client.beta.messages.stream(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                **extra,
            ) as stream:
                message = stream.get_final_message()
        except TypeError as error:
            # The SDK has no error class for "no credentials at all", only a plain
            # TypeError ("Could not resolve authentication method"). A test runs
            # the real SDK without a key, so a changed message fails CI, not users.
            if "authentication" not in str(error):
                raise
            raise _missing_key_error()
        except anthropic.AuthenticationError:
            raise _missing_key_error()
        except anthropic.APIConnectionError:
            raise TranscriptionError(
                "Could not reach the Claude API. Check your internet connection."
            )
        except anthropic.APIStatusError as error:
            raise TranscriptionError(f"Claude API error ({error.status_code}): {error.message}")

        if message.stop_reason == "refusal":
            raise TranscriptionError("Claude declined to write notes for this transcript.")
        return Completion(
            text="".join(block.text for block in message.content if block.type == "text"),
            input_tokens=message.usage.input_tokens,
            cut_off=message.stop_reason == "max_tokens",
        )


def _missing_key_error() -> TranscriptionError:
    return TranscriptionError(
        "Claude API key is missing or invalid.\n"
        "Set it with:  export ANTHROPIC_API_KEY=your_key\n"
        "Get a key at https://console.anthropic.com/settings/keys"
    )
