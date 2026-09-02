from transcribe import config
from transcribe.providers.assemblyai import AssemblyAIProvider
from transcribe.providers.base import TranscriptionProvider
from transcribe.providers.whisper import LocalWhisperProvider
from transcribe.transcription.model import TranscriptionError

PROVIDERS = ("assemblyai", "whisper")


def get_provider(
    name: str = "assemblyai",
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> TranscriptionProvider:
    provider_name = name.strip().lower()

    if provider_name in ("assemblyai", "aai", "cloud"):
        key = api_key or config.load_api_key()
        return AssemblyAIProvider(key)

    elif provider_name in ("whisper", "local", "offline"):
        model_size = model or "base"
        return LocalWhisperProvider(model_size=model_size)

    else:
        raise TranscriptionError(
            f"Unknown provider '{name}'. Supported providers: {', '.join(PROVIDERS)}"
        )
