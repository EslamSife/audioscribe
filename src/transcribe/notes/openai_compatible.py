from transcribe.transcription.model import TranscriptionError

from .base import Completion


class OpenAICompatibleNotesProvider:
    """Any server that speaks the OpenAI Chat Completions API: OpenAI, Gemini, Ollama, LM Studio, ..."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        self.model = model
        self._api_key = api_key
        # None means the official OpenAI endpoint
        self.base_url = base_url

    def complete(self, system: str, user: str) -> Completion:
        try:
            import openai
        except ImportError:
            raise TranscriptionError(
                "This notes provider needs the 'openai' package.\n"
                "Install it with:\n"
                "  uv sync --extra openai"
            )

        server = self.base_url or "the OpenAI API"
        client = openai.OpenAI(api_key=self._api_key, base_url=self.base_url)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except openai.AuthenticationError:
            raise TranscriptionError(f"{server} rejected the API key. Check NOTES_API_KEY.")
        except openai.NotFoundError:
            raise TranscriptionError(
                f"Model '{self.model}' was not found at {server}. Check NOTES_MODEL "
                "(for Ollama, download it first with: ollama pull <model>)."
            )
        except openai.APIConnectionError:
            raise TranscriptionError(
                f"Could not reach {server}. Check the address and that the server is running "
                "(for Ollama: ollama serve)."
            )
        except openai.APIStatusError as error:
            raise TranscriptionError(f"Notes API error ({error.status_code}): {error.message}")

        if not response.choices:
            return Completion(text="")
        choice = response.choices[0]
        return Completion(
            text=choice.message.content or "",
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            cut_off=choice.finish_reason == "length",
        )
