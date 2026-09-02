import json
import os
from pathlib import Path

from dotenv import load_dotenv

API_KEY_ENV = "ASSEMBLYAI_API_KEY"


class ConfigError(Exception):
    pass


def get_config_path() -> Path:
    return Path.home() / ".config" / "transcribe" / "config.json"


def save_api_key(key: str, path: Path | None = None) -> Path:
    target = path or get_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"api_key": key.strip()}, indent=2) + "\n", encoding="utf-8")
    return target


def load_api_key(config_path: Path | None = None) -> str:
    # 1. Try loading from .env file
    load_dotenv()

    # 2. Check environment variable
    key = os.environ.get(API_KEY_ENV, "").strip()
    if key:
        return key

    # 3. Check config file (~/.config/transcribe/config.json)
    cfg_file = config_path or get_config_path()
    if cfg_file.exists() and cfg_file.is_file():
        try:
            data = json.loads(cfg_file.read_text(encoding="utf-8"))
            file_key = (data.get("api_key") or data.get("assemblyai_api_key") or "").strip()
            if file_key:
                return file_key
        except Exception:
            pass

    raise ConfigError(
        f"{API_KEY_ENV} is not set.\n"
        f"Get a key at https://www.assemblyai.com and either:\n"
        f'  - Run: export {API_KEY_ENV}="your-key"\n'
        f"  - Run: transcribe config set-key your-key\n"
        f"  - Add {API_KEY_ENV}=your-key to your .env file"
    )

