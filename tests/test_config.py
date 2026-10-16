import pytest

from transcribe import config


def test_load_api_key_from_env(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "test-env-key")
    assert config.load_api_key() == "test-env-key"


def test_load_api_key_from_custom_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    cfg_file = tmp_path / "config.json"
    config.save_api_key("saved-file-key", cfg_file)
    assert config.load_api_key(config_path=cfg_file) == "saved-file-key"


def test_load_api_key_missing_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    empty_cfg = tmp_path / "non_existent.json"
    with pytest.raises(config.ConfigError, match="ASSEMBLYAI_API_KEY is not set"):
        config.load_api_key(config_path=empty_cfg)


def test_saved_key_file_is_private(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{}")
    cfg_file.chmod(0o644)
    config.save_api_key("secret", cfg_file)
    assert cfg_file.stat().st_mode & 0o777 == 0o600


def test_broken_config_file_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text("{not json")
    with pytest.raises(config.ConfigError, match="Could not read"):
        config.load_api_key(config_path=cfg_file)
