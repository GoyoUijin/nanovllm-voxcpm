import os

import pytest


def test_get_int_env_default(monkeypatch):
    from app.core.config import _get_int_env

    monkeypatch.delenv("NANOVLLM_X", raising=False)
    assert _get_int_env("NANOVLLM_X", 123) == 123

    monkeypatch.setenv("NANOVLLM_X", "")
    assert _get_int_env("NANOVLLM_X", 123) == 123


def test_get_int_env_invalid_raises(monkeypatch):
    from app.core.config import _get_int_env

    monkeypatch.setenv("NANOVLLM_X", "abc")
    with pytest.raises(RuntimeError, match="Invalid env NANOVLLM_X"):
        _get_int_env("NANOVLLM_X", 1)


def test_load_config_validates_mp3_ranges(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_MP3_BITRATE_KBPS", "0")
    with pytest.raises(RuntimeError, match="NANOVLLM_MP3_BITRATE_KBPS must be > 0"):
        load_config()

    monkeypatch.setenv("NANOVLLM_MP3_BITRATE_KBPS", "192")
    monkeypatch.setenv("NANOVLLM_MP3_QUALITY", "3")
    with pytest.raises(RuntimeError, match=r"NANOVLLM_MP3_QUALITY must be in \[0, 2\]"):
        load_config()


def test_load_config_requires_lora_id_when_uri_set(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_LORA_URI", "file:///tmp/lora")
    monkeypatch.delenv("NANOVLLM_LORA_ID", raising=False)
    with pytest.raises(RuntimeError, match="NANOVLLM_LORA_ID is required"):
        load_config()


def test_load_config_expands_user_paths(monkeypatch):
    from app.core.config import load_config

    monkeypatch.setenv("NANOVLLM_MODEL_PATH", "~/VoxCPM1.5")
    monkeypatch.setenv("NANOVLLM_CACHE_DIR", "~/.cache/nanovllm")
    cfg = load_config()
    assert cfg.model_path == os.path.expanduser("~/VoxCPM1.5")
    assert cfg.lora.cache_dir == os.path.expanduser("~/.cache/nanovllm")
