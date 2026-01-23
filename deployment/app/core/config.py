from __future__ import annotations

import os
from dataclasses import dataclass


def _get_int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError as e:
        raise RuntimeError(f"Invalid env {name}={v!r}; expected int") from e


@dataclass(frozen=True)
class Mp3Config:
    bitrate_kbps: int
    quality: int


@dataclass(frozen=True)
class LoRAStartupConfig:
    uri: str | None
    lora_id: str | None
    sha256: str | None
    cache_dir: str


@dataclass(frozen=True)
class ServiceConfig:
    model_path: str
    mp3: Mp3Config
    lora: LoRAStartupConfig


def load_config() -> ServiceConfig:
    model_path = os.path.expanduser(os.environ.get("NANOVLLM_MODEL_PATH", "~/VoxCPM1.5"))

    mp3_bitrate_kbps = _get_int_env("NANOVLLM_MP3_BITRATE_KBPS", 192)
    mp3_quality = _get_int_env("NANOVLLM_MP3_QUALITY", 2)
    if mp3_bitrate_kbps <= 0:
        raise RuntimeError("NANOVLLM_MP3_BITRATE_KBPS must be > 0")
    if mp3_quality < 0 or mp3_quality > 2:
        raise RuntimeError("NANOVLLM_MP3_QUALITY must be in [0, 2]")

    lora_uri = os.environ.get("NANOVLLM_LORA_URI")
    lora_id = os.environ.get("NANOVLLM_LORA_ID")
    lora_sha256 = os.environ.get("NANOVLLM_LORA_SHA256")
    cache_dir = os.path.expanduser(os.environ.get("NANOVLLM_CACHE_DIR", "~/.cache/nanovllm"))

    if lora_uri and not lora_id:
        raise RuntimeError("NANOVLLM_LORA_ID is required when NANOVLLM_LORA_URI is set")

    return ServiceConfig(
        model_path=model_path,
        mp3=Mp3Config(bitrate_kbps=mp3_bitrate_kbps, quality=mp3_quality),
        lora=LoRAStartupConfig(uri=lora_uri, lora_id=lora_id, sha256=lora_sha256, cache_dir=cache_dir),
    )
