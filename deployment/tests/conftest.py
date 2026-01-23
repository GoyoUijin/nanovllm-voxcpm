import sys
from pathlib import Path

import pytest


# Ensure `import app...` resolves to deployment/app.
DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))


# Skip the entire deployment test suite if optional runtime deps are missing.
pytest.importorskip("fastapi")
pytest.importorskip("starlette")
pytest.importorskip("prometheus_client")


class FakeServerPool:
    """CPU-safe fake for AsyncVoxCPMServerPool used by lifespan."""

    def __init__(self, *args, **kwargs):
        self._stopped = False
        self._lora_loaded = False

    async def wait_for_ready(self):
        return None

    async def stop(self):
        self._stopped = True

    async def get_model_info(self):
        return {
            "sample_rate": 16000,
            "channels": 1,
            "feat_dim": 64,
            "patch_size": 2,
            "model_path": "/fake/model",
        }

    async def encode_latents(self, wav: bytes, wav_format: str):
        # Deterministic fake float32 bytes (shape doesn't matter for HTTP layer).
        import numpy as np

        arr = np.arange(0, 64, dtype=np.float32)
        return arr.tobytes()

    async def generate(
        self,
        target_text: str,
        prompt_latents: bytes | None = None,
        prompt_text: str = "",
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.5,
    ):
        import numpy as np

        yield np.zeros((160,), dtype=np.float32)
        yield np.ones((160,), dtype=np.float32) * 0.5

    async def load_lora(self, path: str):
        self._lora_loaded = True

    async def set_lora_enabled(self, enabled: bool):
        return None


@pytest.fixture
def app(monkeypatch):
    import app.core.lifespan as lifespan

    monkeypatch.setattr(lifespan, "AsyncVoxCPMServerPool", FakeServerPool)

    from app.main import create_app

    return create_app()
