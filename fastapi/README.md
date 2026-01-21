# VoxCPM FastAPI Demo

This folder contains a minimal FastAPI demo server that wraps
`nanovllm_voxcpm.models.voxcpm.server.AsyncVoxCPMServerPool` for local testing and prototyping.

Features:

- Prompt management (store a reference audio clip + transcript in an in-memory prompt pool)
- Generation (`/generate` streams raw waveform bytes)
- LoRA load/enable/disable (optional)

Note: this is intentionally lightweight (no auth, no persistence) and assumes a CUDA-capable environment.

Security warning: this demo server is NOT secure. Some endpoints (e.g. `/lora/load`) accept filesystem paths and
access local files on the machine running the server. Do NOT deploy this service directly to production or expose
it to untrusted networks.

## Install (uv)

This repository uses `uv`.

1) Install the core dependencies at the repo root:

```bash
uv sync --frozen
```

2) Install FastAPI demo extras:

```bash
uv pip install -r fastapi/requirements.txt
```

## Configure Model Path

The server loads the model at startup. The default model path is configured in `fastapi/app.py`:

- `fastapi/app.py:124` sets `MODEL_PATH` (defaults to `~/VoxCPM1.5`)

Update it to point to your local model directory (or place the model at the default path).

If you want to automatically load LoRA on startup, configure:

- `fastapi/app.py:125` `LORA_PATH`
- `fastapi/app.py:138` `LORA_CONFIG`

## Start The Server

From the repo root:

```bash
uv run fastapi run fastapi/app.py --host 0.0.0.0 --port 8000
```

The first startup may take a while due to model loading.

OpenAPI docs:

- http://localhost:8000/docs

## Run The Client Example

`fastapi/client.py` sends concurrent `/generate` requests and writes the returned raw float32 audio into WAV files:

```bash
uv run python fastapi/client.py
```

Outputs: `test_*.wav` in the current working directory.

## API

### Health

`GET /health`

- Response: `{"status": "ok"}`

### Prompt Management

`POST /add_prompt`

Request body (JSON):

- `wav_base64`: base64-encoded bytes of the *entire audio file* (not a data URI)
- `wav_format`: container format for decoding (e.g. `wav`, `flac`, `mp3`; passed to torchaudio)
- `prompt_text`: transcript / prompt text for that audio

Response:

- `prompt_id`: id to reference this prompt in `/generate`

Minimal Python example (upload a local WAV as a prompt):

```python
import base64
import requests

with open("prompt.wav", "rb") as f:
    wav_b64 = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post(
    "http://localhost:8000/add_prompt",
    json={
        "wav_base64": wav_b64,
        "wav_format": "wav",
        "prompt_text": "your prompt transcript",
    },
)
print(resp.json())
```

Note: the example above uses `requests`. If you do not have it installed, use `aiohttp` (already used by `fastapi/client.py`) or `curl`.

`POST /remove_prompt`

Request body: `{"prompt_id": "..."}`. Removes the prompt from the in-memory pool.

### Generation (Streaming Raw Audio)

`POST /generate`

Request body (JSON):

- `target_text`: text to synthesize
- `prompt_id` (optional): id from `/add_prompt` (omit for zero-shot)
- `max_generate_length` (optional, default 2000)
- `temperature` (optional, default 1.0)
- `cfg_value` (optional, default 1.5)

Response:

- `Content-Type: audio/raw`
- The HTTP body is a streamed byte sequence
- The byte stream contains contiguous little-endian `float32` PCM samples (mono)
- The response headers describe the format:
  - `X-Audio-Sample-Rate: 44100`
  - `X-Audio-Channels: 1`
  - `X-Audio-DType: float32`

Parse example (write to a WAV file):

```python
import numpy as np
import requests
import soundfile as sf

r = requests.post("http://localhost:8000/generate", json={"target_text": "hello"})
r.raise_for_status()

wav = np.frombuffer(r.content, dtype=np.float32)
sf.write("out.wav", wav, 44100)
```

### LoRA Management (Optional)

Endpoints:

- `POST /lora/load`: `{"lora_path": "/path/to/lora"}`
- `POST /lora/set_enabled`: `{"enabled": true}` / `{"enabled": false}`
- `POST /lora/reset`

Note: whether LoRA structure is enabled is controlled by `LORA_CONFIG` in `fastapi/app.py`. If it is `None`, LoRA is disabled.

## Troubleshooting

- Slow startup: expected during model loading / GPU memory allocation. Check terminal logs.
- Not a WAV response: `/generate` returns `audio/raw` (float32 PCM). Convert it to WAV on the client side (see examples above).
