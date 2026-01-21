# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API (can be wrapped by an HTTP server; see `fastapi/README.md`)

This repository contains a Python package (`nanovllm_voxcpm/`) plus an optional FastAPI demo.

## Installation

Nano-vLLM-VoxCPM is not available on PyPI yet. Install from source.

### Prerequisites

- Linux + NVIDIA GPU (CUDA)
- Python >= 3.10
- `flash-attn` is required (the package imports it at runtime)

The runtime is GPU-centric (Triton + FlashAttention). CPU-only execution is not supported.

### Install with uv (recommended)

This repo uses `uv` and includes a lockfile (`uv.lock`).

```bash
uv sync --frozen
```

Dev deps (tests):

```bash
uv sync --frozen --dev
```

Note: `flash-attn` may require additional system CUDA tooling depending on your environment.

## Basic Usage

See `example.py` for an end-to-end async example.

Quickstart:

```bash
uv run python example.py
```

### Load a model

`VoxCPM.from_pretrained(...)` accepts either:

- a local model directory path, or
- a HuggingFace repo id (it will download via `huggingface_hub.snapshot_download`).

The model directory is expected to contain:

- `config.json`
- one or more `*.safetensors` weight files
- `audiovae.pth` (VAE weights)

### Generate (async)

If you call `from_pretrained()` inside an async event loop, it returns an `AsyncVoxCPMServerPool`.

```python
import asyncio
import numpy as np

from nanovllm_voxcpm import VoxCPM


async def main() -> None:
    server = VoxCPM.from_pretrained(
        model="/path/to/VoxCPM",
        devices=[0],
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        gpu_memory_utilization=0.95,
    )
    await server.wait_for_ready()

    chunks = []
    async for chunk in server.generate(target_text="Hello world"):
        chunks.append(chunk)  # each chunk is a float32 numpy array

    wav = np.concatenate(chunks, axis=0)
    # Write with the model's sample rate (see your model's AudioVAE config; often 16000)
    # import soundfile as sf; sf.write("out.wav", wav, sample_rate)

    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

### Generate (sync)

If you call `from_pretrained()` outside an event loop, it returns a `SyncVoxCPMServerPool`.

```python
import numpy as np

from nanovllm_voxcpm import VoxCPM


server = VoxCPM.from_pretrained(model="/path/to/VoxCPM", devices=[0])
chunks = []
for chunk in server.generate(target_text="Hello world"):
    chunks.append(chunk)
wav = np.concatenate(chunks, axis=0)
server.stop()
```

### Prompting (optional)

The VoxCPM server supports three prompt modes:

- zero-shot: no prompt
- provide `prompt_latents` + `prompt_text`
- provide a stored `prompt_id` (via `add_prompt`) and then generate with that id

See the docstrings in `nanovllm_voxcpm/models/voxcpm/server.py` for details.

## FastAPI demo

The HTTP server demo is documented separately to keep this README focused:

- `fastapi/README.md`

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License

## Known Issue

If you see the errors below:
```
ValueError: Missing parameters: ['base_lm.embed_tokens.weight', 'base_lm.layers.0.self_attn.qkv_proj.weight', ... , 'stop_proj.weight', 'stop_proj.bias', 'stop_head.weight']
[rank0]:[W1106 07:26:04.469150505 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

It's because nanovllm loads model parameters from `*.safetensors`, but some VoxCPM releases ship weights as `.pt`.

Fix:

- use a safetensors-converted checkpoint (or convert the checkpoint yourself)
- ensure the `*.safetensors` files live next to `config.json` in the model directory
