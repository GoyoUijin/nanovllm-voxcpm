# Nano-vLLM-VoxCPM

An inference engine for VoxCPM based on Nano-vLLM.

Features:
- Faster than the pytorch implementation
- Support concurrent requests
- Friendly async API, easy to use in FastAPI (see [fastapi/app.py](fastapi/app.py))

## Installation

Nano-vLLM-VoxCPM is not available on PyPI yet, you need to install it from source.

```
git clone https://github.com/a710128/nanovllm-voxcpm.git
cd nanovllm-voxcpm
pip install -e .
```

## Basic Usage

See the [example.py](example.py) for a usage example.

## Acknowledgments

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)

## License

MIT License
