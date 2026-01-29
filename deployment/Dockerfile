ARG CUDA_BUILD_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

FROM ${CUDA_BUILD_IMAGE} AS builder

# Re-declare build args for this stage so they can be used in ENV/RUN.
ARG TORCH_CUDA_ARCH_LIST

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    UV_NO_PROGRESS=1

WORKDIR /app

# Build deps for native extensions (e.g. flash-attn) and audio libs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      ninja-build \
      git \
      python3 \
      python3-dev \
      python3-pip \
      python3-venv \
      pkg-config \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

COPY . .

# Install workspace packages from the frozen lockfile.
RUN uv sync --all-packages --no-dev -v --compile-bytecode --no-cache --no-editable

FROM ${CUDA_RUNTIME_IMAGE}
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg build-essential python3-dev python3-pip \
&& rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv

ENV NANOVLLM_CACHE_DIR=/var/cache/nanovllm

RUN useradd -m -u 10001 appuser \
 && mkdir -p "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser "$NANOVLLM_CACHE_DIR" \
 && chown -R appuser:appuser /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/deployment /app/deployment
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml /app/pyproject.toml
COPY --from=builder --chown=appuser:appuser /app/uv.lock /app/uv.lock

EXPOSE 8000

USER appuser

# k8s probes can use /health and /ready.
CMD ["uv", "run", "--no-sync", "fastapi", "run", "deployment/app/main.py", "--host", "0.0.0.0", "--port", "8000"]
