# Stage 1: builder - install uv and build the virtual environment
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Install everything except torch/torchvision (and their CUDA-only transitive deps)
# from the frozen lock, then install torch/torchvision from the CPU-only PyTorch
# index: the default PyPI wheels drag in the full CUDA stack (~2GB) that this
# CPU-only serving image can't use, and that was timing out the build. This keeps
# local dev (which uses a GPU) on the regular lockfile-resolved torch, untouched.
RUN uv sync --frozen --no-dev --no-install-project \
    --no-install-package torch \
    --no-install-package torchvision \
    --no-install-package triton \
    --no-install-package nvidia-cublas \
    --no-install-package nvidia-cuda-cupti \
    --no-install-package nvidia-cuda-nvrtc \
    --no-install-package nvidia-cuda-runtime \
    --no-install-package nvidia-cudnn-cu13 \
    --no-install-package nvidia-cufft \
    --no-install-package nvidia-cufile \
    --no-install-package nvidia-curand \
    --no-install-package nvidia-cusolver \
    --no-install-package nvidia-cusparse \
    --no-install-package nvidia-cusparselt-cu13 \
    --no-install-package nvidia-nccl-cu13 \
    --no-install-package nvidia-nvjitlink \
    --no-install-package nvidia-nvshmem-cu13 \
    --no-install-package nvidia-nvtx
RUN uv pip install --python .venv/bin/python --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY src/ src/

# Stage 2: runtime - slim image with just the venv and app source
FROM python:3.13-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
