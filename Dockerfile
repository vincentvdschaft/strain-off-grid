FROM python:3.12-slim

# 1. Grab the uv binary
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

# -------------------------
# Core env configuration
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV KERAS_BACKEND=jax
ENV JAX_DEFAULT_MATMUL_PRECISION=float32
ENV PYTHONPATH=/workdir/src

# -------------------------
# System dependencies
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb-cursor0 \
    libfontconfig1 \
    ca-certificates \
    curl \
    wget \
    ssh \
    && rm -rf /var/lib/apt/lists/*


# -------------------------
# Workspace
# -------------------------
WORKDIR /workdir

# Copy project files (including Rust source if applicable)
COPY pyproject.toml README.md ./
# If your Rust code is in a folder (e.g., ./rust_src), make sure to copy it!
COPY libs/ ./libs/
COPY src/ ./src/

RUN uv pip install --system -e .
RUN uv tool install ruff
RUN uv tool install ruff && uv tool install ty

CMD ["bash"]