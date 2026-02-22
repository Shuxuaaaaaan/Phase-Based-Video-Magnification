FROM python:3.11-slim-bookworm

# Install required system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv tool directly from the official image
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation and set UV behavior
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

# Copy source code and resources
COPY src/ ./src/
COPY data/ ./data/

# Initialize uv sync. 
# It will skip cupy on arm64 and install it on amd64 based on the pyproject.toml marker
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Entrypoint default runs help. Can be overwritten safely.
ENTRYPOINT ["uv", "run", "python", "src/evm_phase.py"]
CMD ["-h"]
