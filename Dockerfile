# AI Image Captioner - Production Dockerfile
# 
# Build arguments come from docker-compose.yml (which references version.json)
# See docker-compose.yml for current configuration values

# Build arguments for base image selection
ARG CUDA_BASE_VERSION
ARG CUDNN_SUFFIX
ARG PYTHON_VERSION

# Build final image with CUDA + Python from Ubuntu repos
FROM nvidia/cuda:${CUDA_BASE_VERSION}-${CUDNN_SUFFIX}-runtime-ubuntu24.04

# Re-declare build arguments (ARG before FROM are not available after)
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG PYTORCH_INDEX_URL
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION
ARG FLASH_ATTENTION_VERSION
ARG FLASH_ATTENTION_CUDA_SUFFIX
ARG BUILD_HELPERS
ARG DOCTR_PACKAGE
ARG ONNXRUNTIME_PACKAGE

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    RUNNING_IN_DOCKER=1

# Install system dependencies including Python from Ubuntu repos
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates \
    # OpenCV dependencies (required by doctr)
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Verify Python installation
RUN python3 --version && pip3 --version

# Create app directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies following the same flow as setup-captioner-gpu.sh
# Step 1: Install PyTorch stack
RUN pip3 install --no-cache-dir --break-system-packages \
        torch==${PYTORCH_VERSION}+${CUDA_VERSION} \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION}+${CUDA_VERSION} \
        --index-url ${PYTORCH_INDEX_URL}

# Step 2: Install build helpers (e.g., ninja, packaging)
RUN if [ -n "${BUILD_HELPERS}" ]; then \
        echo "${BUILD_HELPERS}" | tr ',' '\n' | xargs pip3 install --no-cache-dir --break-system-packages; \
    fi

# Step 3: Install FlashAttention wheel
RUN PY_ABI_TAG=$(echo "cp${PYTHON_VERSION}" | tr -d '.') && \
    FA_WHEEL="flash_attn-${FLASH_ATTENTION_VERSION}+${FLASH_ATTENTION_CUDA_SUFFIX}-${PY_ABI_TAG}-${PY_ABI_TAG}-linux_x86_64.whl" && \
    FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTENTION_VERSION}/${FA_WHEEL}" && \
    wget -q "$FA_URL" && \
    pip3 install --no-cache-dir --break-system-packages "$FA_WHEEL" --no-build-isolation && \
    rm "$FA_WHEEL"

# Step 4: Install doctr package
RUN if [ -n "${DOCTR_PACKAGE}" ]; then \
        pip3 install --no-cache-dir --break-system-packages ${DOCTR_PACKAGE}; \
    fi

# Step 5: Install onnxruntime-gpu
RUN if [ -n "${ONNXRUNTIME_PACKAGE}" ]; then \
        pip3 install --no-cache-dir --break-system-packages ${ONNXRUNTIME_PACKAGE}; \
    fi

# Step 6: Install project dependencies from requirements.txt
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create directory for user config
RUN mkdir -p /app/data

# Expose Flask port
EXPOSE 5000

# Set working directory to backend
WORKDIR /app/backend

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python3", "app.py"]
