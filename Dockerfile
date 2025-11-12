# AI Image Captioner - Production Dockerfile
# 
# Build arguments come from docker-compose.yml (which references version.json)
# See docker-compose.yml for current configuration values

# Build arguments for base image selection
ARG CUDA_BASE_VERSION
ARG CUDNN_SUFFIX

FROM nvidia/cuda:${CUDA_BASE_VERSION}-${CUDNN_SUFFIX}-runtime-ubuntu22.04

# Re-declare build arguments (ARG before FROM are not available after)
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG PYTORCH_VERSION
ARG PYTORCH_INDEX_URL
ARG TORCHVISION_VERSION
ARG TORCHAUDIO_VERSION

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    RUNNING_IN_DOCKER=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    curl \
    jq \
    # OpenCV dependencies (required by doctr)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
    && python3 --version \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3 \
    && python3 -m pip --version

# Create app directory
WORKDIR /app

# Copy version config and requirements first (for layer caching)
COPY version.json .
COPY requirements.txt .

# Install Python dependencies
# Use PyTorch version from build args for complete version control
RUN pip3 install --no-cache-dir \
        torch==${PYTORCH_VERSION}+${CUDA_VERSION} \
        torchvision==${TORCHVISION_VERSION}+${CUDA_VERSION} \
        torchaudio==${TORCHAUDIO_VERSION}+${CUDA_VERSION} \
        --index-url ${PYTORCH_INDEX_URL} && \
    pip3 install --no-cache-dir onnxruntime-gpu>=1.16.0 && \
    pip3 install --no-cache-dir -r requirements.txt

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
