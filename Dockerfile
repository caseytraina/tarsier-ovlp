FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MODEL_PATH="omni-research/Tarsier-34b"
ENV MAX_N_FRAMES=8
ENV HF_HOME=/mnt/models/tarsier
ENV TORCH_HOME=/mnt/models/tarsier
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DOWNLOAD_WORKERS=8
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV AIP_HTTP_PORT=8000
ENV AIP_HEALTH_ROUTE=/health

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    build-essential \
    python3-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create model cache directory with proper permissions
RUN mkdir -p /mnt/models/tarsier && \
    chmod 755 /mnt/models/tarsier

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install hf_transfer for faster downloads
RUN pip install --no-cache-dir hf_transfer

# Install specific version of flash-attention
RUN pip install flash-attn==2.3.6 --no-build-isolation

# Create cache directory with proper permissions
RUN mkdir -p /mnt/models/tarsier && \
    chmod -R 777 /mnt/models/tarsier

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE ${AIP_HTTP_PORT}

# Use ENTRYPOINT to ensure the server runs in the foreground
ENTRYPOINT ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0"]
CMD ["--port", "8000"] 