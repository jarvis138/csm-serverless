FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir runpod torchaudio transformers>=4.52.1 huggingface_hub accelerate

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV HF_HOME=/workspace/.cache/huggingface

# Copy handler
COPY handler.py /workspace/handler.py

# Start the serverless handler
CMD ["python", "-u", "/workspace/handler.py"]
