FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies step by step to avoid conflicts
RUN pip install --no-cache-dir runpod

# Install core dependencies
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    tokenizers==0.21.0 \
    transformers==4.49.0 \
    huggingface_hub==0.28.1

# Install additional dependencies
RUN pip install --no-cache-dir \
    moshi==0.2.2 \
    torchtune==0.4.0 \
    torchao==0.9.0

# Install silentcipher from git (with error handling)
RUN pip install --no-cache-dir git+https://github.com/SesameAILabs/silentcipher@master || echo "silentcipher installation failed, continuing..."

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV HF_HOME=/workspace/.cache/huggingface

# Copy CSM files
COPY . /workspace/

# Start the serverless handler
CMD ["python", "-u", "/workspace/handler.py"]
