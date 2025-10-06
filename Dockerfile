FROM runpod/pytorch:2.4.0-py3.10-cuda12.1-devel

WORKDIR /workspace

# Install system dependencies exactly as per official repo
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies exactly as per official repo
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables exactly as per official repo
ENV NO_TORCH_COMPILE=1
ENV HF_HOME=/workspace/.cache/huggingface

# Copy CSM files
COPY . /workspace/

# Start the serverless handler
CMD ["python", "-u", "/workspace/handler.py"]
