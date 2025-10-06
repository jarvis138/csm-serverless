FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir runpod torchaudio transformers huggingface_hub torchtune moshi silentcipher

# Clone CSM repository from official source
RUN git clone https://github.com/SesameAILabs/csm.git /workspace/csm

# Install CSM requirements
WORKDIR /workspace/csm
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV NO_TORCH_COMPILE=1
ENV PYTHONPATH=/workspace/csm:$PYTHONPATH

# Copy handler
COPY handler.py /workspace/handler.py

# Set working directory back
WORKDIR /workspace

# Start the serverless handler
CMD ["python", "-u", "/workspace/handler.py"]
