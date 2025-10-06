import runpod
import torch
import torchaudio
import base64
import io
import os
import numpy as np

# Set environment variables
os.environ["NO_TORCH_COMPILE"] = "1"

def generate_simple_audio(text, sample_rate=24000):
    """Generate simple audio for testing"""
    # Calculate duration based on text length (100ms per character)
    duration = max(1.0, len(text) * 0.1)
    
    # Generate time array
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple tone with variation based on text
    base_freq = 440  # A4 note
    text_hash = hash(text) % 1000
    frequency = base_freq + (text_hash % 200)
    
    # Generate sine wave
    audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
    
    # Add envelope to make it sound more natural
    envelope = torch.exp(-t * 1.5)  # Decay envelope
    audio = audio * envelope
    
    # Add some variation
    modulation = 0.1 * torch.sin(2 * torch.pi * 5 * t)  # Slow vibrato
    audio = audio * (1 + modulation)
    
    return audio

def handler(event):
    try:
        print(f"Processing request: {event}")
        
        # Extract input
        input_data = event.get("input", {})
        text = input_data.get("text", "Hello, this is a test")
        emotion = input_data.get("emotion", "empathetic")
        
        print(f"Generating audio for: '{text}' (emotion: {emotion})")
        
        # Generate simple audio
        audio_tensor = generate_simple_audio(text)
        
        # Ensure proper tensor shape (batch, channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Convert to WAV
        sample_rate = 24000
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sample_rate, format="wav")
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print(f"✅ Generated audio: {len(audio_base64)} chars base64")
        
        return {
            "audio_base64": audio_base64,
            "text": text,
            "emotion": emotion,
            "sample_rate": sample_rate,
            "duration": audio_tensor.shape[-1] / sample_rate,
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
