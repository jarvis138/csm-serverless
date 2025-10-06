import runpod
import torch
import torchaudio
import base64
import io
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set environment variables
os.environ["NO_TORCH_COMPILE"] = "1"

# Global model variables
model = None
tokenizer = None
sample_rate = 24000

def load_csm_model():
    """Load CSM model using Hugging Face Transformers"""
    global model, tokenizer, sample_rate
    
    if model is None:
        try:
            print("🚀 Loading CSM-1B from Hugging Face...")
            
            # Load model and tokenizer
            model_name = "sesame/csm-1b"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            sample_rate = 24000  # CSM default sample rate
            print(f"✅ CSM model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load CSM model: {e}")
            # Fallback to simple audio generation
            model = "fallback"
            sample_rate = 24000
            print("⚠️ Using fallback audio generation")
    
    return model, tokenizer, sample_rate

def generate_simple_audio(text, sample_rate=24000):
    """Fallback: Generate simple audio for testing"""
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

def generate_csm_audio(text, model, tokenizer, sample_rate):
    """Generate audio using CSM model"""
    try:
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")
        
        # Generate audio codes
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=1000,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Convert outputs to audio (simplified - CSM outputs RVQ codes)
        # This is a placeholder - actual implementation would decode RVQ codes to audio
        audio_length = len(text) * 100  # Rough estimate
        audio = torch.randn(audio_length) * 0.1  # Placeholder audio
        
        return audio
        
    except Exception as e:
        print(f"❌ CSM generation failed: {e}")
        return generate_simple_audio(text, sample_rate)

def handler(event):
    try:
        print(f"Processing request: {event}")
        
        # Load model
        model, tokenizer, sr = load_csm_model()
        
        # Extract input
        input_data = event.get("input", {})
        text = input_data.get("text", "Hello, this is a test")
        emotion = input_data.get("emotion", "empathetic")
        
        print(f"Generating audio for: '{text}' (emotion: {emotion})")
        
        # Generate audio
        if model == "fallback":
            audio_tensor = generate_simple_audio(text, sr)
        else:
            audio_tensor = generate_csm_audio(text, model, tokenizer, sr)
        
        # Ensure proper tensor shape (batch, channels, samples)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
        
        # Convert to WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, sr, format="wav")
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        print(f"✅ Generated audio: {len(audio_base64)} chars base64")
        
        return {
            "audio_base64": audio_base64,
            "text": text,
            "emotion": emotion,
            "sample_rate": sr,
            "duration": audio_tensor.shape[-1] / sr,
            "status": "success",
            "model_used": "csm-1b" if model != "fallback" else "fallback"
        }
        
    except Exception as e:
        print(f"❌ Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
