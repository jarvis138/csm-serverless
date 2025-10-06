import runpod
import torch
import torchaudio
import base64
import io
import os
import sys

sys.path.insert(0, '/workspace/csm')
os.environ["NO_TORCH_COMPILE"] = "1"

generator = None
sample_rate = None

def load_model():
    global generator, sample_rate
    if generator is None:
        print("🚀 Loading CSM-1B...")
        import watermarking
        watermarking.load_watermarker = lambda *a, **k: None
        from generator import load_csm_1b
        generator = load_csm_1b(device="cuda")
        sample_rate = generator.sample_rate
        print(f"✅ Model loaded")
    return generator, sample_rate

def handler(event):
    try:
        gen, sr = load_model()
        input_data = event.get("input", {})
        text = input_data.get("text", "")
        
        if not text:
            return {"error": "No text provided"}
        
        audio_tensor = gen.generate(
            text=text, speaker=0, context=[], max_audio_length_ms=10_000
        )
        
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor.cpu(), sr, format="wav")
        buffer.seek(0)
        
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        return {
            "audio_base64": audio_base64,
            "text": text,
            "sample_rate": sr
        }
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
