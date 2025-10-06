import runpod
import torch
import torchaudio
import base64
import io
import os

# Set environment variables
os.environ["NO_TORCH_COMPILE"] = "1"

# Global CSM generator
csm_generator = None

def load_csm_model():
    """Load CSM model with error handling"""
    global csm_generator
    
    try:
        print("Loading CSM-1B model...")
        
        # Try to import CSM components
        try:
            from generator import load_csm_1b
        except ImportError as e:
            print(f"Failed to import generator: {e}")
            return False
        
        # Device selection
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        print(f"Using device: {device}")
        
        # Load model with error handling
        try:
            csm_generator = load_csm_1b(device=device)
            print(f"CSM model loaded successfully!")
            print(f"Sample rate: {csm_generator.sample_rate}")
            return True
        except Exception as e:
            print(f"Failed to load CSM model: {e}")
            return False
        
    except Exception as e:
        print(f"Error loading CSM model: {e}")
        import traceback
        traceback.print_exc()
        return False

def handler(event):
    """RunPod serverless handler with robust error handling"""
    try:
        print(f"Processing request: {event}")
        
        # Extract input - handle both "prompt" and "text" for compatibility
        input_data = event.get("input", {})
        text = input_data.get("prompt") or input_data.get("text", "Hello from Sesame.")
        emotion = input_data.get("emotion", "empathetic")
        
        print(f"Input received - Text: '{text}', Emotion: '{emotion}'")
        
        # Load model if not loaded
        if csm_generator is None:
            if not load_csm_model():
                return {
                    "error": "Failed to load CSM model - check dependencies",
                    "status": "error",
                    "debug_info": "Model loading failed"
                }
        
        # Generate audio
        print(f"Generating audio for: '{text}'")
        
        try:
            audio = csm_generator.generate(
                text=text,
                speaker=0,
                context=[],
                max_audio_length_ms=10_000,
            )
        except Exception as e:
            print(f"Audio generation failed: {e}")
            return {
                "error": f"Audio generation failed: {e}",
                "status": "error"
            }
        
        # Save audio
        try:
            buffer = io.BytesIO()
            torchaudio.save(buffer, audio.unsqueeze(0).cpu(), csm_generator.sample_rate, format="wav")
            buffer.seek(0)
            audio_bytes = buffer.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"Audio saving failed: {e}")
            return {
                "error": f"Audio saving failed: {e}",
                "status": "error"
            }
        
        print(f"Generated audio: {len(audio_bytes)} bytes")
        
        return {
            "audio_base64": audio_base64,
            "text": text,
            "emotion": emotion,
            "sample_rate": csm_generator.sample_rate,
            "duration": audio.shape[-1] / csm_generator.sample_rate,
            "status": "success",
            "model_used": "csm-1b"
        }
        
    except Exception as e:
        print(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "error"
        }

if __name__ == "__main__":
    print("Starting CSM Serverless Handler...")
    runpod.serverless.start({"handler": handler})
