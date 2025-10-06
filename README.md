# CSM Serverless Endpoint for RunPod

[![Runpod](https://api.runpod.io/badge/jarvis138/csm-serverless)](https://console.runpod.io/hub/jarvis138/csm-serverless)

This is a serverless deployment of the [CSM (Conversational Speech Model)](https://github.com/SesameAILabs/csm) for Oviya Voice AI.

## Usage

POST to endpoint with:
```json
{
  "input": {
    "text": "Hello, this is a test",
    "emotion": "empathetic"
  }
}
```

Response:
```json
{
  "output": {
    "audio_base64": "...",
    "text": "Hello, this is a test",
    "sample_rate": 16000
  }
}
```
