import argparse
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import json
import time

try:
    import openvino_genai as ov_genai
except ImportError:
    ov_genai = None

app = FastAPI(title="Pistak OpenAI Compatible Server")

pipeline = None

@app.on_event("startup")
async def load_model():
    global pipeline
    args = app.state.args
    if not ov_genai:
        print("WARNING: openvino_genai not installed. Server running in dummy mode.")
        return
    
    print(f"Loading model from {args.model_path} on {args.device}...")
    try:
        # Load the pipeline
        # Using openvino_genai LLMPipeline
        pipeline = ov_genai.LLMPipeline(args.model_path, args.device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global pipeline
    if not pipeline and ov_genai:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # In dummy mode (if openvino_genai is missing or failed to load)
    if not pipeline:
        return {"id": "chatcmpl-123", "object": "chat.completion", "created": int(time.time()), 
                "model": "dummy-model", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Dummy response from Pistak server."}, "finish_reason": "stop"}]}

    # Construct prompt from messages (simple handling, a robust one would use tokenizer chat template)
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = body.get("max_tokens", 512)
    config.temperature = body.get("temperature", 0.7)

    if stream:
        async def generate_stream():
            # A real streaming implementation requires an ov_genai.Streamer callback.
            # For simplicity in this demo, we generate fully and then yield.
            # In a production app, use the openvino_genai streamer.
            result = pipeline.generate(prompt, config)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': result}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        result = pipeline.generate(prompt, config)
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "pistak-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}]
        }

def main():
    parser = argparse.ArgumentParser(description="Pistak Inference Server")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the OpenVINO model")
    parser.add_argument("--device", type=str, default="CPU", help="Target device (CPU, GPU, NPU)")
    parser.add_argument("--port", type=int, default=1234, help="Server port")
    args = parser.parse_args()

    app.state.args = args
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
