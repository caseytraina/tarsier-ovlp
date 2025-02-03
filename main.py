from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
import requests
import io
import tempfile
from typing import Optional
import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from transformers import LlavaForConditionalGeneration
from models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
from dataset.processor import Processor
from contextlib import contextmanager

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models and processors
model = None
processor = None
model_lock = threading.Lock()  # Lock for model access
request_semaphore = asyncio.Semaphore(3)  # Limit concurrent requests to 3
executor = ThreadPoolExecutor(max_workers=3)  # Thread pool for CPU-bound tasks

@contextmanager
def temporary_video_file():
    """Context manager to ensure video file cleanup"""
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            temp_file = f.name
            yield temp_file
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception:
                pass

def load_model():
    global model, processor
    if model is None:
        print("Loading Tarsier model and processors...")
        model_config = LlavaConfig.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
        )
        model = TarsierForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            config=model_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        processor = Processor(MODEL_PATH, max_n_frames=8)
        print("Models loaded successfully!")

class GenerateRequest(BaseModel):
    instruction: str
    video_url: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0

async def download_video(url: str, temp_file: str):
    """Download video to a specific temporary file"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(temp_file, 'wb') as f:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

def process_video(prompt: str, video_path: str, request: GenerateRequest):
    """Process video and generate response - CPU/GPU bound operations"""
    with model_lock:  # Ensure exclusive model access
        inputs = processor(prompt, video_path, edit_prompt=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=(request.temperature > 0),
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                use_cache=True
            )
            
            response = processor.tokenizer.decode(
                outputs[0][inputs['input_ids'][0].shape[0]:],
                skip_special_tokens=True
            )
    
    return response

def process_text(request: GenerateRequest):
    """Process text-only request"""
    with model_lock:  # Ensure exclusive model access
        with torch.no_grad():
            inputs = processor(
                text=request.instruction,
                return_tensors="pt"
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                do_sample=(request.temperature > 0),
                temperature=request.temperature,
                max_new_tokens=request.max_new_tokens,
                top_p=request.top_p,
                use_cache=True
            )
            response = processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
    
    return response

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    async with request_semaphore:  # Limit concurrent requests
        try:
            if request.video_url:
                # Use context manager for video file handling
                with temporary_video_file() as video_path:
                    # Download video
                    await download_video(request.video_url, video_path)
                    
                    # Format prompt with video token
                    prompt = f"<video>\n{request.instruction}"
                    
                    # Process video in thread pool
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_video,
                        prompt,
                        video_path,
                        request
                    )
            else:
                # Process text-only request in thread pool
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    process_text,
                    request
                )
            
            return {"response": response.strip()}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import aiohttp
    uvicorn.run(app, host="0.0.0.0", port=8000) 