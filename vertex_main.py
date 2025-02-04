import os
# Set cache directory before importing transformers
os.environ['HF_HOME'] = '/mnt/models/tarsier'  # This is the new recommended way

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import requests
import io
import tempfile
from typing import Optional
import threading
import asyncio
import aiohttp
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from transformers import LlavaForConditionalGeneration
from models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
from dataset.processor import Processor
from contextlib import contextmanager, asynccontextmanager
from huggingface_hub import HfApi

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure parallel downloads
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
os.environ['HF_HUB_DOWNLOAD_WORKERS'] = str(max(1, multiprocessing.cpu_count() // 2))

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPU Memory Configuration - 40GB split across 2 GPUs
max_memory = {0: "40GB", 1: "40GB"}

# Initialize models and processors
model = None
processor = None
model_lock = threading.Lock()
request_semaphore = asyncio.Semaphore(3)
executor = ThreadPoolExecutor(max_workers=3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Loading model on startup...")
    load_model()
    yield
    # Shutdown
    print("Cleaning up on shutdown...")
    global model, processor
    model = None
    processor = None

app = FastAPI(lifespan=lifespan)

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
        # Set explicit dtype for Flash Attention 2.0
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        model = TarsierForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            config=model_config,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",  # New recommended way to enable Flash Attention 2
            trust_remote_code=True
        )
        model.eval()
        processor = Processor(MODEL_PATH, max_n_frames=8)
        print(f"Models loaded successfully with dtype: {dtype}")

class GenerateRequest(BaseModel):
    instruction: str
    video_url: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request path: {request.url.path}")
    logger.info(f"Request method: {request.method}")
    try:
        body = await request.json()
        logger.info(f"Request body: {body}")
    except:
        pass
    response = await call_next(request)
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def download_video(url: str, temp_file: str):
    """Download video to a specific temporary file"""
    logger.info(f"Starting video download from {url}")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                with open(temp_file, 'wb') as f:
                    total_size = 0
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        total_size += len(chunk)
                logger.info(f"Video downloaded successfully: {total_size} bytes")
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

def process_video(prompt: str, video_path: str, request: GenerateRequest):
    """Process video and generate response - CPU/GPU bound operations"""
    try:
        logger.info(f"Processing video from {video_path} with prompt: {prompt}")
        with model_lock:  # Ensure exclusive model access
            logger.info("Processing video with model...")
            inputs = processor(prompt, video_path, edit_prompt=True)
            logger.info("Video processed by processor")
            
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
            logger.info("Inputs moved to device")
            
            with torch.no_grad():
                logger.info("Generating response...")
                outputs = model.generate(
                    **inputs,
                    do_sample=(request.temperature > 0),
                    temperature=request.temperature,
                    max_new_tokens=request.max_new_tokens,
                    top_p=request.top_p,
                    use_cache=True
                )
                logger.info("Response generated")
                
                response = processor.tokenizer.decode(
                    outputs[0][inputs['input_ids'][0].shape[0]:],
                    skip_special_tokens=True
                )
                logger.info("Response decoded")
        
        return response
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

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

@app.post("/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received request: {request}")
    async with request_semaphore:  # Limit concurrent requests
        try:
            if request.video_url:
                logger.info(f"Processing video request: {request.video_url}")
                # Use context manager for video file handling
                with temporary_video_file() as video_path:
                    # Download video
                    await download_video(request.video_url, video_path)
                    
                    # Format prompt with video token
                    prompt = f"<video>\n{request.instruction}"
                    logger.info(f"Processing with prompt: {prompt}")
                    
                    # Process video in thread pool
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_video,
                        prompt,
                        video_path,
                        request
                    )
                    logger.info("Video processing completed")
            else:
                logger.info("Processing text-only request")
                # Process text-only request in thread pool
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    process_text,
                    request
                )
                logger.info("Text processing completed")
            
            return {"response": response.strip()}
        
        except Exception as e:
            logger.error(f"Error in generate endpoint: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 