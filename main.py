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
from typing import Optional, Dict
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
import psutil

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure parallel downloads
# Disable HF_TRANSFER until we ensure it's installed
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "0"
os.environ['HF_HUB_DOWNLOAD_WORKERS'] = str(max(1, multiprocessing.cpu_count() // 2))

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models and processors
model = None
processor = None
model_lock = threading.Lock()
request_semaphore = asyncio.Semaphore(3)
executor = ThreadPoolExecutor(max_workers=3)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        print("Loading model on startup...")
        load_model()
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise
    finally:
        # Shutdown
        print("Cleaning up on shutdown...")
        global model, processor
        model = None
        processor = None

# Create FastAPI app with lifespan
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

def calculate_max_memory() -> Dict[str, str]:
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    # A100 has 40GB or 80GB variants - let's be conservative and reserve some memory
    # gpu_memory_reserve = "35GB"  # For 40GB A100
    gpu_memory_reserve = "75GB"  # For 80GB A100
    
    # Create memory map for all available GPUs
    max_memory = {i: gpu_memory_reserve for i in range(num_gpus)}
    
    return max_memory

def load_model():
    global model, processor
    if model is None:
        try:
            print("Loading Tarsier model and processors...")
            logger.info("Attempting to load model configuration...")
            
            # First try to load config
            try:
                model_config = LlavaConfig.from_pretrained(
                    MODEL_PATH,
                    cache_dir="/mnt/models/tarsier"
                )
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
                raise
                
            # Set explicit dtype for Flash Attention 2.0
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"Using dtype: {dtype}")
            
            logger.info("Loading model with configuration...")
            model = TarsierForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                config=model_config,
                device_map="auto",
                max_memory=calculate_max_memory(),
                torch_dtype=dtype,
                attn_implementation="flash_attention_2",
                cache_dir="/mnt/models/tarsier"
            )
            
            # Tie weights before model goes into eval mode
            if hasattr(model, 'tie_weights'):
                model.tie_weights()
            
            model.eval()
            
            logger.info("Loading processor...")
            processor = Processor(MODEL_PATH, max_n_frames=8)
            logger.info(f"Models loaded successfully with dtype: {dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            raise

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

# Add Vertex AI health check endpoint
@app.get("/v1/endpoints/{endpoint_id}/deployedModels/{deployed_model_id}")
async def health_check(endpoint_id: str, deployed_model_id: str):
    """Health check endpoint for Vertex AI"""
    logger.info(f"Health check request received for endpoint_id: {endpoint_id}, deployed_model_id: {deployed_model_id}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=200,
        content={
            "state": "AVAILABLE",
            "deployment_state": "DEPLOYED"
        }
    )

# Add AIP-Health endpoint
@app.get("/health")
async def health():
    """Basic health check endpoint"""
    return {"status": "healthy"}

# Add Vertex AI prediction endpoint
@app.post("/v1/endpoints/{endpoint_id}/deployedModels/{deployed_model_id}:predict")
async def vertex_predict(endpoint_id: str, deployed_model_id: str, request: Request, background_tasks: BackgroundTasks):
    try:
        # Parse Vertex AI request format
        body = await request.json()
        instances = body.get("instances", [])
        if not instances or len(instances) == 0:
            raise HTTPException(status_code=400, detail="No instances provided")
        
        # Process first instance (we handle one at a time)
        instance = instances[0]
        
        # Convert Vertex AI format to our format
        generate_request = GenerateRequest(
            instruction=instance.get("instruction", ""),
            video_url=instance.get("video_url"),
            max_new_tokens=instance.get("max_new_tokens", 512),
            temperature=instance.get("temperature", 0.0),
            top_p=instance.get("top_p", 1.0)
        )
        
        # Process using existing generate endpoint logic
        async with request_semaphore:
            if generate_request.video_url:
                with temporary_video_file() as video_path:
                    await download_video(generate_request.video_url, video_path)
                    prompt = f"<video>\n{generate_request.instruction}"
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        process_video,
                        prompt,
                        video_path,
                        generate_request
                    )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    process_text,
                    generate_request
                )
        
        # Return in Vertex AI format
        return {
            "predictions": [{
                "response": response.strip()
            }]
        }
        
    except Exception as e:
        logger.error(f"Error in Vertex AI prediction endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")