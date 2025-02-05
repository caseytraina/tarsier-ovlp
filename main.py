import os
# Set cache directory before importing transformers
os.environ['HF_HOME'] = '/mnt/models/tarsier'  # This is the new recommended way

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import requests
import io
import tempfile
from typing import Optional, Dict, List
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
import time

# Configure logging
import logging
import sys

# Create handlers for different log levels
stdout_handler = logging.StreamHandler(sys.stdout)
stderr_handler = logging.StreamHandler(sys.stderr)

# Set level filters
stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)  # INFO and below go to stdout
stderr_handler.setLevel(logging.WARNING)  # WARNING and above go to stderr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[stdout_handler, stderr_handler]
)
logger = logging.getLogger(__name__)

# Configure parallel downloads
# Disable HF_TRANSFER until we ensure it's installed
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "0"
os.environ['HF_HUB_DOWNLOAD_WORKERS'] = str(max(1, multiprocessing.cpu_count() // 2))

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")
NUM_GPUS = torch.cuda.device_count()
logger.info(f"Found {NUM_GPUS} GPUs")

# Initialize models and processors
models: List[TarsierForConditionalGeneration] = []
processors: List[Processor] = []
model_locks = [threading.Lock() for _ in range(NUM_GPUS)]  # One lock per GPU
model_in_use = [False] * NUM_GPUS  # Track which models are in use
model_lock = threading.Lock()  # Global lock for model selection
request_semaphore = asyncio.Semaphore(NUM_GPUS * 3)  # Scale with number of GPUs
executor = ThreadPoolExecutor(max_workers=NUM_GPUS * 3)  # Scale with number of GPUs

# Add model loading state tracking
model_loading = True

# Add conversation history management
class Conversation:
    def __init__(self, max_history: int = 10):
        self.messages = []
        self.max_history = max_history
        self.last_access = time.time()
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_history:
            self.messages.pop(0)
        self.last_access = time.time()
    
    def get_context(self) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages])

# Global conversation store with cleanup
conversations: Dict[str, Conversation] = {}
CONVERSATION_TIMEOUT = 3600  # 1 hour timeout

def cleanup_old_conversations():
    current_time = time.time()
    to_remove = []
    for conv_id, conv in conversations.items():
        if current_time - conv.last_access > CONVERSATION_TIMEOUT:
            to_remove.append(conv_id)
    for conv_id in to_remove:
        del conversations[conv_id]

def get_available_model():
    """Get the index of an available model, or None if all are in use"""
    with model_lock:
        for i in range(len(models)):
            if not model_in_use[i]:
                model_in_use[i] = True
                return i
        return None

def release_model(index):
    """Release a model back to the pool"""
    with model_lock:
        model_in_use[index] = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model_loading
    try:
        print("Loading models on startup...")
        model_loading = True
        load_models()  # Load multiple models
        model_loading = False
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        model_loading = False
        raise
    finally:
        # Shutdown
        print("Cleaning up on shutdown...")
        global models, processors
        models = []
        processors = []

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
    logger.info(f"Calculating max memory for {num_gpus} GPUs with reserve of {gpu_memory_reserve}")
    # Create memory map for all available GPUs
    max_memory = {i: gpu_memory_reserve for i in range(num_gpus)}
    
    return max_memory

def load_models():
    """Load one model instance per available GPU"""
    global models, processors
    if not models:
        try:
            print(f"Loading Tarsier models and processors on {NUM_GPUS} GPUs...")
            logger.info("Attempting to load model configuration...")
            
            # First try to load config
            try:
                logger.info(f"Loading config from {MODEL_PATH}")
                model_config = LlavaConfig.from_pretrained(
                    MODEL_PATH,
                    cache_dir="/mnt/models/tarsier",
                    trust_remote_code=True
                )
                logger.info("Config loaded successfully")
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}", exc_info=True)
                raise
                
            # Set explicit dtype for Flash Attention 2.0
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"Using dtype: {dtype}")
            
            # Load model instances
            for gpu_id in range(NUM_GPUS):
                logger.info(f"Loading model {gpu_id+1} on GPU {gpu_id}")
                try:
                    # Get GPU properties
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # Convert to GB
                    logger.info(f"GPU {gpu_id}: {gpu_name} with {gpu_memory:.1f}GB memory")
                    
                    # Calculate safe memory limit (leave some headroom)
                    memory_limit = f"{int(gpu_memory * 0.9)}GB"  # Use 90% of GPU memory
                    logger.info(f"Setting memory limit to {memory_limit} for GPU {gpu_id}")
                    
                    # Set current device before loading
                    torch.cuda.set_device(gpu_id)
                    
                    # Load model directly to GPU
                    model = TarsierForConditionalGeneration.from_pretrained(
                        MODEL_PATH,
                        config=model_config,
                        torch_dtype=dtype,
                        device_map=None,  # Disable automatic device mapping
                        use_flash_attention_2=True,  # Use this instead of attn_implementation
                        cache_dir="/mnt/models/tarsier",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                        offload_folder=None,  # Disable offloading
                        offload_state_dict=False  # Disable state dict offloading
                    ).to(f"cuda:{gpu_id}")  # Move entire model to GPU immediately
                    
                    # Force model to eval mode
                    model.eval()
                    
                    # Initialize processor
                    processor = Processor(MODEL_PATH, max_n_frames=8)
                    
                    # Verify all parameters are on correct device
                    for param in model.parameters():
                        if param.device != torch.device(f"cuda:{gpu_id}"):
                            param.data = param.data.to(f"cuda:{gpu_id}")
                    
                    models.append(model)
                    processors.append(processor)
                    logger.info(f"Model {gpu_id+1} loaded successfully on GPU {gpu_id}")
                    
                except Exception as e:
                    logger.error(f"Error loading model on GPU {gpu_id}: {str(e)}", exc_info=True)
                    raise
            
            logger.info(f"Successfully loaded {len(models)} models across {NUM_GPUS} GPUs with dtype: {dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}", exc_info=True)
            raise

class GenerateRequest(BaseModel):
    instruction: str
    video_url: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    conversation_id: Optional[str] = None  # Add conversation ID

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

def process_text(request: GenerateRequest):
    """Process text-only request with conversation history"""
    try:
        model_index = get_available_model()
        if model_index is None:
            raise HTTPException(status_code=503, detail="No available models")
        
        try:
            with model_locks[model_index]:
                with torch.no_grad():
                    # Get conversation history if available
                    conversation = None
                    if request.conversation_id and request.conversation_id in conversations:
                        conversation = conversations[request.conversation_id]
                        prompt = f"{conversation.get_context()}\nHuman: {request.instruction}\nAssistant:"
                    else:
                        prompt = f"Human: {request.instruction}\nAssistant:"

                    # Process with the full conversation context
                    inputs = processors[model_index](prompt).to(f"cuda:{model_index}")
                    
                    outputs = models[model_index].generate(
                        **inputs,
                        do_sample=(request.temperature > 0),
                        temperature=request.temperature,
                        max_new_tokens=request.max_new_tokens,
                        top_p=request.top_p,
                        use_cache=True
                    )
                    response = processors[model_index].tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Update conversation history
                    if request.conversation_id:
                        if request.conversation_id not in conversations:
                            conversations[request.conversation_id] = Conversation()
                        conversations[request.conversation_id].add_message("Human", request.instruction)
                        conversations[request.conversation_id].add_message("Assistant", response)
                    
                    return response
        finally:
            release_model(model_index)
    except Exception as e:
        logger.error(f"Error in process_text: {str(e)}", exc_info=True)
        raise

def process_video(prompt: str, video_path: str, request: GenerateRequest):
    """Process video and generate response - CPU/GPU bound operations"""
    try:
        logger.info(f"Processing video from {video_path} with prompt: {prompt}")
        model_index = get_available_model()
        if model_index is None:
            raise HTTPException(status_code=503, detail="No available models")
        
        try:
            with model_locks[model_index]:
                logger.info(f"Processing video with model {model_index}...")
                inputs = processors[model_index](prompt, video_path, edit_prompt=True)
                logger.info("Video processed by processor")
                
                inputs = {k: v.to(f"cuda:{model_index}") for k, v in inputs.items() if v is not None}
                logger.info("Inputs moved to device")
                
                with torch.no_grad():
                    logger.info("Generating response...")
                    outputs = models[model_index].generate(
                        **inputs,
                        do_sample=(request.temperature > 0),
                        temperature=request.temperature,
                        max_new_tokens=request.max_new_tokens,
                        top_p=request.top_p,
                        use_cache=True
                    )
                    logger.info("Response generated")
                    
                    response = processors[model_index].tokenizer.decode(
                        outputs[0][inputs['input_ids'][0].shape[0]:],
                        skip_special_tokens=True
                    )
                    logger.info("Response decoded")
            
            return response
        finally:
            release_model(model_index)
            
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

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

# Add health check endpoint that checks model readiness
@app.get(os.getenv("AIP_HEALTH_ROUTE", "/health"))
async def health():
    """Health check endpoint for Vertex AI.
    Returns:
        JSONResponse: 200 OK if model is loaded and ready to serve
        JSONResponse: 503 Service Unavailable if model is not ready or still loading
    """
    global models, processors, model_loading
    
    try:
        # Check if model is still loading during startup
        if model_loading:
            logger.info("Health check: Model is still loading")
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "reason": "Model is still loading"}
            )
            
        # Check if model and processor are initialized
        if not models or not processors:
            logger.warning("Health check failed: Model or processor not initialized")
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "reason": "Model not initialized"}
            )
            
        # Check if models are in eval mode and on correct device
        for model, processor in zip(models, processors):
            if not model.training and next(model.parameters()).is_cuda:
                # Additional check: try to get GPU memory info
                try:
                    gpu_memory = torch.cuda.memory_allocated()
                    if gpu_memory > 0:  # Model is loaded in GPU
                        logger.info(f"Health check passed: Model {models.index(model)+1} ready and GPU memory allocated")
                        return JSONResponse(
                            status_code=200,
                            content={"status": "healthy", "gpu_memory_allocated": str(gpu_memory)}
                        )
                except Exception as e:
                    logger.warning(f"GPU memory check failed for model {models.index(model)+1}: {str(e)}")
            
            # Even if GPU memory check fails, if model is on CUDA and in eval mode, we're probably okay
            logger.info(f"Health check passed: Model {models.index(model)+1} ready")
            return JSONResponse(
                status_code=200,
                content={"status": "healthy"}
            )
        
        logger.warning("Health check failed: Models not in proper state")
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": "Models not in proper state"}
        )
            
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": str(e)}
        )

# Add Vertex AI prediction endpoint
@app.post("/v1/endpoints/{endpoint_id}/deployedModels/{deployed_model_id}:predict")
async def vertex_predict(endpoint_id: str, deployed_model_id: str, request: Request, background_tasks: BackgroundTasks):
    try:
        # Cleanup old conversations
        cleanup_old_conversations()
        
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
            top_p=instance.get("top_p", 1.0),
            conversation_id=instance.get("conversation_id")  # Get conversation ID from request
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
        
        # Return in Vertex AI format with conversation ID
        return {
            "predictions": [{
                "response": response.strip(),
                "conversation_id": generate_request.conversation_id
            }]
        }
        
    except Exception as e:
        logger.error(f"Error in Vertex AI prediction endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

