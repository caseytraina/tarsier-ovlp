from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
import io
import decord
import numpy as np
from PIL import Image
from typing import Optional
import os
from transformers import AutoProcessor, CLIPVisionModel, CLIPImageProcessor
from transformers import AutoTokenizer, LlavaForConditionalGeneration
import tempfile

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")  # Using the official Tarsier model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models and processors
model = None
tokenizer = None
processor = None

def load_model():
    global model, tokenizer, processor
    if model is None:
        print("Loading Tarsier model and processors...")
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"  # This will automatically handle multi-GPU or single-GPU
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("Models loaded successfully!")

class GenerateRequest(BaseModel):
    instruction: str
    video_url: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    
def download_video(url: str) -> str:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
            return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download video: {str(e)}")

def process_video(video_path: str):
    try:
        # Use decord to load video frames
        video_reader = decord.VideoReader(video_path)
        # Sample 8 frames evenly from the video
        frame_indices = list(range(0, len(video_reader), len(video_reader)//8))[:8]
        video_frames = video_reader.get_batch(frame_indices).asnumpy()
        
        # Convert frames to PIL images and normalize
        pil_frames = []
        for frame in video_frames:
            # Ensure frame is in uint8 range [0, 255]
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(frame))
        
        # Process all frames at once with dummy text
        processed = processor(
            images=pil_frames,
            text="Analyze this image",  # Dummy text to satisfy the processor
            return_tensors="pt"
        )
        
        return processed.pixel_values.to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        if request.video_url:
            # Download and process video
            video_path = download_video(request.video_url)
            processed_frames = process_video(video_path)
            
            # Clean up temporary file
            os.unlink(video_path)
            
            # Process frames and generate response
            with torch.no_grad():
                # Format the conversation prompt
                prompt = f"USER: {request.instruction}\nASSISTANT:"
                
                # Process text and images together
                inputs = processor(
                    text=prompt,
                    images=processed_frames,
                    return_tensors="pt"
                ).to(device)
                
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=(request.temperature > 0)
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the assistant's response
                if "ASSISTANT:" in response:
                    response = response.split("ASSISTANT:", 1)[1].strip()
        else:
            # Text-only generation
            with torch.no_grad():
                # Format the conversation prompt
                prompt = f"USER: {request.instruction}\nASSISTANT:"
                
                # Process text input
                inputs = processor(
                    text=prompt,
                    return_tensors="pt"
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=(request.temperature > 0)
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the assistant's response
                if "ASSISTANT:" in response:
                    response = response.split("ASSISTANT:", 1)[1].strip()
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 