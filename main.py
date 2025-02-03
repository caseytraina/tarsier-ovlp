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
        
        # Process each frame
        processed_frames = []
        for frame in video_frames:
            frame_pil = Image.fromarray(frame)
            processed = processor.image_processor(frame_pil, return_tensors="pt")["pixel_values"]
            processed_frames.append(processed)
            
        # Stack all processed frames
        return torch.cat(processed_frames, dim=0)
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
                # Process the text input
                text_tokens = tokenizer(
                    request.instruction,
                    return_tensors="pt",
                    add_special_tokens=True
                ).to(device)
                
                # Prepare the full input
                inputs = {
                    "input_ids": text_tokens["input_ids"],
                    "attention_mask": text_tokens["attention_mask"],
                    "pixel_values": processed_frames.unsqueeze(0).to(device)  # Add batch dimension
                }
                
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=(request.temperature > 0)
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Text-only generation
            with torch.no_grad():
                inputs = tokenizer(
                    request.instruction,
                    return_tensors="pt",
                    add_special_tokens=True
                ).to(device)
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=(request.temperature > 0)
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 