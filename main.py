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
from transformers import AutoModelForCausalLM, AutoTokenizer
import tempfile

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", os.path.expanduser("~/model_cache"))  # Default to model_cache in home directory
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class GenerateRequest(BaseModel):
    video_url: Optional[str] = None
    text: str
    
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
        video_frames = video_reader.get_batch(list(range(0, len(video_reader), len(video_reader)//8))).asnumpy()
        
        # Process frames here according to model requirements
        # This is a placeholder - adjust according to your model's specific requirements
        processed_frames = []
        for frame in video_frames:
            frame_pil = Image.fromarray(frame)
            # Add any necessary preprocessing steps here
            processed_frames.append(frame_pil)
            
        return processed_frames
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
            frames = process_video(video_path)
            
            # Clean up temporary file
            os.unlink(video_path)
            
            # Generate response using the model
            # This is a placeholder - implement according to your model's specific requirements
            with torch.no_grad():
                # Process frames and text together
                response = model.generate(
                    frames,
                    request.text,
                    max_length=100,
                    num_return_sequences=1
                )
        else:
            # Text-only generation
            with torch.no_grad():
                inputs = tokenizer(request.text, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=100)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 