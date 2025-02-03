from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
import io
import tempfile
from typing import Optional
import os
from transformers import AutoProcessor, LlavaForConditionalGeneration

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")  # Using the official Tarsier model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models and processors
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        print("Loading Tarsier model and processors...")
        model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        print("Models loaded successfully!")

class GenerateRequest(BaseModel):
    instruction: str
    video_url: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
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

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        if request.video_url:
            # Download video
            video_path = download_video(request.video_url)
            
            # Format prompt with video token
            prompt = f"<video>\n{request.instruction}"
            
            # Process using their method
            inputs = processor(prompt, video_path, edit_prompt=True)
            inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
            
            # Clean up temporary file
            os.unlink(video_path)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    do_sample=(request.temperature > 0),
                    temperature=request.temperature,
                    max_new_tokens=request.max_new_tokens,
                    top_p=request.top_p,
                    use_cache=True
                )
                
                # Decode only the new tokens (skip the prompt)
                response = processor.tokenizer.decode(
                    outputs[0][inputs['input_ids'][0].shape[0]:],
                    skip_special_tokens=True
                )
        else:
            # Text-only generation
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
        
        return {"response": response.strip()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 