from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
import io
import tempfile
from typing import Optional, List
import os
from transformers import LlavaForConditionalGeneration
from models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
from dataset.processor import Processor
from tools.conversation import Chat, conv_templates
from copy import deepcopy

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")  # Using the official Tarsier model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models, processor and chat
model = None
processor = None
chat = None

def load_model():
    global model, processor, chat
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
        chat = Chat(model, processor, device=device)
        print("Models loaded successfully!")

class Message(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    instruction: str
    video_url: Optional[str] = None
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    conversation_id: Optional[str] = None  # To track different conversations
    
# Store conversations
conversations = {}
    
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
        # Get or create conversation
        if request.conversation_id and request.conversation_id in conversations:
            conv = conversations[request.conversation_id]
            video_path = conversations[f"{request.conversation_id}_video"]
        else:
            # Determine model type for conversation template
            if '34b' in MODEL_PATH.lower():
                conv_type = 'tarsier-34b'
            elif '13b' in MODEL_PATH.lower():
                conv_type = 'tarsier-13b'
            elif '7b' in MODEL_PATH.lower():
                conv_type = 'tarsier-7b'
            else:
                conv_type = 'tarsier-34b'  # default
            
            conv = deepcopy(conv_templates[conv_type])
            
            if request.video_url:
                # Download and store video for the conversation
                video_path = download_video(request.video_url)
                if request.conversation_id:
                    conversations[f"{request.conversation_id}_video"] = video_path
            else:
                video_path = None

            if request.conversation_id:
                conversations[request.conversation_id] = conv

        # Add user message to conversation
        chat.ask(request.instruction, conv)
        
        # Generate response
        response, conv, _ = chat.answer(
            conv,
            visual_data_file=video_path,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Update conversation if needed
        if request.conversation_id:
            conversations[request.conversation_id] = conv
        
        # Clean up video file if it's not part of a conversation
        if not request.conversation_id and video_path:
            os.unlink(video_path)
        
        return {
            "response": response.strip(),
            "conversation_id": request.conversation_id,
            "messages": [{"role": role, "content": content} for role, content in conv.messages]
        }
    
    except Exception as e:
        # Clean up on error
        if 'video_path' in locals() and not request.conversation_id:
            try:
                os.unlink(video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 