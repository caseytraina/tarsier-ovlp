from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
import io
import tempfile
from typing import Optional, List, Dict
import os
import time
from datetime import datetime, timedelta
from transformers import LlavaForConditionalGeneration
from models.modeling_tarsier import TarsierForConditionalGeneration, LlavaConfig
from dataset.processor import Processor
from tools.conversation import Chat, conv_templates
from copy import deepcopy
import threading
from contextlib import contextmanager

app = FastAPI()

# Model initialization
MODEL_PATH = os.getenv("MODEL_PATH", "omni-research/Tarsier-34b")  # Using the official Tarsier model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models, processor and chat
model = None
processor = None
chat = None

# Conversation settings
MAX_CONVERSATIONS = 100
CONVERSATION_TIMEOUT = 3600
MAX_TURNS = 20

class ConversationState:
    def __init__(self, conv, video_path: Optional[str] = None):
        self.conv = conv
        self.video_path = video_path
        self.last_active = time.time()
        self.turns = 0
        self._lock = threading.Lock()  # Lock for this conversation
    
    @contextmanager
    def lock(self):
        """Context manager for thread-safe access to conversation state"""
        try:
            self._lock.acquire()
            yield
        finally:
            self._lock.release()
    
    def update_activity(self):
        with self.lock():
            self.last_active = time.time()
            self.turns += 1
    
    def cleanup(self):
        with self.lock():
            if self.video_path and os.path.exists(self.video_path):
                try:
                    os.unlink(self.video_path)
                except:
                    pass

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.last_cleanup = time.time()
        self._lock = threading.Lock()  # Global lock for conversation dictionary
    
    @contextmanager
    def lock(self):
        """Context manager for thread-safe access to conversations dictionary"""
        try:
            self._lock.acquire()
            yield
        finally:
            self._lock.release()
    
    def cleanup_old_conversations(self):
        current_time = time.time()
        # Only run cleanup every 5 minutes
        with self.lock():
            if current_time - self.last_cleanup < 300:
                return
            
            self.last_cleanup = current_time
            to_remove = []
            
            for conv_id, conv_state in self.conversations.items():
                with conv_state.lock():
                    if current_time - conv_state.last_active > CONVERSATION_TIMEOUT:
                        to_remove.append(conv_id)
            
            for conv_id in to_remove:
                self.end_conversation(conv_id)
    
    def get_conversation(self, conv_id: str) -> Optional[ConversationState]:
        with self.lock():
            if conv_id in self.conversations:
                conv_state = self.conversations[conv_id]
                conv_state.update_activity()
                return conv_state
            return None
    
    def create_conversation(self, conv_id: str, conv, video_path: Optional[str] = None) -> ConversationState:
        # Clean up old conversations first
        self.cleanup_old_conversations()
        
        with self.lock():
            # Check limits
            if len(self.conversations) >= MAX_CONVERSATIONS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Maximum number of concurrent conversations ({MAX_CONVERSATIONS}) reached. Please try again later."
                )
            
            conv_state = ConversationState(conv, video_path)
            self.conversations[conv_id] = conv_state
            return conv_state
    
    def end_conversation(self, conv_id: str):
        with self.lock():
            if conv_id in self.conversations:
                conv_state = self.conversations[conv_id]
                conv_state.cleanup()
                del self.conversations[conv_id]

# Initialize conversation manager
conversation_manager = ConversationManager()

# Model loading with thread safety
_model_lock = threading.Lock()

def load_model():
    global model, processor, chat
    with _model_lock:
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
        if request.conversation_id:
            conv_state = conversation_manager.get_conversation(request.conversation_id)
            if conv_state:
                with conv_state.lock():
                    if conv_state.turns >= MAX_TURNS:
                        raise HTTPException(
                            status_code=429,
                            detail=f"Maximum number of turns ({MAX_TURNS}) reached for this conversation. Please start a new one."
                        )
                    conv = conv_state.conv
                    video_path = conv_state.video_path
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
                    video_path = download_video(request.video_url)
                else:
                    video_path = None
                
                conv_state = conversation_manager.create_conversation(request.conversation_id, conv, video_path)
        else:
            # Single turn conversation
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
                video_path = download_video(request.video_url)
            else:
                video_path = None

        # Add user message and generate response with thread safety
        with _model_lock:  # Ensure thread-safe access to the model
            chat.ask(request.instruction, conv)
            response, conv, _ = chat.answer(
                conv,
                visual_data_file=video_path,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )

        # Update conversation if needed
        if request.conversation_id:
            with conv_state.lock():
                conv_state.conv = conv
        else:
            # Clean up video file for single turn conversation
            if video_path:
                os.unlink(video_path)
        
        return {
            "response": response.strip(),
            "conversation_id": request.conversation_id,
            "messages": [{"role": role, "content": content} for role, content in conv.messages],
            "turns_remaining": MAX_TURNS - conv_state.turns if request.conversation_id else 0
        }
    
    except Exception as e:
        # Clean up on error
        if 'video_path' in locals() and not request.conversation_id:
            try:
                os.unlink(video_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{conversation_id}")
async def end_conversation(conversation_id: str):
    """Explicitly end a conversation and clean up its resources"""
    conversation_manager.end_conversation(conversation_id)
    return {"status": "success", "message": f"Conversation {conversation_id} ended"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 