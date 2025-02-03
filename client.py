import requests
import uuid
import argparse
import sys
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import print as rprint

class VideoChat:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self.conversation_id: Optional[str] = None
        self.video_url: Optional[str] = None
        self.console = Console()
        
    def start_conversation(self, video_url: Optional[str] = None):
        """Start a new conversation with an optional video"""
        self.conversation_id = str(uuid.uuid4())
        self.video_url = video_url
        
        if video_url:
            self.console.print(f"Starting new conversation with video: {video_url}", style="blue")
        else:
            self.console.print("Starting new conversation without video", style="blue")
    
    def end_conversation(self):
        """End the current conversation and cleanup resources"""
        if self.conversation_id:
            try:
                response = requests.delete(f"{self.server_url}/conversation/{self.conversation_id}")
                response.raise_for_status()
                self.console.print("Conversation ended", style="blue")
            except Exception as e:
                self.console.print(f"Error ending conversation: {e}", style="red")
            finally:
                self.conversation_id = None
                self.video_url = None
    
    def chat(self, instruction: str, temperature: float = 0.0):
        """Send a message and get a response"""
        try:
            payload = {
                "instruction": instruction,
                "video_url": self.video_url if not self.video_url else None,  # Only send video_url on first message
                "temperature": temperature,
                "conversation_id": self.conversation_id
            }
            
            response = requests.post(f"{self.server_url}/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Clear video_url after first message since it's already uploaded
            self.video_url = None
            
            return result
            
        except Exception as e:
            self.console.print(f"Error: {e}", style="red")
            return None

def main():
    parser = argparse.ArgumentParser(description="Video Chat Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--video", help="Video URL to discuss")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for response generation")
    args = parser.parse_args()
    
    chat = VideoChat(args.server)
    console = Console()
    
    try:
        # Start conversation
        chat.start_conversation(args.video)
        
        console.print("\n🤖 Welcome to Video Chat! Type 'quit' to exit or 'new' to start a new conversation.\n", style="bold green")
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'new':
                chat.end_conversation()
                video_url = Prompt.ask("\n[bold yellow]Enter video URL[/bold yellow] (press enter to skip)")
                chat.start_conversation(video_url if video_url else None)
                continue
            
            # Get response
            result = chat.chat(user_input, args.temperature)
            
            if result:
                # Print response
                console.print("\n[bold green]Assistant[/bold green]")
                console.print(Markdown(result["response"]))
                
                # Print turns remaining if in a conversation
                if result.get("turns_remaining") is not None:
                    console.print(f"\nTurns remaining: {result['turns_remaining']}", style="dim")
    
    except KeyboardInterrupt:
        console.print("\nExiting...", style="yellow")
    finally:
        # Cleanup
        chat.end_conversation()

if __name__ == "__main__":
    main() 