#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import json
import base64

class OllamaClient:
    def __init__(self, api_url="http://localhost:11434", default_model="qwen2.5:0.5b-instruct-q4_K_M"):
        self.api_url = api_url
        self.default_model = default_model
        
    def generate(self, model=None, prompt=""):
        """
        Simple generation endpoint
        """
        model = model or self.default_model
        url = f"{self.api_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Ollama generate failed: {e}")
            return {"response": f"Error: {str(e)}"}
    
    def chat(self, model=None, messages=None):
        """
        Chat endpoint with message history
        """
        model = model or self.default_model
        messages = messages or []
        
        url = f"{self.api_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        try:
            print(f"[INFO] Calling Ollama chat with model: {model}")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            # Extract response text
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    return {"response": data["message"].get("content", "")}
                elif "response" in data:
                    return data
            
            return {"response": str(data)}
            
        except Exception as e:
            print(f"[ERROR] Ollama chat failed: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"Error: {str(e)}"}
    
    def chat_with_image(self, model=None, messages=None, image_path=None):
        """
        Chat with image support (for vision models like llava)
        """
        model = model or "llava:7b"
        messages = messages or []
        
        if not image_path:
            return self.chat(model, messages)
        
        url = f"{self.api_url}/api/chat"
        
        # Read and encode image
        try:
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            print(f"[INFO] Image encoded: {len(image_data)} chars")
            
            # Add image to the last user message
            if messages and messages[-1].get("role") == "user":
                messages[-1]["images"] = [image_data]
            else:
                messages.append({
                    "role": "user",
                    "content": "What's in this image?",
                    "images": [image_data]
                })
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            print(f"[INFO] Calling Ollama vision model: {model}")
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            
            # Extract response text
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    content = data["message"].get("content", "")
                    print(f"[INFO] Vision model response: {len(content)} chars")
                    return {"response": content}
                elif "response" in data:
                    return data
            
            return {"response": str(data)}
            
        except FileNotFoundError:
            print(f"[ERROR] Image file not found: {image_path}")
            return {"response": "Error: Image file not found"}
        except Exception as e:
            print(f"[ERROR] Ollama vision chat failed: {e}")
            import traceback
            traceback.print_exc()
            return {"response": f"Error: {str(e)}"}
    
    def list_models(self):
        """
        List available models
        """
        url = f"{self.api_url}/api/tags"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Failed to list models: {e}")
            return {"models": []}
    
    def embeddings(self, model, text):
        """
        Generate embeddings
        """
        url = f"{self.api_url}/api/embeddings"
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
            return {"embedding": []}
