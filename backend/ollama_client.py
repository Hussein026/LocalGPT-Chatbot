import requests
import json
import re

class OllamaClient:
    def __init__(self, api_url="http://localhost:11434", default_model="llama3.2"):
        self.api_url = api_url
        self.default_model = default_model

    # ---------------------------------------
    # MAIN FIX: add the missing generate()
    # ---------------------------------------
    def generate(self, model: str, prompt: str, format: str = ""):
        if not model:
            model = self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 150,
                "repeat_penalty": 1.1,
            }
        }

        if format:
            payload["format"] = format

        try:
            r = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=60
            )
            r.raise_for_status()

            raw = r.text.strip().split("\n")[-1]
            data = json.loads(raw)

            response = data.get("response", "")

            # Remove <think> or <thinking> blocks
            response = re.sub(
                r"<(think|thinking)>.*?</\1>",
                "",
                response,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

            return {"response": response}

        except Exception as e:
            return {"response": f"Error: {str(e)}"}

    # ---------------------------------------
    # List models
    # ---------------------------------------
    def list_models(self):
        try:
            r = requests.get(f"{self.api_url}/tags", timeout=20)
            r.raise_for_status()
            data = r.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models}
        except Exception as e:
            return {"models": [], "error": str(e)}

    # ---------------------------------------
    # Check Ollama is alive
    # ---------------------------------------
    def health(self):
        try:
            r = requests.get(f"{self.api_url}", timeout=10)
            return r.status_code == 200
        except:
            return False
