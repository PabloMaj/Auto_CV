import ollama
from typing import List


class OllamaVisionLLM:
    def __init__(self, model: str):
        self.model = model

    def inference(self, prompt: str, image_paths: List[str], max_words: int) -> str:
        messages = [{
            "role": "user",
            "content": prompt,
            "images": image_paths
        }]
        client = ollama.Client(timeout=60)

        try:
            response = client.chat(model=self.model, messages=messages)
            response_raw = response["message"]["content"].strip().lower()

            words = response_raw.split()
            if not words:
                return ""
            trimmed = " ".join(words[:max_words])
        except Exception as e:
            print(f"Request timed out: {e}")
            return "timeout"

        return trimmed
