import json
import ollama
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class OllamaVisionLLM:
    def __init__(self, model: str, log_dir: Optional[Path] = None):
        self.model = model
        self.log_dir = log_dir
        self._call_count = 0

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
                trimmed = ""
            else:
                trimmed = " ".join(words[:max_words])
        except Exception as e:
            print(f"Request timed out: {e}")
            return "timeout"

        self._save_log(prompt, trimmed)
        return trimmed

    def _save_log(self, prompt: str, response: str) -> None:
        if not self.log_dir:
            return
        self._call_count += 1
        log_dir = Path(self.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"call_{self._call_count:03d}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "call_id": self._call_count,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
            }, f, indent=2, ensure_ascii=False)
