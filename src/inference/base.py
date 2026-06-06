import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path


class BaseInference(ABC):

    def __init__(self):
        self.log_dir: Path = None
        self._call_count: int = 0
        self._last_prompt_text: str = ""
        self._last_system_prompt: str = ""
        self._last_image_paths: list = []

    def _save_llm_log(self, response: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if not self.log_dir:
            return
        self._call_count += 1
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"call_{self._call_count:03d}.json"
        entry = {
            "call_id": self._call_count,
            "timestamp": datetime.now().isoformat(),
            "system_prompt": self._last_system_prompt,
            "image_paths": self._last_image_paths,
            "prompt": self._last_prompt_text,
            "response": response,
        }
        if input_tokens or output_tokens:
            entry["input_tokens"] = input_tokens
            entry["output_tokens"] = output_tokens
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)

    @abstractmethod
    def build_messages(
        self,
        prompt: str,
        image_paths=None,
        system_prompt: str = None
    ):
        pass

    @abstractmethod
    def infer(
        self,
        messages,
        options=None,
        stream=False
    ):
        pass
