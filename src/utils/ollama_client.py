
import ollama


class OllamaInference:

    def __init__(self, model: str):
        self.model = model
        self.client = ollama.Client(timeout=60)

    def infer(self, messages):
        response = self.client.chat(
            model=self.model,
            messages=messages
        )

        response_raw = response["message"]["content"].strip()
        return response_raw
