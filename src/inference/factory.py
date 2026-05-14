from src.inference.ollama_inference import OllamaInference
from src.inference.sonnet_inference import SonnetInference


class InferenceFactory:

    @staticmethod
    def create(
        backend: str,
        **kwargs
    ):

        backend = backend.lower()

        if backend == "ollama":
            return OllamaInference(**kwargs)

        elif backend == "sonnet":
            return SonnetInference(**kwargs)

        raise ValueError(
            f"Unsupported backend: {backend}"
        )