from abc import ABC, abstractmethod


class BaseInference(ABC):

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