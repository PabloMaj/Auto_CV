import base64
import os
import time
from pathlib import Path
from typing import Optional

import anthropic

from src.inference.base import BaseInference
from src.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """
You are an elite Computer Vision Engineer and Python Developer.
"""


class SonnetInference(BaseInference):

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 3
    ):

        if api_key is None:
            api_key = os.getenv("SONNET_API_KEY")

        if not api_key:
            raise ValueError(
                "Sonnet API key is required. "
                "Set SONNET_API_KEY in the environment or pass api_key explicitly."
            )

        self.model = model

        self.client = anthropic.Anthropic(
            api_key=api_key
        )

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    # ======================================================
    # IMAGE
    # ======================================================

    def encode_image_base64(self, image_path):

        image_path = Path(image_path)

        with open(image_path, "rb") as f:

            return base64.b64encode(
                f.read()
            ).decode("utf-8")

    # ======================================================
    # MESSAGE BUILDING
    # ======================================================

    def build_messages(
        self,
        prompt: str,
        image_paths=None,
        system_prompt: str = SYSTEM_PROMPT
    ):

        content = []

        # --------------------------------------------------
        # IMAGES
        # --------------------------------------------------

        if image_paths:

            for image_path in image_paths:

                try:

                    encoded = self.encode_image_base64(
                        image_path
                    )

                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": encoded
                        }
                    })

                except Exception as e:

                    logger.warning(
                        f"Image encoding failed: "
                        f"{image_path} -> {e}"
                    )

        # --------------------------------------------------
        # TEXT
        # --------------------------------------------------

        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [{
            "role": "user",
            "content": content
        }]

        return {
            "system": system_prompt,
            "messages": messages
        }

    # ======================================================
    # INFERENCE
    # ======================================================

    def infer(
        self,
        messages,
        options=None,
        stream=False
    ):

        last_exception = None

        for attempt in range(self.max_retries):

            try:

                logger.info(
                    f"Sonnet inference attempt "
                    f"{attempt + 1}"
                )

                response = self.client.messages.create(
                    model=self.model,
                    system=messages["system"],
                    messages=messages["messages"],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                text = response.content[0].text.strip()

                logger.info(
                    "Sonnet inference successful"
                )

                return text

            except Exception as e:

                logger.warning(
                    f"Sonnet inference failed: {e}"
                )

                last_exception = e

                time.sleep(2)

        raise RuntimeError(
            f"Sonnet inference failed after "
            f"{self.max_retries} retries. "
            f"Last exception: {last_exception}"
        )