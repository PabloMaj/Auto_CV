# src/inference/ollama_inference.py

import base64
import mimetypes
import time
from pathlib import Path

import ollama

from src.inference.base import BaseInference
from src.utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """
You are an elite Computer Vision Engineer and Python Developer.

Your role is to:
- analyze visual inputs carefully
- infer the likely computer vision task
- generate robust executable Python solutions
- follow formatting instructions exactly
- produce production-ready code

When generating code:
- prefer correctness and robustness
- avoid placeholders
- avoid incomplete implementations
- output executable solutions
"""


class OllamaInference(BaseInference):

    def __init__(
        self,
        model: str,
        host: str = None,
        timeout: int = 120,
        temperature: float = 0.1,
        num_ctx: int = 16384,
        num_predict: int = 8192,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        seed: int = 42,
        max_retries: int = 3
    ):

        self.model = model

        self.client = ollama.Client(
            host=host,
            timeout=timeout
        )

        self.max_retries = max_retries

        self.default_options = {
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
        }

    # ==========================================================
    # IMAGE ENCODING
    # ==========================================================

    def encode_image_base64(
        self,
        image_path
    ):

        image_path = Path(image_path)

        with open(image_path, "rb") as f:

            encoded = base64.b64encode(
                f.read()
            ).decode("utf-8")

        return encoded

    # ==========================================================
    # MEDIA TYPE
    # ==========================================================

    def get_media_type(
        self,
        image_path
    ):

        media_type, _ = mimetypes.guess_type(
            str(image_path)
        )

        if media_type is None:

            media_type = "image/jpeg"

        return media_type

    # ==========================================================
    # MESSAGE BUILDING
    # ==========================================================

    def build_messages(
        self,
        prompt: str,
        image_paths=None,
        system_prompt: str = SYSTEM_PROMPT
    ):

        messages = []

        # ------------------------------------------------------
        # SYSTEM MESSAGE
        # ------------------------------------------------------

        if system_prompt:

            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # ------------------------------------------------------
        # USER MESSAGE
        # ------------------------------------------------------

        user_message = {
            "role": "user",
            "content": prompt
        }

        # ------------------------------------------------------
        # MULTIMODAL IMAGES
        # ------------------------------------------------------

        if image_paths:

            encoded_images = []

            for image_path in image_paths:

                try:

                    encoded = self.encode_image_base64(
                        image_path
                    )

                    encoded_images.append(encoded)

                except Exception as e:

                    logger.warning(
                        f"Failed to encode image "
                        f"{image_path}: {e}"
                    )

            if encoded_images:

                user_message["images"] = encoded_images

        messages.append(user_message)

        return messages

    # ==========================================================
    # TOKEN ESTIMATION
    # ==========================================================

    @staticmethod
    def estimate_tokens(text: str):

        return max(
            1,
            len(text) // 4
        )

    # ==========================================================
    # INFERENCE
    # ==========================================================

    def infer(
        self,
        messages,
        options=None,
        stream=False
    ):

        final_options = self.default_options.copy()

        if options:

            final_options.update(options)

        # ------------------------------------------------------
        # TOKEN LOGGING
        # ------------------------------------------------------

        try:

            prompt_text = ""

            for msg in messages:

                prompt_text += msg.get(
                    "content",
                    ""
                )

            estimated_tokens = self.estimate_tokens(
                prompt_text
            )

            logger.info(
                f"Estimated prompt tokens: "
                f"{estimated_tokens}"
            )

        except Exception as e:

            logger.warning(
                f"Token estimation failed: {e}"
            )

        # ------------------------------------------------------
        # RETRIES
        # ------------------------------------------------------

        last_exception = None

        for attempt in range(self.max_retries):

            try:

                logger.info(
                    f"Ollama inference attempt "
                    f"{attempt + 1}/"
                    f"{self.max_retries}"
                )

                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=final_options,
                    stream=stream
                )

                # ==============================================
                # STREAMING
                # ==============================================

                if stream:

                    chunks = []

                    for chunk in response:

                        content = chunk.get(
                            "message",
                            {}
                        ).get(
                            "content",
                            ""
                        )

                        if content:

                            print(
                                content,
                                end="",
                                flush=True
                            )

                            chunks.append(content)

                    response_raw = "".join(
                        chunks
                    )

                # ==============================================
                # NORMAL MODE
                # ==============================================

                else:

                    response_raw = response.get(
                        "message",
                        {}
                    ).get(
                        "content",
                        ""
                    )

                response_raw = response_raw.strip()

                logger.info(
                    "Ollama inference successful"
                )

                return response_raw

            except Exception as e:

                logger.warning(
                    f"Ollama inference failed "
                    f"(attempt={attempt + 1}): {e}"
                )

                last_exception = e

                time.sleep(2)

        raise RuntimeError(
            f"Ollama inference failed after "
            f"{self.max_retries} retries. "
            f"Last exception: {last_exception}"
        )
