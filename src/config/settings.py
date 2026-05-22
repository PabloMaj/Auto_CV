from typing import Any, Dict

from pydantic import BaseModel, Field


# ==========================================================
# LLM CONFIG
# ==========================================================

class LLMSettings(BaseModel):

    backend: str
    model: str

    inference_kwargs: Dict[str, Any] = Field(
        default_factory=dict
    )


# ==========================================================
# SYSTEM SETTINGS
# ==========================================================

class SystemSettings(BaseModel):

    # ======================================================
    # FEATURES FLAGS
    # ======================================================

    enable_dl_model_trainer: bool = False
    enable_dataset_enricher: bool = False

    enable_iterative_improvement: bool = True
    enable_novel_solution_search: bool = True

    # ======================================================
    # LIMITS / CONTROL
    # ======================================================

    max_runner_retries: int = 5
    max_improvement_steps: int = 5
    max_novel_solutions: int = 3

    # ======================================================
    # INITIAL STATE CONTRACT (IMPORTANT)
    # ======================================================

    initial_state_defaults: Dict[str, Any] = Field(
        default_factory=lambda: {
            "retry_count": 0,
            "improvement_count": 0,
            "solution_score": 0.0,
            "execution_success": None,
            "execution_feedback": {},
            "vision_analysis": "",
            "generated_code": "",
        }
    )

    # ======================================================
    # VISION MODEL (QWEN VL - LOCAL)
    # ======================================================

    vision_llm: LLMSettings = LLMSettings(
        backend="ollama",
        model="qwen2.5vl:7b",
        inference_kwargs={
            "temperature": 0.1,
            "num_ctx": 16384,
            "num_predict": 2048,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "seed": 42,
            "max_retries": 3,
        }
    )

    # ======================================================
    # CODING MODEL (SONNET)
    # ======================================================

    programmer_llm: LLMSettings = LLMSettings(
        backend="sonnet",
        model="claude-sonnet-4-5",
        inference_kwargs={
            "temperature": 0.1,
            "max_tokens": 8192,
            "max_retries": 3,
        }
    )

    # ======================================================
    # IMPROVEMENT MODEL (SONNET - MORE CREATIVE)
    # ======================================================

    improvement_llm: LLMSettings = LLMSettings(
        backend="sonnet",
        model="claude-sonnet-4-5",
        inference_kwargs={
            "temperature": 0.2,
            "max_tokens": 4096,
            "max_retries": 3,
        }
    )
