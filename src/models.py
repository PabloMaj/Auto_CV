
"""
Model definitions for Ollama chat interfaces.

This module provides a class-based interface for interacting with Ollama models
for coding and reasoning tasks.
"""

from ollama import chat
from typing import Optional, Dict, Any


class OllamaModels:
    """
    A wrapper class for Ollama models providing easy access to coding and reasoning capabilities.
    """

    def __init__(self, coding_model: str = "qwen2.5-coder:7b", 
                 reasoning_model: str = "qwen2.5vl:7b"):
        """
        Initialize OllamaModels with specified models.

        Args:
            coding_model (str): Model name for code generation (default: qwen2.5-coder:7b)
            reasoning_model (str): Model name for reasoning tasks (default: qwen2.5vl:7b)
        """
        self.coding_model = coding_model
        self.reasoning_model = reasoning_model

    def generate_code(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Generate code using the coding model.

        Args:
            prompt (str): The coding task or question
            temperature (float): Temperature for response variability (default: 0.0)

        Returns:
            str: Generated code response

        Raises:
            Exception: If Ollama connection fails
        """
        response = chat(
            model=self.coding_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False,
            options={
                "temperature": temperature
            }
        )
        return response["message"]["content"]

    def reason(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Generate reasoning response using the reasoning model.

        Args:
            prompt (str): The reasoning question or task
            temperature (float): Temperature for response variability (default: 0.2)

        Returns:
            str: Generated reasoning response

        Raises:
            Exception: If Ollama connection fails
        """
        response = chat(
            model=self.reasoning_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False,
            options={
                "temperature": temperature
            }
        )
        return response["message"]["content"]

    def chat(self, prompt: str, model: Optional[str] = None, 
             temperature: float = 0.7) -> str:
        """
        Generic chat function with any Ollama model.

        Args:
            prompt (str): User message
            model (str): Model name (if None, uses coding model)
            temperature (float): Temperature for response variability (default: 0.7)

        Returns:
            str: Model response

        Raises:
            Exception: If Ollama connection fails
        """
        if model is None:
            model = self.coding_model

        response = chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False,
            options={
                "temperature": temperature
            }
        )
        return response["message"]["content"]

    def get_models_info(self) -> Dict[str, Any]:
        """
        Get information about configured models.

        Returns:
            Dict: Dictionary with model configuration details
        """
        return {
            "coding_model": self.coding_model,
            "reasoning_model": self.reasoning_model,
            "description": "OllamaModels wrapper for easy model access"
        }


# Global instance for convenience
models = OllamaModels()


if __name__ == "__main__":
    print("=" * 70)
    print("OLLAMA MODELS - DEMO")
    print("=" * 70)
    
    # Initialize models instance
    ollama = OllamaModels()
    
    # Display model info
    print("\nConfigured Models:")
    print("-" * 70)
    info = ollama.get_models_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example 1: Code generation
    print("\n" + "=" * 70)
    print("[1] CODE GENERATION EXAMPLE")
    print("-" * 70)
    coding_prompt = "Write a Python function that calculates factorial of a number"
    print(f"Prompt: {coding_prompt}\n")
    
    try:
        code_response = ollama.generate_code(coding_prompt)
        print("Response:")
        print(code_response)
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure Ollama is running on localhost:11434")
    
    # Example 2: Reasoning
    print("\n" + "=" * 70)
    print("[2] REASONING EXAMPLE")
    print("-" * 70)
    reasoning_prompt = "Explain the concept of machine learning in simple terms"
    print(f"Prompt: {reasoning_prompt}\n")
    
    try:
        reasoning_response = ollama.reason(reasoning_prompt)
        print("Response:")
        print(reasoning_response)
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure Ollama is running on localhost:11434")
    
    # Example 3: Generic chat
    print("\n" + "=" * 70)
    print("[3] GENERIC CHAT EXAMPLE")
    print("-" * 70)
    chat_prompt = "What are the benefits of using virtual environments in Python?"
    print(f"Prompt: {chat_prompt}\n")
    print(f"Model: {ollama.coding_model}\n")
    
    try:
        chat_response = ollama.chat(
            prompt=chat_prompt,
            temperature=0.5
        )
        print("Response:")
        print(chat_response)
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure Ollama is running on localhost:11434")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
