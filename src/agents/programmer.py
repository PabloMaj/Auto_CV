
from src.utils.logger import get_logger
from src.utils.ollama_client import OllamaInference

logger = get_logger(__name__)


class ProgrammerAgent:

    def __init__(self, model_name: str):
        self.llm = OllamaInference(model_name)

    def build_prompt(self, state, reasoning_type: str):

        base_prompt = f'''
        User prompt:
        {state["user_prompt"]}

        Task type:
        {state.get("detected_task_type")}

        Previous suggestions:
        {state.get("improvement_suggestions")}

        Previous execution error:
        {state.get("execution_error")}
        '''

        if reasoning_type == "initial_coding":
            base_prompt += "\nGenerate initial CV pipeline."

        elif reasoning_type == "bugfix":
            base_prompt += "\nFix execution errors."

        elif reasoning_type == "improvement":
            base_prompt += "\nImprove current solution."

        elif reasoning_type == "novel_solution":
            base_prompt += "\nGenerate a novel approach."

        return base_prompt

    def run(self, state, reasoning_type="initial_coding"):
        logger.info(f"Running ProgrammerAgent ({reasoning_type})")

        prompt = self.build_prompt(state, reasoning_type)

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        generated_code = self.llm.infer(messages)

        state["generated_code"] = generated_code

        return state
