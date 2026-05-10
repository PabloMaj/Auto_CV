
from src.utils.logger import get_logger
from src.utils.ollama_client import OllamaInference

logger = get_logger(__name__)


class ImprovementSuggesterAgent:

    def __init__(self, model_name: str):
        self.llm = OllamaInference(model_name)

    def build_prompt(self, state):

        prompt = f'''
        Analyze current CV solution.

        Current metric:
        {state.get("evaluation_metric")}

        Current code:
        {state.get("generated_code")}

        Prediction visualizations:
        {state.get("prediction_visualizations")}

        Previous suggestions:
        {state.get("improvement_suggestions")}
        '''

        return prompt

    def run(self, state):
        logger.info("Running ImprovementSuggesterAgent")

        prompt = self.build_prompt(state)

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        suggestions = self.llm.infer(messages)

        state["improvement_suggestions"].append(suggestions)

        return state
