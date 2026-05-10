
import subprocess
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunnerAgent:

    def run(self, state):
        logger.info("Running RunnerAgent")

        code_path = Path("workspace/generated_solution.py")
        code_path.parent.mkdir(exist_ok=True)

        code_path.write_text(state["generated_code"])

        try:
            result = subprocess.run(
                ["python", str(code_path)],
                capture_output=True,
                text=True,
                timeout=300
            )

            state["execution_output"] = result.stdout
            state["execution_error"] = result.stderr

        except Exception as e:
            state["execution_error"] = str(e)

        return state
