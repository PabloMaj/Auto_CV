import subprocess
import traceback
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunnerAgent:

    def __init__(self, timeout: int = 300):
        self.timeout = timeout
        self.repo_root = Path(__file__).resolve().parents[2]

    def run(self, state):

        logger.info("Running RunnerAgent")

        generated_code = state.get("generated_code", "")

        if not generated_code.strip():
            state["runner_success"] = False
            state["runner_error"] = ("Generated code is empty.")
            return state

        exp_workspace = self.repo_root / "workspace" / state.get("exp_id", "default")

        # write code to file for execution
        path_to_save_code = exp_workspace / f"stage_{state.get('stage_id', 0)}_step_{state.get('step_id', 0)}"
        path_to_save_code.mkdir(parents=True, exist_ok=True)
        code_path = (path_to_save_code / "generated_solution.py")
        code_path.write_text(generated_code, encoding="utf-8")

        # execute the code and capture output
        try:
            cmd = ["python", str(code_path)]

            result = subprocess.run(cmd, cwd=str(self.repo_root), capture_output=True,
                                    text=True, timeout=self.timeout)

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            runner_success = (result.returncode == 0)

            # store execution results in state
            state["runner_success"] = (runner_success)
            state["runner_output"] = stdout
            state["runner_error"] = stderr

            logger.info(f"Execution finished (success={runner_success})")

        except subprocess.TimeoutExpired:
            state["runner_success"] = False
            state["runner_error"] = (f"Execution timeout after {self.timeout} seconds")

        except Exception:
            state["runner_success"] = False
            state["runner_error"] = (traceback.format_exc())

        if not state.get("runner_success"):
            state["retry_count"] = state.get("retry_count", 0) + 1
            state["total_retry_count"] = state.get("total_retry_count", 0) + 1
            logger.info(f"Bug fix attempt #{state['retry_count']} (total: {state['total_retry_count']})")

        logger.info("RunnerAgent finished successfully")

        return state
