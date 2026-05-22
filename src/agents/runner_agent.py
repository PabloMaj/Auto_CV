import subprocess
import traceback
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunnerAgent:

    def __init__(self, timeout: int = 300, workspace_dir: str = "workspace"):

        self.timeout = timeout
        workspace_dir_path = Path(workspace_dir)

        if not workspace_dir_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            workspace_dir_path = repo_root / workspace_dir_path
        self.workspace_dir = workspace_dir_path.resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def run(self, state):

        logger.info("Running RunnerAgent")

        generated_code = state.get("generated_code", "")

        if not generated_code.strip():
            state["runner_success"] = False
            state["runner_error"] = ("Generated code is empty.")
            return state

        # write code to file for execution
        path_to_save_code = self.workspace_dir / f"stage_{state.get('stage_id', 0)}_step_{state.get('step_id', 0)}"
        path_to_save_code.mkdir(parents=True, exist_ok=True)
        code_path = (path_to_save_code / "generated_solution.py")
        code_path.write_text(generated_code, encoding="utf-8")

        # execute the code and capture output
        try:
            cmd = ["python", str(code_path)]

            result = subprocess.run(cmd, cwd=str(self.workspace_dir), capture_output=True,
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

        logger.info("RunnerAgent finished successfully")

        return state
