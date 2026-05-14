import subprocess
import traceback
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunnerAgent:

    def __init__(
        self,
        timeout: int = 300,
        workspace_dir: str = "workspace"
    ):

        self.timeout = timeout
        workspace_dir_path = Path(workspace_dir)

        if not workspace_dir_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            workspace_dir_path = repo_root / workspace_dir_path

        self.workspace_dir = workspace_dir_path.resolve()

        self.workspace_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ======================================================
    # RUN
    # ======================================================

    def run(self, state):

        logger.info("Running RunnerAgent")

        generated_code = state.get(
            "generated_code",
            ""
        )

        if not generated_code.strip():

            state["execution_success"] = False

            state["execution_error"] = (
                "Generated code is empty."
            )

            return state

        state["runner_note"] = (
            "Inference output files can only be written inside the workspace directory. "
            "All generated files are stored under: " + str(self.workspace_dir)
        )

        logger.info(state["runner_note"])

        code_path = (
            self.workspace_dir /
            "generated_solution.py"
        )

        code_path.write_text(
            generated_code,
            encoding="utf-8"
        )

        try:
            cmd = ["python", str(code_path)]
            sample_image_path = state.get("sample_image_path")
            if sample_image_path:
                cmd.append(str(sample_image_path))

            result = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            # truncate giant logs
            stdout = stdout[-20000:]
            stderr = stderr[-20000:]

            execution_success = (
                result.returncode == 0
            )

            state["execution_success"] = (
                execution_success
            )

            state["execution_return_code"] = (
                result.returncode
            )

            state["execution_output"] = stdout

            state["execution_error"] = stderr

            logger.info(
                f"Execution finished "
                f"(success={execution_success})"
            )

        except subprocess.TimeoutExpired:

            state["execution_success"] = False

            state["execution_error"] = (
                f"Execution timeout "
                f"after {self.timeout}s"
            )

        except Exception:

            state["execution_success"] = False

            state["execution_error"] = (
                traceback.format_exc()
            )

        print(state)
        import sys
        sys.exit(0)

        return state
