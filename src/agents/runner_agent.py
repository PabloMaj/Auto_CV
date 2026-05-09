
import tempfile
import subprocess
from src.logger import log

def runner_agent(state):

    with tempfile.NamedTemporaryFile(
        suffix=".py",
        delete=False,
        mode="w"
    ) as f:

        f.write(state["predictor_code"])
        path = f.name

    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True
    )

    state["runner_stdout"] = result.stdout
    state["runner_stderr"] = result.stderr

    if result.returncode != 0:
        state["failures"].append({
            "type": "runtime_error",
            "stderr": result.stderr
        })

    print(state)
    log(state, "Runner executed predictor")

    return state
