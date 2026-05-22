import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.agents.runner_agent import RunnerAgent


@pytest.fixture
def runner(tmp_path):
    return RunnerAgent(timeout=5, workspace_dir=str(tmp_path))


def test_run_should_fail_when_generated_code_is_empty(runner):
    state = {"generated_code": ""}

    result = runner.run(state)

    assert result["runner_success"] is False
    assert result["runner_error"] == "Generated code is empty."


@patch("src.agents.runner_agent.subprocess.run")
def test_run_should_execute_code_successfully(mock_subprocess_run, runner):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Hello world"
    mock_result.stderr = ""

    mock_subprocess_run.return_value = mock_result

    state = {"generated_code": "print('Hello world')", "stage_id": 1, "step_id": 2}

    result = runner.run(state)

    assert result["runner_success"] is True
    assert result["runner_output"] == "Hello world"
    assert result["runner_error"] == ""

    mock_subprocess_run.assert_called_once()

    saved_file = (runner.workspace_dir / "stage_1_step_2" / "generated_solution.py")

    assert saved_file.exists()
    assert saved_file.read_text(encoding="utf-8") == "print('Hello world')"


@patch("src.agents.runner_agent.subprocess.run")
def test_run_should_handle_execution_failure(mock_subprocess_run, runner):
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "SyntaxError"

    mock_subprocess_run.return_value = mock_result

    state = {"generated_code": "print("}

    result = runner.run(state)

    assert result["runner_success"] is False
    assert result["runner_output"] == ""
    assert result["runner_error"] == "SyntaxError"


@patch("src.agents.runner_agent.subprocess.run")
def test_run_should_handle_timeout(mock_subprocess_run, runner):
    mock_subprocess_run.side_effect = subprocess.TimeoutExpired(
        cmd="python",
        timeout=5
    )

    state = {"generated_code": "while True: pass"}

    result = runner.run(state)

    assert result["runner_success"] is False
    assert result["runner_error"] == "Execution timeout after 5 seconds"


@patch("src.agents.runner_agent.subprocess.run")
def test_run_should_handle_unexpected_exception(mock_subprocess_run, runner):
    mock_subprocess_run.side_effect = Exception("Unexpected failure")

    state = {"generated_code": "print('test')"}

    result = runner.run(state)

    assert result["runner_success"] is False
    assert "Unexpected failure" in result["runner_error"]


@patch("src.agents.runner_agent.subprocess.run")
def test_run_should_call_subprocess_with_expected_arguments(
    mock_subprocess_run,
    runner
):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "ok"
    mock_result.stderr = ""

    mock_subprocess_run.return_value = mock_result

    state = {"generated_code": "print('ok')", "stage_id": 3, "step_id": 7}

    runner.run(state)

    expected_path = (runner.workspace_dir / "stage_3_step_7" / "generated_solution.py")

    mock_subprocess_run.assert_called_once_with(["python", str(expected_path)], cwd=str(runner.workspace_dir),
                                                capture_output=True, text=True, timeout=runner.timeout)
