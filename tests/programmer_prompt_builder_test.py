import pytest

from src.funcs.programmer_funcs.programmer_prompt_builder import ProgrammerPromptBuilder, ProgrammerReasoningType


@pytest.fixture
def base_state():
    return {
        "user_prompt": "Write a sorting algorithm",
        "desired_output_definition": "Python code",
        "generated_code": "print('old code')",
        "runner_error": "IndexError",
        "verified_problems": "logic error",
        "improvement_suggestions": "use correct loop",
        "raw_images_section": "img1",
        "previous_implementations": "impl history",
    }


@pytest.fixture(autouse=True)
def mock_prompts(monkeypatch):
    """
    Mock all prompt templates so we only test formatting logic.
    """

    monkeypatch.setattr(
        "src.funcs.programmer_funcs.programmer_prompt_builder.BASE_PROGRAMMER_PROMPT",
        "BASE:{user_prompt}|{desired_output_specification}|{code_start_token}|{code_end_token}",
    )

    monkeypatch.setattr(
        "src.funcs.programmer_funcs.programmer_prompt_builder.INITIAL_CODING_PROMPT",
        "INIT:{raw_images_placeholder}|{base_prompt}",
    )

    monkeypatch.setattr(
        "src.funcs.programmer_funcs.programmer_prompt_builder.BUG_FIXING_PROMPT",
        "BUG:{previous_code}|{runner_error}|{base_prompt}",
    )

    monkeypatch.setattr(
        "src.funcs.programmer_funcs.programmer_prompt_builder.IMPROVING_BASED_ON_SUGGESTION_PROMPT",
        "IMPROVE:{previous_code}|{verified_problems}|{improvement_suggestions}|{base_prompt}",
    )

    monkeypatch.setattr(
        "src.funcs.programmer_funcs.programmer_prompt_builder.NOVELTY_CODING_PROMPT",
        "NOVEL:{previous_implementations}|{base_prompt}",
    )


def test_initial_coding_prompt(base_state):
    result = ProgrammerPromptBuilder.build(
        base_state,
        ProgrammerReasoningType.INITIAL_CODING,
        "<START>",
        "<END>",
    )

    assert "INIT:" in result
    assert "img1" in result
    assert "BASE:" in result
    assert "<START>" in result
    assert "<END>" in result


def test_bug_fixing_prompt(base_state):
    result = ProgrammerPromptBuilder.build(
        base_state,
        ProgrammerReasoningType.BUG_FIXING,
        "<S>",
        "<E>",
    )

    assert "BUG:" in result
    assert "print('old code')" in result
    assert "IndexError" in result
    assert "BASE:" in result


def test_improving_based_on_suggestion_prompt(base_state):
    result = ProgrammerPromptBuilder.build(
        base_state,
        ProgrammerReasoningType.IMPROVING_BASED_ON_SUGGESTION,
        "<S>",
        "<E>",
    )

    assert "IMPROVE:" in result
    assert "logic error" in result
    assert "use correct loop" in result
    assert "print('old code')" in result
    assert "BASE:" in result


def test_novelty_coding_prompt(base_state):
    result = ProgrammerPromptBuilder.build(
        base_state,
        ProgrammerReasoningType.NOVELTY_CODING,
        "<S>",
        "<E>",
    )

    assert "NOVEL:" in result
    assert "impl history" in result
    assert "BASE:" in result


def test_initial_coding_default_raw_images(base_state):
    state = dict(base_state)
    state.pop("raw_images_section")

    result = ProgrammerPromptBuilder.build(
        state,
        ProgrammerReasoningType.INITIAL_CODING,
        "<S>",
        "<E>",
    )

    assert "INIT:" in result


def test_unknown_reasoning_type_raises(base_state):
    class FakeReasoningType:
        pass

    with pytest.raises(ValueError, match="Unknown reasoning type"):
        ProgrammerPromptBuilder.build(
            base_state,
            FakeReasoningType,
            "<S>",
            "<E>",
        )
