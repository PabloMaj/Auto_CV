from src.agents.data_analyser import DataAnalyserAgent
from src.inference.ollama_inference import OllamaInference


def test_scalar_output():

    inference = OllamaInference(model="qwen2.5vl:7b")
    agent = DataAnalyserAgent(inference=inference)
    state = {"user_prompt": ("Count all plants in RGB images.")}
    result = agent.run(state)

    assert result["desired_output"] == "scalar"


def test_bounding_boxes_output():

    inference = OllamaInference(model="qwen2.5vl:7b")
    agent = DataAnalyserAgent(inference=inference)
    state = {"user_prompt": ("Detect cars in aerial images.")}
    result = agent.run(state)

    assert result["desired_output"] == ("bounding_boxes")


def test_segmentation_masks_output():

    inference = OllamaInference(model="qwen2.5vl:7b")
    agent = DataAnalyserAgent(inference=inference)
    state = {"user_prompt": ("Segment road pixels from images.")}
    result = agent.run(state)

    assert result["desired_output"] == ("segmentation_masks")


def test_invalid_response_fallback():

    inference = OllamaInference(model="qwen2.5vl:7b")
    agent = DataAnalyserAgent(inference=inference)
    state = {"user_prompt": ("Please buy me a coffee.")}
    result = agent.run(state)

    assert result["desired_output"] == ("unknown")
