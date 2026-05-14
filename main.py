
from src.graph.workflow import build_graph
from src.state.agent_state import AgentState
from src.config.settings import SystemSettings

if __name__ == "__main__":
    settings = SystemSettings()

    initial_state = AgentState(
        user_prompt = "Develop a computer vison method to count all plants in given RGB images. Use only classical CV methods, no deep learning." \
        "You can use ExG index for plant segmentation. For counting, consider using connected components analysis or contour detection.",
        
        dataset_path="./dataset"
    )

    graph = build_graph(settings)
    result = graph.invoke(initial_state.model_dump())

    print("\n=== FINAL RESULT ===")
    print(result)
