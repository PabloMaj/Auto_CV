
from src.graph.workflow import build_graph
from src.state.agent_state import AgentState
from src.config.settings import SystemSettings

if __name__ == "__main__":
    settings = SystemSettings()

    initial_state = AgentState(
        user_prompt="Develop a computer vison method to detect all plants in given RGB images. Use only classical CV methods, no deep learning.",
        dataset_path="data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1"
    )

    graph = build_graph(settings)
    result = graph.invoke(initial_state.model_dump())

    print("\n=== FINAL RESULT ===")
    print(result)
