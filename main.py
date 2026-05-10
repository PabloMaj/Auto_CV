
from src.graph.workflow import build_graph
from src.state.agent_state import AgentState
from src.config.settings import SystemSettings

if __name__ == "__main__":
    settings = SystemSettings()

    initial_state = AgentState(
        user_prompt="Opracuj metodę do zliczania roślin na obrazach RGB",
        dataset_path="./dataset"
    )

    graph = build_graph(settings)
    result = graph.invoke(initial_state.model_dump())

    print("\n=== FINAL RESULT ===")
    print(result)
