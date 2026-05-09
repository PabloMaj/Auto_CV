
from src.graph import build_graph
from src.state import initial_state

if __name__ == "__main__":
    graph = build_graph()

    result = graph.invoke(initial_state())

    print("\n=== FINAL METRICS ===")
    print(result["metrics"])

    print("\n=== LOGS ===")
    for log in result["logs"]:
        print(log)
