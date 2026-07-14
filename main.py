# from src.config.langsmith_integration import setup_langsmith_tracing
# setup_langsmith_tracing()

import json
from datetime import datetime
from pathlib import Path

from src.graph.workflow import build_graph
from src.inference.sonnet_inference import save_cost_report
from src.state.agent_state import AgentState
from src.config.settings import SystemSettings
from src.utils.state_utils import state_to_json

if __name__ == "__main__":

    settings = SystemSettings()

    initial_state = AgentState(
        user_prompt="Develop a computer vison method to detect crops in given RGB images. Return midpoints.",
        dl_dataset_path="data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1_bboxes",
        eval_dataset_path="data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1_midpoints",
        exp_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    graph = build_graph(settings)
    result = graph.invoke(initial_state.model_dump())

    exp_dir = Path("workspace") / initial_state.exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    state_path = exp_dir / "final_state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_to_json(result), f, indent=2, ensure_ascii=False)

    with open(exp_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings.model_dump(), f, indent=2, ensure_ascii=False)

    save_cost_report(exp_dir / "cost_report.json")

    print("\n=== FINAL RESULT ===")
    print(result)
    print(f"\nState saved to: {state_path}")