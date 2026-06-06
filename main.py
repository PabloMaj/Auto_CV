import json
from datetime import datetime
from pathlib import Path

from src.graph.workflow import build_graph
from src.state.agent_state import AgentState
from src.config.settings import SystemSettings


def _state_to_json(state: dict) -> dict:
    from pydantic import BaseModel

    def _convert(v):
        if isinstance(v, BaseModel):
            return _convert(v.model_dump())
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {k: _convert(val) for k, val in v.items()}
        if isinstance(v, list):
            return [_convert(i) for i in v]
        return v
    return {k: _convert(v) for k, v in state.items()}


if __name__ == "__main__":
    settings = SystemSettings()

    initial_state = AgentState(
        user_prompt="Develop a computer vison method to detect crops in given RGB images. You can use DL model as support for solution. Return midpoints.",
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
        json.dump(_state_to_json(result), f, indent=2, ensure_ascii=False)

    print("\n=== FINAL RESULT ===")
    print(result)
    print(f"\nState saved to: {state_path}")
