
from langgraph.graph import StateGraph, END

from src.state import CVAgentState
from src.agents.task_interpreter import task_interpreter
from src.agents.dataset_agent import dataset_agent
from src.agents.evaluator_agent import evaluator_agent
from src.agents.programmer_agent import programmer_agent
from src.agents.runner_agent import runner_agent
from src.agents.improvement_agent import improvement_agent

def build_graph():
    workflow = StateGraph(CVAgentState)

    workflow.add_node("task_interpreter", task_interpreter)
    workflow.add_node("dataset_agent", dataset_agent)
    workflow.add_node("evaluator_agent", evaluator_agent)
    workflow.add_node("programmer_agent", programmer_agent)
    workflow.add_node("runner_agent", runner_agent)
    workflow.add_node("improvement_agent", improvement_agent)

    workflow.set_entry_point("task_interpreter")

    workflow.add_edge("task_interpreter", "dataset_agent")
    workflow.add_edge("dataset_agent", "evaluator_agent")
    workflow.add_edge("evaluator_agent", "programmer_agent")
    workflow.add_edge("programmer_agent", "runner_agent")
    workflow.add_edge("runner_agent", "improvement_agent")
    workflow.add_edge("improvement_agent", END)

    return workflow.compile()
