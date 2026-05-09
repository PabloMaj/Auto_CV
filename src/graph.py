
from langgraph.graph import StateGraph, END

from src.state import AgentState

from src.agents.task_agent import task_agent
from src.agents.dataset_agent import dataset_agent
from src.agents.evaluator_agent import evaluator_agent
from src.agents.programmer_agent import programmer_agent
from src.agents.runner_agent import runner_agent
from src.agents.evaluation_agent import evaluation_agent
from src.agents.failure_analysis_agent import failure_analysis_agent
from src.agents.improvement_agent import improvement_agent

def router(state):

    if state["iteration"] >= state["max_iterations"]:
        return "end"

    if state["metrics"].get("MAE", 999) < 0.5:
        return "end"

    return "continue"

def increment_iteration(state):
    state["iteration"] += 1
    return state

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("task", task_agent)
    graph.add_node("dataset", dataset_agent)
    graph.add_node("evaluator", evaluator_agent)
    graph.add_node("programmer", programmer_agent)
    graph.add_node("runner", runner_agent)
    graph.add_node("evaluation", evaluation_agent)
    graph.add_node("failure_analysis", failure_analysis_agent)
    graph.add_node("improvement", improvement_agent)
    graph.add_node("increment", increment_iteration)

    graph.set_entry_point("task")

    graph.add_edge("task", "dataset")
    graph.add_edge("dataset", "evaluator")
    graph.add_edge("evaluator", "programmer")
    graph.add_edge("programmer", "runner")
    graph.add_edge("runner", "evaluation")
    graph.add_edge("evaluation", "failure_analysis")
    graph.add_edge("failure_analysis", "improvement")
    graph.add_edge("improvement", "increment")

    graph.add_conditional_edges(
        "increment",
        router,
        {
            "continue": "programmer",
            "end": END
        }
    )

    return graph.compile()
