from langgraph.graph import StateGraph, END

from src.agents.data_preprocessor import DataPreprocessorAgent
from src.agents.data_analyser import DataAnalyserAgent
from src.agents.dataset_enricher import DatasetEnricherAgent
from src.agents.dl_model_trainer import DLModelTrainerAgent

from src.agents.programmer import ProgrammerAgent
from src.agents.runner import RunnerAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.improvement_suggester import ImprovementSuggesterAgent

from src.config.settings import SystemSettings


def build_graph(settings: SystemSettings):

    graph = StateGraph(dict)

    # ======================================================
    # BASE AGENTS
    # ======================================================

    preprocessor = DataPreprocessorAgent()
    analyser = DataAnalyserAgent()
    enricher = DatasetEnricherAgent()
    trainer = DLModelTrainerAgent()

    programmer = ProgrammerAgent(
        settings.programmer_llm.backend,
        {
            "model": settings.programmer_llm.model,
            **settings.programmer_llm.inference_kwargs
        }
    )

    runner = RunnerAgent()
    evaluator = EvaluatorAgent()

    suggester = ImprovementSuggesterAgent(
        settings.improvement_llm.model
    )

    # ======================================================
    # NODES
    # ======================================================

    graph.add_node("data_preprocessor", preprocessor.run)
    graph.add_node("data_analyser", analyser.run)

    if settings.enable_dataset_enricher:
        graph.add_node("dataset_enricher", enricher.run)

    if settings.enable_dl_model_trainer:
        graph.add_node("dl_model_trainer", trainer.run)

    graph.add_node(
        "programmer",
        lambda s: programmer.run(s, "initial_coding")
    )

    graph.add_node(
        "programmer_bugfix",
        lambda s: programmer.run(s, "bugfix")
    )

    graph.add_node("runner", runner.run)
    graph.add_node("evaluator", evaluator.run)
    graph.add_node("improvement_suggester", suggester.run)

    # ======================================================
    # ENTRY PIPELINE
    # ======================================================

    graph.set_entry_point("data_preprocessor")

    graph.add_edge("data_preprocessor", "data_analyser")

    if settings.enable_dataset_enricher:
        graph.add_edge("data_analyser", "dataset_enricher")

        if settings.enable_dl_model_trainer:
            graph.add_edge("dataset_enricher", "dl_model_trainer")
            graph.add_edge("dl_model_trainer", "programmer")
        else:
            graph.add_edge("dataset_enricher", "programmer")
    else:
        graph.add_edge("data_analyser", "programmer")

    # ======================================================
    # EXECUTION LOOP
    # ======================================================

    graph.add_edge("programmer", "runner")
    graph.add_edge("programmer_bugfix", "runner")

    # ======================================================
    # ROUTING AFTER RUNNER (CRITICAL)
    # ======================================================

    def route_after_runner(state):

        success = state.get("execution_success", False)
        retries = state.get("retry_count", 0)
        max_retries = settings.max_runner_retries

        # ❌ runtime error → bugfix loop
        if not success and retries < max_retries:
            state["retry_count"] = retries + 1
            return "programmer_bugfix"

        # ❌ too many retries → stop
        if not success:
            return END

        # ✔ success → evaluation
        return "evaluator"

    graph.add_conditional_edges(
        "runner",
        route_after_runner,
        {
            "programmer_bugfix": "programmer_bugfix",
            "evaluator": "evaluator",
            END: END
        }
    )

    # ======================================================
    # QUALITY LOOP
    # ======================================================

    graph.add_edge("evaluator", "improvement_suggester")

    def route_after_improvement(state):

        score = state.get("solution_score", 0.0)

        improvement_count = state.get("improvement_count", 0)

        max_iters = settings.m_improvement_steps

        # ✔ good enough → end
        if score >= 0.85:
            return END

        # ❌ improve again → back to programmer
        if improvement_count < max_iters:
            state["improvement_count"] = improvement_count + 1
            return "programmer"

        return END

    graph.add_conditional_edges(
        "improvement_suggester",
        route_after_improvement,
        {
            "programmer": "programmer",
            END: END
        }
    )

    return graph.compile()