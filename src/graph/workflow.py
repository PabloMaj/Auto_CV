from langgraph.graph import StateGraph, END

from src.agents.data_preprocessor_agent import DataPreprocessorAgent
from src.agents.data_analyser_agent import DataAnalyserAgent
from src.agents.dataset_enricher_agent import DatasetEnricherAgent
from src.agents.dl_model_trainer_agent import DLModelTrainerAgent

from src.agents.programmer_agent import ProgrammerAgent
from src.agents.runner_agent import RunnerAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.improvement_suggester_agent import ImprovementSuggesterAgent
from src.agents.demo_builder_agent import DemoBuilderAgent

from src.config.settings import SystemSettings


# ==========================================================
# STATE LOGIC
# ==========================================================

def resolve_reasoning_type(state, settings: SystemSettings):

    stage_id = state.get("stage_id", 1)
    step_id = state.get("step_id", 1)

    # BUG FIXING (highest priority)
    if state.get("runner_success") is False:
        if state.get("retry_count", 0) < settings.max_runner_retries:
            return "bug_fixing"

    # INITIAL
    if stage_id == 1 and step_id == 1:
        return "initial_coding"

    # NOVELTY
    if stage_id >= 2 and step_id == 1:
        if settings.enable_novel_solution_search:
            return "novelty_coding"
        else:
            return "initial_coding"

    # IMPROVEMENT
    if step_id > 1:
        artifacts = state.get("eval_artifacts", [])
        stage_best = max(
            (a.value for a in artifacts
             if a.step_key.startswith(f"stage_{stage_id}_")),
            default=0.0,
        )
        if stage_best < 0.2:
            return "initial_coding"
        return "improving_based_on_suggestion"

    return "initial_coding"


# ==========================================================
# STATE UPDATE (ONLY PLACE THAT MUTATES EVOLUTION STATE)
# ==========================================================

def update_after_improvement(state, settings: SystemSettings):

    step_id = state.get("step_id", 1) + 1
    stage_id = state.get("stage_id", 1)

    print(f"Updating state after improvement: step_id={step_id}, stage_id={stage_id}")
    if step_id > settings.max_improvement_steps:
        stage_id += 1
        step_id = 1

    return {
        **state,
        "step_id": step_id,
        "stage_id": stage_id,
        "retry_count": 0,
    }


# ==========================================================
# GRAPH
# ==========================================================

def build_graph(settings: SystemSettings):

    graph = StateGraph(dict)

    # ======================================================
    # AGENTS
    # ======================================================

    preprocessor = DataPreprocessorAgent()
    analyser = DataAnalyserAgent()

    enricher = DatasetEnricherAgent(settings=settings) if settings.enable_dataset_enricher else None
    trainer = DLModelTrainerAgent(settings=settings) if settings.enable_dl_model_trainer else None

    programmer = ProgrammerAgent(
        settings.programmer_llm.backend,
        {
            "model": settings.programmer_llm.model,
            **settings.programmer_llm.inference_kwargs
        },
        label_free=settings.enable_label_free_improvement,
    )

    runner = RunnerAgent()
    evaluator = EvaluatorAgent(settings=settings)
    demo_builder = DemoBuilderAgent()

    suggester = ImprovementSuggesterAgent(
        settings.improvement_llm.backend,
        {
            "model": settings.improvement_llm.model,
            **settings.improvement_llm.inference_kwargs
        },
        label_free=settings.enable_label_free_improvement,
    )

    # ======================================================
    # PIPELINE (ONE TIME ONLY)
    # ======================================================

    graph.add_node("data_preprocessor", preprocessor.run)
    graph.add_node("data_analyser", analyser.run)

    if settings.enable_dataset_enricher:
        graph.add_node("dataset_enricher", enricher.run)

    if settings.enable_dl_model_trainer:
        graph.add_node("dl_model_trainer", trainer.run)

    # ======================================================
    # PROGRAMMER NODE
    # ======================================================

    def programmer_node(state):
        reasoning_type = resolve_reasoning_type(state, settings)
        state["reasoning_type"] = reasoning_type
        return programmer.run(state, reasoning_type)

    graph.add_node("programmer", programmer_node)

    # ======================================================
    # EXECUTION NODES
    # ======================================================

    graph.add_node("runner", runner.run)
    graph.add_node("evaluator", evaluator.run)
    graph.add_node("improvement_suggester", suggester.run)
    graph.add_node("demo_builder", demo_builder.run)

    def improvement_state_updater(state):
        return update_after_improvement(state, settings)

    graph.add_node("improvement_state_updater", improvement_state_updater)

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
        if settings.enable_dl_model_trainer:
            graph.add_edge("data_analyser", "dl_model_trainer")
            graph.add_edge("dl_model_trainer", "programmer")
        else:
            graph.add_edge("data_analyser", "programmer")

    # ======================================================
    # MAIN LOOP
    # ======================================================

    graph.add_edge("programmer", "runner")

    # ======================================================
    # RUNNER ROUTING
    # ======================================================

    def route_after_runner(state):
        if state.get("runner_success") is False:
            if state.get("retry_count", 0) < settings.max_runner_retries:
                return "programmer"
        return "evaluator"

    graph.add_conditional_edges(
        "runner",
        route_after_runner,
        {
            "programmer": "programmer",
            "evaluator": "evaluator",
        }
    )

    # ======================================================
    # IMPROVEMENT LOOP
    # ======================================================

    def route_after_improvement(state):

        if state.get("stage_id", 0) > settings.max_novel_solutions:
            return "demo_builder"

        artifacts = state.get("eval_artifacts", [])
        if artifacts and artifacts[-1].value >= 1.0:
            return "demo_builder"

        return "programmer"

    graph.add_edge("evaluator", "improvement_suggester")
    graph.add_edge("improvement_suggester", "improvement_state_updater")

    graph.add_conditional_edges(
        "improvement_state_updater",
        route_after_improvement,
        {
            "programmer": "programmer",
            "demo_builder": "demo_builder",
        }
    )

    graph.add_edge("demo_builder", END)

    return graph.compile()
