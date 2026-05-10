
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

    preprocessor = DataPreprocessorAgent()
    analyser = DataAnalyserAgent()
    enricher = DatasetEnricherAgent()
    trainer = DLModelTrainerAgent()

    programmer = ProgrammerAgent(settings.programmer_model)
    runner = RunnerAgent()
    evaluator = EvaluatorAgent()

    suggester = ImprovementSuggesterAgent(
        settings.improvement_model
    )

    graph.add_node("data_preprocessor", preprocessor.run)
    graph.add_node("data_analyser", analyser.run)

    if settings.enable_dataset_enricher:
        graph.add_node("dataset_enricher", enricher.run)

    if settings.enable_dl_model_trainer:
        graph.add_node("dl_model_trainer", trainer.run)

    graph.add_node(
        "programmer",
        lambda state: programmer.run(state, "initial_coding")
    )

    graph.add_node("runner", runner.run)
    graph.add_node("evaluator", evaluator.run)
    graph.add_node("improvement_suggester", suggester.run)

    graph.set_entry_point("data_preprocessor")

    graph.add_edge("data_preprocessor", "data_analyser")

    if settings.enable_dataset_enricher:
        graph.add_edge("data_analyser", "dataset_enricher")

        if settings.enable_dl_model_trainer:
            graph.add_edge(
                "dataset_enricher",
                "dl_model_trainer"
            )
            graph.add_edge(
                "dl_model_trainer",
                "programmer"
            )
        else:
            graph.add_edge(
                "dataset_enricher",
                "programmer"
            )

    else:
        graph.add_edge("data_analyser", "programmer")

    graph.add_edge("programmer", "runner")
    graph.add_edge("runner", "evaluator")
    graph.add_edge("evaluator", "improvement_suggester")
    graph.add_edge("improvement_suggester", END)

    return graph.compile()
