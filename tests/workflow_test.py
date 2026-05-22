from src.graph.workflow import build_graph
from src.config.settings import SystemSettings


def test_graph_build():
    settings = SystemSettings()
    graph = build_graph(settings)

    assert graph is not None
