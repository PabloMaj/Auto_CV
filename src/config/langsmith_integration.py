"""
LangSmith tracing setup.

Called once, before the graph/agents are imported (see main.py), so that
LangGraph's LangChain-core callback manager picks up tracing from the very
first node. Requires LANGSMITH_API_KEY to be set in the environment; if it
isn't, tracing is silently left disabled and the graph still runs normally.
"""

import os

from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_langsmith_tracing(project: str = "agent_cv") -> bool:
    if not os.environ.get("LANGSMITH_API_KEY"):
        logger.warning("LANGSMITH_API_KEY not set — LangSmith tracing disabled")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"  # back-compat alias, some langchain-core paths still read this
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    logger.info(f"LangSmith tracing enabled — project={os.environ['LANGSMITH_PROJECT']}")
    return True
