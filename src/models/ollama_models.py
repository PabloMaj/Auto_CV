
from langchain_community.chat_models import ChatOllama

coding_llm = ChatOllama(
    model="qwen2.5-coder:14b",
    temperature=0
)

reasoning_llm = ChatOllama(
    model="llava",
    temperature=0.2
)
