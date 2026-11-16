from __future__ import annotations

from ..config import Settings
from ..kb import KnowledgeBase
from ..llm import make_client
from .agent import Orchestrator


def make_agent(kb: KnowledgeBase, settings: Settings) -> Orchestrator:
    """agent_mode: policy = deterministic only; llm_tools = LLM writes + calls tools (needs a provider); auto = llm_tools when a provider is configured."""
    mode = settings.llm.agent_mode
    if mode != "policy":
        client = make_client(settings.llm)
        if client is not None:
            from .llm_agent import LLMOrchestrator
            return LLMOrchestrator(kb, settings, client)
    return Orchestrator(kb, settings)
