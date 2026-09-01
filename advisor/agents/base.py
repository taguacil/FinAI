"""Base advisor agent — reuses FinAI's central llm_config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from src.agents.llm_config import create_llm


@dataclass
class BaseAdvisorAgent:
    """Common scaffold: model + system prompt + bound tool subset."""

    name: str
    system_prompt: str
    model_key: str
    temperature: float = 0.1
    tools: Sequence[BaseTool] = ()

    def build_llm(self) -> BaseChatModel:
        llm = create_llm(self.model_key, temperature=self.temperature)
        if self.tools:
            llm = llm.bind_tools(list(self.tools))
        return llm

    def filter_tools(self, allowed_names: List[str]) -> "BaseAdvisorAgent":
        keep = [t for t in self.tools if t.name in allowed_names]
        return BaseAdvisorAgent(
            name=self.name,
            system_prompt=self.system_prompt,
            model_key=self.model_key,
            temperature=self.temperature,
            tools=keep,
        )
