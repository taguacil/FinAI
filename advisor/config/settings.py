"""Advisor settings loaded from environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class AdvisorSettings:
    """All runtime configuration for the advisor."""

    project_root: Path = _PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    log_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "advisor" / "logs")

    # MCP transport
    mcp_command: str = "uv"
    mcp_args: List[str] = field(
        default_factory=lambda: ["run", "python", "-m", "src.mcp_server", "--stdio"]
    )
    portfolio_name: Optional[str] = None

    # Model selection (per llm_config.py registry)
    orchestrator_model: str = "claude-sonnet-4.5"
    worker_model: str = "claude-sonnet-4.5"
    temperature: float = 0.1

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> AdvisorSettings:
    """Build settings from environment variables (with .env support)."""
    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    s = AdvisorSettings()
    if v := os.getenv("ADVISOR_PORTFOLIO"):
        s.portfolio_name = v
    if v := os.getenv("ADVISOR_DATA_DIR"):
        s.data_dir = Path(v)
    if v := os.getenv("ADVISOR_ORCHESTRATOR_MODEL"):
        s.orchestrator_model = v
    if v := os.getenv("ADVISOR_WORKER_MODEL"):
        s.worker_model = v
    if v := os.getenv("ADVISOR_TEMPERATURE"):
        s.temperature = float(v)
    s.ensure_dirs()
    return s
