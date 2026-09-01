"""
Persistent storage for simulation / what-if runs.

Simulations (what-if, Monte Carlo, optimization) are normally computed on the fly
and discarded. This module persists each run as a JSON record capturing the tool
used, its inputs, the random seed (for reproducibility), and the produced output,
so a run can be re-checked later without re-computing it.
"""

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimulationStore:
    """File-based storage for simulation runs.

    Records are stored as JSON files under ``<data_dir>/simulations/``, one file
    per run, named by a generated id (``sim_<timestamp>_<rand>``).
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize storage with data directory."""
        self.data_dir = Path(data_dir)
        self.simulations_dir = self.data_dir / "simulations"
        self.simulations_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        tool: str,
        inputs: Dict[str, Any],
        output: str,
        portfolio_id: Optional[str] = None,
        portfolio_name: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> str:
        """Persist a simulation run.

        Args:
            tool: Name of the tool that produced the run (e.g. "advanced_what_if").
            inputs: The exact arguments the tool was called with.
            output: The formatted result string returned by the tool.
            portfolio_id: ID of the portfolio the run was based on.
            portfolio_name: Human-readable portfolio name.
            random_seed: Seed used for Monte Carlo runs (enables reproduction).

        Returns:
            The generated simulation id.
        """
        created_at = datetime.now()
        sim_id = f"sim_{created_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        record = {
            "id": sim_id,
            "created_at": created_at.isoformat(),
            "tool": tool,
            "portfolio_id": portfolio_id,
            "portfolio_name": portfolio_name,
            "random_seed": random_seed,
            "inputs": inputs,
            "output": output,
        }

        filepath = self.simulations_dir / f"{sim_id}.json"
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2, default=str)

        return sim_id

    def get(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Load a single simulation record by id, or None if not found."""
        if not self._is_valid_id(simulation_id):
            return None

        filepath = self.simulations_dir / f"{simulation_id}.json"
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def list(
        self,
        portfolio_id: Optional[str] = None,
        tool: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List saved simulations (newest first).

        Returns lightweight summaries (inputs and metadata, without the full
        output) suitable for browsing. Use :meth:`get` to fetch a full record.

        Args:
            portfolio_id: If set, only return runs for this portfolio.
            tool: If set, only return runs from this tool.
            limit: Maximum number of summaries to return.
        """
        summaries: List[Dict[str, Any]] = []

        for filepath in self.simulations_dir.glob("sim_*.json"):
            try:
                with open(filepath, "r") as f:
                    rec = json.load(f)
            except Exception:
                continue

            if portfolio_id and rec.get("portfolio_id") != portfolio_id:
                continue
            if tool and rec.get("tool") != tool:
                continue

            summaries.append(
                {
                    "id": rec.get("id"),
                    "created_at": rec.get("created_at"),
                    "tool": rec.get("tool"),
                    "portfolio_id": rec.get("portfolio_id"),
                    "portfolio_name": rec.get("portfolio_name"),
                    "random_seed": rec.get("random_seed"),
                    "inputs": rec.get("inputs"),
                }
            )

        summaries.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return summaries[:limit]

    def delete(self, simulation_id: str) -> bool:
        """Delete a saved simulation. Returns True if a file was removed."""
        if not self._is_valid_id(simulation_id):
            return False

        filepath = self.simulations_dir / f"{simulation_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    @staticmethod
    def _is_valid_id(simulation_id: str) -> bool:
        """Guard against path traversal: ids are alphanumeric + underscore only."""
        return bool(re.fullmatch(r"[A-Za-z0-9_]+", simulation_id or ""))
