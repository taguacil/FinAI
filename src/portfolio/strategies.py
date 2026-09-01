"""
Pluggable rebalancing strategies for the backtester.

A strategy answers one question at each scheduled rebalance date: given the price
history available *up to and including* that date, what target weights should the
portfolio hold? The backtest engine is strategy-agnostic — it only calls
``target_weights`` and applies the result, so new strategies can be added by
subclassing ``Strategy`` without touching the engine.

Returning ``None`` from ``target_weights`` means "do not rebalance on this date"
(used by buy-and-hold, which allocates once and then drifts).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from .optimizer import OptimizationMethod, OptimizationObjective, PortfolioOptimizer

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for all backtest strategies."""

    #: Human-readable label used in results/charts. Subclasses should set this.
    name: str = "strategy"

    @abstractmethod
    def target_weights(
        self,
        as_of: date,
        prices_window: pd.DataFrame,
        current_weights: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """Return target weights for the rebalance at ``as_of``.

        Args:
            as_of: The rebalance date. ``prices_window`` contains no data after it.
            prices_window: Date-indexed price matrix up to and including ``as_of``,
                with one column per currently-tradeable symbol.
            current_weights: The portfolio's current weights by symbol (pre-rebalance).

        Returns:
            A mapping of symbol -> weight (need not sum to 1.0; the engine
            normalizes over tradeable symbols), or ``None`` to skip rebalancing.
        """
        raise NotImplementedError


class EqualWeightStrategy(Strategy):
    """Allocate equally (1/N) across all tradeable symbols at every rebalance."""

    def __init__(self, name: str = "Equal Weight"):
        self.name = name

    def target_weights(self, as_of, prices_window, current_weights):
        symbols = list(prices_window.columns)
        if not symbols:
            return None
        w = 1.0 / len(symbols)
        return {s: w for s in symbols}


class FixedWeightStrategy(Strategy):
    """Rebalance back to user-supplied static target weights at every rebalance."""

    def __init__(self, weights: Dict[str, float], name: str = "Fixed Weights"):
        total = sum(v for v in weights.values() if v > 0)
        if total <= 0:
            raise ValueError("Fixed weights must contain at least one positive weight")
        self.weights = {s: w / total for s, w in weights.items() if w > 0}
        self.name = name

    def target_weights(self, as_of, prices_window, current_weights):
        available = {
            s: w for s, w in self.weights.items() if s in prices_window.columns
        }
        if not available:
            return None
        return available


class BuyAndHoldStrategy(Strategy):
    """Allocate once at the first rebalance, then never trade again (drifts).

    With no explicit weights this is equal-weight-at-inception, which makes a
    natural passive baseline to compare active strategies against.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        name: str = "Buy & Hold",
    ):
        self.weights = weights
        self.name = name
        self._allocated = False

    def target_weights(self, as_of, prices_window, current_weights):
        if self._allocated:
            return None  # never rebalance after the initial allocation
        self._allocated = True

        symbols = list(prices_window.columns)
        if not symbols:
            self._allocated = False  # nothing tradeable yet; try again next date
            return None

        if self.weights:
            available = {
                s: w for s, w in self.weights.items() if s in symbols and w > 0
            }
            total = sum(available.values())
            if total <= 0:
                return None
            return {s: w / total for s, w in available.items()}

        w = 1.0 / len(symbols)
        return {s: w for s in symbols}


class OptimizerStrategy(Strategy):
    """Walk-forward optimizer: re-run HRP/Markowitz on the trailing window.

    At each rebalance the optimizer sees only ``lookback_days`` of history ending
    at the rebalance date, so target weights are always point-in-time.
    """

    def __init__(
        self,
        method: OptimizationMethod = OptimizationMethod.HRP,
        objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE,
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
        name: Optional[str] = None,
    ):
        self.method = method
        self.objective = objective
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        if name:
            self.name = name
        elif method == OptimizationMethod.MARKOWITZ:
            self.name = f"Markowitz ({objective.value})"
        else:
            self.name = "HRP"

    def target_weights(self, as_of, prices_window, current_weights):
        if prices_window.shape[1] == 0:
            return None
        # Use only the trailing lookback window (approx trading days ~ rows).
        window = prices_window.tail(self.lookback_days)
        weights = PortfolioOptimizer.optimize_weights_from_prices(
            window,
            method=self.method,
            objective=self.objective,
            risk_free_rate=self.risk_free_rate,
        )
        return weights or None


# --- Factory ---------------------------------------------------------------

_METHOD_ALIASES = {
    "hrp": (OptimizationMethod.HRP, OptimizationObjective.MAX_SHARPE),
    "markowitz": (OptimizationMethod.MARKOWITZ, OptimizationObjective.MAX_SHARPE),
    "max_sharpe": (OptimizationMethod.MARKOWITZ, OptimizationObjective.MAX_SHARPE),
    "min_volatility": (
        OptimizationMethod.MARKOWITZ,
        OptimizationObjective.MIN_VOLATILITY,
    ),
    "min_vol": (OptimizationMethod.MARKOWITZ, OptimizationObjective.MIN_VOLATILITY),
    "equal_weight": None,
    "equal": None,
    "buy_and_hold": "buy_and_hold",
    "buyhold": "buy_and_hold",
}


def build_strategy(
    spec: str,
    lookback_days: int = 252,
    risk_free_rate: float = 0.04,
    fixed_weights: Optional[Dict[str, float]] = None,
) -> Strategy:
    """Build a Strategy from a short spec string.

    Recognized specs: ``hrp``, ``markowitz``/``max_sharpe``, ``min_volatility``,
    ``equal_weight``, ``buy_and_hold``, ``fixed`` (requires ``fixed_weights``).
    """
    key = spec.strip().lower()

    if key in ("fixed", "fixed_weights"):
        if not fixed_weights:
            raise ValueError("The 'fixed' strategy requires fixed_weights")
        return FixedWeightStrategy(fixed_weights)

    if key not in _METHOD_ALIASES:
        raise ValueError(
            f"Unknown strategy '{spec}'. Valid: hrp, markowitz, max_sharpe, "
            "min_volatility, equal_weight, buy_and_hold, fixed"
        )

    mapping = _METHOD_ALIASES[key]
    if mapping is None:
        return EqualWeightStrategy()
    if mapping == "buy_and_hold":
        return BuyAndHoldStrategy(weights=fixed_weights)

    method, objective = mapping
    return OptimizerStrategy(
        method=method,
        objective=objective,
        lookback_days=lookback_days,
        risk_free_rate=risk_free_rate,
    )


def build_strategies(
    specs: List[str],
    lookback_days: int = 252,
    risk_free_rate: float = 0.04,
    fixed_weights: Optional[Dict[str, float]] = None,
) -> List[Strategy]:
    """Build multiple strategies from a list of spec strings."""
    return [
        build_strategy(s, lookback_days, risk_free_rate, fixed_weights) for s in specs
    ]
