"""Console entry point: `uv run python -m advisor.run`."""

from __future__ import annotations

import asyncio
import logging

from advisor.config.settings import load_settings
from advisor.console.repl import Repl


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()
    asyncio.run(Repl(settings).run())


if __name__ == "__main__":
    main()
