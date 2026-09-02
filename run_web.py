#!/usr/bin/env python3
"""Run the FinAI FastAPI web app.

    python run_web.py            # http://localhost:8000
    python run_web.py --reload   # dev auto-reload
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser(description="FinAI web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    print("🚀 FinAI web app  ->  http://%s:%d" % (args.host, args.port))
    uvicorn.run(
        "src.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(_ROOT / "src" / "web")] if args.reload else None,
    )


if __name__ == "__main__":
    main()
