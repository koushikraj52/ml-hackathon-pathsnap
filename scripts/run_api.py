"""
FastAPI Server Launcher for Laptop 1.
Usage:
  python scripts/run_api.py [--host 0.0.0.0] [--port 8000]
"""

import os
import sys
import argparse
import uvicorn

repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PathSnap FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting PathSnap API Server on http://{args.host}:{args.port} ...")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=args.reload)
