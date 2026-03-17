"""Uvicorn entry point for the myQuant web application."""

import sys
from pathlib import Path

# Ensure project root is in sys.path so `web.*` imports work
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from web.app import create_app
from web.config import API_HOST, API_PORT

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
