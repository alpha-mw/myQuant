"""Uvicorn entry point for the myQuant web application."""

from pathlib import Path

from web.api import app as workspace_app
from web.config import API_HOST, API_PORT, warn_if_insecure_binding

app = workspace_app

if __name__ == "__main__":
    import uvicorn

    warn_if_insecure_binding(API_HOST)
    uvicorn.run(
        "web.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        reload_dirs=[str(Path(__file__).parent)],
    )
