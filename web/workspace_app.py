"""FastAPI application for the research workspace runtime."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from web.config import CORS_ORIGINS, PROJECT_ROOT
from web.api.data import router as data_router
from web.routers import presets, research, settings, universe
from web.services.run_history_store import history_store


FRONTEND_STATIC_EXTENSIONS = {
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".map",
    ".png",
    ".svg",
    ".txt",
    ".webp",
    ".woff",
    ".woff2",
}


def _serve_frontend_asset(
    frontend_dist: Path,
    requested_path: str,
) -> FileResponse:
    frontend_root = frontend_dist.resolve()
    index_path = frontend_root / "index.html"
    normalized_path = requested_path.lstrip("/")
    candidate = (frontend_root / normalized_path).resolve()

    try:
        candidate.relative_to(frontend_root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Not Found") from exc

    if candidate.is_file():
        return FileResponse(candidate)

    if candidate.suffix.lower() in FRONTEND_STATIC_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Not Found")

    return FileResponse(index_path)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    history_store.init_db()
    yield


def create_app(frontend_dist: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="myQuant Research Workspace",
        version="12.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(research.router)
    app.include_router(presets.router)
    app.include_router(settings.router)
    app.include_router(universe.router)
    app.include_router(data_router, prefix="/api")

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {"ok": True, "version": "12.0.0"}

    frontend_dist = frontend_dist or (PROJECT_ROOT / "frontend" / "dist")
    if frontend_dist.exists():

        @app.get("/", include_in_schema=False)
        async def serve_frontend_root() -> FileResponse:
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            return _serve_frontend_asset(frontend_dist, full_path)

    return app


app = create_app()
