"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from web.config import CORS_ORIGINS


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


def _serve_frontend_asset(frontend_dist: Path, requested_path: str) -> FileResponse:
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


def create_app(frontend_dist: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="myQuant Web API",
        description="Quantitative Investment Research Platform",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from web.api.analysis import router as analysis_router
    from web.api.data import router as data_router
    from web.api.portfolio import router as portfolio_router
    from web.api.settings import router as settings_router

    app.include_router(analysis_router, prefix="/api/v1")
    app.include_router(data_router, prefix="/api/v1")
    app.include_router(portfolio_router, prefix="/api/v1")
    app.include_router(settings_router, prefix="/api/v1")

    # Health check
    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    # Serve frontend static files in production
    frontend_dist = frontend_dist or (Path(__file__).parent.parent / "frontend" / "dist")
    if frontend_dist.exists():
        @app.get("/", include_in_schema=False)
        def serve_frontend_root():
            return FileResponse(frontend_dist / "index.html")

        @app.get("/{full_path:path}", include_in_schema=False)
        def serve_frontend(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            return _serve_frontend_asset(frontend_dist, full_path)

    return app
