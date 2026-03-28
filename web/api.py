"""FastAPI application for the myQuant research workspace."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web.routers import presets, research, settings, universe
from web.services.run_history_store import history_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load .env if present
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    history_store.init_db()
    yield


app = FastAPI(
    title="myQuant Research Workspace",
    version="12.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(research.router)
app.include_router(presets.router)
app.include_router(settings.router)
app.include_router(universe.router)


@app.get("/api/health")
async def health():
    return {"ok": True, "version": "12.0.0"}
