"""Workspace web application exports."""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    from web.workspace_app import create_app as workspace_create_app

    return workspace_create_app(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name == "app":
        from web.workspace_app import app

        return app
    raise AttributeError(name)

__all__ = ["app", "create_app"]
