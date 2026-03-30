"""Preset CRUD endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from web.models.research_models import (
    PresetCreateRequest,
    PresetListResponse,
    PresetResponse,
    PresetUpdateRequest,
)
from web.services.preset_store import preset_store

router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("/", response_model=PresetListResponse)
async def list_presets():
    items = preset_store.list_presets()
    return PresetListResponse(presets=[PresetResponse(**p) for p in items])


@router.get("/{preset_id}", response_model=PresetResponse)
async def get_preset(preset_id: str):
    item = preset_store.get_preset(preset_id)
    if not item:
        raise HTTPException(status_code=404, detail="Preset not found")
    return PresetResponse(**item)


@router.post("/", response_model=PresetResponse)
async def create_preset(request: PresetCreateRequest):
    item = preset_store.create_preset(
        name=request.name,
        description=request.description,
        config=request.config.model_dump(),
    )
    return PresetResponse(**item)


@router.put("/{preset_id}", response_model=PresetResponse)
async def update_preset(preset_id: str, request: PresetUpdateRequest):
    config_dict = request.config.model_dump() if request.config else None
    item = preset_store.update_preset(
        preset_id=preset_id,
        name=request.name,
        description=request.description,
        config=config_dict,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Preset not found")
    return PresetResponse(**item)


@router.delete("/{preset_id}")
async def delete_preset(preset_id: str):
    deleted = preset_store.delete_preset(preset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"ok": True, "deleted": preset_id}
