"""Settings API endpoints."""

from fastapi import APIRouter

from web.models.settings_models import SettingsResponse, SettingsUpdateRequest
from web.services import settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsResponse)
def get_settings():
    data = settings_service.get_settings()
    return data


@router.put("")
def update_settings(req: SettingsUpdateRequest):
    updates = req.model_dump(exclude_none=True)
    return settings_service.update_settings(updates)


@router.get("/credentials/status")
def get_credentials_status():
    return {"credentials": settings_service.get_credentials_status()}
