# backend/routes/cv.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import CVUpdate # Import the update model

from typing import Optional


router = APIRouter()
registry = Registry()

@router.post("/")
def create_cv(name: str, summary: Optional[str] = None):
    """Create a new base CV."""
    return registry.create_cv(name, summary)

@router.get("/")
def list_cvs():
    """List all base CVs."""
    return registry.all_cvs()

@router.get("/{cv_id}")
def get_cv(cv_id: str):
    """Fetch a specific CV by ID."""
    cv = registry.get_cv(cv_id)
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")
    return cv

@router.patch("/{cv_id}")
def update_cv(cv_id: str, data: CVUpdate):
    """Update general CV metadata (name, summary, contact_info)."""
    try:
        return registry.update_cv(cv_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{cv_id}")
def delete_cv(cv_id: str):
    """Delete a CV by ID."""
    try:
        return registry.delete_cv(cv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))