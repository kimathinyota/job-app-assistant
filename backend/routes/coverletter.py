# backend/routes/coverletter.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")
def create_cover_letter(job_id: str, base_cv_id: str, mapping_id: str):
    """Create a new CoverLetter record."""
    return registry.create_cover_letter(job_id, base_cv_id, mapping_id)

@router.get("/{cover_id}")
def get_cover_letter(cover_id: str):
    """Fetch a specific CoverLetter by ID."""
    cover = registry.get_cover_letter(cover_id)
    if not cover:
        raise HTTPException(status_code=404, detail="CoverLetter not found")
    return cover

@router.delete("/{cover_id}")
def delete_cover_letter(cover_id: str):
    """Delete a CoverLetter record by ID."""
    try:
        return registry.delete_cover_letter(cover_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))