# backend/routes/coverletter.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from typing import Optional, List # Ensure List is imported

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


# ---------------------------------------------------------------------
# NESTED ADD ENDPOINTS (Cover Letter Components)
# ---------------------------------------------------------------------

@router.post("/{cover_id}/idea")
def add_idea(cover_id: str, title: str, description: Optional[str] = None, mapping_pair_ids: List[str] = []):
    """Add a new core idea/talking point for the cover letter."""
    try:
        return registry.add_cover_letter_idea(cover_id, title=title, description=description, mapping_pair_ids=mapping_pair_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cover_id}/paragraph")
def add_paragraph(cover_id: str, idea_ids: List[str], purpose: str, draft_text: Optional[str] = None):
    """Add a paragraph structure based on existing ideas (used for generation outline)."""
    try:
        return registry.add_cover_letter_paragraph(cover_id, idea_ids=idea_ids, purpose=purpose, draft_text=draft_text)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))