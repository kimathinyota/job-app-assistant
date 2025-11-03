# backend/routes/coverletter.py

from fastapi import APIRouter, HTTPException, Request
from backend.core.registry import Registry
from typing import Optional, List # Ensure List is imported

router = APIRouter()

@router.post("/")
def create_cover_letter(job_id: str, base_cv_id: str, mapping_id: str, request: Request):
    """Create a new CoverLetter record."""
    registry = request.app.state.registry
    return registry.create_cover_letter(job_id, base_cv_id, mapping_id)

@router.get("/{cover_id}")
def get_cover_letter(cover_id: str, request: Request):
    """Fetch a specific CoverLetter by ID."""
    registry = request.app.state.registry
    cover = registry.get_cover_letter(cover_id)
    if not cover:
        raise HTTPException(status_code=404, detail="CoverLetter not found")
    return cover

@router.delete("/{cover_id}")
def delete_cover_letter(cover_id: str, request: Request):
    """Delete a CoverLetter record by ID."""
    try:
        registry = request.app.state.registry
        return registry.delete_cover_letter(cover_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------
# NESTED ADD ENDPOINTS (Cover Letter Components)
# ---------------------------------------------------------------------

@router.post("/{cover_id}/idea")
def add_idea(cover_id: str, title: str, request: Request, description: Optional[str] = None, mapping_pair_ids: List[str] = []):
    """Add a new core idea/talking point for the cover letter."""
    try:
        registry = request.app.state.registry
        return registry.add_cover_letter_idea(cover_id, title=title, description=description, mapping_pair_ids=mapping_pair_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cover_id}/paragraph")
def add_paragraph(cover_id: str, idea_ids: List[str], purpose: str, request: Request, draft_text: Optional[str] = None):
    """Add a paragraph structure based on existing ideas (used for generation outline)."""
    try:
        registry = request.app.state.registry
        return registry.add_cover_letter_paragraph(cover_id, idea_ids=idea_ids, purpose=purpose, draft_text=draft_text)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))