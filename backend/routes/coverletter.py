# backend/routes/coverletter.py

from fastapi import APIRouter, HTTPException, Request, Query, Depends
from typing import Optional, List, Literal
from backend.core.registry import Registry
from backend.core.models import IdeaUpdate, ParagraphUpdate, User
from backend.routes.auth import get_current_user # Adjust import path if located elsewhere

router = APIRouter()

@router.post("/")
def create_cover_letter(
    job_id: str, 
    base_cv_id: str, 
    mapping_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Create a new CoverLetter record."""
    registry: Registry = request.app.state.registry
    return registry.create_cover_letter(user.id, job_id, base_cv_id, mapping_id)

@router.get("/{cover_id}")
def get_cover_letter(
    cover_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific CoverLetter by ID."""
    registry: Registry = request.app.state.registry
    # Pass user.id to ensure ownership
    cover = registry.get_cover_letter(cover_id, user.id)
    if not cover:
        raise HTTPException(status_code=404, detail="CoverLetter not found")
    return cover

@router.delete("/{cover_id}")
def delete_cover_letter(
    cover_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete a CoverLetter record by ID."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_cover_letter(user.id, cover_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------
# NESTED ADD ENDPOINTS (Cover Letter Components)
# ---------------------------------------------------------------------

@router.post("/{cover_id}/idea")
def add_idea(
    cover_id: str, 
    title: str, 
    request: Request, 
    description: Optional[str] = None, 
    mapping_pair_ids: List[str] = Query([]),
    annotation: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Add a new core idea/talking point for the cover letter."""
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cover_letter_idea(
            user.id, # <--- Security
            cover_id, 
            title=title, 
            description=description, 
            mapping_pair_ids=mapping_pair_ids,
            annotation=annotation
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cover_id}/paragraph")
def add_paragraph(
    cover_id: str, 
    purpose: str, 
    request: Request, 
    idea_ids: List[str] = Query([]), 
    draft_text: Optional[str] = None,
    order: Optional[int] = Query(None),
    user: User = Depends(get_current_user)
):
    """Add a paragraph structure. Accepts optional order for insertion."""
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cover_letter_paragraph(
            user.id, # <--- Security
            cover_id, 
            idea_ids=idea_ids, 
            purpose=purpose, 
            draft_text=draft_text,
            order=order
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ---------------------------------------------------------------------
# NESTED UPDATE (PATCH) ROUTES
# ---------------------------------------------------------------------

@router.patch("/{cover_id}/idea/{idea_id}")
def update_idea(
    cover_id: str, 
    idea_id: str, 
    data: IdeaUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update an idea (e.g., its title, or its list of pairs)."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_cover_letter_idea(user.id, cover_id, idea_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cover_id}/paragraph/{para_id}")
def update_paragraph(
    cover_id: str, 
    para_id: str, 
    data: ParagraphUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update a paragraph (e.g., its purpose, or its list of ideas)."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_cover_letter_paragraph(user.id, cover_id, para_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ---------------------------------------------------------------------
# NESTED DELETE ROUTES
# ---------------------------------------------------------------------

@router.delete("/{cover_id}/idea/{idea_id}")
def delete_idea(
    cover_id: str, 
    idea_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Deletes an idea from the cover letter."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_cover_letter_idea(user.id, cover_id, idea_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{cover_id}/paragraph/{para_id}")
def delete_paragraph(
    cover_id: str, 
    para_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Deletes a paragraph from the cover letter."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_cover_letter_paragraph(user.id, cover_id, para_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.patch("/{cover_id}/metadata")
def update_cover_letter_metadata(
    cover_id: str, 
    request: Request, 
    name: str = Query(...),
    user: User = Depends(get_current_user)
):
    """Rename the document."""
    try:
        registry: Registry = request.app.state.registry
        return registry.rename_cover_letter(user.id, cover_id, name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{cover_id}/autofill")
def autofill_cover_letter(
    cover_id: str, 
    request: Request,
    strategy: Literal["standard", "mission_driven", "specialist"] = Query("standard"),
    mode: Literal["reset", "augment"] = Query("reset"),
    user: User = Depends(get_current_user)
):
    """
    Auto-generates or re-orchestrates the cover letter outline.
    """
    try:
        registry: Registry = request.app.state.registry
        cover_letter_with_outline = registry.autofill_cover_letter(
            user.id, # <--- Security
            cover_id, 
            strategy=strategy, 
            mode=mode
        )
        return cover_letter_with_outline
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error during autofill: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during outline generation")