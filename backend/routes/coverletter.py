# backend/routes/coverletter.py

from fastapi import APIRouter, HTTPException, Request, Query # <-- ADD Query
from backend.core.registry import Registry
from typing import Optional, List, Literal
# --- 1. IMPORT THE UPDATE MODELS ---
from backend.core.models import IdeaUpdate, ParagraphUpdate


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
def add_idea(cover_id: str, title: str, request: Request, 
             description: Optional[str] = None, 
             # vvv THIS IS THE FIX vvv
             mapping_pair_ids: List[str] = Query([]),
             annotation: Optional[str] = None):
    """Add a new core idea/talking point for the cover letter."""
    try:
        registry = request.app.state.registry
        return registry.add_cover_letter_idea(
            cover_id, 
            title=title, 
            description=description, 
            mapping_pair_ids=mapping_pair_ids,
            annotation=annotation # <-- 2. Pass annotation here
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cover_id}/paragraph")
def add_paragraph(cover_id: str, purpose: str, request: Request, 
                  idea_ids: List[str] = [],  # <-- THIS IS THE FIX
                  draft_text: Optional[str] = None):
    """Add a paragraph structure based on existing ideas (used for generation outline)."""
    try:
        registry = request.app.state.registry
        return registry.add_cover_letter_paragraph(
            cover_id, 
            idea_ids=idea_ids, 
            purpose=purpose, 
            draft_text=draft_text
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- 2. ADD THESE NEW UPDATE (PATCH) ROUTES ---

@router.patch("/{cover_id}/idea/{idea_id}")
def update_idea(cover_id: str, idea_id: str, data: IdeaUpdate, request: Request):
    """Update an idea (e.g., its title, or its list of pairs)."""
    try:
        registry = request.app.state.registry
        # This registry method already exists and supports this
        return registry.update_cover_letter_idea(cover_id, idea_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cover_id}/paragraph/{para_id}")
def update_paragraph(cover_id: str, para_id: str, data: ParagraphUpdate, request: Request):
    """Update a paragraph (e.g., its purpose, or its list of ideas)."""
    try:
        registry = request.app.state.registry
        # This registry method already exists and supports this
        return registry.update_cover_letter_paragraph(cover_id, para_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- 3. (Optional but recommended) ADD DELETE ROUTES ---

@router.delete("/{cover_id}/idea/{idea_id}")
def delete_idea(cover_id: str, idea_id: str, request: Request):
    """Deletes an idea from the cover letter."""
    try:
        registry = request.app.state.registry
        return registry.delete_cover_letter_idea(cover_id, idea_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{cover_id}/paragraph/{para_id}")
def delete_paragraph(cover_id: str, para_id: str, request: Request):
    """Deletes a paragraph from the cover letter."""
    try:
        registry = request.app.state.registry
        return registry.delete_cover_letter_paragraph(cover_id, para_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.post("/{cover_id}/autofill")
def autofill_cover_letter(
    cover_id: str, 
    request: Request,
    strategy: Literal["standard", "mission_driven", "specialist"] = Query("standard"),
    mode: Literal["reset", "augment"] = Query("reset")
):
    """
    Auto-generates or re-orchestrates the cover letter outline.
    Based on the "Ownership & Re-Classification Engine" logic.
    
    - strategy: Defines the paragraph order.
    - mode: 'reset' (default) surgically clears all AI-generated content 
            and rebuilds it around user content.
            'augment' (future) would only add new, un-used evidence.
    """
    try:
        registry = request.app.state.registry
        # All the complex logic is delegated to the registry method
        # we designed earlier (autofill_cover_letter in registry.py)
        cover_letter_with_outline = registry.autofill_cover_letter(
            cover_id, 
            strategy=strategy, 
            mode=mode
        )
        return cover_letter_with_outline
    except ValueError as e:
        # This will catch "Cover letter not found" etc.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # General catch-all for any unexpected clustering errors
        print(f"Error during autofill: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred during outline generation.")