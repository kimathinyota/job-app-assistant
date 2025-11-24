# backend/routes/prompt.py


from fastapi import APIRouter, HTTPException, Query, Request # <-- 1. Import Query
from backend.core.registry import Registry
from backend.core.models import AIPromptResponse, CoverLetterPromptPayload # Import the response model
from typing import Optional, List # <-- 2. Import Optional and List

router = APIRouter()

@router.post("/generate-cv-prompt", response_model=AIPromptResponse)
def generate_cv_prompt(
    base_cv_id: str, 
    job_id: str,
    request: Request,
    # 3. Add the new query parameter
    selected_skill_ids: Optional[List[str]] = Query(None) 
):
    """
    Generates a structured JSON prompt payload for an AI service to create a Derived CV.
    Requires an existing CV, Job, and an implicit Mapping between them.
    Can optionally accept a specific list of skills to include.
    """
    try:
        # 4. Pass the new parameter to the registry
        registry = request.app.state.registry
        return registry.generate_cv_prompt(base_cv_id, job_id, selected_skill_ids)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    
@router.get("/cover-letter-payload/{cover_id}", response_model=CoverLetterPromptPayload)
def get_cover_letter_ai_payload(cover_id: str, request: Request):
    """
    Generates the advanced 'Greedy' Context Assembler payload.
    Provides the AI with the Full Graph (Mapped + Unmapped) to allow creative connections.
    """
    try:
        registry = request.app.state.registry
        return registry.construct_advanced_cover_letter_prompt(cover_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Log the full error in production
        print(f"Error generating prompt payload: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error while assembling prompt.")