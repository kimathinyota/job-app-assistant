# backend/routes/prompt.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import AIPromptResponse # Import the response model

router = APIRouter()
registry = Registry()

@router.post("/generate-cv-prompt", response_model=AIPromptResponse)
def generate_cv_prompt(base_cv_id: str, job_id: str):
    """
    Generates a structured JSON prompt payload for an AI service to create a Derived CV.
    Requires an existing CV, Job, and an implicit Mapping between them.
    """
    try:
        return registry.generate_cv_prompt(base_cv_id, job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/generate-coverletter-prompt", response_model=AIPromptResponse)
def generate_cover_letter_prompt(mapping_id: str):
    """
    Generates a structured JSON prompt payload for an AI service to draft a Cover Letter.
    Requires an existing Mapping and related Cover Letter Ideas/Job/CV data.
    """
    try:
        return registry.generate_coverletter_prompt(mapping_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))