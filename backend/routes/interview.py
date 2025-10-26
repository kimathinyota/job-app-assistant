# backend/routes/interview.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")
def create_interview(application_id: str):
    """Create a new Interview process record linked to an Application."""
    return registry.create_interview(application_id)

@router.post("/{interview_id}/stage")
def add_stage(interview_id: str, name: str, description: Optional[str] = None):
    """Add a new stage (e.g., 'Technical Screening') to the interview process."""
    try:
        return registry.add_interview_stage(interview_id, name, description)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{interview_id}/question")
def add_question(interview_id: str, stage_name: str, question: str, answer: Optional[str] = None):
    """Add a question/answer pair to a specific stage of the interview."""
    try:
        return registry.add_interview_question(interview_id, stage_name, question, answer)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{interview_id}")
def get_interview(interview_id: str):
    """Fetch a specific Interview record by ID."""
    interview = registry.get_interview(interview_id)
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    return interview

@router.delete("/{interview_id}")
def delete_interview(interview_id: str):
    """Delete an Interview record by ID."""
    try:
        return registry.delete_interview(interview_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
