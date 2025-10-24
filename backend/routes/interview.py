from fastapi import APIRouter
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")
def create_interview(application_id: str):
    return registry.create_interview(application_id)

@router.post("/{interview_id}/stage")
def add_stage(interview_id: str, name: str, description: Optional[str] = None):
    return registry.add_interview_stage(interview_id, name, description)

@router.post("/{interview_id}/question")
def add_question(interview_id: str, stage_name: str, question: str, answer: Optional[str] = None):
    return registry.add_interview_question(interview_id, stage_name, question, answer)

@router.get("/{interview_id}")
def get_interview(interview_id: str):
    return registry.get_interview(interview_id)
