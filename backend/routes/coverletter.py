from fastapi import APIRouter
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")
def create_cover_letter(job_id: str, base_cv_id: str, mapping_id: str):
    return registry.create_cover_letter(job_id, base_cv_id, mapping_id)

@router.get("/{cover_id}")
def get_cover_letter(cover_id: str):
    return registry.get_cover_letter(cover_id)
