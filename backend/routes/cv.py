from fastapi import APIRouter
from backend.core.registry import Registry

from typing import Optional



router = APIRouter()
registry = Registry()

@router.post("/")
def create_cv(name: str, summary: Optional[str] = None):
    return registry.create_cv(name, summary)

@router.get("/")
def list_cvs():
    return registry.all_cvs()

@router.get("/{cv_id}")
def get_cv(cv_id: str):
    return registry.get_cv(cv_id)
