from fastapi import APIRouter
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")
def create_application(job_id: str, base_cv_id: str, mapping_id: Optional[str] = None, derived_cv_id: Optional[str] = None):
    return registry.create_application(job_id, base_cv_id, mapping_id, derived_cv_id)

@router.get("/")
def list_applications():
    return registry.all_applications()

@router.get("/{app_id}")
def get_application(app_id: str):
    return registry.get_application(app_id)
