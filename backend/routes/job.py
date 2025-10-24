from fastapi import APIRouter
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")

def create_job(title: str, company: str, notes: Optional[str] = None):
    return registry.create_job(title, company, notes)

@router.post("/{job_id}/feature")
def add_feature(job_id: str, description: str, type: str = "requirement"):
    return registry.add_job_feature(job_id, description, type)

@router.get("/")
def list_jobs():
    return registry.all_jobs()

@router.get("/{job_id}")
def get_job(job_id: str):
    return registry.get_job(job_id)
