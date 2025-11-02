# backend/routes/job.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import JobDescriptionUpdate # Import the update model
from typing import Optional
from backend.core.dependencies import registry 

router = APIRouter()

@router.post("/")
def create_job(title: str, company: str, notes: Optional[str] = None):
    """Create a new job description."""
    return registry.create_job(title, company, notes)

@router.get("/")
def list_jobs():
    """List all job descriptions."""
    return registry.all_jobs()

@router.get("/{job_id}")
def get_job(job_id: str):
    """Fetch a specific job by ID."""
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.patch("/{job_id}")
def update_job(job_id: str, data: JobDescriptionUpdate):
    """Update job metadata (title, company, notes)."""
    try:
        return registry.update_job(job_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{job_id}")
def delete_job(job_id: str):
    """Delete a job description by ID."""
    try:
        return registry.delete_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{job_id}/feature")
def add_feature(job_id: str, description: str, type: str = "requirement"):
    """Add a feature/requirement to a job description."""
    try:
        return registry.add_job_feature(job_id, description, type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))