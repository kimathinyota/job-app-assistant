# backend/routes/job.py

from fastapi import APIRouter, HTTPException, Request, Depends
from backend.core.registry import Registry
from backend.core.models import JobDescriptionUpdate, JobUpsertPayload, User
from backend.routes.auth import get_current_user # Adjust import path if needed
from typing import Optional
from pydantic import BaseModel, ValidationError
from backend.tasks import task_parse_job
import logging as log
from redis import Redis
from rq import Queue

router = APIRouter()
redis_conn = Redis(host='localhost', port=6379)
q = Queue(connection=redis_conn)

class JobTextRequest(BaseModel):
    text: str

@router.post("/parse")
async def parse_job_description(
    request: Request, 
    job_request: JobTextRequest,
    user: User = Depends(get_current_user) # Secured endpoint
):
    """
    Parses a raw job description using the loaded Llama 3 model.
    """
    parser = getattr(request.app.state, "job_parser", None)
    
    if not parser:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded on the server. Check server logs."
        )
    
    # Run inference
    result = await parser.fast_parse(job_request.text)
    
    if "error" in result and result["error"] != "Failed to generate valid JSON":
         raise HTTPException(status_code=400, detail=result["error"])

    return result



@router.post("/parse_external")
async def parse_external_job(payload: JobTextRequest):
    # 1. Send text to Redis
    job = q.enqueue(task_parse_job, payload.text)
    
    # 2. Return the Ticket ID
    return {"job_id": job.get_id(), "status": "processing"}

@router.get("/status/{job_id}")
def check_status(job_id: str):
    job = q.fetch_job(job_id)
    
    if job.is_finished:
        # RETURN THE PARSED JSON
        return {"status": "finished", "data": job.result} 
    elif job.is_failed:
        return {"status": "failed", "error": str(job.exc_info)}
    else:
        return {"status": "processing"}

@router.post("/")
def create_job(
    title: str, 
    company: str, 
    request: Request, 
    notes: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Create a new job description."""
    registry: Registry = request.app.state.registry
    return registry.create_job(user.id, title, company, notes)

@router.get("/")
def list_jobs(
    request: Request,
    user: User = Depends(get_current_user)
):
    """List all job descriptions belonging to the user."""
    registry: Registry = request.app.state.registry
    return registry.all_jobs(user.id)

@router.get("/{job_id}")
def get_job(
    job_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific job by ID."""
    registry: Registry = request.app.state.registry
    job = registry.get_job(job_id, user.id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.patch("/{job_id}")
def update_job(
    job_id: str, 
    data: JobDescriptionUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update job metadata (title, company, notes)."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_job(user.id, job_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{job_id}")
def delete_job(
    job_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete a job description by ID."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_job(user.id, job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{job_id}/feature")
def add_feature(
    job_id: str, 
    description: str, 
    request: Request, 
    type: str = "requirement",
    user: User = Depends(get_current_user)
):
    """Add a feature/requirement to a job description."""
    try:
        registry: Registry = request.app.state.registry
        return registry.add_job_feature(user.id, job_id, description, type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.delete("/{job_id}/feature/{feature_id}")
def delete_feature(
    job_id: str, 
    feature_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Deletes a feature/requirement from a job description."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_job_feature(user.id, job_id, feature_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.post("/upsert")
def upsert_job(
    payload: JobUpsertPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Creates a new job (if payload.id is null) or
    fully updates an existing job (if payload.id is provided).
    """
    try:
        registry: Registry = request.app.state.registry
        return registry.upsert_job(user.id, payload)
    except ValidationError as e:
        log.error(f"Validation error in upsert_job: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")
    except ValueError as e:
        log.error(f"Value error in upsert_job: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Unexpected error in upsert_job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")