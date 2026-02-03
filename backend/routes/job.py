# backend/routes/job.py

from fastapi import APIRouter, HTTPException, Request
from backend.core.registry import Registry
from backend.core.models import JobDescriptionUpdate, JobUpsertPayload # Import the update model
from typing import Optional
from pydantic import BaseModel
router = APIRouter()

class JobTextRequest(BaseModel):
    text: str

@router.post("/parse")
async def parse_job_description(request: Request, job_request: JobTextRequest):
    """
    Parses a raw job description using the loaded Llama 3 model.
    """
    parser = getattr(request.app.state, "job_parser", None)
    
    if not parser:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded on the server. Check server logs."
        )
    
    # Run inference (this is blocking CPU code, so fast/short requests are okay, 
    # but for heavy loads you might want to run this in a threadpool)
    result = await parser.fast_parse(job_request.text)
    
    if "error" in result and result["error"] != "Failed to generate valid JSON":
         raise HTTPException(status_code=400, detail=result["error"])

    return result

@router.post("/")
def create_job(title: str, company: str, request: Request, notes: Optional[str] = None):
    """Create a new job description."""
    registry = request.app.state.registry
    return registry.create_job(title, company, notes)

@router.get("/")
def list_jobs(request: Request):
    """List all job descriptions."""
    registry = request.app.state.registry
    return registry.all_jobs()

@router.get("/{job_id}")
def get_job(job_id: str, request: Request):
    """Fetch a specific job by ID."""
    registry = request.app.state.registry
    job = registry.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.patch("/{job_id}")
def update_job(job_id: str, data: JobDescriptionUpdate, request: Request):
    """Update job metadata (title, company, notes)."""
    try:
        registry = request.app.state.registry
        return registry.update_job(job_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{job_id}")
def delete_job(job_id: str, request: Request):
    """Delete a job description by ID."""
    try:
        registry = request.app.state.registry
        return registry.delete_job(job_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{job_id}/feature")
def add_feature(job_id: str, description: str, request: Request, type: str = "requirement"):
    """Add a feature/requirement to a job description."""
    try:
        registry = request.app.state.registry
        return registry.add_job_feature(job_id, description, type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

# --- ADD THIS NEW ROUTE ---
@router.delete("/{job_id}/feature/{feature_id}")
def delete_feature(job_id: str, feature_id: str, request: Request):
    """Deletes a feature/requirement from a job description."""
    try:
        registry = request.app.state.registry
        return registry.delete_job_feature(job_id, feature_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.post("/upsert")
def upsert_job(payload: JobUpsertPayload, request: Request):
    """
    Creates a new job (if payload.id is null) or
    fully updates an existing job (if payload.id is provided).
    This single endpoint replaces create, update, and feature management.
    """
    try:
        registry = request.app.state.registry
        return registry.upsert_job(payload)
    # --- 2. ADD THIS CATCH BLOCK ---
    except ValidationError as e:
        log.error(f"Validation error in upsert_job: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(e)}")
    # --- END OF FIX ---
    except ValueError as e:
        log.error(f"Value error in upsert_job: {e}", exc_info=True)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Unexpected error in upsert_job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")