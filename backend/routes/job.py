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
from backend.core.services.scoring import ScoringService

from datetime import datetime

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

# @router.get("/")
# def list_jobs(
#     request: Request,
#     user: User = Depends(get_current_user)
# ):
#     """List all job descriptions belonging to the user."""
#     registry: Registry = request.app.state.registry
#     return registry.all_jobs(user.id)


@router.get("/")
def list_jobs(
    request: Request,
    sort: str = "recommended", # 'recommended', 'date', 'deadline', 'score'
    q: Optional[str] = None,   # Search Query
    user: User = Depends(get_current_user)
):
    """List jobs with Advanced Sorting, Search, and Date Intelligence."""
    registry: Registry = request.app.state.registry
    jobs = registry.all_jobs(user.id)
    
    # 1. SEARCH FILTER (Keyword & Semantic Lite)
    if q:
        query = q.lower()
        filtered_jobs = []
        for job in jobs:
            # Basic text search
            text_corpus = f"{job.title} {job.company} {job.notes or ''}".lower()
            # Check features too
            feature_text = " ".join([f.description.lower() for f in job.features])
            
            if query in text_corpus or query in feature_text:
                filtered_jobs.append(job)
        jobs = filtered_jobs

    # 2. DATE PARSING & PRIORITY LOGIC
    now = datetime.utcnow()
    
    def get_date_obj(date_str):
        if not date_str: return None
        try: return date_parser.parse(date_str)
        except: return None

    def calculate_priority(job):
        # 1. Deadline Panic (< 7 days)
        closing = get_date_obj(job.date_closing or job.application_end_date)
        if closing:
            days_left = (closing - now).days
            if 0 <= days_left <= 7: return 1000 - days_left # Highest priority
        
        # 2. Freshness (< 2 days)
        posted = get_date_obj(job.date_posted or job.date_extracted or job.created_at)
        if posted:
            hours_old = (now - posted).total_seconds() / 3600
            if hours_old < 48: return 500
            if hours_old > (30 * 24): return -100 # Stale
            
        return 0

    # 3. SORTING LOGIC
    if sort == "score":
        # Sort by match_score DESC
        jobs.sort(key=lambda x: getattr(x, 'match_score', 0), reverse=True)
        
    elif sort == "deadline":
        # Sort by date_closing ASC (Future only)
        # Push None to end
        def deadline_key(j):
            d = get_date_obj(j.date_closing or j.application_end_date)
            return d if d and d > now else datetime.max
        jobs.sort(key=deadline_key)
        
    elif sort == "recommended":
        # Smart Sort: Match Score + Priority Boost
        def smart_key(j):
            base_score = getattr(j, 'match_score', 0)
            priority_boost = calculate_priority(j)
            return base_score + priority_boost
        jobs.sort(key=smart_key, reverse=True)
        
    else: # Default: Newest Created/Posted
        def date_key(j):
            return get_date_obj(j.date_posted or j.created_at) or datetime.min
        jobs.sort(key=date_key, reverse=True)

    return jobs


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


@router.post("/score-all")
def score_all_jobs(
    request: Request,
    cv_id: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Runs AI analysis on ALL jobs against a specific CV (or user default)."""
    registry: Registry = request.app.state.registry
    inferer = request.app.state.inferer # Re-use the loaded model
    
    # 1. Determine CV
    target_cv_id = cv_id or user.primary_cv_id
    if not target_cv_id:
        # Fallback: Get first CV
        all_cvs = registry.all_cvs(user.id)
        if all_cvs: target_cv_id = all_cvs[0].id
    
    if not target_cv_id:
        raise HTTPException(400, "No CV provided and no default CV found.")

    service = ScoringService(registry, inferer)
    jobs = registry.all_jobs(user.id)
    scored_count = 0
    
    # In production, this should be a background task!
    for job in jobs:
        try:
            service.score_job(user.id, job.id, target_cv_id)
            scored_count += 1
        except Exception as e:
            log.error(f"Failed to score job {job.id}: {e}")
            
    return {"status": "success", "scored_count": scored_count, "cv_id": target_cv_id}



from backend.core.services.scoring import ScoringService

@router.get("/{job_id}/match-preview")
def preview_job_match(
    job_id: str,
    cv_id: str,
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Calculates a score on-the-fly for a Job + CV combo.
    Does NOT update the Job's default cache.
    """
    registry: Registry = request.app.state.registry
    
    # Init service
    inferer = getattr(request.app.state, "inferer", None)
    service = ScoringService(registry, inferer)
    
    # 1. Fetch Entities
    job = registry.get_job(job_id, user.id)
    cv = registry.get_cv(cv_id, user.id)

    print(f" [Preview] Fetching Job and CV for User ID: {user.id}...", "\n\n")
    print(f" [Preview] Job ID: {job_id}, CV ID: {cv_id}",  "\n\n")
    if not job or not cv:
        raise HTTPException(404, "Job or CV not found")
        
    # 2. Run the Logic (Reusing internal helper)
    # We use _get_or_create_smart_mapping to ensure we leverage existing AI work
    mapping = service._get_or_create_smart_mapping(user.id, job, cv)

    print(f" [Preview] Mapping Mapping ID: {mapping.id if mapping else 'N/A'}, Pairs Count: {len(mapping.pairs) if mapping else 'N/A'}"  )

    # 3. Calculate Score
    analysis = service.forensics.calculate(job, mapping)
    print(f" [Preview] Analysis for Job ID: {job_id} with CV ID: {cv_id}: Stats:", analysis.stats, "\n\n")
    stats = analysis.stats
    
    # 4. Generate Badges
    badges = service._generate_badges(stats, job)
    
    return {
        "score": stats.overall_match_score,
        "badges": badges
    }

class JobFeatureUpdate(BaseModel):
    description: str
    type: str

# Add this endpoint to your router
@router.patch("/{job_id}/feature/{feature_id}")
def update_feature(
    job_id: str, 
    feature_id: str, 
    payload: JobFeatureUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Updates a specific job feature's text or type."""
    registry: Registry = request.app.state.registry
    # You will need to ensure your Registry class has this method!
    return registry.update_job_feature(user.id, job_id, feature_id, payload.description, payload.type)