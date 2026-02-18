# backend/routes/forensics.py

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Literal, Optional, List
from pydantic import BaseModel
from redis import Redis
from rq import Queue

from backend.core.services.mapping_service import MappingOptimizer
from backend.core.forensics import ForensicCalculator
from backend.core.models import ForensicAnalysis, Mapping, LineageItem, MatchCandidate, MatchingMeta, User, ApplicationUpdate
from backend.core.registry import Registry
from backend.routes.auth import get_current_user
from backend.core.services.scoring import ScoringService 

# IMPORT TASKS
from backend.tasks import task_generate_role_case, task_score_application

router = APIRouter()

# --- REDIS QUEUE SETUP ---
redis_conn = Redis(host='localhost', port=6379)
q_inference = Queue('q_inference', connection=redis_conn)


# --- [NEW] TRIGGER ANALYSIS ENDPOINT ---
@router.post("/applications/{app_id}/generate-analysis")
def trigger_application_analysis(
    app_id: str,
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Triggers the background worker to run inference for a specific Application.
    
    frontend usage: client.post(`/forensics/applications/${appId}/generate-analysis`)
    """
    registry: Registry = request.app.state.registry
    
    # 1. Validation: Ensure App exists and belongs to User
    app = registry.get_application(app_id, user.id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")

    # 2. Enqueue Task to 'q_inference' (The Fast Lane)
    # This task will:
    #   a. Run inference (if needed)
    #   b. Calculate Score/Grade
    #   c. Save to DB
    #   d. Send 'APP_SCORED' WebSocket event
    q_inference.enqueue(
        task_score_application,
        args=(user.id, app.id),
        job_timeout='3m'
    )

    return {
        "status": "queued",
        "message": "Analysis started. Waiting for results..."
    }

# --- REQUEST MODELS ---
class GenerateRoleCaseRequest(BaseModel):
    job_id: str
    cv_id: str
    mode: Literal[
        "super_eager", 
        "eager_mode", 
        "balanced_default", 
        "picky_mode", 
        "super_picky"
    ] = "balanced_default"

class PromoteRequest(BaseModel):
    alternative_id: str

class ManualMatchRequest(BaseModel):
    evidence_text: str 
    cv_item_id: Optional[str] = None
    cv_item_type: Optional[str] = None 
    cv_item_name: Optional[str] = None 


def _update_job_cache_from_mapping(user_id: str, job_id: str, mapping: Mapping, registry: Registry):
    """
    Recalculates the score based on the current mapping and updates the Job Cache.
    """
    job = registry.get_job(job_id, user_id)
    if not job: return

    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping)
    stats = analysis.stats

    job.match_score = stats.overall_match_score
    
    if stats.overall_match_score >= 85: job.match_grade = "A"
    elif stats.overall_match_score >= 65: job.match_grade = "B"
    elif stats.overall_match_score >= 40: job.match_grade = "C"
    else: job.match_grade = "D"

    badges = []
    if stats.critical_gaps_count > 0: badges.append("Missing Critical Skills")
    if stats.overall_match_score > 90: badges.append("Top Match")
    job.cached_badges = badges

    registry.update_job(user_id, job.id, job)


@router.post("/applications/{app_id}/mappings/{feature_id}/promote")
def promote_match(
    app_id: str, 
    feature_id: str, 
    payload: PromoteRequest, 
    request: Request,
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair: raise HTTPException(404, "Pair not found")
    
    if MappingOptimizer.promote_alternative(pair, payload.alternative_id):
        registry.save_mapping(user.id, mapping)
    
    job = registry.get_job(app.job_id, user.id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    return new_analysis


@router.post("/applications/{app_id}/mappings/{feature_id}/approve")
def approve_match(
    app_id: str, 
    feature_id: str, 
    request: Request, 
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "App not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    if not mapping: raise HTTPException(404, "Mapping not found")

    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair: raise HTTPException(404, "Pair not found")
    
    pair.status = "user_approved"
    registry.save_mapping(user.id, mapping)

    _update_job_cache_from_mapping(user.id, app.job_id, mapping, registry)

    return {"status": "success", "new_status": pair.status}


# -----------------------------------------------------------------------------
# 1. GENERATE ROLE CASE (Async / WebSocket)
# -----------------------------------------------------------------------------
@router.post("/generate")
def generate_role_case(
    payload: GenerateRoleCaseRequest, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    The 'One-Click' Endpoint:
    Enqueues the inference task to the background worker.
    The frontend should wait for WebSocket event: ROLE_CASE_GENERATED
    """
    registry: Registry = request.app.state.registry
    
    # 1. Validate Inputs (Secured)
    job = registry.get_job(payload.job_id, user.id)
    cv = registry.get_cv(payload.cv_id, user.id)
    
    if not job or not cv:
        raise HTTPException(404, "Job or CV not found")

    # 2. Enqueue Task
    # The task handles finding/creating the mapping internally
    q_inference.enqueue(
        task_generate_role_case,
        args=(user.id, payload.job_id, payload.cv_id, payload.mode),
        job_timeout='2m'
    )

    return {
        "status": "queued", 
        "message": "Generating RoleCase... Please wait for results."
    }


@router.get("/applications/{app_id}/forensic-analysis", response_model=ForensicAnalysis)
def get_forensic_analysis(
    app_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    READ-ONLY Endpoint.
    Fetches the existing analysis. Does NOT run heavy inference synchronously.
    If no analysis exists, it returns a 404 or empty state, prompting user to click 'Generate'.
    """
    registry: Registry = request.app.state.registry
    
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    job = registry.get_job(app.job_id, user.id)
    
    # Handle missing mapping gracefully
    if not app.mapping_id:
        # Create empty mapping just so we can return empty analysis
        mapping = registry.create_mapping(user.id, app.job_id, app.base_cv_id)
        app.mapping_id = mapping.id
        registry.update_application(user.id, app.id, ApplicationUpdate(mapping_id=mapping.id))
    else:
        mapping = registry.get_mapping(app.mapping_id, user.id)
    
    if not job or not mapping:
        raise HTTPException(404, "Job or Mapping data missing")

    # Calculate Analysis (Fast - just math, no AI)
    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping) 

    analysis.application_id = app_id
    return analysis


@router.post("/applications/{app_id}/mappings/{feature_id}/reject")
def reject_match(
    app_id: str, 
    feature_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    if not mapping: raise HTTPException(404, "Mapping not found")

    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair:
        raise HTTPException(404, "Mapping pair not found")

    # Optimize & Save
    MappingOptimizer.reject_current_match(pair)
    registry.save_mapping(user.id, mapping)

    # Recalculate & Cache Update
    job = registry.get_job(app.job_id, user.id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    
    # Sync to Job Cache
    stats = new_analysis.stats
    job.match_score = stats.overall_match_score
    
    if stats.overall_match_score >= 85: job.match_grade = "A"
    elif stats.overall_match_score >= 65: job.match_grade = "B"
    elif stats.overall_match_score >= 40: job.match_grade = "C"
    else: job.match_grade = "D"

    badges = []
    if stats.critical_gaps_count > 0: badges.append("Missing Critical Skills")
    if stats.overall_match_score > 90: badges.append("Top Match")
    
    job.cached_badges = badges
    registry.update_job(user.id, job.id, job)

    new_analysis.application_id = app_id

    return {
        "success": True,
        "updated_pair": pair,
        "new_forensics": new_analysis 
    }


@router.post("/applications/{app_id}/mappings/{feature_id}/manual", response_model=ForensicAnalysis)
def create_manual_match(
    app_id: str, 
    feature_id: str, 
    payload: ManualMatchRequest, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    User forces a match. Overrides existing AI match.
    """
    registry: Registry = request.app.state.registry
    
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    job = registry.get_job(app.job_id, user.id)
    
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair:
        raise HTTPException(404, "Mapping pair container not found")

    manual_lineage = []
    if payload.cv_item_id:
        manual_lineage.append(LineageItem(
            id=payload.cv_item_id, 
            type=payload.cv_item_type or "manual", 
            name=payload.cv_item_name or "Manual Link"
        ))

    manual_candidate = MatchCandidate(
        segment_text=payload.evidence_text,
        segment_type="manual_override",
        score=1.0, 
        lineage=manual_lineage
    )

    if not pair.meta:
        pair.meta = MatchingMeta(best_match=manual_candidate, summary_note="")
    
    pair.meta.best_match = manual_candidate
    pair.strength = 1.0 
    pair.context_item_id = payload.cv_item_id
    pair.context_item_type = payload.cv_item_type
    pair.context_item_text = payload.cv_item_name or "Manual Match"
    
    pair.meta.summary_note = f"Manual Match: \"{payload.evidence_text[:50]}...\""
    
    registry.save_mapping(user.id, mapping)
    
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    
    return new_analysis