# backend/routes/mapping.py

from fastapi import APIRouter, HTTPException, Request, Depends
from backend.core.registry import Registry
from backend.core.models import MappingUpdate, MappingPair, User
from typing import Optional, List, Literal
from backend.core.tuning import TUNING_MODES
from backend.routes.auth import get_current_user
import logging as log
from redis import Redis
from rq import Queue

# IMPORT THE TASK
from backend.tasks import task_infer_mapping_suggestions

router = APIRouter()

# --- REDIS QUEUE SETUP ---
redis_conn = Redis(host='localhost', port=6379)
q_inference = Queue('q_inference', connection=redis_conn)


@router.post("/")
def create_mapping(
    job_id: str, 
    base_cv_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Create a new mapping between a Job and a Base CV."""
    registry: Registry = request.app.state.registry
    return registry.create_mapping(user.id, job_id, base_cv_id)

@router.get("/")
def list_mappings(
    request: Request,
    user: User = Depends(get_current_user)
):
    """List all mappings."""
    registry: Registry = request.app.state.registry
    return registry.all_mappings(user.id)

@router.get("/{mapping_id}")
def get_mapping(
    mapping_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific mapping by ID."""
    registry: Registry = request.app.state.registry
    mapping = registry.get_mapping(mapping_id, user.id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    return mapping

@router.patch("/{mapping_id}")
def update_mapping(
    mapping_id: str, 
    data: MappingUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update mapping metadata."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_mapping(user.id, mapping_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{mapping_id}")
def delete_mapping(
    mapping_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete a mapping by ID."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_mapping(user.id, mapping_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{mapping_id}/pair")
def add_mapping_pair(
    mapping_id: str, 
    feature_id: str, 
    context_item_id: str,
    context_item_type: str, 
    request: Request,
    annotation: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Add a specific link (pair) between a job feature and a CV item."""
    registry: Registry = request.app.state.registry
    
    # 1. Find the Job and Feature (Secured)
    user_jobs = registry.all_jobs(user.id)
    job = next((j for j in user_jobs if any(f.id == feature_id for f in j.features)), None)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job Feature's parent Job not found")
    
    feature = next((f for f in job.features if f.id == feature_id), None)
    if not feature:
        raise HTTPException(status_code=404, detail="Job Feature not found")

    # 2. Get Mapping (Secured)
    mapping = registry.get_mapping(mapping_id, user.id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
        
    # 3. Get CV (Secured)
    cv = registry.get_cv(mapping.base_cv_id, user.id)
    if not cv:
        raise HTTPException(status_code=404, detail="Base CV not found for this mapping")

    # 4. Get the generic CV item
    try:
        context_item = registry._get_nested_entity(cv, context_item_type, context_item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    if not context_item:
        raise HTTPException(status_code=404, detail=f"CV Item {context_item_id} not found in {context_item_type}")

    # 5. Call the updated registry function
    try:
        return registry.add_mapping_pair(
            user.id, 
            mapping_id, 
            feature, 
            context_item, 
            context_item_type, 
            annotation
        ) 
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{mapping_id}/pair/{pair_id}")
def delete_mapping_pair(
    mapping_id: str, 
    pair_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Deletes a specific pair from a mapping."""
    try:
        registry: Registry = request.app.state.registry
        registry.delete_mapping_pair(user.id, mapping_id, pair_id)
        return {"status": "success", "detail": "Pair deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- MODIFIED INFERENCE ENDPOINT ---
@router.post("/{mapping_id}/infer")
def infer_mapping_pairs(
    mapping_id: str, 
    request: Request,
    mode: Literal[
        "super_eager", 
        "eager_mode", 
        "balanced_default", 
        "picky_mode", 
        "super_picky"
    ] = "balanced_default",
    user: User = Depends(get_current_user)
):
    """
    Submits an inference request to the background worker.
    The worker will update the mapping with suggestions and notify via WebSocket.
    """
    registry: Registry = request.app.state.registry
    
    # 1. Validation (Fast)
    mapping = registry.get_mapping(mapping_id, user.id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    
    # 2. Enqueue Task
    # We pass 'mode' so the worker can look up tuning params
    q_inference.enqueue(
        task_infer_mapping_suggestions,
        args=(user.id, mapping_id, mode),
        job_timeout='2m' # Inference is fast, usually <10s
    )
    
    return {
        "status": "queued", 
        "message": f"Inference started in '{mode}' mode. Please wait for results."
    }