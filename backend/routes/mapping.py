# backend/routes/mapping.py

from fastapi import APIRouter, HTTPException, Request, Depends
from backend.core.registry import Registry
from backend.core.models import MappingUpdate, MappingPair, User
from typing import Optional, List, Literal
from backend.core.tuning import TUNING_MODES
from backend.routes.auth import get_current_user
import logging as log

router = APIRouter()

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
    """Update mapping metadata (mostly handled via touch/update_at)."""
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
    # We must scan only THIS user's jobs to find the feature
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
    # This uses the in-memory object helper, so it is safe because 'cv' is already secured
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

@router.post("/{mapping_id}/infer", response_model=List[MappingPair])
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
    Runs the NLP inference engine to suggest new mapping pairs.
    """
    registry: Registry = request.app.state.registry
    
    # 1. Fetch Secured Data
    mapping = registry.get_mapping(mapping_id, user.id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    
    job = registry.get_job(mapping.job_id, user.id)
    cv = registry.get_cv(mapping.base_cv_id, user.id)
    
    if not job or not cv:
        raise HTTPException(status_code=404, detail="Job or CV not found")
    
    try:
        inferer = request.app.state.inferer
        mode_settings = TUNING_MODES.get(mode, TUNING_MODES["balanced_default"])
        config_params = mode_settings.get("config", {})
        
        log.info(f"Running inference for mapping {mapping_id} with mode: {mode}")
        
        suggestions = inferer.infer_mappings(job, cv, **config_params)
        
        return suggestions
        
    except Exception as e:
        log.error(f"Error during inference for mapping {mapping_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")