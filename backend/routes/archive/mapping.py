# backend/routes/mapping.py

from fastapi import APIRouter, HTTPException, Request
from backend.core.registry import Registry
from backend.core.models import MappingUpdate, MappingPair # <-- 1. Import MappingPair
from typing import Optional, List # <-- 2. Import List
# from backend.core.dependencies import registry, inferer # <-- 3. Import the new 'inferer'

router = APIRouter()

@router.post("/")
def create_mapping(job_id: str, base_cv_id: str, request: Request):
    """Create a new mapping between a Job and a Base CV."""
    registry = request.app.state.registry
    return registry.create_mapping(job_id, base_cv_id)

@router.get("/")
def list_mappings(request: Request):
    """List all mappings."""
    registry = request.app.state.registry
    return registry.all_mappings()

@router.get("/{mapping_id}")
def get_mapping(mapping_id: str, request: Request):
    """Fetch a specific mapping by ID."""
    registry = request.app.state.registry
    mapping = registry.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    return mapping

@router.patch("/{mapping_id}")
def update_mapping(mapping_id: str, data: MappingUpdate, request: Request):
    """Update mapping metadata (mostly handled via touch/update_at)."""
    try:
        registry = request.app.state.registry
        return registry.update_mapping(mapping_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{mapping_id}")
def delete_mapping(mapping_id: str, request: Request):
    """Delete a mapping by ID."""
    try:
        registry = request.app.state.registry
        return registry.delete_mapping(mapping_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{mapping_id}/pair")
def add_mapping_pair(
    mapping_id: str, 
    feature_id: str, 
    context_item_id: str,
    context_item_type: str, 
    request: Request,
    annotation: Optional[str] = None  # <-- ADD THIS
):
    """Add a specific link (pair) between a job feature and a CV item."""
    registry = request.app.state.registry
    
    # ... (Get Feature, Mapping, and CV logic is unchanged) ...
    job = next((j for j in registry.all_jobs() if any(f.id == feature_id for f in j.features)), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job Feature's parent Job not found")
    feature = next((f for f in job.features if f.id == feature_id), None)
    if not feature:
        raise HTTPException(status_code=404, detail="Job Feature not found")

    mapping = registry.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
        
    cv = registry.get_cv(mapping.base_cv_id)
    if not cv:
        raise HTTPException(status_code=404, detail="Base CV not found for this mapping")

    # 3. Get the generic CV item
    try:
        context_item = registry._get_nested_entity(cv, context_item_type, context_item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not context_item:
        raise HTTPException(status_code=404, detail=f"CV Item {context_item_id} not found in {context_item_type}")

    # 4. Call the updated registry function
    try:
        # Pass annotation through to the registry
        return registry.add_mapping_pair(mapping_id, feature, context_item, context_item_type, annotation) 
    except ValueError as e:
        # This will now catch duplicate errors as well
        raise HTTPException(status_code=400, detail=str(e)) # 400 Bad Request is better for duplicates


# --- ADD THIS ENTIRE NEW ROUTE FOR DELETING ---
@router.delete("/{mapping_id}/pair/{pair_id}")
def delete_mapping_pair(mapping_id: str, pair_id: str, request: Request):
    """Deletes a specific pair from a mapping."""
    try:
        registry = request.app.state.registry
        registry.delete_mapping_pair(mapping_id, pair_id)
        return {"status": "success", "detail": "Pair deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- 4. ADD THIS NEW ENDPOINT AT THE END OF THE FILE ---
@router.post("/{mapping_id}/infer", response_model=List[MappingPair])
def infer_mapping_pairs(mapping_id: str, request: Request):
    """
    Runs the NLP inference engine to suggest new mapping pairs
    for the given job and CV.
    """
    registry = request.app.state.registry
    mapping = registry.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    
    job = registry.get_job(mapping.job_id)
    cv = registry.get_cv(mapping.base_cv_id)
    
    if not job or not cv:
        raise HTTPException(status_code=404, detail="Job or CV not found")
    
    try:
        # Call the singleton inferer
        inferer = request.app.state.inferer
        suggestions = inferer.infer_mappings(job, cv)
        
        # Note: We just return the suggestions, we don't save them.
        # The frontend can decide what to do with them.
        return suggestions
        
    except Exception as e:
        # Catch any potential NLP errors
        log.error(f"Error during inference for mapping {mapping_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    
