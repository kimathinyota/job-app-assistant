# backend/routes/mapping.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import MappingUpdate # Import the update model
from typing import Optional
from backend.core.dependencies import registry 

router = APIRouter()

@router.post("/")
def create_mapping(job_id: str, base_cv_id: str):
    """Create a new mapping between a Job and a Base CV."""
    return registry.create_mapping(job_id, base_cv_id)

@router.get("/")
def list_mappings():
    """List all mappings."""
    return registry.all_mappings()

@router.get("/{mapping_id}")
def get_mapping(mapping_id: str):
    """Fetch a specific mapping by ID."""
    mapping = registry.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
    return mapping

@router.patch("/{mapping_id}")
def update_mapping(mapping_id: str, data: MappingUpdate):
    """Update mapping metadata (mostly handled via touch/update_at)."""
    try:
        return registry.update_mapping(mapping_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{mapping_id}")
def delete_mapping(mapping_id: str):
    """Delete a mapping by ID."""
    try:
        return registry.delete_mapping(mapping_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))



@router.post("/{mapping_id}/pair")
# --- START CHANGES ---
def add_mapping_pair(
    mapping_id: str, 
    feature_id: str, 
    context_item_id: str,   # <-- Change from experience_id
    context_item_type: str, # <-- Add this
    annotation: Optional[str] = None
):
# --- END CHANGES ---
    """Add a specific link (pair) between a job feature and a CV item."""
    
    # 1. Get Feature (No change)
    job = next((j for j in registry.all_jobs() if any(f.id == feature_id for f in j.features)), None)
    if not job:
        raise HTTPException(status_code=404, detail="Job Feature's parent Job not found")
    feature = next((f for f in job.features if f.id == feature_id), None)
    if not feature:
        raise HTTPException(status_code=404, detail="Job Feature not found")

    # 2. Get Mapping & CV (No change)
    mapping = registry.get_mapping(mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")
        
    cv = registry.get_cv(mapping.base_cv_id)
    if not cv:
        raise HTTPException(status_code=404, detail="Base CV not found for this mapping")

    # --- START CHANGES ---
    # 3. Get the generic CV item using the registry helper
    try:
        context_item = registry._get_nested_entity(cv, context_item_type, context_item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not context_item:
        raise HTTPException(status_code=404, detail=f"CV Item {context_item_id} not found in {context_item_type}")

    # 4. Call the updated registry function
    try:
        return registry.add_mapping_pair(mapping_id, feature, context_item, context_item_type, annotation)
    # --- END CHANGES ---
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))