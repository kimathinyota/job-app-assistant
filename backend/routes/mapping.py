# backend/routes/mapping.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import MappingUpdate # Import the update model
from typing import Optional

router = APIRouter()
registry = Registry()

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
def add_mapping_pair(mapping_id: str, feature_id: str, experience_id: str, annotation: Optional[str] = None):
    """Add a specific link (pair) between a job feature and a CV experience."""
    # Logic to fetch objects to pass to registry is better handled in the route for validation
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
    experience = next((e for e in cv.experiences if e.id == experience_id), None)
    if not experience:
        raise HTTPException(status_code=404, detail="CV Experience not found")

    try:
        return registry.add_mapping_pair(mapping_id, feature, experience, annotation)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))