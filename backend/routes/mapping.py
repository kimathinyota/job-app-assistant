from fastapi import APIRouter
from backend.core.registry import Registry
from typing import Optional

router = APIRouter()
registry = Registry()

@router.post("/")
def create_mapping(job_id: str, base_cv_id: str):
    return registry.create_mapping(job_id, base_cv_id)

@router.post("/{mapping_id}/pair")
def add_mapping_pair(mapping_id: str, feature_id: str, experience_id: str, annotation: Optional[str] = None):
    job = next((j for j in registry.all_jobs() if any(f.id == feature_id for f in j.features)), None)
    if not job:
        return {"error": "Feature not found"}
    feature = next(f for f in job.features if f.id == feature_id)

    cv = next((c for c in registry.all_cvs() if any(e.id == experience_id for e in c.experiences)), None)
    if not cv:
        return {"error": "Experience not found"}
    experience = next(e for e in cv.experiences if e.id == experience_id)

    return registry.add_mapping_pair(mapping_id, feature, experience, annotation)

@router.get("/{mapping_id}")
def get_mapping(mapping_id: str):
    return registry.get_mapping(mapping_id)
