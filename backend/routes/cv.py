# backend/routes/cv.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import CVUpdate # Import the update model

from typing import Optional, List # Ensure List is imported


router = APIRouter()
registry = Registry()

# ... (Existing top-level CRUD endpoints) ...

@router.post("/")
def create_cv(name: str, summary: Optional[str] = None):
    """Create a new base CV."""
    return registry.create_cv(name, summary)

@router.get("/")
def list_cvs():
    """List all base CVs."""
    return registry.all_cvs()

@router.get("/{cv_id}")
def get_cv(cv_id: str):
    """Fetch a specific CV by ID."""
    cv = registry.get_cv(cv_id)
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")
    return cv

@router.patch("/{cv_id}")
def update_cv(cv_id: str, data: CVUpdate):
    """Update general CV metadata (name, summary, contact_info)."""
    try:
        return registry.update_cv(cv_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{cv_id}")
def delete_cv(cv_id: str):
    """Delete a CV by ID."""
    try:
        return registry.delete_cv(cv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------
# NESTED ADD ENDPOINTS (CV Components)
# ---------------------------------------------------------------------

@router.post("/{cv_id}/experience")
def add_experience(
    cv_id: str, 
    title: str, 
    company: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    description: Optional[str] = None
):
    """Add a new experience entry to the CV."""
    try:
        return registry.add_cv_experience(cv_id, title=title, company=company, start_date=start_date, end_date=end_date, description=description)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/education")
def add_education(
    cv_id: str, 
    institution: str, 
    degree: str, 
    field: str, 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None
):
    """Add a new education entry to the CV."""
    try:
        return registry.add_cv_education(cv_id, institution=institution, degree=degree, field=field, start_date=start_date, end_date=end_date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/skill")
def add_skill(
    cv_id: str, 
    name: str, 
    category: str = "technical", 
    level: Optional[str] = None, 
    importance: Optional[int] = None, 
    description: Optional[str] = None
):
    """Add a new skill to the CV's master list."""
    try:
        return registry.add_cv_skill(cv_id, name=name, category=category, level=level, importance=importance, description=description)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project")
def add_project(
    cv_id: str, 
    title: str, 
    description: str, 
    related_experience_id: Optional[str] = None, 
    related_education_id: Optional[str] = None
):
    """Add a new project to the CV."""
    try:
        return registry.add_cv_project(cv_id, title=title, description=description, related_experience_id=related_experience_id, related_education_id=related_education_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/hobby")
def add_hobby(cv_id: str, name: str, description: Optional[str] = None):
    """Add a new hobby to the CV."""
    try:
        return registry.add_cv_hobby(cv_id, name=name, description=description)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/achievement")
def add_achievement(cv_id: str, text: str, context: Optional[str] = None):
    """Add a new global achievement to the CV's master list."""
    try:
        return registry.add_cv_achievement(cv_id, text=text, context=context)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------
# NESTED LINKING ENDPOINTS (Skills and Achievements)
# ---------------------------------------------------------------------

# --- Skill Linking ---

@router.post("/{cv_id}/experience/{exp_id}/skill/{skill_id}")
def link_skill_to_experience(cv_id: str, exp_id: str, skill_id: str):
    """Links a master skill to a specific experience."""
    try:
        return registry.link_skill_to_entity(cv_id, exp_id, skill_id, 'experiences')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/skill/{skill_id}")
def link_skill_to_project(cv_id: str, proj_id: str, skill_id: str):
    """Links a master skill to a specific project."""
    try:
        return registry.link_skill_to_entity(cv_id, proj_id, skill_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/achievement/{ach_id}/skill/{skill_id}")
def link_skill_to_achievement(cv_id: str, ach_id: str, skill_id: str):
    """Links a master skill to a specific achievement (tags the achievement)."""
    try:
        return registry.link_skill_to_entity(cv_id, ach_id, skill_id, 'achievements')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Achievement Linking ---

@router.post("/{cv_id}/experience/{exp_id}/achievement/{ach_id}")
def link_achievement_to_experience(cv_id: str, exp_id: str, ach_id: str):
    """Links a master achievement to a specific experience."""
    try:
        return registry.link_achievement_to_context(cv_id, exp_id, ach_id, 'experiences')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/achievement/{ach_id}")
def link_achievement_to_project(cv_id: str, proj_id: str, ach_id: str):
    """Links a master achievement to a specific project."""
    try:
        return registry.link_achievement_to_context(cv_id, proj_id, ach_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))