# backend/routes/cv.py

from fastapi import APIRouter, HTTPException, Query, Request, Depends
from backend.core.registry import Registry
from backend.core.models import (
    CVUpdate, ExperienceUpdate, ExperienceComplexPayload, 
    EducationComplexPayload, HobbyComplexPayload, ProjectComplexPayload, 
    CVImportRequest, CV, Experience, Project, Education, Skill, Hobby, User
)
from typing import Optional, List
import logging as log
from backend.routes.auth import get_current_user # Adjust import path

router = APIRouter()

@router.post("/import")
async def import_cv_text(
    request: Request, 
    payload: CVImportRequest,
    user: User = Depends(get_current_user)
):
    """
    Imports a CV from raw text using the Fast Parse engine.
    """
    parser = getattr(request.app.state, "cv_parser", None)
    if not parser:
        raise HTTPException(status_code=503, detail="LLM Model is not loaded.")

    try:
        # 1. Parse into a full Pydantic CV object
        structured_cv = await parser.parse_cv(payload.text, cv_name=payload.name)
        
        # 2. Persist to TinyDB using the Registry
        # Enforce User Ownership on the new object
        structured_cv.user_id = user.id
        
        registry: Registry = request.app.state.registry
        registry._insert("cvs", structured_cv)
        
        return structured_cv
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"CV Import Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse CV: {str(e)}")

@router.post("/")
def create_cv(
    name: str, 
    request: Request, 
    first_name: Optional[str] = None, 
    last_name: Optional[str] = None, 
    summary: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Create a new base CV."""
    registry: Registry = request.app.state.registry
    return registry.create_cv(user.id, name=name, first_name=first_name, last_name=last_name, summary=summary)


@router.get("/")
def list_cvs(
    request: Request,
    user: User = Depends(get_current_user)
):
    """List all base CVs belonging to the user."""
    registry: Registry = request.app.state.registry
    print(f"Listing CVs for user_id: {user.id}")
    return registry.all_cvs(user.id)

@router.get("/{cv_id}")
def get_cv(
    cv_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific CV by ID."""
    registry: Registry = request.app.state.registry
    # Pass user.id to enforce ownership
    cv = registry.get_cv(cv_id, user.id)
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")
    
    if cv and cv.experiences:
        cv.experiences.sort(
            key=lambda exp: exp.start_date or '0000-00-00', 
            reverse=True
        )
    return cv

@router.patch("/{cv_id}")
def update_cv(
    cv_id: str, 
    data: CVUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update general CV metadata (name, summary, contact_info)."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_cv(user.id, cv_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{cv_id}")
def delete_cv(
    cv_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete a CV by ID."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_cv(user.id, cv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------
# NESTED ADD ENDPOINTS (CV Components)
# ---------------------------------------------------------------------

@router.post("/{cv_id}/experience/complex")
def add_experience_complex(
    cv_id: str, 
    payload: ExperienceComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.create_experience_from_payload(user.id, cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/experience/{exp_id}/complex")
def update_experience_complex(
    cv_id: str, 
    exp_id: str, 
    payload: ExperienceComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.update_experience_from_payload(user.id, cv_id, exp_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/education/complex")
def add_education_complex(
    cv_id: str, 
    payload: EducationComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.create_education_from_payload(user.id, cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/education/{edu_id}/complex")
def update_education_complex(
    cv_id: str, 
    edu_id: str, 
    payload: EducationComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.update_education_from_payload(user.id, cv_id, edu_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/hobby/complex")
def add_hobby_complex(
    cv_id: str, 
    payload: HobbyComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.create_hobby_from_payload(user.id, cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/hobby/{hobby_id}/complex")
def update_hobby_complex(
    cv_id: str, 
    hobby_id: str, 
    payload: HobbyComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.update_hobby_from_payload(user.id, cv_id, hobby_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/education")
def add_education(
    cv_id: str, 
    institution: str, 
    degree: str, 
    field: str,
    request: Request,
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None),
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cv_education(
            user.id,
            cv_id, 
            institution=institution, 
            degree=degree, 
            field=field, 
            start_date=start_date, 
            end_date=end_date, 
            skill_ids=skill_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/skill")
def add_skill(
    cv_id: str, 
    name: str,
    request: Request,
    category: str = "technical",
    level: Optional[str] = None, 
    importance: Optional[int] = None, 
    description: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cv_skill(user.id, cv_id, name=name, category=category, level=level, importance=importance, description=description)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project")
def add_project(
    cv_id: str, 
    title: str, 
    description: str,
    request: Request,
    related_experience_id: Optional[str] = None, 
    related_education_id: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None),
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cv_project(
            user.id,
            cv_id, 
            title=title, 
            description=description, 
            related_experience_id=related_experience_id, 
            related_education_id=related_education_id, 
            skill_ids=skill_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/hobby")
def add_hobby(
    cv_id: str, 
    name: str, 
    request: Request,
    description: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None), 
    achievement_ids: Optional[List[str]] = Query(None),
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cv_hobby(
            user.id,
            cv_id, 
            name=name, 
            description=description, 
            skill_ids=skill_ids,
            achievement_ids=achievement_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.post("/{cv_id}/project/complex")
def add_project_complex(
    cv_id: str, 
    payload: ProjectComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.create_project_from_payload(user.id, cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/project/{project_id}/complex")
def update_project_complex(
    cv_id: str, 
    project_id: str, 
    payload: ProjectComplexPayload, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.update_project_from_payload(user.id, cv_id, project_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/achievement")
def add_achievement(
    cv_id: str, 
    text: str,
    request: Request,
    context: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None),
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.add_cv_achievement(
            user.id,
            cv_id,
            text=text, 
            context=context, 
            skill_ids=skill_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ---------------------------------------------------------------------
# NESTED LINKING ENDPOINTS (Skills and Achievements)
# ---------------------------------------------------------------------

@router.post("/{cv_id}/experience/{exp_id}/skill/{skill_id}")
def link_skill_to_experience(
    cv_id: str, 
    exp_id: str, 
    skill_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.link_skill_to_entity(user.id, cv_id, exp_id, skill_id, 'experiences')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/skill/{skill_id}")
def link_skill_to_project(
    cv_id: str, 
    proj_id: str, 
    skill_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.link_skill_to_entity(user.id, cv_id, proj_id, skill_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/achievement/{ach_id}/skill/{skill_id}")
def link_skill_to_achievement(
    cv_id: str, 
    ach_id: str, 
    skill_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.link_skill_to_entity(user.id, cv_id, ach_id, skill_id, 'achievements')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{cv_id}/experience/{exp_id}/achievement/{ach_id}")
def link_achievement_to_experience(
    cv_id: str, 
    exp_id: str, 
    ach_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.link_achievement_to_context(user.id, cv_id, exp_id, ach_id, 'experiences')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/achievement/{ach_id}")
def link_achievement_to_project(
    cv_id: str, 
    proj_id: str, 
    ach_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        registry: Registry = request.app.state.registry
        return registry.link_achievement_to_context(user.id, cv_id, proj_id, ach_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{cv_id}/{entity_list_name}/{entity_id}/skills/aggregated")
def get_aggregated_skills(
    cv_id: str, 
    entity_list_name: str, 
    entity_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    try:
        valid_lists = ['experiences', 'projects', 'achievements', 'education', 'hobbies']
        if entity_list_name not in valid_lists:
            raise ValueError(f"Invalid entity list name. Must be one of: {valid_lists}")
        registry: Registry = request.app.state.registry
        return registry.get_aggregated_skills_for_entity(user.id, cv_id, entity_list_name, entity_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.delete("/{cv_id}/{list_name}/{item_id}")
def delete_nested_item(
    cv_id: str, 
    list_name: str, 
    item_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Deletes a specific nested item (Experience, Skill, Education, etc.)
    from a CV's master list.
    """
    registry: Registry = request.app.state.registry
    delete_functions = {
        "experiences": registry.delete_cv_experience,
        "education": registry.delete_cv_education,
        "skills": registry.delete_cv_skill,
        "achievements": registry.delete_cv_achievement,
        "projects": registry.delete_cv_project,
        "hobbies": registry.delete_cv_hobby,
    }

    func = delete_functions.get(list_name)

    if not func:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid list name '{list_name}'. Cannot delete."
        )

    try:
        # Pass user.id to the delete function
        return func(user.id, cv_id, item_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# -----------------------------------------------------------------------------
# Specific Item Access (For Modals & Forensics)
# -----------------------------------------------------------------------------

@router.get("/experience/{item_id}", response_model=Experience)
def get_experience_item(
    item_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    item = request.app.state.registry.fetch_item(item_id, 'experience', user.id)
    if not item:
        raise HTTPException(status_code=404, detail="Experience not found")
    return item

@router.get("/project/{item_id}", response_model=Project)
def get_project_item(
    item_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    item = request.app.state.registry.fetch_item(item_id, 'project', user.id)
    if not item:
        raise HTTPException(status_code=404, detail="Project not found")
    return item

@router.get("/education/{item_id}", response_model=Education)
def get_education_item(
    item_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    item = request.app.state.registry.fetch_item(item_id, 'education', user.id)
    if not item:
        raise HTTPException(status_code=404, detail="Education not found")
    return item

@router.get("/skill/{item_id}", response_model=Skill)
def get_skill_item(
    item_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    item = request.app.state.registry.fetch_item(item_id, 'skill', user.id)
    if not item:
        raise HTTPException(status_code=404, detail="Skill not found")
    return item

@router.get("/hobby/{item_id}", response_model=Hobby)
def get_hobby_item(
    item_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    item = request.app.state.registry.fetch_item(item_id, 'hobby', user.id)
    if not item:
        raise HTTPException(status_code=404, detail="Hobby not found")
    return item

@router.get("/item-details/{item_id}")
def get_item_details(
    item_id: str, 
    type: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Returns the item AND its resolved relationships.
    """
    registry: Registry = request.app.state.registry
    
    data = registry.fetch_item_details(item_id, type, user.id)
    
    if not data:
        raise HTTPException(404, "Item not found")
        
    return data