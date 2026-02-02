# backend/routes/cv.py

from fastapi import APIRouter, HTTPException, Query, Request # <-- Import Query
from backend.core.registry import Registry
from backend.core.models import CVUpdate, ExperienceUpdate, ExperienceComplexPayload, EducationComplexPayload, HobbyComplexPayload, ProjectComplexPayload, CVImportRequest # Import the update model

from typing import Optional, List # Ensure List is imported
import logging as log


router = APIRouter()
# ... (Existing top-level CRUD endpoints: create_cv, list_cvs, etc. No changes needed here) ...

# ... (Existing top-level CRUD endpoints: create_cv, list_cvs, etc. No changes needed here) ...

@router.post("/import")
async def import_cv_text(request: Request, payload: CVImportRequest):
    """
    Imports a CV from raw text using the Fast Parse engine.
    """
    parser = getattr(request.app.state, "job_parser", None)
    if not parser:
        raise HTTPException(status_code=503, detail="LLM Model is not loaded.")

    try:
        # 1. Parse into a full Pydantic CV object
        # The parser now handles all ID generation and object linking internally
        structured_cv = parser.fast_parse_cv(payload.text, cv_name=payload.name)
        
        # 2. Persist to TinyDB using the Registry
        registry = request.app.state.registry
        
        # Accessing the internal _insert method directly 
        # (Since Registry doesn't expose a 'save_cv' method that accepts an object)
        # 'cvs' matches the table name used in registry.create_cv
        registry._insert("cvs", structured_cv)
        
        return structured_cv
        
    except Exception as e:
        # Log the full error for debugging
        import logging
        logging.getLogger(__name__).error(f"CV Import Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse CV: {str(e)}")

@router.post("/")
def create_cv(name: str, request: Request, first_name: Optional[str] = None, last_name: Optional[str] = None, summary: Optional[str] = None):
    """Create a new base CV."""
    registry = request.app.state.registry
    return registry.create_cv(name=name, first_name=first_name, last_name=last_name, summary=summary)


@router.get("/")
def list_cvs(request: Request):
    """List all base CVs."""
    registry = request.app.state.registry
    return registry.all_cvs()

@router.get("/{cv_id}")
def get_cv(cv_id: str, request: Request):
    """Fetch a specific CV by ID."""
    registry = request.app.state.registry
    cv = registry.get_cv(cv_id)
    if not cv:
        raise HTTPException(status_code=404, detail="CV not found")
    
    if cv and cv.experiences:
        # Sort experiences by start_date, descending (newest first).
        # We use a default value ('0000-00-00') for any None or empty dates
        # to ensure they are sorted to the bottom as the "oldest".
        cv.experiences.sort(
            key=lambda exp: exp.start_date or '0000-00-00', 
            reverse=True
        )
    return cv

@router.patch("/{cv_id}")
def update_cv(cv_id: str, data: CVUpdate, request: Request):
    """Update general CV metadata (name, summary, contact_info)."""
    try:
        registry = request.app.state.registry
        return registry.update_cv(cv_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{cv_id}")
def delete_cv(cv_id: str, request: Request):
    """Delete a CV by ID."""
    try:
        registry = request.app.state.registry
        return registry.delete_cv(cv_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------------------------------------------------------------------
# NESTED ADD ENDPOINTS (CV Components)
# ---------------------------------------------------------------------

# --- *** NEW: Complex Experience Endpoint (CREATE) *** ---
@router.post("/{cv_id}/experience/complex")
def add_experience_complex(cv_id: str, payload: ExperienceComplexPayload, request: Request):
    """
    Creates a new experience and all its dependencies (new skills, new achievements)
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.create_experience_from_payload(cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- *** NEW: Complex Experience Endpoint (UPDATE) *** ---
@router.patch("/{cv_id}/experience/{exp_id}/complex")
def update_experience_complex(cv_id: str, exp_id: str, payload: ExperienceComplexPayload, request: Request):
    """
    Updates an existing experience and all its dependencies (new/modified skills
    and achievements) from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.update_experience_from_payload(cv_id, exp_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- *** OLD Endpoints Removed *** ---
# @router.post("/{cv_id}/experience") ... (REMOVED)
# @router.patch("/{cv_id}/experience/{exp_id}") ... (REMOVED)

 # --- *** NEW: Complex Education Endpoints *** ---
@router.post("/{cv_id}/education/complex")
def add_education_complex(cv_id: str, payload: EducationComplexPayload, request: Request): # <--- FIXED
    """
    Creates a new education entry and all its dependencies (new skills, new achievements)
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.create_education_from_payload(cv_id, payload) # <--- FIXED
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/education/{edu_id}/complex")
def update_education_complex(cv_id: str, edu_id: str, payload: EducationComplexPayload, request: Request):
    """
    Updates an existing education entry and all its dependencies
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.update_education_from_payload(cv_id, edu_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
# --- *** END NEW ENDPOINTS *** ---   

# --- *** NEW: Complex Hobby Endpoints *** ---
@router.post("/{cv_id}/hobby/complex")
def add_hobby_complex(cv_id: str, payload: HobbyComplexPayload, request: Request):
    """
    Creates a new hobby and all its dependencies (new skills, new achievements)
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.create_hobby_from_payload(cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/hobby/{hobby_id}/complex")
def update_hobby_complex(cv_id: str, hobby_id: str, payload: HobbyComplexPayload, request: Request):
    """
    Updates an existing hobby and all its dependencies
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.update_hobby_from_payload(cv_id, hobby_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
# --- *** END NEW ENDPOINTS *** ---

@router.post("/{cv_id}/education")
def add_education(
    cv_id: str, 
    institution: str, 
    degree: str, 
    field: str,
    request: Request,
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None) # <-- *** ADDED THIS LINE ***
):
    """Add a new education entry to the CV."""
    try:
        registry = request.app.state.registry
        # *** ADDED skill_ids TO THE CALL ***
        return registry.add_cv_education(
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
    description: Optional[str] = None
):
    """Add a new skill to the CV's master list."""
    try:
        registry = request.app.state.registry
        return registry.add_cv_skill(cv_id, name=name, category=category, level=level, importance=importance, description=description)
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
    skill_ids: Optional[List[str]] = Query(None) # <-- *** ADDED THIS LINE ***
):
    """Add a new project to the CV."""
    try:
        registry = request.app.state.registry
        # *** ADDED skill_ids TO THE CALL ***
        return registry.add_cv_project(
            cv_id, 
            title=title, 
            description=description, 
            related_experience_id=related_experience_id, 
            related_education_id=related_education_id, 
            skill_ids=skill_ids
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ... (rest of the file, e.g., add_hobby, add_achievement, and linking endpoints, remains unchanged) ...


@router.post("/{cv_id}/hobby")
def add_hobby(
    cv_id: str, 
    name: str, 
    request: Request,
    description: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None), 
    achievement_ids: Optional[List[str]] = Query(None) # <-- *** ADDED THIS LINE ***
):
    """Add a new hobby to the CV."""
    try:
        registry = request.app.state.registry
        # Pass achievement_ids to the registry method
        return registry.add_cv_hobby(
            cv_id, 
            name=name, 
            description=description, 
            skill_ids=skill_ids,
            achievement_ids=achievement_ids # <-- *** ADDED HERE ***
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

# --- *** NEW: Complex Project Endpoints *** ---
@router.post("/{cv_id}/project/complex")
def add_project_complex(cv_id: str, payload: ProjectComplexPayload, request: Request):
    """
    Creates a new project and all its dependencies (new skills, new achievements)
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.create_project_from_payload(cv_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.patch("/{cv_id}/project/{project_id}/complex")
def update_project_complex(cv_id: str, project_id: str, payload: ProjectComplexPayload, request: Request):
    """
    Updates an existing project and all its dependencies
    from a single complex payload.
    """
    try:
        registry = request.app.state.registry
        return registry.update_project_from_payload(cv_id, project_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
# --- *** END NEW ENDPOINTS *** ---

@router.post("/{cv_id}/achievement")
def add_achievement(
    cv_id: str, # <-- Corrected parameter name from cvId
    text: str,
    request: Request,
    context: Optional[str] = None,
    skill_ids: Optional[List[str]] = Query(None) # <-- *** ADDED THIS LINE ***
):
    """Add a new global achievement to the CV's master list."""
    try:
        # Pass skill_ids to the registry method
        registry = request.app.state.registry
        return registry.add_cv_achievement(
            cv_id, # <-- Corrected parameter name
            text=text, 
            context=context, 
            skill_ids=skill_ids # <-- *** ADDED HERE ***
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
# ---------------------------------------------------------------------
# NESTED LINKING ENDPOINTS (Skills and Achievements)
# ---------------------------------------------------------------------


# --- Skill Linking ---

@router.post("/{cv_id}/experience/{exp_id}/skill/{skill_id}")
def link_skill_to_experience(cv_id: str, exp_id: str, skill_id: str, request: Request):
    """Links a master skill to a specific experience."""
    try:
        registry = request.app.state.registry
        return registry.link_skill_to_entity(cv_id, exp_id, skill_id, 'experiences')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/skill/{skill_id}")
def link_skill_to_project(cv_id: str, proj_id: str, skill_id: str, request: Request):
    """Links a master skill to a specific project."""
    try:
        registry = request.app.state.registry
        return registry.link_skill_to_entity(cv_id, proj_id, skill_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/achievement/{ach_id}/skill/{skill_id}")
def link_skill_to_achievement(cv_id: str, ach_id: str, skill_id: str, request: Request):
    """Links a master skill to a specific achievement (tags the achievement)."""
    try:
        registry = request.app.state.registry
        return registry.link_skill_to_entity(cv_id, ach_id, skill_id, 'achievements')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Achievement Linking ---

@router.post("/{cv_id}/experience/{exp_id}/achievement/{ach_id}")
def link_achievement_to_experience(cv_id: str, exp_id: str, ach_id: str, request: Request):
    """Links a master achievement to a specific experience."""
    try:
        registry = request.app.state.registry
        return registry.link_achievement_to_context(cv_id, exp_id, ach_id, 'experiences')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/achievement/{ach_id}")
def link_achievement_to_project(cv_id: str, proj_id: str, ach_id: str, request: Request):
    """Links a master achievement to a specific project."""
    try:
        registry = request.app.state.registry
        return registry.link_achievement_to_context(cv_id, proj_id, ach_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{cv_id}/project/{proj_id}/achievement/{ach_id}")
def link_achievement_to_project(cv_id: str, proj_id: str, ach_id: str, request: Request):
    """Links a master achievement to a specific project."""
    try:
        registry = request.app.state.registry
        return registry.link_achievement_to_context(cv_id, proj_id, ach_id, 'projects')
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- NEW AGGREGATION ENDPOINT ---

@router.get("/{cv_id}/{entity_list_name}/{entity_id}/skills/aggregated")
def get_aggregated_skills(cv_id: str, entity_list_name: str, entity_id: str, request: Request):
    """
    Fetches all unique skills for an entity and its children.
    Valid entity_list_name examples: 'experiences', 'projects', 'achievements'.
    """
    try:
        # Validate entity_list_name to prevent arbitrary calls
        valid_lists = ['experiences', 'projects', 'achievements', 'education', 'hobbies']
        if entity_list_name not in valid_lists:
            raise ValueError(f"Invalid entity list name. Must be one of: {valid_lists}")
        registry = request.app.state.registry
        return registry.get_aggregated_skills_for_entity(cv_id, entity_list_name, entity_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    

@router.delete("/{cv_id}/{list_name}/{item_id}")
def delete_nested_item(cv_id: str, list_name: str, item_id: str, request: Request):
    """
    Deletes a specific nested item (Experience, Skill, Education, etc.)
    from a CV's master list.
    """
    
    # Map the list_name from the URL to the correct registry function
    # (These functions already exist in your registry.py)
    registry = request.app.state.registry
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
        # Call the corresponding registry function
        # e.g., registry.delete_cv_experience(cv_id, item_id)
        return func(cv_id, item_id)
    except ValueError as e:
        # This catches errors if the CV or the nested item isn't found
        raise HTTPException(status_code=404, detail=str(e))