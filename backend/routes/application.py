# backend/routes/application.py

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional
from backend.core.registry import Registry
from backend.core.models import ApplicationUpdate, ApplicationStatus, AppSuiteData, User, DerivedCV
from backend.routes.auth import get_current_user # Adjust import path if you put the dependency in backend.core.security

router = APIRouter()

@router.post("/")
def create_application(
    job_id: str, 
    base_cv_id: str, 
    request: Request, 
    mapping_id: Optional[str] = None, 
    derived_cv_id: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Create a new job application record."""
    registry: Registry = request.app.state.registry
    return registry.create_application(user.id, job_id, base_cv_id, mapping_id, derived_cv_id)

@router.get("/")
def list_applications(
    request: Request,
    user: User = Depends(get_current_user)
):
    """List all application records belonging to the user."""
    registry: Registry = request.app.state.registry
    return registry.all_applications(user.id)

# --- 2. ADD THE NEW ENDPOINT (at the end of the file) ---
@router.get("/app-suite-data/", response_model=AppSuiteData)
def get_app_suite_data(
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    Fetch all jobs and applications in a single call
    for the Application Suite view.
    """
    try:
        # This calls the new registry method with user_id
        registry: Registry = request.app.state.registry
        return registry.get_app_suite_data(user.id)
    except Exception as e:
        # Generic catch-all in case of read errors
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{app_id}")
def get_application(
    app_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific application by ID."""
    registry: Registry = request.app.state.registry
    # Pass user.id to ensure they own it
    app = registry.get_application(app_id, user.id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    return app


@router.patch("/{app_id}")
def update_application(
    app_id: str, 
    data: ApplicationUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update application metadata (status, notes)."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_application(user.id, app_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{app_id}/status")
def update_application_status(
    app_id: str, 
    data: ApplicationStatus, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Quickly update the application status only."""
    try:
        # Re-use ApplicationUpdate to pass data structure to registry
        registry: Registry = request.app.state.registry
        return registry.update_application(user.id, app_id, ApplicationUpdate(status=data.status))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{app_id}")
def delete_application(
    app_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete an application record by ID."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_application(user.id, app_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


from backend.core.models import ApplicationUpdate # Ensure this is imported

@router.post("/{app_id}/tailored-cv", response_model=DerivedCV)
def get_or_create_tailored_cv(
    app_id: str,
    force_refresh: bool = False,
    request: Request = None,
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    app = registry.get_application(app_id, user.id)
    if not app:
        raise HTTPException(404, "Application not found")

    # 1. Return existing if valid
    if app.derived_cv_id and not force_refresh:
        existing_cv = registry.get_cv(app.derived_cv_id, user.id)
        if existing_cv:
            return existing_cv

    # 2. Generate New
    base_cv = registry.get_cv(app.base_cv_id, user.id)
    mapping = registry.get_mapping(app.mapping_id, user.id)
    
    if not base_cv or not mapping:
        raise HTTPException(400, "Missing Base CV or Mapping")

    new_derived_cv = DerivedCV.from_mapping(base_cv, app.job_id, mapping)
    
    # 3. Save CV
    registry.save_derived_cv(user.id, new_derived_cv)
    
    # 4. Link to Application (Using specific Update model)
    # This ensures we don't accidentally wipe other fields or fail validation
    update_payload = ApplicationUpdate(derived_cv_id=new_derived_cv.id)
    registry.update_application(user.id, app.id, update_payload)
    
    return new_derived_cv