# backend/routes/application.py

from fastapi import APIRouter, HTTPException
from backend.core.registry import Registry
from backend.core.models import ApplicationUpdate, ApplicationStatus, AppSuiteData # Import update models
from typing import Optional
from backend.core.dependencies import registry 


router = APIRouter()

@router.post("/")
def create_application(job_id: str, base_cv_id: str, mapping_id: Optional[str] = None, derived_cv_id: Optional[str] = None):
    """Create a new job application record."""
    return registry.create_application(job_id, base_cv_id, mapping_id, derived_cv_id)

@router.get("/")
def list_applications():
    """List all application records."""
    return registry.all_applications()

# --- 2. ADD THE NEW ENDPOINT (at the end of the file) ---
@router.get("/app-suite-data/", response_model=AppSuiteData)
def get_app_suite_data():
    """
    Fetch all jobs and applications in a single call
    for the Application Suite view.
    """
    try:
        # This calls the new registry method
        return registry.get_app_suite_data()
    except Exception as e:
        # Generic catch-all in case of read errors
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/{app_id}")
def get_application(app_id: str):
    """Fetch a specific application by ID."""
    app = registry.get_application(app_id)
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    return app


@router.patch("/{app_id}")
def update_application(app_id: str, data: ApplicationUpdate):
    """Update application metadata (status, notes)."""
    try:
        return registry.update_application(app_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.put("/{app_id}/status")
def update_application_status(app_id: str, data: ApplicationStatus):
    """Quickly update the application status only."""
    try:
        # Re-use ApplicationUpdate to pass data structure to registry
        return registry.update_application(app_id, ApplicationUpdate(status=data.status))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{app_id}")
def delete_application(app_id: str):
    """Delete an application record by ID."""
    try:
        return registry.delete_application(app_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))