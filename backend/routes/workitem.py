# backend/routes/workitem.py

from fastapi import APIRouter, HTTPException
from typing import Optional, List # Ensure List is imported
from backend.core.registry import Registry
from backend.core.models import WorkItemUpdate # Import the update model

router = APIRouter()
registry = Registry()

# ---------------------------------------------------------------------
# WorkItem Endpoints
# ---------------------------------------------------------------------

@router.post("/")
def create_work_item(
    title: str,
    work_type: str = "research", # Renamed parameter to avoid conflict with Python 'type'
    related_application_id: Optional[str] = None,
    related_interview_id: Optional[str] = None,
    related_job_id: Optional[str] = None,
    related_goal_id: Optional[str] = None,
    effort_hours: Optional[float] = None,
    tags: Optional[List[str]] = None,
    reflection: Optional[str] = None,
    outcome: Optional[str] = None,
):
    """Create a new work item."""
    return registry.create_work_item(
        title=title,
        work_type=work_type, # Pass the corrected parameter name
        related_application_id=related_application_id,
        related_interview_id=related_interview_id,
        related_job_id=related_job_id,
        related_goal_id=related_goal_id,
        effort_hours=effort_hours,
        tags=tags or [],
        reflection=reflection,
        outcome=outcome,
    )

@router.get("/")
def list_work_items():
    """List all work items."""
    return registry.all_work_items()

@router.get("/{work_id}")
def get_work_item(work_id: str):
    """Fetch a specific work item by ID."""
    work = registry.get_work_item(work_id)
    if not work:
        raise HTTPException(status_code=404, detail="WorkItem not found")
    return work

@router.patch("/{work_id}")
def update_work_item(work_id: str, data: WorkItemUpdate):
    """Update a work item."""
    try:
        return registry.update_work_item(work_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{work_id}")
def delete_work_item(work_id: str):
    """Delete a work item by ID."""
    try:
        return registry.delete_work_item(work_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{work_id}/complete")
def mark_work_item_complete(work_id: str, reflection: Optional[str] = None):
    """Mark a work item as completed."""
    try:
        return registry.mark_work_item_completed(work_id, reflection)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))