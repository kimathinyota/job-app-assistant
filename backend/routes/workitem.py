# backend/routes/workitem.py

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional, List
from backend.core.registry import Registry
from backend.core.models import WorkItemUpdate, User
from backend.routes.auth import get_current_user # Adjust import path if needed

router = APIRouter()

# ---------------------------------------------------------------------
# WorkItem Endpoints
# ---------------------------------------------------------------------

@router.post("/")
def create_work_item(
    title: str,
    request: Request,
    work_type: str = "research", 
    related_application_id: Optional[str] = None,
    related_interview_id: Optional[str] = None,
    related_job_id: Optional[str] = None,
    related_goal_id: Optional[str] = None,
    effort_hours: Optional[float] = None,
    tags: Optional[List[str]] = None,
    reflection: Optional[str] = None,
    outcome: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Create a new work item."""
    registry: Registry = request.app.state.registry
    return registry.create_work_item(
        user.id,
        title=title,
        work_type=work_type,
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
def list_work_items(
    request: Request,
    user: User = Depends(get_current_user)
):
    """List all work items belonging to the user."""
    registry: Registry = request.app.state.registry
    return registry.all_work_items(user.id)

@router.get("/{work_id}")
def get_work_item(
    work_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific work item by ID."""
    registry: Registry = request.app.state.registry
    work = registry.get_work_item(work_id, user.id)
    if not work:
        raise HTTPException(status_code=404, detail="WorkItem not found")
    return work

@router.patch("/{work_id}")
def update_work_item(
    work_id: str, 
    data: WorkItemUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update a work item."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_work_item(user.id, work_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{work_id}")
def delete_work_item(
    work_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete a work item by ID."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_work_item(user.id, work_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{work_id}/complete")
def mark_work_item_complete(
    work_id: str, 
    request: Request, 
    reflection: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Mark a work item as completed."""
    try:
        registry: Registry = request.app.state.registry
        return registry.mark_work_item_completed(user.id, work_id, reflection)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))