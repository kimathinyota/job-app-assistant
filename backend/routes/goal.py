# backend/routes/goal.py

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Optional
from backend.core.registry import Registry
from backend.core.models import GoalUpdate, User
from backend.routes.auth import get_current_user # Adjust import path if needed

router = APIRouter()

# ---------------------------------------------------------------------
# Goal Endpoints
# ---------------------------------------------------------------------

@router.post("/")
def create_goal(
    title: str, 
    request: Request, 
    description: Optional[str] = None, 
    metric: Optional[str] = None,
    user: User = Depends(get_current_user)
):
    """Create a new job search goal."""
    registry: Registry = request.app.state.registry
    return registry.create_goal(user.id, title, description, metric)

@router.get("/")
def list_goals(
    request: Request,
    user: User = Depends(get_current_user)
):
    """List all goals belonging to the user."""
    registry: Registry = request.app.state.registry
    return registry.all_goals(user.id)

@router.get("/{goal_id}")
def get_goal(
    goal_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Fetch a specific goal by ID."""
    registry: Registry = request.app.state.registry
    # Pass user.id to enforce ownership
    goal = registry.get_goal(goal_id, user.id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal

@router.patch("/{goal_id}")
def update_goal(
    goal_id: str, 
    data: GoalUpdate, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Update goal metadata (title, status, due_date, etc.)."""
    try:
        registry: Registry = request.app.state.registry
        return registry.update_goal(user.id, goal_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{goal_id}")
def delete_goal(
    goal_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Delete a goal by ID, and unlink it from any WorkItems."""
    try:
        registry: Registry = request.app.state.registry
        return registry.delete_goal(user.id, goal_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{goal_id}/add-work/{work_id}")
def add_work_item_to_goal(
    goal_id: str, 
    work_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """Link an existing WorkItem to a Goal and update goal progress."""
    try:
        registry: Registry = request.app.state.registry
        return registry.add_work_to_goal(user.id, goal_id, work_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))