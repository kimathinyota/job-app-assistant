# backend/routes/goal.py

from fastapi import APIRouter, HTTPException, Request
from typing import Optional
from backend.core.registry import Registry
from backend.core.models import GoalUpdate # Import the update model

router = APIRouter()

# ---------------------------------------------------------------------
# Goal Endpoints
# ---------------------------------------------------------------------

@router.post("/")
def create_goal(title: str, request: Request, description: Optional[str] = None, metric: Optional[str] = None):
    """Create a new job search goal."""
    registry = request.app.state.registry
    return registry.create_goal(title, description, metric)

@router.get("/")
def list_goals(request: Request):
    """List all goals."""
    registry = request.app.state.registry
    return registry.all_goals()

@router.get("/{goal_id}")
def get_goal(goal_id: str, request: Request):
    """Fetch a specific goal by ID."""
    registry = request.app.state.registry
    goal = registry.get_goal(goal_id)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal

@router.patch("/{goal_id}")
def update_goal(goal_id: str, data: GoalUpdate, request: Request):
    """Update goal metadata (title, status, due_date, etc.)."""
    try:
        registry = request.app.state.registry
        return registry.update_goal(goal_id, data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.delete("/{goal_id}")
def delete_goal(goal_id: str, request: Request):
    """Delete a goal by ID, and unlink it from any WorkItems."""
    try:
        registry = request.app.state.registry
        return registry.delete_goal(goal_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/{goal_id}/add-work/{work_id}")
def add_work_item_to_goal(goal_id: str, work_id: str, request: Request):
    """Link an existing WorkItem to a Goal and update goal progress."""
    try:
            
        return registry.add_work_to_goal(goal_id, work_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))