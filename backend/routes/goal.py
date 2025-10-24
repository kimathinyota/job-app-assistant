# backend/routes/goal.py

from fastapi import APIRouter
from typing import Optional
from backend.core.registry import Registry

router = APIRouter()
registry = Registry()

# ---------------------------------------------------------------------
# Goal Endpoints
# ---------------------------------------------------------------------

@router.post("/")
def create_goal(title: str, description: Optional[str] = None, metric: Optional[str] = None):
    """Create a new job search goal."""
    return registry.create_goal(title, description, metric)

@router.get("/")
def list_goals():
    """List all goals."""
    return registry.all_goals()

@router.get("/{goal_id}")
def get_goal(goal_id: str):
    """Fetch a specific goal by ID."""
    return registry.get_goal(goal_id)

@router.post("/{goal_id}/add-work/{work_id}")
def add_work_item_to_goal(goal_id: str, work_id: str):
    """Link an existing WorkItem to a Goal and update goal progress."""
    return registry.add_work_to_goal(goal_id, work_id)