from fastapi import APIRouter
from typing import Optional
from backend.core.registry import Registry

router = APIRouter()
registry = Registry()

# ---------------------------------------------------------------------
# WorkItem Endpoints
# ---------------------------------------------------------------------

@router.post("/")
def create_work_item(
    title: str,
    type: str = "research",
    related_application_id: Optional[str] = None,
    related_interview_id: Optional[str] = None,
    related_job_id: Optional[str] = None,
    related_goal_id: Optional[str] = None,
    effort_hours: Optional[float] = None,
    tags: Optional[list[str]] = None,
    reflection: Optional[str] = None,
    outcome: Optional[str] = None,
):
    """Create a new work item."""
    return registry.create_work_item(
        title=title,
        type=type,
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
    return registry.get_work_item(work_id)


@router.post("/{work_id}/complete")
def mark_work_item_complete(work_id: str, reflection: Optional[str] = None):
    """Mark a work item as completed."""
    return registry.mark_work_item_completed(work_id, reflection)
