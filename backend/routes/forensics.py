from fastapi import APIRouter, HTTPException, Request
from typing import Literal, Optional, List
from pydantic import BaseModel

from backend.core.services.mapping_service import MappingOptimizer
from backend.core.forensics import ForensicCalculator
from backend.core.models import ForensicAnalysis, Mapping
# You might need to import your TUNING_MODES constant if it's shared, 
# otherwise I have defined it below for self-containment.
import hashlib
router = APIRouter()

# --- Tuning Configuration (Same as your inference route) ---
TUNING_MODES = {
    "super_eager": {
        "description": "Matches almost anything remotely related. High recall, low precision.",
        "config": {"min_score": 0.15, "top_k": 50}
    },
    "eager_mode": {
        "description": "Generous matching. Good for brainstorming.",
        "config": {"min_score": 0.22, "top_k": 30}
    },
    "balanced_default": {
        "description": "The standard balance of relevance and coverage.",
        "config": {"min_score": 0.28, "top_k": None} # None = Allow all valid matches
    },
    "picky_mode": {
        "description": "Only shows solid matches.",
        "config": {"min_score": 0.35, "top_k": None}
    },
    "super_picky": {
        "description": "Strict. Only high-confidence matches.",
        "config": {"min_score": 0.45, "top_k": None}
    }
}

# --- Request Model ---
class GenerateRoleCaseRequest(BaseModel):
    job_id: str
    cv_id: str
    mode: Literal[
        "super_eager", 
        "eager_mode", 
        "balanced_default", 
        "picky_mode", 
        "super_picky"
    ] = "balanced_default"



class PromoteRequest(BaseModel):
    alternative_id: str

@router.post("/applications/{app_id}/mappings/{feature_id}/promote")
def promote_match(app_id: str, feature_id: str, payload: PromoteRequest, request: Request):
    registry = request.app.state.registry
    app = registry.get_application(app_id)
    mapping = registry.get_mapping(app.mapping_id)
    
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair: raise HTTPException(404, "Pair not found")
    
    if MappingOptimizer.promote_alternative(pair, payload.alternative_id):
        registry.save_mapping(mapping)
    
    job = registry.get_job(app.job_id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    return new_analysis

@router.post("/applications/{app_id}/mappings/{feature_id}/approve")
def approve_match(app_id: str, feature_id: str, request: Request):
    registry = request.app.state.registry
    app = registry.get_application(app_id)
    mapping = registry.get_mapping(app.mapping_id)
    
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair: raise HTTPException(404, "Pair not found")
    
    MappingOptimizer.approve_current_match(pair)
    registry.save_mapping(mapping)
    
    job = registry.get_job(app.job_id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    return new_analysis


# -----------------------------------------------------------------------------
# 1. GENERATE ROLE CASE (Inference + Forensics Pipeline)
# -----------------------------------------------------------------------------
@router.post("/generate", response_model=ForensicAnalysis)
def generate_role_case(payload: GenerateRoleCaseRequest, request: Request):
    """
    The 'One-Click' Endpoint:
    1. Finds or Creates a Mapping for the Job/CV pair.
    2. Runs the NLP Inference (fresh).
    3. Saves the results to DB.
    4. Calculates and returns the Forensic Analysis (RoleCase).
    """
    registry = request.app.state.registry
    inferer = request.app.state.inferer
    
    # 1. Validate Inputs
    job = registry.get_job(payload.job_id)
    cv = registry.get_cv(payload.cv_id)
    
    if not job or not cv:
        raise HTTPException(404, "Job or CV not found")

    # 2. Find or Create Mapping
    # We check if a mapping already exists for this pair
    existing_mappings = registry.all_mappings()
    mapping = next(
        (m for m in existing_mappings if m.job_id == payload.job_id and m.base_cv_id == payload.cv_id), 
        None
    )

    if not mapping:
        # Create a new blank mapping container
        mapping = registry.create_mapping(job_id=payload.job_id, base_cv_id=payload.cv_id)
    
    # 3. RUN INFERENCE (The "Brain")
    try:
        mode_settings = TUNING_MODES.get(payload.mode, TUNING_MODES["balanced_default"])
        config_params = mode_settings.get("config", {})
        
        # This returns a list of MappingPair objects (in memory)
        suggestions = inferer.infer_mappings(job, cv, **config_params)
        
    except Exception as e:
        raise HTTPException(500, f"Inference Engine Failed: {str(e)}")

    # 4. SAVE RESULTS
    # We overwrite the existing pairs with the fresh AI suggestions
    mapping.pairs = suggestions
    registry.save_mapping(mapping)

    # 5. RUN FORENSICS (The "Verdict")
    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping)
    
    # Inject context
    analysis.application_id = None # Optional, or you could look up related application ID

    return analysis


# -----------------------------------------------------------------------------
# 2. GET EXISTING ANALYSIS (Read-Only)
# -----------------------------------------------------------------------------
@router.get("/applications/{app_id}/forensic-analysis", response_model=ForensicAnalysis)
def get_forensic_analysis(app_id: str, request: Request):
    registry = request.app.state.registry
    
    app = registry.get_application(app_id)
    if not app: raise HTTPException(404, "Application not found")
    
    job = registry.get_job(app.job_id)
    mapping = registry.get_mapping(app.mapping_id)
    
    if not job or not mapping:
        raise HTTPException(404, "Job or Mapping data missing")

    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping)
    
    analysis.application_id = app_id
    return analysis


# -----------------------------------------------------------------------------
# 3. REJECT MATCH (Interactive Triage)
# -----------------------------------------------------------------------------
@router.post("/applications/{app_id}/mappings/{feature_id}/reject")
def reject_match(app_id: str, feature_id: str, request: Request):
    registry = request.app.state.registry
    
    # A. Fetch Context
    app = registry.get_application(app_id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id)
    if not mapping: raise HTTPException(404, "Mapping not found")

    # B. Find Pair
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair:
        raise HTTPException(404, "Mapping pair not found")

    # C. OPTIMIZE (Modify in Memory)
    MappingOptimizer.reject_current_match(pair)

    # D. SAVE (Persist changes)
    registry.save_mapping(mapping)

    # E. RE-CALCULATE (Instant Feedback)
    job = registry.get_job(app.job_id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    
    new_analysis.application_id = app_id

    return {
        "success": True,
        "updated_pair": pair,
        "new_forensics": new_analysis 
    }

class ManualMatchRequest(BaseModel):
    # The user provides the text evidence or links to a CV item
    evidence_text: str 
    cv_item_id: Optional[str] = None
    cv_item_type: Optional[str] = None # e.g. "experiences", "projects"
    cv_item_name: Optional[str] = None # e.g. "Senior Dev at Google"

@router.post("/applications/{app_id}/mappings/{feature_id}/manual", response_model=ForensicAnalysis)
def create_manual_match(app_id: str, feature_id: str, payload: ManualMatchRequest, request: Request):
    """
    User forces a match for a specific requirement.
    This OVERRIDES any existing AI match or rejection.
    """
    registry = request.app.state.registry
    
    # 1. Setup Context
    app = registry.get_application(app_id)
    if not app: raise HTTPException(404, "Application not found")
    mapping = registry.get_mapping(app.mapping_id)
    job = registry.get_job(app.job_id)
    
    # 2. Get or Create the Pair
    # (If it was previously missing/rejected, we might need to reset it)
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair:
        # Create a blank pair if somehow missing (rare)
        # In practice, you might just fetch the feature and append a new pair
        raise HTTPException(404, "Mapping pair container not found")

    # 3. Construct the "Manual" Candidate
    # We treat user input as 100% confidence (1.0)
    manual_lineage = []
    if payload.cv_item_id:
        manual_lineage.append(LineageItem(
            id=payload.cv_item_id, 
            type=payload.cv_item_type or "manual", 
            name=payload.cv_item_name or "Manual Link"
        ))

    manual_candidate = MatchCandidate(
        segment_text=payload.evidence_text,
        segment_type="manual_override",
        score=1.0, # User says it matches, so it matches.
        lineage=manual_lineage
    )

    # 4. Update the Pair State
    if not pair.meta:
        pair.meta = MatchingMeta(best_match=manual_candidate, summary_note="")
    
    # Force this candidate to be the winner
    pair.meta.best_match = manual_candidate
    pair.strength = 1.0 # Max score
    pair.context_item_id = payload.cv_item_id
    pair.context_item_type = payload.cv_item_type
    pair.context_item_text = payload.cv_item_name or "Manual Match"
    
    # Create a distinct note so UI knows it's manual
    pair.meta.summary_note = f"Manual Match: \"{payload.evidence_text[:50]}...\""
    
    # 5. Save & Recalculate
    registry.save_mapping(mapping)
    
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    
    return new_analysis