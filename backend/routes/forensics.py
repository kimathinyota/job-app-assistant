# backend/routes/forensics.py

from fastapi import APIRouter, HTTPException, Request, Depends
from typing import Literal, Optional, List
from pydantic import BaseModel

from backend.core.services.mapping_service import MappingOptimizer
from backend.core.forensics import ForensicCalculator
from backend.core.models import ForensicAnalysis, Mapping, LineageItem, MatchCandidate, MatchingMeta, User, ApplicationUpdate
from backend.core.registry import Registry
from backend.routes.auth import get_current_user
from backend.core.services.scoring import ScoringService # Import ScoringService

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
        "config": {"min_score": 0.4, "top_k": None} # None = Allow all valid matches
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


def _update_job_cache_from_mapping(user_id: str, job_id: str, mapping: Mapping, registry: Registry):
    """
    Recalculates the score based on the current mapping and updates the Job Cache.
    """
    # 1. Fetch Job
    job = registry.get_job(job_id, user_id)
    if not job: return

    # 2. Recalculate Forensics (Uses the NEW Approved/Rejected state)
    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping)
    stats = analysis.stats

    # 3. Update the Job Cache (Score, Grade, Badges)
    job.match_score = stats.overall_match_score
    
    # Grade Logic
    if stats.overall_match_score >= 85: job.match_grade = "A"
    elif stats.overall_match_score >= 65: job.match_grade = "B"
    elif stats.overall_match_score >= 40: job.match_grade = "C"
    else: job.match_grade = "D"

    # Badge Logic
    badges = []
    if stats.critical_gaps_count > 0: badges.append("Missing Critical Skills")
    if stats.overall_match_score > 90: badges.append("Top Match")
    # Add other badges as needed...
    job.cached_badges = badges

    # 4. Save
    registry.update_job(user_id, job.id, job)

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
def promote_match(
    app_id: str, 
    feature_id: str, 
    payload: PromoteRequest, 
    request: Request,
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair: raise HTTPException(404, "Pair not found")
    
    if MappingOptimizer.promote_alternative(pair, payload.alternative_id):
        registry.save_mapping(user.id, mapping)
    
    job = registry.get_job(app.job_id, user.id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    return new_analysis

@router.post("/applications/{app_id}/mappings/{feature_id}/approve")
def approve_match(
    app_id: str, 
    feature_id: str, 
    request: Request, 
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    
    # 1. Fetch Data
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "App not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    if not mapping: raise HTTPException(404, "Mapping not found")

    # 2. Find and Update the Pair (Your existing logic)
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair: raise HTTPException(404, "Pair not found")
    
    pair.status = "user_approved"  # <--- The User Action
    registry.save_mapping(user.id, mapping) # <--- Saved to DB

    # 3. NEW: Trigger Background Score Update
    # This ensures the Job Board immediately reflects the higher score
    _update_job_cache_from_mapping(user.id, app.job_id, mapping, registry)

    return {"status": "success", "new_status": pair.status}

# -----------------------------------------------------------------------------
# 1. GENERATE ROLE CASE (Inference + Forensics Pipeline)
# -----------------------------------------------------------------------------
@router.post("/generate", response_model=ForensicAnalysis)
def generate_role_case(
    payload: GenerateRoleCaseRequest, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    The 'One-Click' Endpoint:
    1. Finds or Creates a Mapping for the Job/CV pair.
    2. Runs the NLP Inference (fresh).
    3. Saves the results to DB.
    4. Calculates and returns the Forensic Analysis (RoleCase).
    """
    registry: Registry = request.app.state.registry
    inferer = request.app.state.inferer
    
    # 1. Validate Inputs (Secured)
    job = registry.get_job(payload.job_id, user.id)
    cv = registry.get_cv(payload.cv_id, user.id)
    
    if not job or not cv:
        raise HTTPException(404, "Job or CV not found")

    # 2. Find or Create Mapping
    # We check if a mapping already exists for this pair
    existing_mappings = registry.all_mappings(user.id)
    mapping = next(
        (m for m in existing_mappings if m.job_id == payload.job_id and m.base_cv_id == payload.cv_id), 
        None
    )

    if not mapping:
        # Create a new blank mapping container
        mapping = registry.create_mapping(user.id, job_id=payload.job_id, base_cv_id=payload.cv_id)
    
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
    registry.save_mapping(user.id, mapping)

    # 5. RUN FORENSICS (The "Verdict")
    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping)
    
    # Inject context
    analysis.application_id = None # Optional, or you could look up related application ID

    return analysis


# backend/routes/forensics.py

@router.get("/applications/{app_id}/forensic-analysis", response_model=ForensicAnalysis)
def get_forensic_analysis(
    app_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry

    print(f"[Forensics] Fetching forensic analysis for Application {app_id}..." )
    
    # 1. Fetch Context
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    job = registry.get_job(app.job_id, user.id)

    print(f"[Forensics] Found Application for Job {job.id}: {job.title if job else 'N/A'}"  )
    
    # Handle missing mapping gracefully (Auto-create if missing)
    if not app.mapping_id:
        print(f"[Forensics] No mapping found for Application {app.id}. Creating new mapping...")
        mapping = registry.create_mapping(user.id, app.job_id, app.base_cv_id)
        print(f"[Forensics] Created Mapping {mapping.id} for Job {app.job_id} and CV {app.base_cv_id}")
        app.mapping_id = mapping.id
        registry.update_application(user.id, app.id, ApplicationUpdate(mapping_id=mapping.id))
    else:
        mapping = registry.get_mapping(app.mapping_id, user.id)
    
    print(f"[Forensics] Mapping ID: {mapping.id if mapping else 'N/A'}, Pairs Count: {len(mapping.pairs) if mapping else 'N/A'}"  )

    if not job or not mapping:
        raise HTTPException(404, "Job or Mapping data missing")

    # --- LAZY INFERENCE FIX ---
    # 2. Check if mapping is empty. If so, run the AI "Just in Time".
    if not mapping.pairs:
        print(f"[Forensics] Mapping {mapping.id} is empty. Running Lazy Inference...")
        try:
            inferer = request.app.state.inferer
            cv = registry.get_cv(app.base_cv_id, user.id)
            
            # Use default balanced settings or config
            mode_settings = "balanced_default" 
            # (Assuming you have access to TUNING_MODES or defaults)


            print(f"[Forensics] Running inference for Job {job.id} with CV {cv.id} using mode: {mode_settings}")
            
            # Run AI
            suggestions = inferer.infer_mappings(job, cv, min_score=0.20)
            
            # Save results
            mapping.pairs = suggestions
            registry.save_mapping(user.id, mapping)
            print(f"[Forensics] Mapping after inference: Mapping ID: {mapping.id if mapping else 'N/A'}, Pairs Count: {len(mapping.pairs) if mapping else 'N/A'}"  )

            print(f"[Forensics] Lazy Inference complete. {len(suggestions)} suggestions saved to Mapping {mapping.id}.")
            print(f"[Forensics] Inference complete. Found {len(suggestions)} matches.")
            
        except Exception as e:
            # Log error but don't crash if AI fails (return empty analysis)
            print(f"[Forensics] Lazy Inference Failed: {str(e)}")

    # 3. Calculate Analysis
    print(f"[Forensics] Calculating forensic analysis for Job {job.id} with Mapping {mapping.id}..."   )
    calculator = ForensicCalculator()
    analysis = calculator.calculate(job, mapping)
    print(f"[Forensics] Analysis complete. Stats: {analysis.stats}")

    # --- CACHE WRITE-BACK (THE NEW FIX) ---
    # If the deep analysis disagrees with the cached dashboard score, sync them.
    current_score = analysis.stats.overall_match_score
    
    if (app.match_score != current_score) or (not app.cached_badges):
        print(f"[Forensics] Syncing Application Cache for {app.id} (New Score: {current_score})...")
        
        try:
            # Instantiate service with existing inferer (lightweight) to access badge logic
            inferer = getattr(request.app.state, "inferer", None)
            service = ScoringService(registry, inferer)
            
            # Update fields
            app.match_score = current_score
            app.cached_badges = service._generate_badges(analysis.stats, job)
            
            # Update Grade (Simple logic mirroring ScoringService)
            if current_score >= 85: app.match_grade = "A"
            elif current_score >= 65: app.match_grade = "B"
            elif current_score >= 40: app.match_grade = "C"
            else: app.match_grade = "D"
            
            # Save to DB
            registry.update_application(user.id, app.id, app)
            
        except Exception as e:
            print(f"[Forensics] Warning: Failed to update cache: {e}")
    # --------------------------------------

    analysis.application_id = app_id
    return analysis


# -----------------------------------------------------------------------------
# 3. REJECT MATCH (Interactive Triage)
# -----------------------------------------------------------------------------
@router.post("/applications/{app_id}/mappings/{feature_id}/reject")
def reject_match(
    app_id: str, 
    feature_id: str, 
    request: Request,
    user: User = Depends(get_current_user)
):
    registry: Registry = request.app.state.registry
    
    # A. Fetch Context
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    if not mapping: raise HTTPException(404, "Mapping not found")

    # B. Find Pair
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair:
        raise HTTPException(404, "Mapping pair not found")

    # C. OPTIMIZE (Modify in Memory)
    # This sets status='rejected' and clears the score/annotation so it doesn't count
    MappingOptimizer.reject_current_match(pair)

    # D. SAVE (Persist changes to Mapping)
    registry.save_mapping(user.id, mapping)

    # E. RE-CALCULATE (Get new Forensic Analysis)
    # We calculate this immediately so we can update the Job Cache
    job = registry.get_job(app.job_id, user.id)
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    
    # F. UPDATE JOB CACHE (Spotlight Persistence)
    # This logic matches your _update_job_cache_from_mapping helper
    stats = new_analysis.stats
    job.match_score = stats.overall_match_score
    
    # Update Grade
    if stats.overall_match_score >= 85: job.match_grade = "A"
    elif stats.overall_match_score >= 65: job.match_grade = "B"
    elif stats.overall_match_score >= 40: job.match_grade = "C"
    else: job.match_grade = "D"

    # Update Badges
    badges = []
    if stats.critical_gaps_count > 0: badges.append("Missing Critical Skills")
    if stats.overall_match_score > 90: badges.append("Top Match")
    if stats.coverage_pct > 80 and stats.overall_match_score < 60: badges.append("Broad but Weak")
    
    job.cached_badges = badges

    # Save Job Updates (So the dashboard updates instantly)
    registry.update_job(user.id, job.id, job)

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
def create_manual_match(
    app_id: str, 
    feature_id: str, 
    payload: ManualMatchRequest, 
    request: Request,
    user: User = Depends(get_current_user)
):
    """
    User forces a match for a specific requirement.
    This OVERRIDES any existing AI match or rejection.
    """
    registry: Registry = request.app.state.registry
    
    # 1. Setup Context
    app = registry.get_application(app_id, user.id)
    if not app: raise HTTPException(404, "Application not found")
    
    mapping = registry.get_mapping(app.mapping_id, user.id)
    job = registry.get_job(app.job_id, user.id)
    
    # 2. Get or Create the Pair
    pair = next((p for p in mapping.pairs if p.feature_id == feature_id), None)
    if not pair:
        raise HTTPException(404, "Mapping pair container not found")

    # 3. Construct the "Manual" Candidate
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
        score=1.0, 
        lineage=manual_lineage
    )

    # 4. Update the Pair State
    if not pair.meta:
        pair.meta = MatchingMeta(best_match=manual_candidate, summary_note="")
    
    pair.meta.best_match = manual_candidate
    pair.strength = 1.0 
    pair.context_item_id = payload.cv_item_id
    pair.context_item_type = payload.cv_item_type
    pair.context_item_text = payload.cv_item_name or "Manual Match"
    
    pair.meta.summary_note = f"Manual Match: \"{payload.evidence_text[:50]}...\""
    
    # 5. Save & Recalculate
    registry.save_mapping(user.id, mapping)
    
    calculator = ForensicCalculator()
    new_analysis = calculator.calculate(job, mapping)
    new_analysis.application_id = app_id
    
    return new_analysis