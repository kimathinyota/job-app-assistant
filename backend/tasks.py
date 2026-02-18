# backend/tasks.py
import asyncio
import logging
import json
from typing import Optional, List, Dict, Any
from redis import Redis
from rq import Queue

# --- IMPORTS ---
from backend.core.llm_manager import LLMManager
from backend.core.registry import Registry
from backend.core.inferer import CVParser, JobParser, MappingInferer
from backend.core.services.scoring import ScoringService
from backend.core.forensics import ForensicCalculator 
from backend.core.models import ForensicAnalysis 

# --- GLOBAL WORKER STATE ---
parsing_llm: Optional[LLMManager] = None         # Heavy (6GB)
inference_bot: Optional[MappingInferer] = None   # Light (~300MB)

# --- REDIS CONNECTION ---
redis_client = Redis(host='localhost', port=6379)

def _publish_inference_update(user_id: str, type: str, payload: dict):
    """
    Helper to publish FULL DATA payloads to Redis/WebSockets.
    """
    message = {
        "user_id": user_id,
        "type": type,
        "payload": payload
    }
    redis_client.publish("job_updates", json.dumps(message))


# --- INITIALIZERS ---
def initialize_parsing_worker():
    global parsing_llm
    if parsing_llm is None:
        print("🐢 PARSING WORKER: Loading Llama 3...")
        parsing_llm = LLMManager()
        parsing_llm.load_local_models(
            model_path="backend/core/llama3_job_cpu_8b.gguf", 
            max_instances=1, machine_type="mac"
        )

def initialize_inference_worker():
    global inference_bot
    if inference_bot is None:
        print("🐇 INFERENCE WORKER: Loading spaCy + MiniLM...")
        inference_bot = MappingInferer()
        inference_bot.load_models()
        print("✅ INFERENCE WORKER: Ready.")

# =========================================================================
#  QUEUE: 'q_parsing' (HEAVY WORKER)
# =========================================================================
# ... (task_parse_job and task_import_cv remain the same) ...
def task_parse_job(text_data: str):
    if parsing_llm is None: initialize_parsing_worker()
    parser = JobParser(parsing_llm)
    return asyncio.run(parser.fast_parse(text_data))

def task_import_cv(user_id: str, cv_id: str, text_data: str, cv_name: str):
    if parsing_llm is None: initialize_parsing_worker()
    registry = Registry()
    try:
        parser = CVParser(parsing_llm)
        parsed_cv = asyncio.run(parser.parse_cv(text_data, user_id, cv_name=cv_name))
        parsed_cv.id = cv_id
        parsed_cv.is_importing = False
        registry._update("cvs", parsed_cv, user_id)
        return {"status": "success", "id": cv_id}
    except Exception as e:
        logging.error(f"Import failed: {e}")
        raise e

# =========================================================================
#  QUEUE: 'q_inference' (LIGHT WORKER)
# =========================================================================

# --- 1. JOB SCORING (Library View) ---
def task_score_job(user_id: str, job_id: str, cv_id: str):
    if inference_bot is None: initialize_inference_worker()
    
    print(f"🚀 TASK: Score Job {job_id}")
    try:
        registry = Registry()
        service = ScoringService(registry, inferer=inference_bot)
        
        # 1. Update Database
        service.score_job(user_id, job_id, cv_id)
        
        # 2. Fetch Result to broadcast
        updated_job = registry.get_job(job_id, user_id)
        
        # 3. NOTIFY with FULL DATA (Score, Badges, Grade)
        _publish_inference_update(user_id, "JOB_SCORED", {
            "job_id": job_id,
            "score": updated_job.match_score,
            "grade": updated_job.match_grade,
            "badges": updated_job.cached_badges # <--- FULL BADGES
        })
        return {"status": "success", "job_id": job_id}

    except Exception as e:
        _publish_inference_update(user_id, "JOB_SCORE_FAILED", {"job_id": job_id, "error": str(e)})
        raise e


# --- 2. APPLICATION SCORING (Dashboard View) ---
def task_score_application(user_id: str, app_id: str):
    if inference_bot is None: initialize_inference_worker()
    
    print(f"🚀 TASK: Score Application {app_id}")
    try:
        registry = Registry()
        service = ScoringService(registry, inferer=inference_bot)
        
        # 1. Update Database
        service.score_application(user_id, app_id)
        
        # 2. Fetch Result
        updated_app = registry.get_application(app_id, user_id)

        # 3. NOTIFY with FULL DATA
        _publish_inference_update(user_id, "APP_SCORED", {
            "app_id": app_id,
            "score": updated_app.match_score,
            "grade": updated_app.match_grade,
            "badges": updated_app.cached_badges # <--- FULL BADGES
        })
        return {"status": "success", "app_id": app_id}
        
    except Exception as e:
        _publish_inference_update(user_id, "APP_SCORE_FAILED", {"app_id": app_id, "error": str(e)})
        raise e


# --- 3. GENERATE ROLE CASE (Forensics View) ---
def task_generate_role_case(user_id: str, job_id: str, cv_id: str, mode: str = "balanced_default"):
    """
    Generates and broadcasts the FULL ForensicAnalysis object.
    """
    if inference_bot is None: initialize_inference_worker()

    print(f"🚀 TASK: Generate RoleCase ({mode}) for Job {job_id}")
    try:
        registry = Registry()
        
        job = registry.get_job(job_id, user_id)
        cv = registry.get_cv(cv_id, user_id)
        
        service = ScoringService(registry, inferer=inference_bot)
        mapping = service._get_or_create_smart_mapping(user_id, job, cv)
        
        calculator = ForensicCalculator()
        analysis = calculator.calculate(job, mapping)
        
        # PUSH FULL FORENSIC ANALYSIS OBJECT
        # Using .dict() or .model_dump() ensures the whole JSON structure is sent
        payload = analysis.model_dump(mode='json') if hasattr(analysis, 'model_dump') else analysis.dict()

        _publish_inference_update(
            user_id, 
            "ROLE_CASE_GENERATED", 
            payload # <--- THE FULL OBJECT
        )
        
        return {"status": "success", "job_id": job_id}
    
    except Exception as e:
        _publish_inference_update(user_id, "ROLE_CASE_FAILED", {"job_id": job_id, "error": str(e)})
        raise e


# --- 4. MAPPING SUGGESTIONS (Manual Editor) ---
def task_infer_mapping_suggestions(user_id: str, mapping_id: str, mode: str = "balanced_default"):
    """
    Infers suggestions and broadcasts the FULL LIST of pairs.
    """
    if inference_bot is None: initialize_inference_worker()
    
    print(f"🚀 TASK: Infer Pairs for Mapping {mapping_id}")
    try:
        registry = Registry()
        
        mapping = registry.get_mapping(mapping_id, user_id)
        job = registry.get_job(mapping.job_id, user_id)
        cv = registry.get_cv(mapping.base_cv_id, user_id)
        
        # Run inference
        suggestions = inference_bot.infer_mappings(job, cv, min_score=0.2) 
        
        # Save to DB
        mapping.pairs = suggestions
        registry.save_mapping(user_id, mapping)
        
        # NOTIFY WITH FULL DATA
        # Convert Pydantic list to JSON-compatible list
        pairs_data = [p.model_dump(mode='json') for p in suggestions]

        _publish_inference_update(user_id, "MAPPING_SUGGESTIONS_READY", {
            "mapping_id": mapping_id,
            "suggestions": pairs_data # <--- FULL LIST OF PAIRS
        })
        
        return {"status": "success", "count": len(suggestions)}

    except Exception as e:
        _publish_inference_update(user_id, "MAPPING_FAILED", {"mapping_id": mapping_id, "error": str(e)})
        raise e


# --- 5. [NEW] PREVIEW MATCH (Job Card Dropdown) ---
def task_preview_job_match(user_id: str, job_id: str, cv_id: str):
    """
    Calculates score/badges/grade WITHOUT updating the Job Cache.
    """
    if inference_bot is None: initialize_inference_worker()
    
    print(f"🚀 TASK: Preview Match Job {job_id} <> CV {cv_id}")
    try:
        registry = Registry()
        service = ScoringService(registry, inferer=inference_bot)
        
        job = registry.get_job(job_id, user_id)
        cv = registry.get_cv(cv_id, user_id)

        # 1. Run Logic (Saves Mapping, but NOT Job)
        mapping = service._get_or_create_smart_mapping(user_id, job, cv)
        
        # 2. Calculate Stats
        analysis = service.forensics.calculate(job, mapping)
        
        # 3. NOTIFY with FULL PREVIEW DATA
        _publish_inference_update(user_id, "MATCH_PREVIEW_GENERATED", {
            "job_id": job_id,
            "cv_id": cv_id,
            "score": analysis.stats.overall_match_score,
            "grade": analysis.suggested_grade,
            "badges": analysis.suggested_badges # <--- FULL BADGES
        })
        
        return {"status": "success"}

    except Exception as e:
        _publish_inference_update(user_id, "PREVIEW_FAILED", {"job_id": job_id, "error": str(e)})
        raise e

# ... (task_dispatch_cv_updates remains the same) ...
def task_dispatch_cv_updates(user_id: str, cv_id: str):
    # (Same as previous code block)
    if inference_bot is None: initialize_inference_worker()
    
    print(f"🔄 BULK: Starting update fan-out for CV {cv_id}...")
    registry = Registry()
    
    redis_conn = Redis(host='localhost', port=6379)
    q_bg = Queue('q_background', connection=redis_conn)

    # A. Update Linked Applications
    all_apps = registry.all_applications(user_id)
    affected_apps = [app for app in all_apps if app.base_cv_id == cv_id]
    
    for app in affected_apps:
        q_bg.enqueue(task_score_application, args=(user_id, app.id), job_timeout='30s')

    # B. Update Job Board (if Primary)
    user = registry.get_user(user_id)
    if user and user.primary_cv_id == cv_id:
        jobs = registry.all_jobs(user_id)
        for job in jobs:
            q_bg.enqueue(task_score_job, args=(user_id, job.id, cv_id), job_timeout='30s')