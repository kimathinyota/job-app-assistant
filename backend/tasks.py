# backend/tasks.py
import asyncio
import logging
import json
from backend.core.llm_manager import LLMManager
from backend.core.inferer import CVParser, JobParser
from backend.core.registry import Registry

# --- GLOBAL WORKER STATE ---
# This variable lives in the Worker Process's memory.
# It persists between jobs, so the model stays loaded.
worker_llm = None

MODEL_PATH = "backend/core/llama3_job_cpu_8b.gguf"


def initialize_worker():
    """
    Called once when the worker process starts to load the model.
    """
    global worker_llm
    if worker_llm is None:
        print("🔌 WORKER: Initializing LLM Manager...")
        worker_llm = LLMManager()
        # Adjust path as needed for your local setup
        worker_llm.load_local_models(
            model_path=MODEL_PATH, 
            max_instances=1, # 1 model per worker process is usually best
            machine_type="mac"
        )
        print("✅ WORKER: Model Loaded.")

# ... imports ...

# ---------------------------------------------------------
# TASK 1: PARSE CV
# ---------------------------------------------------------
def task_import_cv(user_id: str, cv_id: str, text_data: str,  cv_name: str):
    """
    Parses a CV and updates the existing placeholder in the database.
    """
    # 1. Ensure Model is Ready
    if worker_llm is None:
        initialize_worker()

    # 2. Setup Dependencies
    registry = Registry()
    parser = CVParser(worker_llm)

    print(f"📄 START: Parsing content for CV ID '{cv_id}'...")

    try:
        # 3. Run Async Parsing
        # Note: We pass a temp name, but we will override the ID shortly
        parsed_cv = asyncio.run(
            parser.parse_cv(text_data, user_id, cv_name=cv_name)
        )

        # 4. MERGE/OVERWRITE LOGIC
        # We take the parsed object, assign it the ORIGINAL ID, and save it.
        # This effectively replaces the placeholder with the real data.
        parsed_cv.id = cv_id  # <--- CRITICAL: Adopt the placeholder's ID
        
        # Restore the original name if the parser didn't find a better one, 
        # or just trust the user's input name from the placeholder if you wish.
        # Here we keep the parsed structure but ensure flags are reset.
        parsed_cv.is_importing = False
        parsed_cv.import_task_id = None
        
        # 5. Save (Overwrite) to DB
        registry._update("cvs", parsed_cv,  user_id)
        print(f"✅ DONE: Updated CV {cv_id} with parsed data.")
        
        return {"id": cv_id, "status": "success"}

    except Exception as e:
        # On failure, mark the specific CV as failed so the user sees it
        try:
            error_cv = registry.get_cv(cv_id, user_id)
            if error_cv:
                error_cv.summary = f"Import Failed: {str(e)}"
                error_cv.is_importing = False
                error_cv.import_task_id = None
                # Keep is_importing=True or add an is_failed flag if you prefer
                # For now, let's leave it so the user can delete it
                registry._update("cvs", error_cv,  user_id)
        except:
            pass
            
        logging.error(f"CV Parse Failed: {e}")
        raise e
# ---------------------------------------------------------
# TASK 2: PARSE JOB (For your Browser Extension)
# ---------------------------------------------------------
def task_parse_job(text_data: str):
    """
    Parses a raw Job Description text into structured JSON.
    Returns: A Dict containing the structured job data.
    """
    # 1. Ensure Model is Ready
    if worker_llm is None:
        initialize_worker()

    # 2. Setup Dependencies
    parser = JobParser(worker_llm)

    print(f"💼 START: Parsing Job Description...")

    try:
        # 3. Run Async Parsing
        # fast_parse returns a Dictionary (JSON)
        result_json = asyncio.run(
            parser.fast_parse(text_data)
        )
        
        print(f"✅ DONE: Parsed Job ({len(result_json.get('features', []))} features)")
        
        # We return the JSON directly so the API can send it back to the extension
        return result_json

    except Exception as e:
        logging.error(f"Job Parse Failed: {e}")
        raise e