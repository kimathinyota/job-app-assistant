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

# ---------------------------------------------------------
# TASK 1: PARSE CV (For your Import Feature)
# ---------------------------------------------------------
def task_import_cv(user_id: str, cv_name: str, text_data: str):
    """
    Parses a CV and saves it to the database.
    Returns: The ID of the new CV.
    """
    # 1. Ensure Model is Ready
    if worker_llm is None:
        initialize_worker()

    # 2. Setup Dependencies
    registry = Registry()
    parser = CVParser(worker_llm)

    print(f"📄 START: Parsing CV '{cv_name}'...")

    try:
        # 3. Run Async Parsing Synchronously (Bridge for RQ)
        # We assume parse_cv returns a CV object
        structured_cv = asyncio.run(
            parser.parse_cv(text_data, user_id, cv_name=cv_name)
        )

        # 4. Save to DB
        registry._insert("cvs", structured_cv)
        print(f"✅ DONE: Saved CV {structured_cv.id}")
        
        return {"id": structured_cv.id, "status": "success"}

    except Exception as e:
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