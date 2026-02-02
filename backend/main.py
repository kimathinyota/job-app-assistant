# backend/main.py
from contextlib import asynccontextmanager
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the *classes*, not instances
from backend.core.registry import Registry
from backend.core.inferer import MappingInferer

# Import your routes
from backend.routes import (
    cv, job, mapping, application, coverletter, 
    prompt, interview, workitem, goal
)
from backend.core.inferer import JobDescriptionParser # Import the new class
# Setup logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
# Ensure this path is correct relative to where you run `python run.py`
# MODEL_PATH = "backend/core/llama3_job_cpu_FINAL.gguf" 
MODEL_PATH = "backend/core/llama3_job_cpu_8b.gguf"
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Loads the ML model once on startup.
    """
    # 1. Startup: Load the Model
    try:
        if os.path.exists(MODEL_PATH):
            log.info("🚀 Startup: Initializing Llama Model...")
            app.state.job_parser = JobDescriptionParser(model_path=MODEL_PATH)
        else:
            log.warning(f"⚠️ Model file '{MODEL_PATH}' not found. Job parsing will be disabled.")
            app.state.job_parser = None
        
        # --- STEP 2: LOAD NLP MODELS FIRST ---
        log.info("Creating Inferer singleton...")
        app.state.inferer = MappingInferer()
        
        log.info("Loading NLP models...")
        app.state.inferer.load_models()
        log.info("NLP models loaded successfully.")
        
        # --- STEP 3: CREATE REGISTRY AND DB LOCK SECOND ---
        log.info("Creating Registry singleton...")
        app.state.registry = Registry("./backend/data/db.json")
        log.info("Registry created successfully.")
        
        log.info("Startup event: All singletons created and models loaded.")
    except Exception as e:
        log.error(f"❌ Failed to load model: {e}")
        app.state.job_parser = None
        
    yield  # The application runs here
    
    # 2. Shutdown: Clean up resources (if needed)
    log.info("🛑 Shutdown: Cleaning up resources...")
    if hasattr(app.state, 'job_parser'):
        del app.state.job_parser



app = FastAPI(title="Job Application Assistant API", lifespan=lifespan)

# @app.on_event("startup")
# async def startup_event():
#     """
#     This code runs *inside* the worker process.
#     We will load the C-heavy NLP models FIRST, then create the
#     Python-based threading.Lock.
#     """
#     log.info("--- Application Startup Event ---")
#     try:
#         # --- STEP 1: LOAD NLP MODELS FIRST ---
#         log.info("Creating Inferer singleton...")
#         app.state.inferer = MappingInferer()
        
#         log.info("Loading NLP models...")
#         app.state.inferer.load_models()
#         log.info("NLP models loaded successfully.")
        
#         # --- STEP 2: CREATE REGISTRY AND DB LOCK SECOND ---
#         log.info("Creating Registry singleton...")
#         app.state.registry = Registry("./backend/data/db.json")
#         log.info("Registry created successfully.")
        
#         log.info("Startup event: All singletons created and models loaded.")
#     except Exception as e:
#         log.critical(f"Startup event: Failed: {e}", exc_info=True)
#     log.info("--- Application Startup Complete ---")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://192.168.1.161:5173", "chrome-extension://fcodmekeoifocfcljbhbgecmbacffhlm", "chrome-extension://afamfigjdefgbepikbhinhchmdldoani"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cv.router, prefix="/api/cv", tags=["CV"])
app.include_router(job.router, prefix="/api/job", tags=["Job"])
app.include_router(application.router, prefix="/api/application", tags=["Application"])
app.include_router(mapping.router, prefix="/api/mapping", tags=["Mapping"])
app.include_router(prompt.router, prefix="/api/prompt", tags=["AI Prompt"])
app.include_router(workitem.router, prefix="/api/workitem", tags=["Work Item"])
app.include_router(goal.router, prefix="/api/goal", tags=["Goal"])
app.include_router(coverletter.router, prefix="/api/coverletter", tags=["Cover Letter"])
app.include_router(interview.router, prefix="/api/interview", tags=["Interview"])


@app.get("/api")
async def root():
    return {"message": "Welcome to the Job Application Assistant API!"}