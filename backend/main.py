# backend/main.py
import os
import logging
from fastapi import FastAPI, Depends # <-- Remove Depends, it's not needed here
from fastapi.middleware.cors import CORSMiddleware

# Import the *classes*, not instances
from backend.core.registry import Registry
from backend.core.inferer import MappingInferer

# Import your routes
from backend.routes import (
    cv, job, mapping, application, coverletter, 
    prompt, interview, workitem, goal
)

# REMOVE all imports from backend.core.dependencies
# from backend.core.dependencies import get_db, get_inferer, registry # <--- DELETE THIS LINE

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Job Application Assistant API")

@app.on_event("startup")
async def startup_event():
    """
    This code runs *inside* the worker process, *after* the fork.
    This is the safe place to create locks and load models.
    """
    log.info("--- Application Startup Event ---")
    try:
        # --- UNCOMMENT THESE LINES ---
        log.info("Creating Registry singleton...")
        app.state.registry = Registry("./backend/data/db.json")
        
        log.info("Creating Inferer singleton...")
        app.state.inferer = MappingInferer()
        
        log.info("Loading NLP models...")
        app.state.inferer.load_models()
        
        log.info("Startup event: All singletons created and models loaded.")
        # --- END UNCOMMENT ---
    except Exception as e:
        log.critical(f"Startup event: Failed: {e}", exc_info=True)
    log.info("--- Application Startup Complete ---")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include your routers
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