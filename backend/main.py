# backend/main.py
import os
import logging

# REMOVE the os.environ lines from here. They are now in run.py.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import your routes
from backend.routes import (
    cv, job, mapping, application, coverletter, 
    prompt, interview, workitem, goal
)
# Import your singletons (this is fine now)
from backend.core.dependencies import registry, inferer 

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Job Application Assistant API")

@app.on_event("startup")
async def startup_event():
    """
    Load NLP models *after* the fork, in the worker process.
    """
    log.info("--- Application Startup Event ---")
    try:
        # The env vars are set, so this load is now safe.
        inferer.load_models()
        log.info("Startup event: NLP models loaded successfully.")
    except Exception as e:
        log.critical(f"Startup event: Failed to load NLP models: {e}", exc_info=True)
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