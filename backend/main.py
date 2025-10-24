from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.registry import Registry
from backend.routes import (
    cv,
    job,
    mapping,
    application,
    coverletter,
    prompt,
    interview,
    workitem,
)


app = FastAPI(title="Job Application Assistant API")

# Initialize global registry with TinyDB
registry = Registry("./backend/data/db.json")

# Allow frontend on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routes
app.include_router(cv.router, prefix="/cv", tags=["CV"])
app.include_router(job.router, prefix="/job", tags=["Job"])
app.include_router(mapping.router, prefix="/mapping", tags=["Mapping"])
app.include_router(application.router, prefix="/application", tags=["Application"])
app.include_router(coverletter.router, prefix="/coverletter", tags=["CoverLetter"])
app.include_router(prompt.router, prefix="/prompt", tags=["Prompt"])
app.include_router(interview.router, prefix="/interview", tags=["Interview"])
app.include_router(workitem.router, prefix="/workitem", tags=["WorkItem"])


@app.get("/")
def root():
    return {"status": "ok", "message": "Job Application Assistant backend running"}
