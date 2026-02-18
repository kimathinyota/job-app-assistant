# backend/main.py
from contextlib import asynccontextmanager
import os
import logging
import json
import asyncio
from redis import asyncio as aioredis # pip install redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# --- IMPORTS FOR WEB SOCKETS ---
# Ensure you have created backend/core/websocket_manager.py as discussed previously
from backend.core.websocket_manager import manager

# Import classes
from backend.core.registry import Registry
from backend.core.inferer import MappingInferer
from backend.core.llm_manager import LLMManager
from backend.core.inferer import JobParser, CVParser

# Import routes
from backend.routes import (
    cv, job, mapping, application, coverletter, 
    prompt, interview, workitem, goal, forensics, auth
)

# Setup logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
load_dotenv("backend/.env")
SECRET_KEY = os.getenv("SECRET_KEY")

# Ensure this path is correct relative to where you run `python run.py`
MODEL_PATH = "backend/core/llama3_job_cpu_8b.gguf"
MAX_MODEL_INSTANCES = 1

# --- BACKGROUND REDIS LISTENER ---
async def redis_listener():
    """
    Listens to Redis Pub/Sub 'job_updates' channel.
    When a worker publishes a message (Scoring/Parsing complete),
    this listener forwards it to the specific user via WebSocket.
    """
    try:
        # Connect to Redis (Async)
        redis = await aioredis.from_url("redis://localhost:6379")
        pubsub = redis.pubsub()
        await pubsub.subscribe("job_updates") # Channel name must match backend/tasks.py

        log.info("🎧 Redis Listener: Subscribed to 'job_updates'")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    user_id = data.get("user_id")
                    
                    # Forward to the specific user if they are connected
                    if user_id:
                        await manager.send_personal_message(data, user_id)
                except Exception as e:
                    log.error(f"Error processing Redis message: {e}")
                    
    except Exception as e:
        log.error(f"❌ Redis Listener Connection Failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    1. Starts Redis Listener.
    2. Loads Lightweight API Singletons (Registry).
    """
    # --- STARTUP ---
    
    # 1. Start the Redis Listener in the background
    redis_task = asyncio.create_task(redis_listener())

    try:
        # 2. Initialize Logic Classes
        # Note: We do NOT load the Heavy LLM here anymore (moved to workers).
        # We might still load MappingInferer if you need simple synchronous fallbacks,
        # but ideally, that is also fully offloaded now.
        
        # Checking for model just to set state, but we rely on Workers for heavy lifting
        if os.path.exists(MODEL_PATH):
            log.info("Main API: Model file found (Workers will use it).")
            # We keep these as None or lightweight wrappers in the Main API
            app.state.llm_manager = None 
            app.state.job_parser = None
        else:
            log.warning(f"⚠️ Model file '{MODEL_PATH}' not found.")
        
        # --- STEP 2: LOAD NLP MODELS (Optional on API side) ---
        # If you want the API to feel "instant" on boot, you might skip loading this
        # and rely 100% on the worker. For now, we keep it as requested.
        log.info("Creating Inferer singleton...")
        app.state.inferer = MappingInferer()
        
        log.info("Loading lightweight NLP models (Spacy/MiniLM)...")
        app.state.inferer.load_models()
        log.info("NLP models loaded successfully.")
        
        # --- STEP 3: CREATE REGISTRY AND DB LOCK ---
        log.info("Creating Registry singleton...")
        app.state.registry = Registry("./backend/data/db.json")
        log.info("Registry created successfully.")
        
        log.info("🚀 Startup event: API Ready & Listening for Worker updates.")
        
    except Exception as e:
        log.error(f"❌ Startup Error: {e}")
        
    yield  # The application runs here
    
    # --- SHUTDOWN ---
    log.info("🛑 Shutdown: Cancelling Redis Listener...")
    redis_task.cancel()
    try:
        await redis_task
    except asyncio.CancelledError:
        pass
        
    log.info("Shutdown: Cleaning up resources...")


app = FastAPI(title="Job Application Assistant API", lifespan=lifespan)

# 3. Add Session Middleware
if not SECRET_KEY:
    log.warning("⚠️ SECRET_KEY not found in .env, using unsafe default for development.")
    SECRET_KEY = "unsafe_dev_secret"

app.add_middleware(
    SessionMiddleware, 
    secret_key=SECRET_KEY, 
    max_age=3600  # 1 hour session
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://192.168.1.161:5173", "chrome-extension://fcodmekeoifocfcljbhbgecmbacffhlm", "chrome-extension://afamfigjdefgbepikbhinhchmdldoani"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(user_id, websocket)
    try:
        while True:
            # Keep connection alive (wait for messages from client, though we mostly push)
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user_id, websocket)


# --- ROUTES ---
app.include_router(cv.router, prefix="/api/cv", tags=["CV"])
app.include_router(job.router, prefix="/api/job", tags=["Job"])
app.include_router(application.router, prefix="/api/application", tags=["Application"])
app.include_router(mapping.router, prefix="/api/mapping", tags=["Mapping"])
app.include_router(prompt.router, prefix="/api/prompt", tags=["AI Prompt"])
app.include_router(workitem.router, prefix="/api/workitem", tags=["Work Item"])
app.include_router(goal.router, prefix="/api/goal", tags=["Goal"])
app.include_router(coverletter.router, prefix="/api/coverletter", tags=["Cover Letter"])
app.include_router(interview.router, prefix="/api/interview", tags=["Interview"])
app.include_router(forensics.router, prefix="/api/forensics", tags=["Forensics"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])

@app.get("/api")
async def root():
    return {"message": "Welcome to the Job Application Assistant API!"}