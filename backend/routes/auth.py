import os
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException, Response
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from jose import jwt
from dotenv import load_dotenv

# Import your internal modules
from backend.core.models import User
from backend.core.registry import Registry

# Setup explicit logging for auth debugging
log = logging.getLogger("auth_router")

# 1. Load Environment Variables
# Calculates path to: .../backend/.env
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# 2. Configuration & Validation
SECRET_KEY = os.getenv("SECRET_KEY")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

# --- DEBUGGING CHECK ---
if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    log.error(f"❌ Failed to load Google Credentials.")
    log.error(f"   Looking for .env at: {env_path}")
    log.error(f"   Current Working Directory: {os.getcwd()}")
    raise ValueError("FATAL: GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET is missing from .env")
# -----------------------

if not SECRET_KEY:
    raise ValueError("FATAL: SECRET_KEY is missing from .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 Days

router = APIRouter()
oauth = OAuth()

# 3. Initialize Google OAuth (Explicitly passing credentials)
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,         # <--- Pass Explicitly
    client_secret=GOOGLE_CLIENT_SECRET, # <--- Pass Explicitly
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

def create_session_token(user_id: str, tier: str) -> str:
    """Creates a signed JWT containing user ID and Tier."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": user_id, "tier": tier, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.get("/login/google")
async def login_google(request: Request):
    """Redirects user to Google Login page."""
    # Ensure the callback URL matches what is in Google Cloud Console exactly
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@router.get("/callback/google", name="auth_google_callback")
async def auth_google_callback(request: Request):
    """Handles the return from Google after login."""
    try:
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get('userinfo')
        
        if not user_info:
            raise HTTPException(400, "Failed to retrieve user info from Google")

        # --- Database Logic ---
        registry: Registry = request.app.state.registry
        
        # Check if user exists using the Google 'sub' (Subject ID)
        existing_user = registry.get_user_by_provider_id(user_info['sub'])
        
        if existing_user:
            user = existing_user
            # Update avatar/name if they changed on Google
            if user.avatar_url != user_info.get('picture') or user.full_name != user_info.get('name'):
                user.avatar_url = user_info.get('picture')
                user.full_name = user_info.get('name', user.full_name)
                registry.update_user(user)
        else:
            # Create new Free Tier user
            user = User(
                id=user_info['sub'], 
                email=user_info['email'],
                oauth_provider="google",
                provider_id=user_info['sub'],
                full_name=user_info.get('name', 'User'),
                avatar_url=user_info.get('picture'),
                tier="free"
            )
            registry.create_user(user)

        # --- Session Logic ---
        access_token = create_session_token(user.provider_id, user.tier)

        # --- Cookie Logic ---
        # Redirect to Frontend Dashboard
        response = RedirectResponse(url="http://localhost:5173/dashboard")
        
        response.set_cookie(
            key="session_token",
            value=access_token,
            httponly=True,   
            max_age=60 * 60 * 24 * 7,
            samesite="lax",  
            secure=False     
        )
        return response

    except Exception as e:
        log.error(f"Auth Error: {e}")
        return RedirectResponse(url="http://localhost:5173/login?error=auth_failed")

@router.post("/logout")
def logout(response: Response):
    """Clears the session cookie."""
    response.delete_cookie("session_token")
    return {"message": "Logged out"}

@router.get("/me")
def get_current_user(request: Request):
    """
    Frontend calls this to check if logged in.
    """
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(401, "Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        registry = request.app.state.registry
        user = registry.get_user_by_provider_id(user_id)
        
        if not user:
            raise HTTPException(401, "User not found")
            
        return user
    except Exception as e:
        log.error(f"Token verification failed: {e}")
        raise HTTPException(401, "Invalid session")