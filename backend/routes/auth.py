import os
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, HTTPException, Response
from starlette.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from jose import jwt
from dotenv import load_dotenv

# Import your internal modules
from backend.core.models import User
from backend.core.registry import Registry

# 1. Load Environment Variables
# This points to backend/.env assuming this file is in backend/routes/
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# 2. Configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("FATAL: SECRET_KEY is missing from .env")

# Google Credentials are automatically picked up by Authlib from 
# GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 Days

router = APIRouter()
oauth = OAuth()

# 3. Initialize Google OAuth
oauth.register(
    name='google',
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
            # Using User.create() (or standard constructor depending on your BaseEntity)
            user = User(
                id=user_info['sub'], # Use Google ID as ID, or generate a UUID
                email=user_info['email'],
                oauth_provider="google",
                provider_id=user_info['sub'],
                full_name=user_info.get('name', 'User'),
                avatar_url=user_info.get('picture'),
                tier="free"
            )
            # Ensure your Registry has this method
            registry.create_user(user)

        # --- Session Logic ---
        access_token = create_session_token(user.provider_id, user.tier)

        # --- Cookie Logic ---
        # Redirect to Frontend Dashboard
        response = RedirectResponse(url="http://localhost:5173/dashboard")
        
        response.set_cookie(
            key="session_token",
            value=access_token,
            httponly=True,   # Critical: JavaScript cannot read this (prevents XSS)
            max_age=60 * 60 * 24 * 7,
            samesite="lax",  # CSRF Protection
            secure=False     # Set to True in Production (HTTPS)
        )
        return response

    except Exception as e:
        print(f"Auth Error: {e}")
        # Redirect to login with error param
        return RedirectResponse(url="http://localhost:5173/login?error=auth_failed")

@router.post("/logout")
def logout(response: Response):
    """Clears the session cookie."""
    response.delete_cookie("session_token")
    return {"message": "Logged out"}

@router.get("/me")
def get_current_user_info(request: Request):
    """
    Frontend calls this to check if logged in.
    Decodes the JWT cookie and fetches user from DB.
    """
    token = request.cookies.get("session_token")
    if not token:
        raise HTTPException(401, "Not authenticated")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        registry = request.app.state.registry
        # Ensure Registry has get_user_by_provider_id or get_user
        user = registry.get_user_by_provider_id(user_id)
        
        if not user:
            raise HTTPException(401, "User not found")
            
        return user
    except Exception as e:
        print(f"Token verification failed: {e}")
        raise HTTPException(401, "Invalid session")