# main.py
from fastapi import FastAPI, Depends
from app.database.database import supabase  # import your initialized Supabase client
from fastapi.middleware.cors import CORSMiddleware
from app.auth import router as auth_router
from app.interview import router as interview_router
from app.database.dependencies import get_current_user
from app.campaign import router as campaign_router
from app.template import router as template_router
from fastapi.staticfiles import StaticFiles

app = FastAPI()
# Allow specific origins (your frontend URLs)
origins = [
    "http://localhost:3000",   # React local dev
    "http://127.0.0.1:3000",   # Alternate localhost
    "https://your-frontend.com"  # Production domain
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,           # allowed domains
    allow_credentials=True,
    allow_methods=["*"],             # GET, POST, PUT, DELETE, OPTIONS
    allow_headers=["*"],             # Accept, Authorization, Content-Type, etc.
)
app.include_router(auth_router)
app.include_router(interview_router)
app.include_router(campaign_router)
app.include_router(template_router)
UPLOAD_DIR = "uploads"
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

