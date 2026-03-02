"""
FastAPI main app entry point.
"""
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from backend.db.database import init_db
from backend.api.meetings import router as meetings_router
from backend.api.speakers import router as speakers_router

app = FastAPI(
    title="Meeting Minutes AI",
    description="Tu dong tao bien ban cuoc hop bang AI",
    version="1.0.0"
)

# Initialize DB on startup
@app.on_event("startup")
def startup():
    init_db()

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Include routers
app.include_router(meetings_router)
app.include_router(speakers_router)

# Serve frontend
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health():
    return {"status": "ok", "service": "Meeting Minutes AI"}
