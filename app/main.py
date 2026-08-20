from fastapi import FastAPI
from app.config import DEMO_MODE
from app.api.mission import router as mission_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CoolPath Mission Planner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "demo_mode": DEMO_MODE
    }

app.include_router(mission_router, prefix="/api")
