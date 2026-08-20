from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional, List

from app.models.mission import Mission, Coordinate
from app.decision.engine import optimize_mission
from app.config import DEMO_MODE
from app.services.thermal_provider import SyntheticThermalProvider, FortyGuardThermalProvider
from app.agent.gemini_agent import parse_user_intent_with_gemini, generate_gemini_briefing
import httpx
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class IntentRequest(BaseModel):
    prompt: str

class MissionRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate
    departure_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    planning_mode: str = "instant"  # "instant" or "scheduled"
    deadline_minutes: Optional[int] = 60
    activity: str = "walking"
    pace: str = "normal"
    prompt: Optional[str] = None
    special_tags: Optional[List[str]] = None

@router.post("/parse-intent")
def parse_intent(req: IntentRequest):
    """
    Agentic Intent Parser powered by Gemini API.
    Extracts structured activity, pace, deadline, and medical profile parameters from natural language prompts.
    """
    if not req.prompt or len(req.prompt.strip()) < 3:
        raise HTTPException(status_code=400, detail={"error": True, "message": "Prompt too short"})
    
    try:
        intent = parse_user_intent_with_gemini(req.prompt)
        return {"status": "ok", "intent": intent}
    except Exception as e:
        logger.error(f"Intent parsing failed: {e}")
        raise HTTPException(status_code=500, detail={"error": True, "message": str(e)})

@router.post("/mission")
async def plan_mission(req: MissionRequest):
    """
    Evaluates the mission using NetworkX Pareto routing and FortyGuard STRtree thermal data,
    and synthesizes a Gemini Agentic Persona Briefing.
    """
    try:
        activity = req.activity
        pace = req.pace
        tags = req.special_tags or []

        # If a natural language prompt is supplied directly in the mission request
        if req.prompt:
            intent = parse_user_intent_with_gemini(req.prompt)
            if intent.get("activity"):
                activity = intent["activity"]
            if intent.get("pace"):
                pace = intent["pace"]
            if intent.get("special_profile_tags"):
                tags.extend(intent["special_profile_tags"])

        now = datetime.now()
        dep_time = req.departure_time or now
        dl_minutes = req.deadline_minutes or 60
        deadline_dt = req.deadline or (dep_time + timedelta(minutes=dl_minutes))

        mission = Mission(
            origin=req.origin,
            destination=req.destination,
            departure_time=dep_time,
            deadline=deadline_dt,
            activity=activity,
            pace=pace,
            planning_mode=req.planning_mode,
            deadline_minutes=dl_minutes
        )

        provider = SyntheticThermalProvider() if DEMO_MODE else FortyGuardThermalProvider()
        result = await optimize_mission(mission, provider)


        # Synthesize Agentic Gemini Briefing
        best_route = (result.get("route_options") or [{}])[0]
        mission_facts = {
            "decision": result.get("decision"),
            "wait_minutes": result.get("wait_minutes"),
            "activity": activity,
            "pace": pace,
            "thermal_reduction_percent": result.get("thermal_reduction_percent", 0.0),
            "special_profile_tags": list(set(tags)),
            "best_route": best_route,
            "total_routes_found": len(result.get("route_options", []))
        }

        briefing = generate_gemini_briefing(mission_facts)
        result["gemini_briefing"] = briefing
        result["parsed_profile_tags"] = list(set(tags))

        return result

    except ValueError as e:
        logger.warning(f"Validation error in plan_mission: {e}")
        raise HTTPException(status_code=422, detail={"error": True, "message": str(e), "type": "validation_error"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Unhandled error in plan_mission: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": True, "message": str(e), "type": type(e).__name__}
        )

@router.get("/geocode")
async def geocode_location(q: str):
    """Converts a place name to coordinates using Nominatim."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(status_code=400, detail={"error": True, "message": "Query too short"})

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={
                    "q": q,
                    "format": "json",
                    "limit": 5,
                    "addressdetails": 1,
                },
                headers={"User-Agent": "CoolPath-HeatNav/1.0"},
                timeout=8.0
            )
            resp.raise_for_status()
            data = resp.json()

        results = []
        for item in data:
            results.append({
                "display_name": item.get("display_name", ""),
                "lat": float(item["lat"]),
                "lng": float(item["lon"]),
                "type": item.get("type", ""),
                "importance": float(item.get("importance", 0))
            })

        return {"results": results}

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail={"error": True, "message": "Geocoding service timed out"})
    except Exception as e:
        logger.error(f"Geocode error: {e}")
        raise HTTPException(status_code=500, detail={"error": True, "message": str(e), "type": "geocode_error"})
