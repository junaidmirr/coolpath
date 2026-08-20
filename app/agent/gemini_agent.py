import os
import json
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", message=".*Automatic function calling.*")
warnings.filterwarnings("ignore", message=".*AFC.*")

GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.5-flash"
]


# Pydantic schemas for structured outputs
class ParsedIntent(BaseModel):
    activity: str = Field(default="walking", description="Activity type: walking, running, biking, driving")
    pace: str = Field(default="normal", description="Pace intensity: slow, normal, fast")
    origin_query: Optional[str] = Field(default=None, description="Origin location query if specified by user")
    destination_query: Optional[str] = Field(default=None, description="Destination location query if specified by user")
    deadline_minutes: Optional[int] = Field(default=30, description="Estimated deadline minutes from now")
    thermal_sensitivity: float = Field(default=0.5, description="Heat sensitivity rating from 0.0 (heat tolerant) to 1.0 (extreme sensitivity)")
    special_profile_tags: List[str] = Field(default_factory=list, description="Extracted profile tags e.g. dog_walking, asthma, child, heat_stroke_prone, shade_priority")
    summary: str = Field(default="Heat-aware mission request", description="Concise 1-sentence summary of user intent")

class GeminiBriefing(BaseModel):
    headline: str = Field(description="Actionable, punchy recommendation headline")
    narrative: str = Field(description="Personalized contextual explanation of the recommended route and thermal conditions")
    health_alert: str = Field(description="Safety & medical advice tailored to user's profile and heat exposure")
    timing_advice: str = Field(description="Advice on departure timing e.g. depart immediately or wait for microclimate transition")

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key or not api_key.strip():
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key.strip())
    except Exception as e:
        logger.warning(f"Google GenAI SDK init error: {e}")
        return None

def parse_user_intent_with_gemini(user_prompt: str) -> dict:
    """
    Agentic Intent Parser: Uses Gemini 2.0/1.5 Flash structured output to translate
    raw natural language queries into parameter inputs for our deterministic routing engine.
    """
    client = get_gemini_client()
    if client:
        from google.genai import types
        prompt = f"""
        You are the Intent Orchestrator for CoolPath, an urban heat-aware routing engine.
        Analyze the following user prompt and extract routing parameters and medical/profile constraints.
        
        User Prompt: "{user_prompt}"
        
        Rules:
        - Map activity strictly to one of: "walking", "running", "biking", "driving".
        - Map pace strictly to one of: "slow", "normal", "fast".
        - Extract thermal_sensitivity (0.0 to 1.0) based on user's mentioned health, pets, or comfort needs.
        - If walking a pet/dog, set activity="walking", thermal_sensitivity>=0.8, and add "dog_walking" to special_profile_tags.
        """
        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ParsedIntent,
                        temperature=0.1
                    ),
                )
                data = json.loads(response.text)
                return data
            except Exception as e:
                logger.warning(f"Gemini Intent Parsing with model {model_name} failed: {e}")
                continue

    # High-Intelligence Rule-Based Fallback
    text = user_prompt.lower()
    activity = "walking"
    if any(k in text for k in ["run", "jog", "sprint"]):
        activity = "running"
    elif any(k in text for k in ["bike", "cycle", "biking", "ride"]):
        activity = "biking"
    elif any(k in text for k in ["drive", "car", "taxi"]):
        activity = "driving"

    pace = "normal"
    if any(k in text for k in ["slow", "relax", "gentle", "leisurely", "easy"]):
        pace = "slow"
    elif any(k in text for k in ["fast", "quick", "hurry", "speed"]):
        pace = "fast"

    tags = []
    sens = 0.5
    if any(k in text for k in ["dog", "puppy", "pet", "paws"]):
        tags.append("dog_walking")
        tags.append("pavement_heat_sensitivity")
        sens = 0.9
    if any(k in text for k in ["asthma", "breath", "dizzy", "heart"]):
        tags.append("respiratory_sensitivity")
        sens = 0.9
    if any(k in text for k in ["child", "kid", "baby", "stroller"]):
        tags.append("child_care")
        sens = 0.8

    return ParsedIntent(
        activity=activity,
        pace=pace,
        thermal_sensitivity=sens,
        special_profile_tags=tags,
        summary=f"Parsed '{user_prompt[:40]}…' for {activity} ({pace} pace)"
    ).model_dump()

def generate_gemini_briefing(mission_facts: dict) -> dict:
    """
    Agentic Narrative & Safety Agent: Takes deterministic route options, FortyGuard heat metrics,
    and profile tags, and synthesizes a personalized safety briefing.
    """
    client = get_gemini_client()
    if client:
        from google.genai import types
        prompt = f"""
        You are the Medical & Contextual Persona Brain of CoolPath.
        Synthesize a hyper-personalized, human-like safety briefing based on these routing facts:
        {json.dumps(mission_facts, indent=2)}
        """
        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=GeminiBriefing,
                        temperature=0.3
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                logger.warning(f"Gemini Briefing generation with model {model_name} failed: {e}")
                continue

    # High-Intelligence Rule-Based Briefing Synthesizer
    decision = mission_facts.get("decision", "GO")
    activity = mission_facts.get("activity", "walking")
    tags = mission_facts.get("special_profile_tags", [])
    reduction = mission_facts.get("thermal_reduction_percent", 0.0)
    best_route = mission_facts.get("best_route", {})
    avg_temp = best_route.get("avg_temp_c", 32.5)

    if reduction > 10:
        headline = f"Avoid Asphalt Corridors; Save {reduction}% Heat Exposure via Side Streets"
    elif reduction > 0:
        headline = f"Cooler {activity.capitalize()} Corridor Selected — {reduction}% Heat Reduction"
    elif "dog_walking" in tags:
        headline = "Protect Paw Pads: Shaded Concrete Corridor Recommended"
    else:
        headline = f"Direct {activity.capitalize()} Route is Optimal — Low Thermal Strain"

    if reduction > 0:
        narrative = (
            f"CoolPath's spatial R-Tree thermal engine analyzed microclimate heatmaps along your trip. "
            f"The recommended path keeps average surface temperatures at ~{avg_temp}°C, reducing thermal strain by {reduction}% vs direct asphalt."
        )
    else:
        narrative = (
            f"CoolPath's spatial R-Tree thermal engine analyzed microclimate heatmaps along your trip. "
            f"The direct path maintains an optimal low surface temperature (~{avg_temp}°C) without needing long detours."
        )

    if "dog_walking" in tags:
        narrative += " Pavement in direct sunlight can reach 50°C+; this route maximizes tree canopy cover."


    health_alert = "Hydrate well and seek shade whenever available during peak midday solar irradiance."
    if "dog_walking" in tags or avg_temp > 33.0:
        health_alert = "⚠️ Caution: High asphalt surface heat detected. Check pavement temperature with your hand before letting pets walk."
    elif activity == "running":
        health_alert = "🏃 Hyperthermia Risk: High metabolic heat buildup expected during running. Keep pace steady."

    timing = "Departure recommended immediately for optimal shade coverage."
    if mission_facts.get("wait_minutes", 0) > 0:
        timing = f"⏰ Delay departure by {mission_facts['wait_minutes']} minutes to allow urban solar irradiance to decrease."

    return GeminiBriefing(
        headline=headline,
        narrative=narrative,
        health_alert=health_alert,
        timing_advice=timing
    ).model_dump()
