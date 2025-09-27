from __future__ import annotations
from typing import Dict, Any, List
from google.adk.agents import Agent
from .tools import shortlist_routes, build_display_url

def pick_routes(routes: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    """
    Tool: take a list of {id, encoded_polyline, duration_sec, distance_m (opt), risk}
    and return top 2–3 picks + bullets for the user.
    """
    return shortlist_routes(routes, limit)

def make_display_url(route: Dict[str, Any], base_url: str = "http://localhost:8000/mapOutput.html") -> Dict[str, str]:
    """
    Tool: return {"url": "..."} that opens mapOutput.html with #poly=&dist=&dur=
    """
    return {"url": build_display_url(route, base=base_url)}

route_picker = Agent(
    name="route_picker",
    model="gemini-2.0-flash",  # or "gemini-2.0-flash" if you prefer
    description="Shortlists 2–3 routes (short, safer, alternative) from duration+risk, then produces a display link.",
    instruction=(
        "You will receive JSON containing an array 'routes', each with fields:\n"
        "id, encoded_polyline, duration_sec, distance_m (optional), risk.\n"
        "1) Call pick_routes(routes) to get your top 2–3.\n"
        "2) Show routes and ask the user to choose by id, indicate the shortest route by time, safest, and your suggusted one.\n"
        "3) When the user chooses, call make_display_url(route=<that route>), "
        "and return ONLY the URL.\n"
        "Do not print raw polylines in chat."
    ),
    tools=[pick_routes, make_display_url],
)