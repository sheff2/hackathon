import os
import httpx
from google.adk.agents import Agent

MAPS_KEY = "AIzaSyCkB45eQTx6VTU98khj9YcPSJAazXJmuqE"  # or set via env

def get_routes(origin: str, destination: str, count: int = 10) -> dict:
    """
    Fetch up to 10 walking routes between origin and destination
    using Google Maps Routes API.
    """
    if not MAPS_KEY or MAPS_KEY in {"apikey", "YOUR_REAL_KEY_HERE"}:
        return {"status": "error", "error_message": "Missing/placeholder MAPS_KEY"}

    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    body = {
        "origin": {"address": origin},         # <- fixed
        "destination": {"address": destination},# <- fixed
        "travelMode": "WALK",
        "computeAlternativeRoutes": True,
        "polylineQuality": "OVERVIEW"
    }
    headers = {
        "X-Goog-Api-Key": MAPS_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline",
    }

    try:
        # HTTP/2 can help with some Google endpoints
        with httpx.Client(http2=True, timeout=20.0) as client:
            r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        routes = []
        for i, rt in enumerate(data.get("routes", [])[:count]):
            # durations are like "123s"
            dur = rt.get("duration", "0s")
            dur_sec = int(dur[:-1]) if dur.endswith("s") and dur[:-1].isdigit() else 0
            routes.append({
                "id": f"route_{i}",
                "duration_sec": dur_sec,
                "distance_m": rt.get("distanceMeters", 0),
                "encoded_polyline": rt.get("polyline", {}).get("encodedPolyline", ""),
            })
        return {"status": "success", "routes": routes}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error_message": f"{e.response.status_code} {e.response.text}"}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

root_agent = Agent(
    name="routing_agent",
    model="gemini-2.0-flash",
    description="Agent to fetch up to 10 walking routes from A to B using Google Maps.",
    instruction="Call get_routes with an origin and destination.",
    tools=[get_routes],
)

if __name__ == "__main__":
    print(get_routes("New York, NY", "Brooklyn, NY", count=1))
