#!/usr/bin/env python3
"""
routing_agent.py

Requirements:
  pip install httpx "httpx[http2]" google-adk-agents  # http2 extra pulls in 'h2'

Env:
  export MAPS_KEY=YOUR_ROUTES_API_KEY  # enable Routes API + billing in the GCP project

Run:
  python routing_agent.py
"""

import os
import json
import math
from typing import List, Tuple, Dict, Any
import httpx

from google.adk.agents import Agent



# ---- Config -----------------------------------------------------------------

MAPS_KEY = "AIzaSyCkB45eQTx6VTU98khj9YcPSJAazXJmuqE"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
FIELD_MASK = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"


# ---- Polyline helpers --------------------------------------------------------

def decode_polyline(enc: str) -> List[Tuple[float, float]]:
    """Decode a Google encoded polyline into [(lat, lng), ...]."""
    coords: List[Tuple[float, float]] = []
    idx = lat = lng = 0
    while idx < len(enc):
        shift = result = 0
        while True:
            b = ord(enc[idx]) - 63
            idx += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = result = 0
        while True:
            b = ord(enc[idx]) - 63
            idx += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coords.append((lat * 1e-5, lng * 1e-5))
    return coords


def directions_link_from_polyline(
    encoded_polyline: str,
    travelmode: str = "walking",
    max_waypoints: int = 20
) -> str:
    """
    Build a Google Maps Directions URL that follows the given polyline by
    sampling interior points as waypoints (web link).
    """
    from urllib.parse import urlencode

    pts = decode_polyline(encoded_polyline)
    if len(pts) < 2:
        raise ValueError("Polyline must have at least 2 points")

    origin = f"{pts[0][0]},{pts[0][1]}"
    destination = f"{pts[-1][0]},{pts[-1][1]}"

    interior = pts[1:-1]
    waypoints = ""
    if interior:
        step = max(1, len(interior) // max_waypoints)
        sampled = interior[::step][:max_waypoints]
        waypoints = "|".join(f"{lat},{lng}" for (lat, lng) in sampled)

    base = "https://www.google.com/maps/dir/?api=1"
    q = {
        "origin": origin,
        "destination": destination,
        "travelmode": travelmode,
    }
    if waypoints:
        q["waypoints"] = waypoints
    query = urlencode(q, safe="|,")
    return f"{base}&{query}"


# ---- Routes API tool ---------------------------------------------------------

def get_routes(origin: str, destination: str, count: int = 10) -> Dict[str, Any]:
    """
    Fetch up to `count` walking routes between origin and destination
    using Google Maps Routes API.

    Returns:
        {"status":"success","routes":[{id,duration_sec,distance_m,encoded_polyline,maps_link},...]}
        or {"status":"error","error_message": "..."}
    """
    if not MAPS_KEY:
        return {"status": "error", "error_message": "Missing MAPS_KEY env var"}

    body = {
        # NOTE: For computeRoutes, address goes directly under origin/destination
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "WALK",
        "computeAlternativeRoutes": True,
        "polylineQuality": "OVERVIEW",
    }
    headers = {
        "X-Goog-Api-Key": MAPS_KEY,
        "X-Goog-FieldMask": FIELD_MASK,
    }

    try:
        # http2=True requires 'h2' package (install via httpx[http2])
        with httpx.Client(http2=True, timeout=20.0) as client:
            resp = client.post(ROUTES_URL, json=body, headers=headers)
        resp.raise_for_status()
        payload = resp.json()

        routes_out: List[Dict[str, Any]] = []
        for i, rt in enumerate(payload.get("routes", [])[: max(1, min(10, count))]):
            dur_s = 0
            d = rt.get("duration", "0s")
            # duration is an RFC3339 duration like "123s" (Maps currently returns seconds)
            if d.endswith("s"):
                try:
                    dur_s = int(d[:-1])
                except ValueError:
                    dur_s = 0

            enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
            link = directions_link_from_polyline(enc, travelmode="walking") if enc else None

            routes_out.append(
                {
                    "id": f"route_{i}",
                    "duration_sec": dur_s,
                    "distance_m": int(rt.get("distanceMeters", 0)),
                    "encoded_polyline": enc,
                    "maps_link": link,
                }
            )

        return {"status": "success", "routes": routes_out}

    # Bubble up server’s error text for easier debugging
    except httpx.HTTPStatusError as e:
        text = e.response.text
        return {
            "status": "error",
            "error_message": f"{e.response.status_code} {text}".strip(),
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}





# ---- CLI / demo --------------------------------------------------------------

def _pretty_distance(m: int) -> str:
    if m >= 1000:
        return f"{m/1000:.1f} km"
    return f"{m} m"


root_agent = Agent(
        name="routing_agent",
        model="gemini-2.0-flash",
        description="Fetch up to 10 walking routes from A to B using Google Maps, and provide a Google Maps link for each.",
        instruction="Call get_routes with an origin and destination.",
        tools=[get_routes],
)

