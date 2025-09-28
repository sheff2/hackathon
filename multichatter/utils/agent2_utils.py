# utils/agent2_utils.py
#
# Crime-aware route scoring utilities and tools for the Crime Agent.
# Core addition: rank_supplied_routes(routes, origin, destination) which
# accepts Agent 1's routes and scores them locally (no extra API fetch).

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional: only used by get_ranked_routes (kept for parity; not used by chain)
try:
    import googlemaps  # type: ignore
except ImportError:
    googlemaps = None  # type: ignore

# ---- Data / scoring params ----
DATA_PATH = Path(__file__).parent / "data" / "mock_data.json"
MAX_ROUTE_CACHE = 10
SAMPLE_INTERVAL_M = 120.0
INCIDENT_SEARCH_RADIUS_M = 1000.0
DEFAULT_RISK_FLOOR = 0.01

SEVERITY_WEIGHTS = {
    "homicide": 1.0, "murder": 1.0, "shooting": 0.9,
    "aggravated assault": 0.85, "robbery": 0.8, "carjacking": 0.75,
    "burglary": 0.6, "motor vehicle theft": 0.55, "theft": 0.5,
    "drug": 0.45, "vandalism": 0.35, "other": 0.3,
}

CATEGORY_WEIGHTS = {"person": 0.85, "society": 0.55, "property": 0.5}


@dataclass(frozen=True)
class CrimeIncident:
    lat: float
    lon: float
    offense: str
    timestamp: datetime
    severity: float
    raw: Dict[str, object]


_INCIDENT_CACHE: Optional[List[CrimeIncident]] = None
_ROUTE_CACHE: "OrderedDict[str, Dict[str, object]]" = OrderedDict()


def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value:
        return datetime.utcnow()
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError:
        return datetime.utcnow()


def _severity_weight(incident: Dict[str, object]) -> float:
    offense = (incident.get("incident_offense") or "").lower()
    detail = (incident.get("incident_offense_detail_description") or "").lower()
    category = (incident.get("incident_offense_crime_against") or "").lower()
    score = DEFAULT_RISK_FLOOR
    for label, weight in SEVERITY_WEIGHTS.items():
        if label in offense or label in detail:
            score = max(score, weight)
    if category in CATEGORY_WEIGHTS:
        score = max(score, CATEGORY_WEIGHTS[category])
    return min(max(score, DEFAULT_RISK_FLOOR), 1.0)


def load_incidents() -> List[CrimeIncident]:
    global _INCIDENT_CACHE
    if _INCIDENT_CACHE is not None:
        return _INCIDENT_CACHE
    with DATA_PATH.open() as fp:
        payload = json.load(fp)
    incidents: List[CrimeIncident] = []
    for raw in payload.get("incidents", []):
        lat = raw.get("incident_latitude")
        lon = raw.get("incident_longitude")
        if lat is None or lon is None:
            continue
        incidents.append(
            CrimeIncident(
                lat=float(lat),
                lon=float(lon),
                offense=str(raw.get("incident_offense", "")),
                timestamp=_parse_timestamp(str(raw.get("incident_date", ""))),
                severity=_severity_weight(raw),
                raw=raw,
            )
        )
    _INCIDENT_CACHE = incidents
    return incidents


def get_crime_locations(limit: Optional[int] = None) -> List[Dict[str, object]]:
    """Optional helper for quick peeks at incidents."""
    results: List[Dict[str, object]] = []
    for incident in load_incidents():
        results.append(
            {
                "street": incident.raw.get("incident_address", ""),
                "city": incident.raw.get("city_key", ""),
                "lat": incident.lat,
                "lon": incident.lon,
                "offense": incident.offense,
                "timestamp": incident.timestamp.isoformat() + "Z",
            }
        )
        if limit is not None and len(results) >= limit:
            break
    return results


# ---------------- Geometry helpers ----------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def decode_polyline(polyline: str) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0
    while index < len(polyline):
        shift = 0
        result = 0
        while True:
            b = ord(polyline[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        delta = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += delta

        shift = 0
        result = 0
        while True:
            b = ord(polyline[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        delta = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += delta

        points.append((lat / 1e5, lon / 1e5))
    return points


def _sample_points(points: Sequence[Tuple[float, float]], interval: float) -> List[Tuple[float, float]]:
    if not points:
        return []
    if len(points) == 1:
        return [(points[0][0], points[0][1])]
    samples: List[Tuple[float, float]] = [points[0]]
    distance_since_sample = 0.0
    for start, end in zip(points[:-1], points[1:]):
        seg_len = haversine_distance(start[0], start[1], end[0], end[1])
        if seg_len == 0:
            continue
        remaining = seg_len
        start_point = start
        while distance_since_sample + remaining >= interval:
            ratio = (interval - distance_since_sample) / remaining
            lat = start_point[0] + (end[0] - start_point[0]) * ratio
            lon = start_point[1] + (end[1] - start_point[1]) * ratio
            samples.append((lat, lon))
            remaining -= (interval - distance_since_sample)
            distance_since_sample = 0.0
            start_point = (lat, lon)
        distance_since_sample += remaining
    if samples[-1] != points[-1]:
        samples.append(points[-1])
    return samples


def _find_incidents_within_radius(lat: float, lon: float, radius_m: float) -> List[Tuple[CrimeIncident, float]]:
    hits: List[Tuple[CrimeIncident, float]] = []
    for incident in load_incidents():
        distance = haversine_distance(lat, lon, incident.lat, incident.lon)
        if distance <= radius_m:
            hits.append((incident, distance))
    return hits


def _time_decay(timestamp: datetime, reference: Optional[datetime] = None) -> float:
    reference = reference or datetime.utcnow()
    delta = reference - timestamp
    hours = max(delta.total_seconds() / 3600.0, 0.0)
    half_life_hours = 168
    decay = math.exp(-hours / half_life_hours)
    return max(decay, 0.05)


def _distance_decay(distance_m: float) -> float:
    if distance_m <= 0:
        return 1.0
    # Use the new larger radius for decay calculation
    return max(0.01, 1.0 - (distance_m / INCIDENT_SEARCH_RADIUS_M))


def estimate_lighting(samples: Sequence[Tuple[float, float]]) -> Dict[str, object]:
    return {
        "score": None,
        "sampled_points": len(samples),
        "notes": "Lighting analysis pending Street View integration.",
        "multiplier": 1.0,
    }


def _summarize_route_risk(incidents: Iterable[Dict[str, object]], lighting: Dict[str, object]) -> str:
    incidents_list = list(incidents)
    if not incidents_list:
        return "No recent incidents detected within 120 meters of this route."
    incidents_list.sort(key=lambda item: item["score_contribution"], reverse=True)
    top = incidents_list[:3]
    parts = []
    for item in top:
        offense = item.get("offense", "incident")
        distance_m = round(item.get("distance_m", 0))
        parts.append(f"{offense} about {distance_m} meters away")
    note = "; ".join(parts)
    if lighting.get("score") is not None:
        note += f"; lighting score {lighting['score']}"
    return note


def score_polyline(polyline: str) -> Dict[str, object]:
    points = decode_polyline(polyline)
    samples = _sample_points(points, SAMPLE_INTERVAL_M)
    
    # Use a tuple of (lat, lon, offense) as the key instead of the incident object
    incident_rollup: Dict[Tuple[float, float, str], Dict[str, object]] = {}
    total_risk = 0.0
    
    for lat, lon in samples:
        nearby = _find_incidents_within_radius(lat, lon, INCIDENT_SEARCH_RADIUS_M)
        for incident, distance_m in nearby:
            contribution = incident.severity * _time_decay(incident.timestamp) * _distance_decay(distance_m)
            total_risk += contribution
            
            # Create a hashable key from incident data
            incident_key = (incident.lat, incident.lon, incident.offense)
            
            data = incident_rollup.setdefault(
                incident_key,
                {
                    "offense": incident.offense,
                    "lat": incident.lat,
                    "lon": incident.lon,
                    "timestamp": incident.timestamp.isoformat() + "Z",
                    "distance_m": distance_m,
                    "score_contribution": 0.0,
                },
            )
            data["distance_m"] = min(data["distance_m"], distance_m)
            data["score_contribution"] += contribution

    lighting = estimate_lighting(samples)
    risk_score = total_risk * float(lighting.get("multiplier", 1.0))
    
    # Add route-specific factors for more variation
    route_length_km = len(points) * 0.01 if points else 0  # Rough approximation
    length_penalty = 1.0 + (route_length_km * 0.02)  # 2% increase per km
    
    # Density factor - more incidents nearby = higher risk  
    incident_density = len(incident_rollup) / max(route_length_km, 0.1)
    density_multiplier = 1.0 + (incident_density * 0.1)
    
    # Apply factors
    risk_score = risk_score * length_penalty * density_multiplier
    risk_score = max(risk_score, DEFAULT_RISK_FLOOR)
    
    return {
        "risk_score": risk_score,
        "incidents": list(incident_rollup.values()),
        "lighting": lighting,
        "samples": len(samples),
        "summary": _summarize_route_risk(incident_rollup.values(), lighting),
        "route_factors": {  # Add debug info
            "length_penalty": length_penalty,
            "density_multiplier": density_multiplier,
            "incidents_found": len(incident_rollup)
        }
    }

def _store_route(route_id: str, route_payload: Dict[str, object]) -> None:
    _ROUTE_CACHE[route_id] = route_payload
    while len(_ROUTE_CACHE) > MAX_ROUTE_CACHE:
        _ROUTE_CACHE.popitem(last=False)


# --------- The tool your pipeline will use ---------
def rank_supplied_routes(
    routes: List[Dict[str, object]],
    origin: Optional[str] = None,
    destination: Optional[str] = None,
) -> Dict[str, object]:
    """
    Accepts Agent 1's list of routes:
      [{"id": "...", "encoded_polyline": "...", "distance_m": int, "duration_sec": int, "maps_link": "..."}]
    Scores them locally and returns a ranked list (lowest risk first).
    """
    ranked: List[Dict[str, object]] = []
    for r in routes:
        rid = str(r.get("id", f"route_{len(ranked)}"))
        enc = r.get("encoded_polyline")
        if not isinstance(enc, str) or not enc:
            continue
        metrics = score_polyline(enc)
        # inside rank_supplied_routes(...)
        # utils/agent2_utils.py (inside rank_supplied_routes)
        result = {
            "route_id": rid,
            # NOTE: do NOT include "polyline" here
            "distance_meters": r.get("distance_m"),
            "duration_seconds": r.get("duration_sec"),
            "risk_score": metrics["risk_score"],
            "risk_summary": metrics["summary"],
            "incidents_nearby": metrics["incidents"],
            "lighting": metrics["lighting"],
            "maps_link": r.get("maps_link"),
        }
        ranked.append(result)

        # keep the polyline ONLY in cache for get_route_details()
        _store_route(rid, {**result, "polyline": enc})



    if not ranked:
        return {
            "origin": origin,
            "destination": destination,
            "routes": [],
            "best_route_id": None,
            "error": "No valid supplied routes to rank.",
        }

    ranked.sort(key=lambda item: (item["risk_score"], item.get("duration_seconds") or float("inf")))
    best = ranked[0]
    baseline = best["risk_score"] or DEFAULT_RISK_FLOOR
    for item in ranked:
        item["relative_risk"] = item["risk_score"] / baseline if baseline else 1.0

    return {
        "origin": origin,
        "destination": destination,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "routes": ranked,
        "best_route_id": best["route_id"],
    }


# --------- Optional (kept for dev/testing parity) ---------
def get_ranked_routes(origin: str, destination: str) -> Dict[str, object]:
    """Fetch 3–5 routes via googlemaps library and rank them (not used in chain)."""
    if googlemaps is None:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": "googlemaps library not installed; cannot fetch routes."}
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": "GOOGLE_MAPS_API_KEY not set; configure Maps access."}
    client = googlemaps.Client(key=api_key)
    try:
        routes = client.directions(origin, destination, mode="walking", alternatives=True)
    except Exception as exc:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": f"Directions API failed: {exc}"}

    ranked: List[Dict[str, object]] = []
    for idx, route in enumerate(routes):
        enc = route.get("overview_polyline", {}).get("points")
        if not enc:
            continue
        leg = route.get("legs", [{}])[0]
        metrics = score_polyline(enc)
        rid = f"route_{idx}_{abs(hash(enc)) % 10_000_000}"
        result = {
            "route_id": rid,
            "polyline": enc,
            "distance_meters": leg.get("distance", {}).get("value"),
            "duration_seconds": leg.get("duration", {}).get("value"),
            "risk_score": metrics["risk_score"],
            "risk_summary": metrics["summary"],
            "incidents_nearby": metrics["incidents"],
            "lighting": metrics["lighting"],
        }
        ranked.append(result)
        _store_route(rid, result)

    if not ranked:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": "No valid walking routes returned by Directions API."}

    ranked.sort(key=lambda item: (item["risk_score"], item.get("duration_seconds", float("inf"))))
    best = ranked[0]
    base = best["risk_score"] or DEFAULT_RISK_FLOOR
    for r in ranked:
        r["relative_risk"] = r["risk_score"] / base if base else 1.0

    return {
        "origin": origin,
        "destination": destination,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "routes": ranked,
        "best_route_id": best["route_id"],
    }


def get_route_details(route_id: str) -> Dict[str, object]:
    route = _ROUTE_CACHE.get(route_id)
    if route is None:
        available = list(_ROUTE_CACHE.keys())
        raise ValueError(f"Unknown route_id '{route_id}'. Cached routes: {available}")
    return route

def debug_agent_context(context_data: str = "") -> Dict[str, object]:
    """
    Debug tool to see exactly what Agent2 receives from Agent1
    """
    print("🔍 DEBUGGING AGENT CONTEXT")
    print(f"📥 Type: {type(context_data)}")
    print(f"📥 Length: {len(str(context_data))}")
    print(f"📥 First 500 chars: {str(context_data)[:500]}")
    print(f"📥 Last 500 chars: {str(context_data)[-500:]}")
    print("=" * 80)
    
    # Try different parsing approaches
    parsing_attempts = {
        "direct_json": None,
        "string_contains_json": None,
        "function_call_result": None,
        "raw_text": str(context_data)
    }
    
    # Attempt 1: Direct JSON parse
    try:
        import json
        if isinstance(context_data, str) and context_data.strip().startswith('{'):
            parsing_attempts["direct_json"] = json.loads(context_data)
            print("✅ SUCCESS: Direct JSON parsing worked")
    except Exception as e:
        print(f"❌ Direct JSON failed: {e}")
    
    # Attempt 2: Find JSON in text
    try:
        import re
        json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', str(context_data))
        if json_matches:
            parsing_attempts["string_contains_json"] = json_matches
            print(f"✅ Found {len(json_matches)} JSON-like patterns")
    except Exception as e:
        print(f"❌ JSON pattern search failed: {e}")
    
    # Attempt 3: Look for function call results
    try:
        if "plan_routes" in str(context_data) or "routes" in str(context_data):
            parsing_attempts["function_call_result"] = "Found routing-related content"
            print("✅ Found routing-related content")
    except Exception as e:
        print(f"❌ Function call search failed: {e}")
    
    return {
        "debug_info": parsing_attempts,
        "raw_context": str(context_data),
        "analysis": "Check console output for detailed debugging info"
    }


def parse_and_rank_routes() -> Dict[str, object]:
    """
    Find and parse routing context from ADK SequentialAgent
    """
    print("🔍 DEBUGGING: Looking for routing context...")
    
    import inspect
    import sys
    
    # Try multiple ways to find the routing data
    context_sources = []
    
    # Method 1: Check all stack frames
    frame = inspect.currentframe()
    frame_count = 0
    while frame and frame_count < 10:
        frame_locals = frame.f_locals
        frame_globals = frame.f_globals
        
        # Look for anything with route data
        for name, value in {**frame_locals, **frame_globals}.items():
            if isinstance(value, (str, dict)):
                str_val = str(value)
                if len(str_val) > 100 and any(keyword in str_val.lower() for keyword in ['routes', 'encoded_polyline', 'duration_sec']):
                    context_sources.append({
                        "source": f"frame_{frame_count}_{name}",
                        "data": str_val,
                        "type": type(value).__name__
                    })
        
        frame = frame.f_back
        frame_count += 1
    
    # Method 2: Check sys modules for ADK context
    for module_name, module in sys.modules.items():
        if 'adk' in module_name.lower() or 'agent' in module_name.lower():
            try:
                for attr_name in dir(module):
                    if not attr_name.startswith('_'):
                        attr_value = getattr(module, attr_name, None)
                        if attr_value and 'route' in str(attr_value).lower():
                            context_sources.append({
                                "source": f"module_{module_name}_{attr_name}",
                                "data": str(attr_value)[:500],
                                "type": type(attr_value).__name__
                            })
            except:
                continue
    
    # Method 3: Check globals for agent context
    import builtins
    if hasattr(builtins, '__dict__'):
        for name, value in builtins.__dict__.items():
            if 'route' in str(value).lower() and len(str(value)) > 100:
                context_sources.append({
                    "source": f"builtins_{name}",
                    "data": str(value)[:500],
                    "type": type(value).__name__
                })
    
    print(f"🔍 Found {len(context_sources)} potential context sources")
    
    # Try to parse the most promising source
    for source in context_sources:
        try:
            data = source["data"]
            
            # Try JSON parsing
            import json
            import re
            
            # Look for JSON patterns
            json_match = re.search(r'\{.*"routes".*\}', data, re.DOTALL)
            if json_match:
                routing_data = json.loads(json_match.group())
                print(f"✅ Parsed routing data from {source['source']}")
                
                if routing_data.get("status") == "success" and routing_data.get("routes"):
                    # Found valid routing data! Now rank it
                    return rank_supplied_routes(
                        routes=routing_data["routes"],
                        origin=routing_data.get("origin"),
                        destination=routing_data.get("destination")
                    )
        except Exception as e:
            print(f"❌ Failed to parse {source['source']}: {e}")
            continue
    
    # If we get here, return debug info
    return {
        "error": "No routing context found",
        "debug_sources_found": len(context_sources),
        "sources": [{"source": s["source"], "preview": s["data"][:100]} for s in context_sources[:3]]
    }
    """
    Parse routing context from the agent's execution environment
    """
    # In ADK, try to access the previous agent's context
    # This might be available through globals, context, or other mechanisms
    
    import inspect
    
    # Try to get the context from the calling frame
    frame = inspect.currentframe()
    if frame and frame.f_back:
        local_vars = frame.f_back.f_locals
        global_vars = frame.f_back.f_globals
        
        # Look for routing data in various possible locations
        context_candidates = []
        
        # Check if there's routing data in locals/globals
        for var_name, var_value in {**local_vars, **global_vars}.items():
            if isinstance(var_value, (str, dict)):
                str_value = str(var_value)
                if "routes" in str_value and "status" in str_value:
                    context_candidates.append(str_value)
        
        if context_candidates:
            routing_context = context_candidates[0]  # Use the first match
        else:
            return {"error": "No routing context found in execution environment"}
    else:
        return {"error": "Could not access execution context"}
    
    # Your existing parsing logic here...
    print(f"🔍 Found routing context: {len(routing_context)} chars")
    
    # Rest of your parsing code...

    
    """
    Bulletproof function to parse routing agent output and rank routes by safety.
    Handles multiple input formats and provides detailed error reporting.
    """
    print(f"🔍 Crime agent received context of length: {len(routing_context)}")
    
    if not routing_context or len(routing_context) < 10:
        return {
            "error": "No routing context received from previous agent",
            "debug_info": f"Received: '{routing_context[:100]}...'"
        }
    
    # Parse the routing data
    routing_data = None
    
    # Method 1: Direct JSON parse
    try:
        import json
        if routing_context.strip().startswith('{'):
            routing_data = json.loads(routing_context.strip())
            print("✅ Parsed as direct JSON")
    except json.JSONDecodeError:
        pass
    
    # Method 2: Find JSON in text
    if not routing_data:
        try:
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', routing_context, re.DOTALL)
            if json_match:
                routing_data = json.loads(json_match.group())
                print("✅ Extracted JSON from text")
        except json.JSONDecodeError:
            pass
    
    # Method 3: Look for specific patterns
    if not routing_data:
        try:
            # Look for function call results in various formats
            patterns = [
                r'"status":\s*"success".*?"routes":\s*\[(.*?)\]',
                r'plan_routes.*?(\{.*?\})',
                r'status.*?success.*?routes.*?\[(.*?)\]'
            ]
            for pattern in patterns:
                match = re.search(pattern, routing_context, re.DOTALL | re.IGNORECASE)
                if match:
                    # Try to reconstruct the JSON
                    break
        except:
            pass
    
    if not routing_data:
        return {
            "error": "Could not parse routing data from context",
            "debug_info": f"Context preview: '{routing_context[:200]}...'"
        }
    
    # Validate the routing data structure
    if not isinstance(routing_data, dict):
        return {"error": "Routing data is not a valid dictionary"}
    
    if routing_data.get("status") != "success":
        error_msg = routing_data.get("error_message", "Unknown routing error")
        return {"error": f"Routing failed: {error_msg}"}
    
    routes = routing_data.get("routes", [])
    if not routes:
        return {"error": "No routes found in routing data"}
    
    if len(routes) < 1:
        return {"error": "Insufficient routes for analysis"}
    
    print(f"✅ Found {len(routes)} routes to analyze")
    
    # Call the ranking function
    try:
        ranking_result = rank_supplied_routes(
            routes=routes,
            origin=routing_data.get("origin"),
            destination=routing_data.get("destination")
        )
        
        if ranking_result.get("error"):
            return {"error": f"Ranking failed: {ranking_result['error']}"}
        
        # Format for the agent
        formatted_routes = []
        for i, route in enumerate(ranking_result["routes"], 1):
            duration_min = round((route.get("duration_seconds") or 0) / 60, 1)
            distance_km = round((route.get("distance_meters") or 0) / 1000, 2)
            
            formatted_routes.append({
                "rank": i,
                "route_id": route.get("route_id"),
                "duration_minutes": duration_min,
                "distance_km": distance_km,
                "safety_score": round(route.get("risk_score", 0), 4),
                "risk_summary": route.get("risk_summary", "No summary available"),
                "maps_link": route.get("maps_link", ""),
                "incidents_count": len(route.get("incidents_nearby", []))
            })
        
        return {
            "success": True,
            "origin": ranking_result.get("origin"),
            "destination": ranking_result.get("destination"), 
            "total_routes": len(formatted_routes),
            "routes": formatted_routes,
            "safest_route_id": ranking_result.get("best_route_id")
        }
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}