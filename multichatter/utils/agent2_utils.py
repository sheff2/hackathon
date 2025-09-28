from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging
log = logging.getLogger(__name__)

try:
    import googlemaps  # optional
except ImportError:
    googlemaps = None  # type: ignore

DATA_PATH = Path(__file__).parent / "data" / "mock_data.json"
MAX_ROUTE_CACHE = 12
SAMPLE_INTERVAL_M = 80.0
INCIDENT_SEARCH_RADIUS_M = 500
DEFAULT_RISK_FLOOR = 0.05
MAX_CONTRIB_PER_INCIDENT = 1.5

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

# ---------------- Safe polyline decoder ----------------
class PolylineDecodeError(Exception):
    pass

def decode_polyline_safe(enc: str) -> List[Tuple[float, float]]:
    if not isinstance(enc, str) or not enc:
        return []
    pts: List[Tuple[float, float]] = []
    idx = 0
    lat = 0
    lon = 0
    n = len(enc)
    try:
        while idx < n:
            shift = 0; result = 0
            while True:
                if idx >= n:
                    raise PolylineDecodeError("Truncated at latitude chunk")
                b = ord(enc[idx]) - 63; idx += 1
                result |= (b & 0x1F) << shift; shift += 5
                if b < 0x20: break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += delta

            shift = 0; result = 0
            while True:
                if idx >= n:
                    raise PolylineDecodeError("Truncated at longitude chunk")
                b = ord(enc[idx]) - 63; idx += 1
                result |= (b & 0x1F) << shift; shift += 5
                if b < 0x20: break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            lon += delta

            pts.append((lat / 1e5, lon / 1e5))
    except PolylineDecodeError:
        return pts
    except Exception:
        return []
    return pts

# ---- Data / scoring ----
def _parse_timestamp(value: Optional[str]) -> datetime:
    if not value: return datetime.utcnow()
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
        if label in offense or label in detail: score = max(score, weight)
    if category in CATEGORY_WEIGHTS: score = max(score, CATEGORY_WEIGHTS[category])
    return min(max(score, DEFAULT_RISK_FLOOR), 1.0)

def load_incidents() -> List[CrimeIncident]:
    global _INCIDENT_CACHE
    if _INCIDENT_CACHE is not None: return _INCIDENT_CACHE
    with DATA_PATH.open() as fp:
        payload = json.load(fp)
    incidents: List[CrimeIncident] = []
    for raw in payload.get("incidents", []):
        lat = raw.get("incident_latitude"); lon = raw.get("incident_longitude")
        if lat is None or lon is None: continue
        incidents.append(CrimeIncident(
            lat=float(lat), lon=float(lon),
            offense=str(raw.get("incident_offense", "")),
            timestamp=_parse_timestamp(str(raw.get("incident_date", ""))),
            severity=_severity_weight(raw), raw=raw
        ))
    _INCIDENT_CACHE = incidents
    return incidents

def get_crime_locations(limit: Optional[int] = None) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for incident in load_incidents():
        results.append({
            "street": incident.raw.get("incident_address", ""),
            "city": incident.raw.get("city_key", ""),
            "lat": incident.lat, "lon": incident.lon,
            "offense": incident.offense,
            "timestamp": incident.timestamp.isoformat() + "Z",
        })
        if limit is not None and len(results) >= limit: break
    return results

# ---------------- Geometry helpers ----------------
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1); d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)); return r * c

def _sample_points(points: Sequence[Tuple[float, float]], interval: float) -> List[Tuple[float, float]]:
    if not points: return []
    if len(points) == 1: return [(points[0][0], points[0][1])]
    samples: List[Tuple[float, float]] = [points[0]]; distance_since_sample = 0.0
    for start, end in zip(points[:-1], points[1:]):
        seg_len = haversine_distance(start[0], start[1], end[0], end[1])
        if seg_len == 0: continue
        remaining = seg_len; start_point = start
        while distance_since_sample + remaining >= interval:
            ratio = (interval - distance_since_sample) / remaining
            lat = start_point[0] + (end[0] - start_point[0]) * ratio
            lon = start_point[1] + (end[1] - start_point[1]) * ratio
            samples.append((lat, lon))
            remaining -= (interval - distance_since_sample); distance_since_sample = 0.0
            start_point = (lat, lon)
        distance_since_sample += remaining
    if samples[-1] != points[-1]: samples.append(points[-1])
    return samples

def _find_incidents_within_radius(lat: float, lon: float, radius_m: float) -> List[Tuple[CrimeIncident, float]]:
    hits: List[Tuple[CrimeIncident, float]] = []
    for incident in load_incidents():
        distance = haversine_distance(lat, lon, incident.lat, incident.lon)
        if distance <= radius_m: hits.append((incident, distance))
    return hits

def _time_decay(timestamp: datetime, reference: Optional[datetime] = None) -> float:
    reference = reference or datetime.utcnow()
    hours = max((reference - timestamp).total_seconds() / 3600.0, 0.0)
    half_life_hours = 72.0
    decay = math.exp(-hours / half_life_hours)
    return max(decay, 0.05)

def _distance_decay(distance_m: float) -> float:
    if distance_m <= 0: return 1.0
    return max(0.05, 1.0 - (distance_m / INCIDENT_SEARCH_RADIUS_M))

def estimate_lighting(samples: Sequence[Tuple[float, float]]) -> Dict[str, object]:
    return {"score": None, "sampled_points": len(samples), "notes": "Lighting analysis pending Street View integration.", "multiplier": 1.0}

def _summarize_route_risk(incidents: Iterable[Dict[str, object]], lighting: Dict[str, object]) -> str:
    incidents_list = list(incidents)
    if not incidents_list: return "No recent incidents detected within 120 meters of this route."
    incidents_list.sort(key=lambda item: item["score_contribution"], reverse=True)
    top = incidents_list[:3]; parts = []
    for item in top:
        offense = item.get("offense", "incident"); distance_m = round(item.get("distance_m", 0))
        parts.append(f"{offense} about {distance_m} meters away")
    note = "; ".join(parts)
    if lighting.get("score") is not None: note += f"; lighting score {lighting['score']}"
    return note

# --------- Scoring with hashable keys only ---------
def score_polyline(polyline: str) -> Dict[str, object]:
    points = decode_polyline_safe(polyline)
    if len(points) < 2:
        return {
            "risk_score": DEFAULT_RISK_FLOOR,
            "incidents": [],
            "lighting": {"score": None, "sampled_points": 0, "notes": "Invalid or truncated polyline.", "multiplier": 1.0},
            "samples": 0,
            "summary": "Route geometry invalid/truncated; unable to evaluate incidents.",
        }

    samples = _sample_points(points, SAMPLE_INTERVAL_M)
    total_risk = 0.0

    # Aggregate using a STRING key (no dicts as keys → avoids 'unhashable type: dict')
    by_key: Dict[str, Dict[str, object]] = {}

    for lat, lon in samples:
        nearby = _find_incidents_within_radius(lat, lon, INCIDENT_SEARCH_RADIUS_M)
        for incident, distance_m in nearby:
            contribution = incident.severity * _time_decay(incident.timestamp) * _distance_decay(distance_m)
            contribution = min(contribution, MAX_CONTRIB_PER_INCIDENT)
            total_risk += contribution

            key = f"{round(incident.lat,5)},{round(incident.lon,5)}|{incident.offense.lower()}|{incident.timestamp.date().isoformat()}"
            entry = by_key.get(key)
            if entry is None:
                by_key[key] = {
                    "offense": incident.offense,
                    "lat": float(incident.lat),
                    "lon": float(incident.lon),
                    "timestamp": incident.timestamp.isoformat() + "Z",
                    "distance_m": float(distance_m),
                    "score_contribution": float(contribution),
                }
            else:
                entry["distance_m"] = min(float(entry["distance_m"]), float(distance_m))
                entry["score_contribution"] = float(entry["score_contribution"]) + float(contribution)

    lighting = estimate_lighting(samples)
    risk_score = max(total_risk * float(lighting.get("multiplier", 1.0)), DEFAULT_RISK_FLOOR)
    incidents = list(by_key.values())
    summary = _summarize_route_risk(incidents, lighting)

    return {"risk_score": risk_score, "incidents": incidents, "lighting": lighting, "samples": len(samples), "summary": summary}

def _store_route(route_id: str, route_payload: Dict[str, object]) -> None:
    _ROUTE_CACHE[route_id] = route_payload
    while len(_ROUTE_CACHE) > MAX_ROUTE_CACHE: _ROUTE_CACHE.popitem(last=False)

# NOTE: no default param values (Gemini tool schema requirement)
def rank_supplied_routes(routes: List[Dict[str, object]], origin: str, destination: str) -> Dict[str, object]:
    ranked: List[Dict[str, object]] = []
    for r in routes:
        base_id = str(r.get("id", f"route_{len(ranked)}"))
        enc = r.get("encoded_polyline")
        if not isinstance(enc, str) or not enc:
            continue
        salted = f"{base_id}_{abs(hash(enc)) % 10_000_000}"
        metrics = score_polyline(enc)
        result = {
            "route_id": salted,
            "distance_meters": r.get("distance_m"),
            "duration_seconds": r.get("duration_sec"),
            "risk_score": metrics["risk_score"],
            "risk_summary": metrics["summary"],
            "incidents_nearby": metrics["incidents"],
            "lighting": metrics["lighting"],
            "maps_link": r.get("maps_link"),
        }
        ranked.append(result)
        _store_route(salted, {**result, "polyline": enc})

    if not ranked:
        return {
            "schema_version": "1.0",
            "origin": origin,
            "destination": destination,
            "routes": [],
            "best_route_id": None,
            "error": "No valid supplied routes to rank."
        }

    ranked.sort(key=lambda item: (item["risk_score"], item.get("duration_seconds") or float("inf")))
    best = ranked[0]
    baseline = best["risk_score"] or DEFAULT_RISK_FLOOR
    for item in ranked:
        item["relative_risk"] = item["risk_score"] / baseline if baseline else 1.0

    return {
        "schema_version": "1.0",
        "origin": origin,
        "destination": destination,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "routes": ranked,
        "best_route_id": best["route_id"],
    }

def get_ranked_routes(origin: str, destination: str) -> Dict[str, object]:
    if googlemaps is None:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": "googlemaps not installed; cannot fetch routes."}
    api_key = os.getenv("MAPS_KEY") or os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": "MAPS_KEY not set; configure Maps access."}
    client = googlemaps.Client(key=api_key)
    try:
        routes = client.directions(origin, destination, mode="walking", alternatives=True)
    except Exception as exc:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": f"Directions API failed: {exc}"}
    ranked: List[Dict[str, object]] = []
    for idx, route in enumerate(routes):
        enc = route.get("overview_polyline", {}).get("points")
        if not enc: continue
        leg = route.get("legs", [{}])[0]; from_hash = abs(hash(enc)) % 10_000_000
        rid = f"route_{idx}_{from_hash}"
        metrics = score_polyline(enc)
        result = {"route_id": rid, "polyline": enc, "distance_meters": leg.get("distance", {}).get("value"),
                  "duration_seconds": leg.get("duration", {}).get("value"),
                  "risk_score": metrics["risk_score"], "risk_summary": metrics["summary"],
                  "incidents_nearby": metrics["incidents"], "lighting": metrics["lighting"]}
        ranked.append(result); _store_route(rid, result)
    if not ranked:
        return {"origin": origin, "destination": destination, "routes": [], "best_route_id": None,
                "error": "No valid walking routes returned by Directions API."}
    ranked.sort(key=lambda item: (item["risk_score"], item.get("duration_seconds", float("inf"))))
    best = ranked[0]; base = best["risk_score"] or DEFAULT_RISK_FLOOR
    for r in ranked: r["relative_risk"] = r["risk_score"] / base if base else 1.0
    return {"origin": origin, "destination": destination, "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "routes": ranked, "best_route_id": best["route_id"]}

def get_route_details(route_id: str) -> Dict[str, object]:
    route = _ROUTE_CACHE.get(route_id)
    if route is None:
        available = list(_ROUTE_CACHE.keys())
        raise ValueError(f"Unknown route_id '{route_id}'. Cached routes: {available}")
    return route