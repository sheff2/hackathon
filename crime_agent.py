"""
crime_agent.py
--------------
Safety-aware route scoring agent using Google ADK tools.

Loads crime incidents from data/mock_data.json, exposes helper tools
for retrieving incident snapshots, ranking candidate walking routes,
and returning cached route details for a parent agent.
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from google.adk.agents import Agent

try:  # Optional dependency; required for live route fetching.
    import googlemaps  # type: ignore
except ImportError:  # pragma: no cover - library not installed in some envs
    googlemaps = None  # type: ignore

DATA_PATH = Path(__file__).parent / "data" / "mock_data.json"
MAX_ROUTE_CACHE = 10
SAMPLE_INTERVAL_M = 80.0
INCIDENT_SEARCH_RADIUS_M = 120.0
DEFAULT_RISK_FLOOR = 0.05

# Severity lookup tuned for late-night walking risk.
SEVERITY_WEIGHTS = {
    "homicide": 1.0,
    "murder": 1.0,
    "shooting": 0.9,
    "aggravated assault": 0.85,
    "robbery": 0.8,
    "carjacking": 0.75,
    "burglary": 0.6,
    "motor vehicle theft": 0.55,
    "theft": 0.5,
    "drug": 0.45,
    "vandalism": 0.35,
    "other": 0.3,
}

CATEGORY_WEIGHTS = {
    "person": 0.85,
    "society": 0.55,
    "property": 0.5,
}


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
    """Return simplified incident data for quick lookups."""
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


def _time_decay(timestamp: datetime, reference: Optional[datetime] = None) -> float:
    reference = reference or datetime.utcnow()
    delta = reference - timestamp
    hours = max(delta.total_seconds() / 3600.0, 0.0)
    half_life_hours = 72.0
    decay = math.exp(-hours / half_life_hours)
    return max(decay, 0.05)


def _distance_decay(distance_m: float) -> float:
    if distance_m <= 0:
        return 1.0
    return max(0.05, 1.0 - (distance_m / INCIDENT_SEARCH_RADIUS_M))


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
    incident_rollup: Dict[CrimeIncident, Dict[str, object]] = {}
    total_risk = 0.0

    for lat, lon in samples:
        nearby = _find_incidents_within_radius(lat, lon, INCIDENT_SEARCH_RADIUS_M)
        for incident, distance_m in nearby:
            contribution = (
                incident.severity
                * _time_decay(incident.timestamp)
                * _distance_decay(distance_m)
            )
            total_risk += contribution
            data = incident_rollup.setdefault(
                incident,
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
    risk_score = max(risk_score, DEFAULT_RISK_FLOOR)

    return {
        "risk_score": risk_score,
        "incidents": list(incident_rollup.values()),
        "lighting": lighting,
        "samples": len(samples),
        "summary": _summarize_route_risk(incident_rollup.values(), lighting),
    }


def _store_route(route_id: str, route_payload: Dict[str, object]) -> None:
    _ROUTE_CACHE[route_id] = route_payload
    while len(_ROUTE_CACHE) > MAX_ROUTE_CACHE:
        _ROUTE_CACHE.popitem(last=False)


def get_ranked_routes(origin: str, destination: str) -> Dict[str, object]:
    """Fetch up to five walking routes and rank them by safety."""
    if googlemaps is None:
        return {
            "origin": origin,
            "destination": destination,
            "routes": [],
            "best_route_id": None,
            "error": "googlemaps library not installed; cannot fetch routes.",
        }

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {
            "origin": origin,
            "destination": destination,
            "routes": [],
            "best_route_id": None,
            "error": "GOOGLE_MAPS_API_KEY not set; configure Maps access.",
        }

    client = googlemaps.Client(key=api_key)
    try:
        routes = client.directions(
            origin,
            destination,
            mode="walking",
            alternatives=True,
        )
    except Exception as exc:  # pragma: no cover - passthrough for runtime errors
        return {
            "origin": origin,
            "destination": destination,
            "routes": [],
            "best_route_id": None,
            "error": f"Directions API failed: {exc}",
        }

    ranked: List[Dict[str, object]] = []
    for idx, route in enumerate(routes):
        overview = route.get("overview_polyline", {})
        polyline = overview.get("points")
        if not polyline:
            continue

        metrics = score_polyline(polyline)
        leg = route.get("legs", [{}])[0]
        route_id = f"route_{idx}_{abs(hash(polyline)) % 10_000_000}"

        result = {
            "route_id": route_id,
            "summary": route.get("summary") or leg.get("end_address") or "Unnamed route",
            "polyline": polyline,
            "distance_meters": leg.get("distance", {}).get("value"),
            "duration_seconds": leg.get("duration", {}).get("value"),
            "risk_score": metrics["risk_score"],
            "risk_summary": metrics["summary"],
            "incidents_nearby": metrics["incidents"],
            "lighting": metrics["lighting"],
        }
        ranked.append(result)
        _store_route(route_id, result)

    if not ranked:
        return {
            "origin": origin,
            "destination": destination,
            "routes": [],
            "best_route_id": None,
            "error": "No valid walking routes returned by Directions API.",
        }

    ranked.sort(key=lambda item: (item["risk_score"], item.get("duration_seconds", float("inf"))))
    best_route = ranked[0]
    best_score = best_route["risk_score"] or DEFAULT_RISK_FLOOR
    for route in ranked:
        route["relative_risk"] = route["risk_score"] / best_score if best_score else 1.0

    return {
        "origin": origin,
        "destination": destination,
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "routes": ranked,
        "best_route_id": best_route["route_id"],
    }


def get_route_details(route_id: str) -> Dict[str, object]:
    route = _ROUTE_CACHE.get(route_id)
    if route is None:
        available = list(_ROUTE_CACHE.keys())
        raise ValueError(f"Unknown route_id '{route_id}'. Cached routes: {available}")
    return route


crime_agent = Agent(
    name="crime_route_agent",
    model="gemini-2.0-flash",
    description="Scores late-night walking routes using nearby crime data.",
    instruction="Use get_ranked_routes to analyze candidate paths and get_route_details for follow-ups.",
    tools=[get_crime_locations, get_ranked_routes, get_route_details],
)


if __name__ == "__main__":
    preview = get_crime_locations(limit=50)
    print(json.dumps(preview, indent=2))
