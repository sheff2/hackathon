"""Lightweight crime-risk scoring helpers for Agent 2.

Given precomputed mock crime data, these helpers evaluate a batch of
candidate routes (encoded as Google polylines) and return scores that
another agent can consume. The scoring model mirrors ``crime_agent`` but
removes network dependencies and keeps the output focused on the
requested metrics (raw risk, normalization by distance, and incident
context).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

###############################################################################
# Data model and loading
###############################################################################


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mock_data.json"
SAMPLE_INTERVAL_M = 80.0
INCIDENT_SEARCH_RADIUS_M = 120.0
DEFAULT_RISK_FLOOR = 0.05
MIN_DISTANCE_FALLOFF_M = 25.0  # Prevents divide-by-zero explosions near hits.

# Severity lookup tuned for late-night walking risk. Mirrors crime_agent.
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


###############################################################################
# Geometry helpers
###############################################################################


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
        lat, lon = points[0]
        return [(lat, lon)]

    samples: List[Tuple[float, float]] = [points[0]]
    distance_since_sample = 0.0

    for start, end in zip(points[:-1], points[1:]):
        seg_len = haversine_distance(start[0], start[1], end[0], end[1])
        if seg_len == 0:
            continue
        remaining = seg_len
        current = start

        while distance_since_sample + remaining >= interval:
            ratio = (interval - distance_since_sample) / remaining
            lat = current[0] + (end[0] - current[0]) * ratio
            lon = current[1] + (end[1] - current[1]) * ratio
            samples.append((lat, lon))
            remaining -= (interval - distance_since_sample)
            distance_since_sample = 0.0
            current = (lat, lon)
        distance_since_sample += remaining

    if samples[-1] != points[-1]:
        samples.append(points[-1])
    return samples


def _polyline_length(points: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points[:-1], points[1:]):
        total += haversine_distance(start[0], start[1], end[0], end[1])
    return total


###############################################################################
# Scoring helpers
###############################################################################


def _time_decay(timestamp: datetime, reference: Optional[datetime] = None) -> float:
    reference = reference or datetime.utcnow()
    delta = reference - timestamp
    hours = max(delta.total_seconds() / 3600.0, 0.0)
    half_life_hours = 72.0
    return max(math.exp(-hours / half_life_hours), 0.05)


def _distance_weight(distance_m: float) -> float:
    clamped = max(distance_m, MIN_DISTANCE_FALLOFF_M)
    return 1.0 / clamped


def _find_incidents_within_radius(lat: float, lon: float, radius_m: float) -> List[Tuple[CrimeIncident, float]]:
    hits: List[Tuple[CrimeIncident, float]] = []
    for incident in load_incidents():
        distance = haversine_distance(lat, lon, incident.lat, incident.lon)
        if distance <= radius_m:
            hits.append((incident, distance))
    return hits


def _summarize_route_risk(incidents: Iterable[Dict[str, object]]) -> str:
    incidents_list = sorted(
        list(incidents),
        key=lambda item: item.get("score_contribution", 0.0),
        reverse=True,
    )
    if not incidents_list:
        return "No recent incidents detected within 120 meters of this route."

    top = incidents_list[:3]
    parts = []
    for item in top:
        offense = item.get("offense", "incident")
        distance_m = round(item.get("distance_m", 0))
        parts.append(f"{offense} about {distance_m} meters away")
    return "; ".join(parts)


def score_polyline(
    polyline: str,
    *,
    route_distance_m: Optional[float] = None,
    departure_time: Optional[datetime] = None,
) -> Dict[str, object]:
    points = decode_polyline(polyline)
    samples = _sample_points(points, SAMPLE_INTERVAL_M)
    incident_rollup: Dict[str, Dict[str, object]] = {}
    total_risk = 0.0

    for lat, lon in samples:
        nearby = _find_incidents_within_radius(lat, lon, INCIDENT_SEARCH_RADIUS_M)
        for incident, distance_m in nearby:
            contribution = (
                incident.severity
                * _time_decay(incident.timestamp, departure_time)
                * _distance_weight(distance_m)
            )
            total_risk += contribution
            incident_id = str(
                incident.raw.get("incident_code")
                or f"{incident.lat:.6f},{incident.lon:.6f}"
            )
            data = incident_rollup.setdefault(
                incident_id,
                {
                    "incident_code": incident.raw.get("incident_code"),
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

    computed_distance = route_distance_m or _polyline_length(points)
    distance_km = computed_distance / 1000.0 if computed_distance else 0.0
    risk_per_km = total_risk / distance_km if distance_km else total_risk
    risk_score = max(total_risk, DEFAULT_RISK_FLOOR)

    return {
        "risk_score": risk_score,
        "risk_per_km": risk_per_km,
        "route_distance_m": computed_distance,
        "incidents": list(incident_rollup.values()),
        "samples": len(samples),
        "summary": _summarize_route_risk(incident_rollup.values()),
    }


###############################################################################
# Public API
###############################################################################


def _coerce_routes_payload(
    routes: Sequence[Union[str, Dict[str, object], Sequence[object]]]
) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for idx, entry in enumerate(routes):
        if entry is None:
            continue
        if isinstance(entry, str):
            normalized.append({"route_id": f"route_{idx}", "polyline": entry})
            continue
        if isinstance(entry, dict):
            payload = dict(entry)
            polyline = payload.get("polyline") or payload.get("encoded_polyline")
            if polyline is None:
                continue
            payload.setdefault("polyline", polyline)
            payload.setdefault("encoded_polyline", polyline)
            primary_id = payload.get("id") or payload.get("route_id")
            payload["route_id"] = primary_id or f"route_{idx}"
            payload.setdefault("id", payload["route_id"])
            normalized.append(payload)
            continue
        if isinstance(entry, Sequence):  # tuple like (polyline, duration, distance)
            try:
                polyline = entry[0]  # type: ignore[index]
            except IndexError:
                continue
            if not isinstance(polyline, str):
                continue
            duration = entry[1] if len(entry) > 1 else None  # type: ignore[index]
            distance = entry[2] if len(entry) > 2 else None  # type: ignore[index]
            route_id = f"route_{idx}"
            normalized.append(
                {
                    "route_id": route_id,
                    "id": route_id,
                    "polyline": polyline,
                    "encoded_polyline": polyline,
                    "duration_s": duration,
                    "distance_m": distance,
                }
            )
            continue
    return normalized


def score_routes(
    routes: Sequence[Union[str, Dict[str, object], Sequence[object]]],
    *,
    departure_time: Optional[str | datetime] = None,
) -> Dict[str, object]:
    """Evaluate a batch of routes and return crime-risk metrics.

    ``routes`` is expected to be an iterable of dictionaries containing at minimum
    ``route_id`` and ``polyline``. Optional keys ``distance_m`` and ``duration_s``
    (or their *_meters/_seconds variants) are used when available to avoid
    recomputing the length.
    """

    if not routes:
        return {
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "routes": [],
            "best_route_id": None,
        }

    normalized = _coerce_routes_payload(list(routes))
    if not normalized:
        return {
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "routes": [],
            "best_route_id": None,
        }

    if isinstance(departure_time, str):
        try:
            departure_dt = datetime.fromisoformat(departure_time.replace("Z", "+00:00"))
        except ValueError:
            departure_dt = datetime.utcnow()
    else:
        departure_dt = departure_time

    scored: List[Dict[str, object]] = []
    for route in normalized:
        polyline = route.get("polyline")
        if not polyline:
            continue

        distance_m = (
            route.get("distance_m")
            or route.get("distance_meters")
            or route.get("distance")
        )
        duration_s = (
            route.get("duration_s")
            or route.get("duration_seconds")
            or route.get("duration")
            or route.get("duration_sec")
        )

        metrics = score_polyline(
            polyline,
            route_distance_m=float(distance_m) if distance_m else None,
            departure_time=departure_dt,
        )

        final_distance = metrics["route_distance_m"] or (float(distance_m) if distance_m else 0.0)
        risk_per_km = metrics["risk_per_km"]
        risk_value = risk_per_km if risk_per_km else metrics["risk_score"]

        route_id = route.get("route_id") or route.get("id")
        encoded = route.get("encoded_polyline") or polyline
        duration_value = None if duration_s is None else float(duration_s)
        scored.append(
            {
                "route_id": route_id,
                "id": route_id,
                "polyline": polyline,
                "encoded_polyline": encoded,
                "distance_m": final_distance,
                "duration_sec": duration_value,
                "risk": risk_value,
                "risk_raw": metrics["risk_score"],
                "risk_per_km": risk_per_km,
                "incidents": metrics["incidents"],
                "summary": metrics["summary"],
            }
        )

    if not scored:
        return {
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "routes": [],
            "best_route_id": None,
        }

    scored.sort(key=lambda item: item["risk_per_km"])
    best_route_id = scored[0]["route_id"]
    best_score = scored[0]["risk_per_km"] or DEFAULT_RISK_FLOOR
    for item in scored:
        item["relative_risk"] = item["risk_per_km"] / best_score if best_score else 1.0

    return {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "routes": scored,
        "best_route_id": best_route_id,
    }


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


# Compatibility hooks kept for existing imports; these simply proxy to score_routes
# using minimal arguments to avoid breaking callers that still expect the old
# function names.


def get_ranked_routes(routes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return score_routes(routes)


def get_route_details(route_id: str) -> Dict[str, object]:
    for item in load_incidents():
        if item.raw.get("incident_code") == route_id:
            return item.raw
    raise ValueError(f"No cached route for id '{route_id}'.")


__all__ = [
    "score_routes",
    "score_polyline",
    "get_crime_locations",
    "get_ranked_routes",
    "get_route_details",
]
