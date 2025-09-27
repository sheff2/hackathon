#!/usr/bin/env python3
# routing_agent.py — single-tool ADK version (error-resilient)
# Requirements:
#   pip install httpx "httpx[http2]" google-adk-agents
#
# Environment:
#   export MAPS_KEY=YOUR_GOOGLE_MAPS_API_KEY   # Enable Places API + Routes API; billing on.

import os
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import httpx
from google.adk.agents import Agent

# ----------------------- Config -----------------------
MAPS_KEY = "AIzaSyCkB45eQTx6VTU98khj9YcPSJAazXJmuqE"  # keep secrets out of code
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROUTES_FIELDMASK = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELDMASK = "places.id,places.displayName,places.location"

RETRY_STATUSES = {429, 500, 502, 503, 504}

# Rectangle bias roughly around Miami, Florida
MIAMI_BOUNDS = {
    "low":  {"latitude": 25.40, "longitude": -80.60},   # south/west of Miami
    "high": {"latitude": 25.95, "longitude": -80.05},   # north/east of Miami
}


# ---------------- Polyline helpers -------------------
def decode_polyline(enc: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    idx = 0
    lat = 0
    lng = 0
    while idx < len(enc):
        shift = 0
        result = 0
        while True:
            b = ord(enc[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift = 0
        result = 0
        while True:
            b = ord(enc[idx]) - 63
            idx += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        pts.append((lat * 1e-5, lng * 1e-5))
    return pts


def directions_link_from_polyline(encoded_polyline: str, travelmode: str = "walking", max_waypoints: int = 20) -> str:
    from urllib.parse import urlencode

    print("\n\n" + encoded_polyline + "\n\n")


    pts = decode_polyline(encoded_polyline)
    if len(pts) < 2:
        return ""
    
    origin = f"{pts[0][0]},{pts[0][1]}"
    destination = f"{pts[-1][0]},{pts[-1][1]}"
    interior = pts[1:-1]
    waypoints = ""
    if interior:
        step = max(1, len(interior) // max_waypoints)
        sampled = interior[::step][:max_waypoints]
        waypoints = "|".join(f"{lat},{lng}" for (lat, lng) in sampled)
    base = "https://www.google.com/maps/dir/?api=1"
    q = {"origin": origin, "destination": destination, "travelmode": travelmode}
    if waypoints:
        q["waypoints"] = waypoints
    return f"{base}&{urlencode(q, safe='|,')}"


# --------------- Retry / Backoff ----------------------
def post_with_backoff(
    url: str,
    *,
    json_body: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float = 20.0,
    max_tries: int = 7,
    base: float = 1.0,
    cap: float = 16.0,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            with httpx.Client(http2=True, timeout=timeout) as client:
                resp = client.post(url, json=json_body, headers=headers)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in RETRY_STATUSES and attempt < max_tries - 1:
                sleep = min(cap, base * (2 ** attempt)) * (1.0 + 0.25 * random.random())
                time.sleep(sleep)
                attempt += 1
                continue
            raise
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.RemoteProtocolError):
            if attempt < max_tries - 1:
                sleep = min(cap, base * (2 ** attempt)) * (1.0 + 0.25 * random.random())
                time.sleep(sleep)
                attempt += 1
                continue
            raise


# ---------------- Geocoding (Places) ------------------
def resolve_place(query: str) -> Tuple[str, Tuple[float, float]]:
    """
    Return (place_id, (lat, lng)) for a text query.
    Biases to New Jersey and retries with small variations to avoid ambiguous hits.
    Raises on hard failure so caller can do a fallback path.
    """
    if not MAPS_KEY:
        raise RuntimeError("Missing MAPS_KEY")

    headers = {
        "X-Goog-Api-Key": MAPS_KEY,
        "X-Goog-FieldMask": PLACES_FIELDMASK,
        "Content-Type": "application/json",
    }

    attempts = [
        {"textQuery": query, "languageCode": "en", "regionCode": "US", "locationBias": {"rectangle": MIAMI_BOUNDS}},
        {"textQuery": f"{query}, Miami, FL, USA", "languageCode": "en", "regionCode": "US", "locationBias": {"rectangle": MIAMI_BOUNDS}},
        {"textQuery": query, "languageCode": "en", "regionCode": "US"},  # last try, no bias
    ]


    last_err: Optional[Exception] = None
    for body in attempts:
        try:
            resp = post_with_backoff(PLACES_SEARCH_URL, json_body=body, headers=headers)
            js = resp.json()
            places = js.get("places", [])
            if places:
                p = places[0]
                loc = p["location"]
                return p["id"], (loc["latitude"], loc["longitude"])
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Places geocode failed for '{query}': {last_err or 'no candidates'}")


# ---------- Geo + simplify + distinctness -----------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371000.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    R = 6371000.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dr = distance_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(br))
    lon2 = lon1 + math.atan2(
        math.sin(br) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180)


def midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    bx = math.cos(lat2) * math.cos(lon2 - lon1)
    by = math.cos(lat2) * math.sin(lon2 - lon1)
    lat3 = math.atan2(math.sin(lat1) + math.sin(lat2), math.sqrt((math.cos(lat1) + bx) ** 2 + by ** 2))
    lon3 = lon1 + math.atan2(by, math.cos(lat1) + bx)
    return (math.degrees(lat3), (math.degrees(lon3) + 540) % 360 - 180)


def ring_points(center: Tuple[float, float], radii_m: Tuple[int, ...], bearings: List[int]) -> List[Tuple[float, float]]:
    return [destination_point(center[0], center[1], b, r) for r in radii_m for b in bearings]


def rdp(points: List[Tuple[float, float]], epsilon_m: float) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return points

    def point_seg_dist_m(p, a, b):
        if a == b:
            return haversine_m(p, a)
        lat0 = math.radians((a[0] + b[0]) / 2)

        def to_xy(pt):
            x = (pt[1] - a[1]) * math.cos(lat0) * 111320
            y = (pt[0] - a[0]) * 110540
            return (x, y)

        px, py = to_xy(p)
        ax, ay = to_xy(a)
        bx, by = to_xy(b)
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        c2 = vx * vx + vy * vy
        t = 0 if c2 == 0 else max(0, min(1, c1 / c2))
        cx, cy = ax + t * vx, ay + t * vy
        return math.hypot(px - cx, py - cy)

    def _rdp(pts):
        if len(pts) < 3:
            return pts
        a, b = pts[0], pts[-1]
        idx, dmax = 0, -1.0
        for i in range(1, len(pts) - 1):
            d = point_seg_dist_m(pts[i], a, b)
            if d > dmax:
                idx, dmax = i, d
        if dmax > epsilon_m:
            left = _rdp(pts[: idx + 1])
            right = _rdp(pts[idx:])
            return left[:-1] + right
        return [a, b]

    return _rdp(points)


def grid_bins(points: List[Tuple[float, float]], cell_m: float = 120) -> set:
    bins = set()
    for lat, lon in points:
        y = int(round(lat * (110540 / cell_m)))
        x = int(round(lon * (111320 * math.cos(math.radians(lat)) / cell_m)))
        bins.add((x, y))
    return bins


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


# --------------- Low-level Routes call ----------------
def _compute_route_with_waypoints(origin_ll: Tuple[float, float], dest_ll: Tuple[float, float], waypoints_ll: List[Tuple[float, float]]):
    headers = {"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": ROUTES_FIELDMASK}
    body: Dict[str, Any] = {
        "origin": {"location": {"latLng": {"latitude": origin_ll[0], "longitude": origin_ll[1]}}},
        "destination": {"location": {"latLng": {"latitude": dest_ll[0], "longitude": dest_ll[1]}}},
        "travelMode": "WALK",
        "polylineQuality": "OVERVIEW",
    }
    if waypoints_ll:
        body["intermediates"] = [{"location": {"latLng": {"latitude": lat, "longitude": lon}}} for (lat, lon) in waypoints_ll]
    resp = post_with_backoff(ROUTES_URL, json_body=body, headers=headers)
    return resp.json().get("routes", [None])[0]


# ---------------- Candidate generation ---------------
def _build_candidates(
    origin_ll: Tuple[float, float],
    dest_ll: Tuple[float, float],
    bearings_step: int = 45,
    short: Tuple[int, ...] = (300, 500),
    medium: Tuple[int, ...] = (800, 1200),
    longr: Tuple[int, ...] = (1800, 2600),
    extra_long: Tuple[int, ...] = (3200, 4000),
) -> List[List[Tuple[float, float]]]:
    mid = midpoint(origin_ll, dest_ll)
    bearings = list(range(0, 360, bearings_step))

    o_ring_s = ring_points(origin_ll, short, bearings)
    d_ring_s = ring_points(dest_ll, short, bearings)
    m_ring_s = ring_points(mid, short, bearings)

    o_ring_m = ring_points(origin_ll, medium, bearings)
    d_ring_m = ring_points(dest_ll, medium, bearings)
    m_ring_m = ring_points(mid, medium, bearings)

    o_ring_l = ring_points(origin_ll, longr, bearings)
    d_ring_l = ring_points(dest_ll, longr, bearings)
    m_ring_l = ring_points(mid, longr, bearings)

    m_ring_xl = ring_points(mid, extra_long, bearings)

    cands: List[List[Tuple[float, float]]] = [[]]  # baseline

    # 1-waypoint around midpoint (short/med/long/xl)
    mid_pool = m_ring_s + random.sample(m_ring_m, min(16, len(m_ring_m))) \
               + random.sample(m_ring_l, min(12, len(m_ring_l))) \
               + random.sample(m_ring_xl, min(12, len(m_ring_xl)))
    for p in mid_pool:
        cands.append([p])

    # 1-waypoint “kick-off” near origin (short/med/long)
    kick_pool = random.sample(o_ring_s, min(10, len(o_ring_s))) \
                + random.sample(o_ring_m, min(10, len(o_ring_m))) \
                + random.sample(o_ring_l, min(10, len(o_ring_l)))
    for p in kick_pool:
        cands.append([p])

    # 2-waypoint origin arc + dest arc (include long)
    om_ol = o_ring_m + o_ring_l
    dm_dl = d_ring_m + d_ring_l
    for p in random.sample(om_ol, min(18, len(om_ol))):
        for q in random.sample(dm_dl, min(18, len(dm_dl))):
            cands.append([p, q])

    # Random jitter around midpoint to fill gaps
    for _ in range(18):
        b = random.randrange(0, 360)
        r = random.choice([350, 700, 1100, 1600, 2200, 3000])
        cands.append([destination_point(mid[0], mid[1], b, r)])

    random.shuffle(cands)
    return cands


# ---------------- Core single tool -------------------
def plan_routes(origin: str, destination: str, count: int = 10, output: str = "markdown"):
    """
    Single-call tool: robust geocoding (with Miami bias & fallbacks), baseline route,
    fan-out via waypoints, de-dup by geometry, and return Markdown links or JSON.
    """
    if not MAPS_KEY:
        return "**Error:** Missing MAPS_KEY env var" if output == "markdown" else {"status": "error", "error_message": "Missing MAPS_KEY"}

    # --- Resolve to precise points (with fallbacks) ---
    origin_ll: Optional[Tuple[float, float]] = None
    dest_ll: Optional[Tuple[float, float]] = None

    def try_parse_latlng(s: str) -> Optional[Tuple[float, float]]:
        try:
            parts = [p.strip() for p in s.split(",")]
            if len(parts) != 2:
                return None
            return (float(parts[0]), float(parts[1]))
        except Exception:
            return None

    try:
        _, origin_ll = resolve_place(origin)
        _, dest_ll = resolve_place(destination)
    except Exception:
        # Fallback A: accept "lat,lon" inputs if provided
        origin_guess = try_parse_latlng(origin)
        dest_guess = try_parse_latlng(destination)
        if origin_guess and dest_guess:
            origin_ll, dest_ll = origin_guess, dest_guess
        else:
            # Fallback B: proceed with Routes address baseline (let Routes resolve)
            origin_ll = None
            dest_ll = None

    headers = {"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": ROUTES_FIELDMASK}

    # Baseline: if we have lat/lng, use them; else let Routes resolve the address
    if origin_ll and dest_ll:
        base_body: Dict[str, Any] = {
            "origin": {"location": {"latLng": {"latitude": origin_ll[0], "longitude": origin_ll[1]}}},
            "destination": {"location": {"latLng": {"latitude": dest_ll[0], "longitude": dest_ll[1]}}},
            "travelMode": "WALK",
            "computeAlternativeRoutes": True,
            "polylineQuality": "OVERVIEW",
        }
    else:
        base_body = {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": "WALK",
            "computeAlternativeRoutes": True,
            "polylineQuality": "OVERVIEW",
        }

    try:
        base_resp = post_with_backoff(ROUTES_URL, json_body=base_body, headers=headers)
        base_routes = base_resp.json().get("routes", [])
        if not base_routes:
            msg = "No baseline route found"
            if origin_ll is None or dest_ll is None:
                msg += " (Routes couldn't resolve the addresses)"
            return "**Error:** " + msg if output == "markdown" else {"status": "error", "error_message": msg}
    except Exception as e:
        return ("**Error:** Baseline failed: " + str(e)) if output == "markdown" else {"status": "error", "error_message": f"Baseline failed: {e}"}

    # If we didn't have lat/lng (address path), extract them from the baseline polyline
    if origin_ll is None or dest_ll is None:
        enc0 = (base_routes[0].get("polyline") or {}).get("encodedPolyline", "")
        pts0 = decode_polyline(enc0) if enc0 else []
        if len(pts0) >= 2:
            origin_ll, dest_ll = pts0[0], pts0[-1]
        else:
            msg = "Could not infer coordinates from baseline polyline"
            return "**Error:** " + msg if output == "markdown" else {"status": "error", "error_message": msg}

    # Distinctness bookkeeping
    kept: List[Dict[str, Any]] = []
    sigs: List[set] = []

    def route_to_sig(enc: str, eps: int = 90, cell: int = 120) -> set:
        p = decode_polyline(enc)
        if len(p) < 2:
            return set()
        return grid_bins(rdp(p, epsilon_m=eps), cell_m=cell)

    # Seed 1–3 baseline alternates
    for rt in base_routes[: min(3, count)]:
        enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
        if not enc:
            continue
        sig = route_to_sig(enc)
        if sigs and max(jaccard(sig, s) for s in sigs) > 0.55:
            continue
        kept.append(
            {
                "id": f"route_{len(kept)}",
                "duration_sec": int((rt.get("duration", "0s") or "0s").rstrip("s") or 0),
                "distance_m": int(rt.get("distanceMeters", 0) or 0),
                "encoded_polyline": enc,
                "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                "waypoints_used": [],
            }
        )
        sigs.append(sig)
        if len(kept) >= count:
            break

    # Fan-out with adaptive widening to hit count
    attempts = 0
    max_attempts = max(300, 22 * count)
    widen_plan = [
        {"bear_step": 45, "eps": 90,  "cell": 120, "cutoff": 0.55},
        {"bear_step": 30, "eps": 100, "cell": 140, "cutoff": 0.58},
        {"bear_step": 22, "eps": 110, "cell": 160, "cutoff": 0.62},
        {"bear_step": 15, "eps": 120, "cell": 180, "cutoff": 0.65},
    ]

    for phase in widen_plan:
        if len(kept) >= count:
            break
        cands = _build_candidates(origin_ll, dest_ll, bearings_step=phase["bear_step"])
        for wps in cands:
            if len(kept) >= count or attempts >= max_attempts:
                break
            attempts += 1
            try:
                rt = _compute_route_with_waypoints(origin_ll, dest_ll, wps)
                if not rt:
                    continue
                enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
                if not enc:
                    continue
                sig = route_to_sig(enc, eps=phase["eps"], cell=phase["cell"])
                if sigs and max(jaccard(sig, s) for s in sigs) > phase["cutoff"]:
                    continue
                kept.append(
                    {
                        "id": f"route_{len(kept)}",
                        "duration_sec": int((rt.get("duration", "0s") or "0s").rstrip("s") or 0),
                        "distance_m": int(rt.get("distanceMeters", 0) or 0),
                        "encoded_polyline": enc,
                        "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                        "waypoints_used": wps,
                    }
                )
                sigs.append(sig)
            except Exception:
                # POST already retried inside; skip on persistent failure
                continue

    # Final permissive pass if still short
    if len(kept) < count:
        cands = _build_candidates(origin_ll, dest_ll, bearings_step=12)
        for wps in cands:
            if len(kept) >= count or attempts >= max_attempts:
                break
            attempts += 1
            try:
                rt = _compute_route_with_waypoints(origin_ll, dest_ll, wps)
                if not rt:
                    continue
                enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
                if not enc:
                    continue
                sig = route_to_sig(enc, eps=130, cell=200)
                if sigs and max(jaccard(sig, s) for s in sigs) > 0.70:
                    continue
                kept.append(
                    {
                        "id": f"route_{len(kept)}",
                        "duration_sec": int((rt.get("duration", "0s") or "0s").rstrip("s") or 0),
                        "distance_m": int(rt.get("distanceMeters", 0) or 0),
                        "encoded_polyline": enc,
                        "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                        "waypoints_used": wps,
                    }
                )
                sigs.append(sig)
            except Exception:
                continue

    if output == "json":
        return {"status": "success", "routes": kept}

    # Markdown (clickable links)
    if not kept:
        return f"No routes found for **{origin}** → **{destination}**."
    lines = [f"### Routes from **{origin}** → **{destination}**"]
    for r in kept:
        km = (r.get("distance_m") or 0) / 1000.0
        mins = round((r.get("duration_sec") or 0) / 60)
        meta_parts = []
        if mins:
            meta_parts.append(f"{mins} min")
        if km:
            meta_parts.append(f"{km:.1f} km")
        meta = " • " + " ".join(meta_parts) if meta_parts else ""
        link = r.get("maps_link") or ""
        rid = r.get("id") or "route"
        lines.append(f"- [{rid}]({link}){meta}")
    return "\n".join(lines)


# ---------------- ADK Agent wiring -------------------
root_agent = Agent(
    name="routing_agent",
    model="gemini-2.0-flash",
    description="Single-call tool that returns up to 10 distinct walking routes as clickable Google Maps links.",
    instruction=(
        "Call plan_routes(origin, destination, count) once when the user provides start and end points. "
        "Return the tool output exactly as-is (Markdown)."
    ),
    tools=[plan_routes],
)
