#!/usr/bin/env python3
import os
import math
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import logging
log = logging.getLogger(__name__)

HTTP2_OK = True
try:
    import httpx
    try:
        import h2  # noqa: F401
    except Exception:
        HTTP2_OK = False
except Exception:
    httpx = None  # type: ignore
    HTTP2_OK = False

MAPS_KEY = "AIzaSyCkB45eQTx6VTU98khj9YcPSJAazXJmuqE"

# Env caps to control fan-out / cost
MAX_ATTEMPTS_ENV = int(os.getenv("PLAN_ROUTES_MAX_ATTEMPTS", "120"))
TARGET_COUNT_ENV  = int(os.getenv("PLAN_ROUTES_TARGET_COUNT", "6"))

ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
ROUTES_FIELDMASK = "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"

PLACES_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
PLACES_FIELDMASK = "places.id,places.displayName,places.location"

RETRY_STATUSES = {429, 500, 502, 503, 504}

# Rectangle bias (Miami)
MIAMI_BOUNDS = {
    "low":  {"latitude": 25.40, "longitude": -80.60},
    "high": {"latitude": 25.95, "longitude": -80.05},
}

# Common aliases / typo fixes
ALIASES = {
    "umiami": "University of Miami, Coral Gables, FL",
    "fiu": "Florida International University, Modesto Maidique Campus, Miami, FL",
    "brickel": "Brickell, Miami, FL",
    "brickle": "Brickell, Miami, FL",
    "brickle street miami": "Brickell, Miami, FL",
    "brickell miami": "Brickell, Miami, FL",
    "downtown miami": "Downtown Miami, FL",
    "south point beach": "South Pointe Beach, Miami Beach, FL",
}

# ---------------- Safe polyline decoder -------------------
class PolylineDecodeError(Exception):
    pass

def decode_polyline_safe(enc: str) -> List[Tuple[float, float]]:
    """Robust decoder: never raises IndexError; returns partial or empty on truncation."""
    if not isinstance(enc, str) or not enc:
        return []
    pts: List[Tuple[float, float]] = []
    idx = 0
    lat = 0
    lng = 0
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
            dlat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += dlat

            shift = 0; result = 0
            while True:
                if idx >= n:
                    raise PolylineDecodeError("Truncated at longitude chunk")
                b = ord(enc[idx]) - 63; idx += 1
                result |= (b & 0x1F) << shift; shift += 5
                if b < 0x20: break
            dlng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += dlng

            pts.append((lat * 1e-5, lng * 1e-5))
    except PolylineDecodeError:
        return pts
    except Exception:
        return []
    return pts

def directions_link_from_polyline(encoded_polyline: str, travelmode: str = "walking", max_waypoints: int = 20) -> str:
    from urllib.parse import urlencode
    pts = decode_polyline_safe(encoded_polyline)
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
def _sleep_time(attempt: int, base: float = 0.7, cap: float = 8.0) -> float:
    return min(cap, base * (2 ** attempt)) * (1.0 + 0.25 * random.random())

def _http_client(timeout: float):
    return httpx.Client(http2=HTTP2_OK, timeout=timeout)

def post_with_backoff(url: str, *, json_body: Dict[str, Any], headers: Dict[str, str], timeout: float = 18.0, max_tries: int = 6):
    if httpx is None:
        raise RuntimeError("Dependency missing: install httpx and httpx[h2]")
    attempt = 0
    while True:
        try:
            with _http_client(timeout) as client:
                resp = client.post(url, json=json_body, headers=headers)
            resp.raise_for_status()
            return resp
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < max_tries - 1:
                time.sleep(_sleep_time(attempt)); attempt += 1; continue
            raise RuntimeError(f"Network error after retries: {e}") from e
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code in RETRY_STATUSES and attempt < max_tries - 1:
                time.sleep(_sleep_time(attempt)); attempt += 1; continue
            try:
                body = e.response.json()
            except Exception:
                body = e.response.text
            raise RuntimeError(f"HTTP {code}: {body}") from e

# ---------------- Geocoding (Places) ------------------
def _normalize_place(q: str) -> str:
    key = (q or "").strip().lower()
    return ALIASES.get(key, q)

def resolve_place(query: str) -> Tuple[str, Tuple[float, float]]:
    if not MAPS_KEY:
        raise RuntimeError("CONFIG: MAPS_KEY not set")
    headers = {
        "X-Goog-Api-Key": MAPS_KEY,
        "X-Goog-FieldMask": PLACES_FIELDMASK,
        "Content-Type": "application/json",
    }
    q = _normalize_place(query)
    attempts = [
        {"textQuery": q, "languageCode": "en", "regionCode": "US", "locationBias": {"rectangle": MIAMI_BOUNDS}},
        {"textQuery": f"{q}, Miami, FL, USA", "languageCode": "en", "regionCode": "US", "locationBias": {"rectangle": MIAMI_BOUNDS}},
        {"textQuery": q, "languageCode": "en", "regionCode": "US"},
    ]
    last_err: Optional[Exception] = None
    for body in attempts:
        try:
            resp = post_with_backoff(PLACES_SEARCH_URL, json_body=body, headers=headers)
            js = resp.json()
            places = js.get("places", [])
            if places:
                p = places[0]; loc = p["location"]
                return p["id"], (loc["latitude"], loc["longitude"])
        except Exception as e:
            last_err = e; continue
    raise RuntimeError(f"GEOCODE: failed for '{query}': {last_err or 'no candidates'}")

# ---------- Geo helpers ----------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371000.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) ** 2 * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))

def destination_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    R = 6371000.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat); lon1 = math.radians(lon); dr = distance_m / R
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * math.sin(dr) * math.cos(lat1), math.cos(dr) - math.sin(lat1) * math.sin(lat2))
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
        px, py = to_xy(p); ax, ay = to_xy(a); bx, by = to_xy(b)
        vx, vy = bx - ax, by - ay; wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy; c2 = vx * vx + vy * vy
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
            left = _rdp(pts[: idx + 1]); right = _rdp(pts[idx:])
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
def _build_candidates(origin_ll: Tuple[float, float], dest_ll: Tuple[float, float], bearings_step: int = 45,
                      short: Tuple[int, ...] = (300, 500), medium: Tuple[int, ...] = (800, 1200),
                      longr: Tuple[int, ...] = (1800, 2600), extra_long: Tuple[int, ...] = (3200, 4000)) -> List[List[Tuple[float, float]]]:
    mid = midpoint(origin_ll, dest_ll); bearings = list(range(0, 360, bearings_step))
    o_ring_s = ring_points(origin_ll, short, bearings); d_ring_s = ring_points(dest_ll, short, bearings); m_ring_s = ring_points(mid, short, bearings)
    o_ring_m = ring_points(origin_ll, medium, bearings); d_ring_m = ring_points(dest_ll, medium, bearings); m_ring_m = ring_points(mid, medium, bearings)
    o_ring_l = ring_points(origin_ll, longr, bearings); d_ring_l = ring_points(dest_ll, longr, bearings); m_ring_l = ring_points(mid, longr, bearings)
    m_ring_xl = ring_points(mid, extra_long, bearings)
    cands: List[List[Tuple[float, float]]] = [[]]
    mid_pool = m_ring_s + random.sample(m_ring_m, min(16, len(m_ring_m))) \
               + random.sample(m_ring_l, min(12, len(m_ring_l))) \
               + random.sample(m_ring_xl, min(12, len(m_ring_xl)))
    for p in mid_pool: cands.append([p])
    kick_pool = random.sample(o_ring_s, min(10, len(o_ring_s))) \
                + random.sample(o_ring_m, min(10, len(o_ring_m))) \
                + random.sample(o_ring_l, min(10, len(o_ring_l)))
    for p in kick_pool: cands.append([p])
    om_ol = o_ring_m + o_ring_l; dm_dl = d_ring_m + d_ring_l
    for p in random.sample(om_ol, min(18, len(om_ol))):
        for q in random.sample(dm_dl, min(18, len(dm_dl))):
            cands.append([p, q])
    for _ in range(16):
        b = random.randrange(0, 360); r = random.choice([350, 700, 1100, 1600, 2200, 3000])
        cands.append([destination_point(mid[0], mid[1], b, r)])
    random.shuffle(cands); return cands

# ---------------- Core single tool -------------------
# NOTE: no default param values (Gemini tool schema requirement)
def plan_routes(origin: str, destination: str, count: int, output: str):
    """
    Tool: robust geocoding (with Miami bias & fallbacks), baseline route,
    fan-out via waypoints, de-dup by geometry, return JSON with polylines.
    """
    # Internal defaults to preserve behavior
    if not isinstance(count, int):
        count = 10
    if not output:
        output = "json"
    count = min(count, TARGET_COUNT_ENV)

    if not MAPS_KEY:
        return {"status": "error", "error_code": "CONFIG", "error_message": "Missing MAPS_KEY"}

    if httpx is None:
        return {"status": "error", "error_code": "DEPENDENCY", "error_message": "Missing httpx. pip install httpx 'httpx[h2]'"}

    # --- Resolve to precise points (with fallbacks) ---
    origin_q = _normalize_place(origin)
    dest_q = _normalize_place(destination)

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
        _, origin_ll = resolve_place(origin_q)
        _, dest_ll = resolve_place(dest_q)
    except Exception:
        origin_guess = try_parse_latlng(origin_q)
        dest_guess = try_parse_latlng(dest_q)
        if origin_guess and dest_guess:
            origin_ll, dest_ll = origin_guess, dest_guess

    headers = {"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": ROUTES_FIELDMASK}

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
            "origin": {"address": origin_q},
            "destination": {"address": dest_q},
            "travelMode": "WALK",
            "computeAlternativeRoutes": True,
            "polylineQuality": "OVERVIEW",
        }

    # Baseline
    try:
        base_resp = post_with_backoff(ROUTES_URL, json_body=base_body, headers=headers)
        base_routes = base_resp.json().get("routes", [])
        if not base_routes:
            msg = "No baseline route found"
            if origin_ll is None or dest_ll is None:
                msg += " (Routes couldn't resolve the addresses)"
            return {"status": "error", "error_code": "NO_RESULTS", "error_message": msg}
    except Exception as e:
        return {"status": "error", "error_code": "BASELINE", "error_message": f"Baseline failed: {e}"}

    # If address path, infer coords from baseline polyline
    if origin_ll is None or dest_ll is None:
        enc0 = (base_routes[0].get("polyline") or {}).get("encodedPolyline", "")
        pts0 = decode_polyline_safe(enc0)
        if len(pts0) >= 2:
            origin_ll, dest_ll = pts0[0], pts0[-1]
        else:
            return {"status": "error", "error_code": "INFER_COORDS", "error_message": "Baseline polyline invalid/truncated"}

    # Distinctness bookkeeping
    kept: List[Dict[str, Any]] = []
    sigs: List[set] = []

    def route_to_sig(enc: str, eps: int = 90, cell: int = 120) -> set:
        p = decode_polyline_safe(enc)
        if len(p) < 2:
            return set()
        return grid_bins(rdp(p, epsilon_m=eps), cell_m=cell)

    # Seed baseline alternates
    for rt in base_routes[: min(3, count)]:
        enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
        if not enc:
            continue
        pts = decode_polyline_safe(enc)
        if len(pts) < 2:
            log.warning("Skipping malformed baseline polyline"); continue
        sig = route_to_sig(enc)
        if sigs and max((jaccard(sig, s) for s in sigs), default=0.0) > 0.55:
            continue
        dur_raw = rt.get("duration", "0s") or "0s"
        try:
            duration_sec = int(str(dur_raw).rstrip("s"))
        except Exception:
            duration_sec = int(dur_raw) if isinstance(dur_raw, (int, float)) else 0
        kept.append(
            {
                "id": f"route_{len(kept)}",
                "duration_sec": duration_sec,
                "distance_m": int(rt.get("distanceMeters", 0) or 0),
                "encoded_polyline": enc,
                "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                "waypoints_used": [],
            }
        )
        sigs.append(sig)
        if len(kept) >= count:
            break

    # Fan-out with adaptive widening
    attempts = 0
    max_attempts = min(max(120, 12 * count), MAX_ATTEMPTS_ENV)
    widen_plan = [
        {"bear_step": 45, "eps": 95,  "cell": 150, "cutoff": 0.60},
        {"bear_step": 30, "eps": 110, "cell": 170, "cutoff": 0.65},
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
                pts = decode_polyline_safe(enc)
                if len(pts) < 2:
                    log.warning("Skipping malformed candidate polyline"); continue
                sig = route_to_sig(enc, eps=phase["eps"], cell=phase["cell"])
                if sigs and max((jaccard(sig, s) for s in sigs), default=0.0) > phase["cutoff"]:
                    continue
                dur_raw = rt.get("duration", "0s") or "0s"
                try:
                    duration_sec = int(str(dur_raw).rstrip("s"))
                except Exception:
                    duration_sec = int(dur_raw) if isinstance(dur_raw, (int, float)) else 0
                kept.append(
                    {
                        "id": f"route_{len(kept)}",
                        "duration_sec": duration_sec,
                        "distance_m": int(rt.get("distanceMeters", 0) or 0),
                        "encoded_polyline": enc,
                        "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                        "waypoints_used": wps,
                    }
                )
                sigs.append(sig)
            except Exception:
                continue

    # Final permissive pass if still short
    if len(kept) < count and attempts < max_attempts:
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
                pts = decode_polyline_safe(enc)
                if len(pts) < 2:
                    continue
                sig = route_to_sig(enc, eps=130, cell=200)
                if sigs and max((jaccard(sig, s) for s in sigs), default=0.0) > 0.70:
                    continue
                dur_raw = rt.get("duration", "0s") or "0s"
                try:
                    duration_sec = int(str(dur_raw).rstrip("s"))
                except Exception:
                    duration_sec = int(dur_raw) if isinstance(dur_raw, (int, float)) else 0
                kept.append(
                    {
                        "id": f"route_{len(kept)}",
                        "duration_sec": duration_sec,
                        "distance_m": int(rt.get("distanceMeters", 0) or 0),
                        "encoded_polyline": enc,
                        "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                        "waypoints_used": wps,
                    }
                )
                sigs.append(sig)
            except Exception:
                continue

    # Sanity guard: reject obviously mis-resolved trips (e.g., 100km+ walk)
    if kept and max((r.get("distance_m") or 0) for r in kept) > 100_000:
        return {
            "status": "error",
            "error_code": "MISGEO",
            "error_message": "Destination likely mis-resolved; please include city and state (e.g., 'Brickell, Miami, FL')."
        }

    if not kept:
        return {"status": "error", "error_code": "NO_RESULTS", "error_message": f"No routes found for {origin_q} -> {dest_q}"}

    log.info("plan_routes kept=%d attempts=%d origin=%s destination=%s", len(kept), attempts, origin_q, dest_q)
    return {
        "status": "success",
        "origin": origin_q,
        "destination": dest_q,
        "routes": kept[:count],
        "schema_version": "1.0",
        "notes": {"http2": HTTP2_OK},
    }