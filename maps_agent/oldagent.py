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
import random

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

    print(pts)
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

# ---------- geo helpers ----------

def haversine_m(a, b):
    R = 6371000.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def destination_point(lat, lon, bearing_deg, distance_m):
    R = 6371000.0
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat); lon1 = math.radians(lon)
    dr = distance_m / R
    lat2 = math.asin(math.sin(lat1)*math.cos(dr) + math.cos(lat1)*math.sin(dr)*math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), (math.degrees(lon2)+540)%360 - 180)

def midpoint(a, b):
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    bx = math.cos(lat2)*math.cos(lon2-lon1)
    by = math.cos(lat2)*math.sin(lon2-lon1)
    lat3 = math.atan2(math.sin(lat1)+math.sin(lat2),
                      math.sqrt((math.cos(lat1)+bx)**2 + by**2))
    lon3 = lon1 + math.atan2(by, math.cos(lat1)+bx)
    return (math.degrees(lat3), (math.degrees(lon3)+540)%360 - 180)

def ring_points(center, radii_m, bearings):
    pts = []
    for r in radii_m:
        for b in bearings:
            pts.append(destination_point(center[0], center[1], b, r))
    return pts

# ---------- simplify + distinctness ----------
def rdp(points, epsilon_m):
    if len(points) < 3: return points
    def point_seg_dist_m(p, a, b):
        if a == b: return haversine_m(p, a)
        lat0 = math.radians((a[0]+b[0])/2)
        def to_xy(pt):
            x = (pt[1]-a[1]) * math.cos(lat0) * 111320
            y = (pt[0]-a[0]) * 110540
            return (x,y)
        px,py = to_xy(p); ax,ay = to_xy(a); bx,by = to_xy(b)
        vx,vy = bx-ax, by-ay; wx,wy = px-ax, py-ay
        c1 = vx*wx + vy*wy; c2 = vx*vx + vy*vy
        t = 0 if c2 == 0 else max(0, min(1, c1/c2))
        cx,cy = ax + t*vx, ay + t*vy
        return math.hypot(px-cx, py-cy)
    def _rdp(pts):
        if len(pts) < 3: return pts
        a, b = pts[0], pts[-1]
        idx, dmax = 0, -1.0
        for i in range(1, len(pts)-1):
            d = point_seg_dist_m(pts[i], a, b)
            if d > dmax: idx, dmax = i, d
        if dmax > epsilon_m:
            left = _rdp(pts[:idx+1]); right = _rdp(pts[idx:])
            return left[:-1] + right
        return [a, b]
    return _rdp(points)

def grid_bins(points, cell_m=120):
    # coarse lat/lon grid ~120m cells
    bins = set()
    for lat, lon in points:
        y = int(round(lat * (110540/cell_m)))
        x = int(round(lon * (111320*math.cos(math.radians(lat))/cell_m)))
        bins.add((x,y))
    return bins

def jaccard(a, b):
    if not a or not b: return 0.0
    u = len(a|b)
    return len(a&b)/u if u else 0.0

# ---------- low-level route call that accepts intermediates ----------
def _compute_route_with_waypoints(origin_latlng, dest_latlng, waypoints_latlng):
    body = {
        "origin": {"location": {"latLng": {"latitude": origin_latlng[0], "longitude": origin_latlng[1]}}},
        "destination": {"location": {"latLng": {"latitude": dest_latlng[0], "longitude": dest_latlng[1]}}},
        "travelMode": "WALK",
        "polylineQuality": "OVERVIEW",
    }
    if waypoints_latlng:
        body["intermediates"] = [
            {"location": {"latLng": {"latitude": lat, "longitude": lon}}}
            for (lat, lon) in waypoints_latlng
        ]
    headers = {"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": FIELD_MASK}
    with httpx.Client(http2=True, timeout=20.0) as client:
        r = client.post(ROUTES_URL, json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data.get("routes", [None])[0]

# ---------- candidate generator: cardinal + diagonal + jitter, multiple radii ----------
def _build_candidates(origin_ll, dest_ll):
    mid = midpoint(origin_ll, dest_ll)
    # bearings: N,E,S,W + NE,SE,SW,NW (cardinal & diagonal)
    bearings = [0,45,90,135,180,225,270,315]
    # radii sets for different distance classes (short/med/long)
    short = [250, 400]
    medium = [700, 1000]
    long = [1400, 2000]

    o_ring_s = ring_points(origin_ll, short, bearings)
    d_ring_s = ring_points(dest_ll, short, bearings)
    m_ring_s = ring_points(mid, short, bearings)
    o_ring_m = ring_points(origin_ll, medium, bearings)
    d_ring_m = ring_points(dest_ll, medium, bearings)
    m_ring_m = ring_points(mid, medium, bearings)
    o_ring_l = ring_points(origin_ll, long, bearings)
    d_ring_l = ring_points(dest_ll, long, bearings)
    m_ring_l = ring_points(mid, long, bearings)

    cands = []
    # 0-waypoint baselines (very short/fast)
    cands += [ [] ]

    # 1-waypoint: near-mid (short/med/long) – creates arcs
    for p in m_ring_s + random.sample(m_ring_m, min(12, len(m_ring_m))) + random.sample(m_ring_l, min(8, len(m_ring_l))):
        cands.append([p])

    # 1-waypoint: “kick-off” near origin to enforce early different choices
    for p in random.sample(o_ring_s, min(8, len(o_ring_s))) + random.sample(o_ring_m, min(8, len(o_ring_m))):
        cands.append([p])

    # 2-waypoint: origin arc + dest arc (more variety/length)
    for p in random.sample(o_ring_m + o_ring_l, min(12, len(o_ring_m + o_ring_l))):
        for q in random.sample(d_ring_m + d_ring_l, min(12, len(d_ring_m + d_ring_l))):
            cands.append([p, q])

    # Random jitter around midpoint to fill gaps
    for _ in range(12):
        b = random.randrange(0, 360)
        r = random.choice([300, 600, 900, 1400, 1800])
        cands.append([destination_point(mid[0], mid[1], b, r)])

    random.shuffle(cands)
    return cands

# ---------- upgraded get_routes with fan-out + distinctness ----------
def get_routes(origin: str, destination: str, count: int = 10) -> Dict[str, Any]:
    if not MAPS_KEY:
        return {"status": "error", "error_message": "Missing MAPS_KEY env var"}

    # First: a baseline call using addresses (for quick success & to extract coords)
    base_body = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "WALK",
        "computeAlternativeRoutes": True,
        "polylineQuality": "OVERVIEW",
    }
    headers = {"X-Goog-Api-Key": MAPS_KEY, "X-Goog-FieldMask": FIELD_MASK}

    try:
        with httpx.Client(http2=True, timeout=20.0) as client:
            base_resp = client.post(ROUTES_URL, json=base_body, headers=headers)
        base_resp.raise_for_status()
        base_payload = base_resp.json()
        base_routes = base_payload.get("routes", [])
        if not base_routes:
            return {"status": "error", "error_message": "No route found for baseline query"}

        # Pull approximate endpoints from baseline polyline
        base_enc = (base_routes[0].get("polyline") or {}).get("encodedPolyline", "")
        base_pts = decode_polyline(base_enc)
        if len(base_pts) < 2:
            return {"status": "error", "error_message": "Polyline decode failed"}
        origin_ll = base_pts[0]
        dest_ll = base_pts[-1]

        # Build diverse waypoint candidates
        candidates = _build_candidates(origin_ll, dest_ll)

        # Distinctness bookkeeping
        kept = []
        sigs = []  # grid-bin signatures
        def route_to_sig(enc):
            pts = decode_polyline(enc)
            simp = rdp(pts, epsilon_m=90)       # simplify ~90 m
            return grid_bins(simp, cell_m=120)  # bin ~120 m cells

        # Always consider baseline itself as a candidate
        seed_routes = []
        for rt in base_routes[: min(3, count)]:
            enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
            if not enc: continue
            sig = route_to_sig(enc)
            if all(jaccard(sig, s) <= 0.55 for s in sigs):
                seed_routes.append(rt)
                sigs.append(sig)

        # Keep seed(s)
        for i, rt in enumerate(seed_routes):
            dur_s = int(rt.get("duration", "0s").rstrip("s") or 0)
            enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
            kept.append({
                "id": f"route_{len(kept)}",
                "duration_sec": dur_s,
                "distance_m": int(rt.get("distanceMeters", 0)),
                "encoded_polyline": enc,
                "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                "waypoints_used": [],
            })
            if len(kept) >= count:
                return {"status":"success","routes": kept}

        # Fan-out loop
        attempts = 0
        max_attempts = max(150, 12*count)
        for wps in candidates:
            if len(kept) >= count or attempts >= max_attempts: break
            attempts += 1
            try:
                rt = _compute_route_with_waypoints(origin_ll, dest_ll, wps)
                if not rt: continue
                enc = (rt.get("polyline") or {}).get("encodedPolyline", "")
                if not enc: continue
                sig = route_to_sig(enc)
                # distinct if max overlap <= 0.55
                if sigs and max(jaccard(sig, s) for s in sigs) > 0.55:
                    continue
                dur_s = int(rt.get("duration", "0s").rstrip("s") or 0)
                kept.append({
                    "id": f"route_{len(kept)}",
                    "duration_sec": dur_s,
                    "distance_m": int(rt.get("distanceMeters", 0)),
                    "encoded_polyline": enc,
                    "maps_link": directions_link_from_polyline(enc, travelmode="walking"),
                    "waypoints_used": wps,
                })
                sigs.append(sig)
            except httpx.HTTPError:
                continue

        return {"status": "success", "routes": kept}

    except httpx.HTTPStatusError as e:
        return {"status": "error", "error_message": f"{e.response.status_code} {e.response.text}".strip()}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}


root_agent = Agent(
        name="routing_agent",
        model="gemini-2.0-flash",
        description="Fetch up to 10 walking routes from A to B using Google Maps, and provide a Google Maps link for each.",
        instruction="Call get_routes with an origin and destination.",
        tools=[get_routes],
)

