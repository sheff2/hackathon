from __future__ import annotations
from typing import List, Dict, Any
from urllib.parse import quote

MAP_PAGE = "http://localhost:8000/mapOutput.html"  # your existing viewer

def build_display_url(route: Dict[str, Any], base: str = MAP_PAGE) -> str:
    """
    Turn a route into a URL that mapOutput.html can render via hash params.
    Expects: encoded_polyline, duration_sec, distance_m (optional)
    """
    poly = quote(route["encoded_polyline"], safe="")
    dur_iso = f"PT{int(route['duration_sec'])}S"
    frag = f"#poly={poly}&dur={dur_iso}"
    if route.get("distance_m"):
        frag += f"&dist={int(route['distance_m'])}"
    return f"{base}{frag}"

def _pareto_prune(routes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop routes strictly dominated on (duration_sec, risk)."""
    keep = []
    for i, r in enumerate(routes):
        dominated = False
        for j, s in enumerate(routes):
            if i == j: 
                continue
            if (
                s["duration_sec"] <= r["duration_sec"]
                and s["risk"] <= r["risk"]
                and (s["duration_sec"] < r["duration_sec"] or s["risk"] < r["risk"])
            ):
                dominated = True
                break
        if not dominated:
            keep.append(r)
    return keep

def shortlist_routes(routes: List[Dict[str, Any]], limit: int = 3) -> Dict[str, Any]:
    """
    Input route schema per item:
      { id, encoded_polyline, duration_sec, distance_m (opt), risk }
    Returns:
      {
        "chosen": [route, ... up to 3],
        "bullets": ["- R1: 13.5 km, 97 min, safer", ...]
      }
    """
    if not routes:
        return {"chosen": [], "bullets": ["(no routes)"]}

    routes = _pareto_prune(routes)

    # A1: shortest among “reasonably safe” (<= median risk); fallback to absolute shortest
    risks = sorted(r["risk"] for r in routes)
    med = risks[len(risks)//2]
    pool = [r for r in routes if r["risk"] <= med] or routes
    A1 = min(pool, key=lambda r: r["duration_sec"])

    # A2: safest within +180 s of A1 and meaningfully safer (≥10% reduction)
    detour_cap = A1["duration_sec"] + 180
    near = [r for r in routes if r["duration_sec"] <= detour_cap]
    A2 = min(near, key=lambda r: r["risk"]) if near else None
    if A2 and (A2["id"] == A1["id"] or A2["risk"] >= A1["risk"] * 0.9):
        A2 = None

    # A3: one distinct remaining option if any
    remaining = [r for r in routes if r["id"] not in {A1["id"], *( {A2["id"]} if A2 else set() )}]
    A3 = remaining[0] if remaining else None

    picks = [x for x in (A1, A2, A3) if x][:limit]

    def label(risk: float) -> str:
        if risk <= med*0.8: return "safer"
        if risk <= med*1.1: return "moderate"
        return "riskier"

    def pretty_km(dist_m: int | None) -> str:
        if not dist_m:
            return "—"
        return f"{dist_m/1000:.1f} km"

    bullets = [
        f"- {r['id']}: {pretty_km(r.get('distance_m'))}, {round(r['duration_sec']/60)} min, {label(r['risk'])}"
        for r in picks
    ]
    return {"chosen": picks, "bullets": bullets}