"""Utility to build mock Miami routes for testing crime scoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class RouteDefinition:
    route_id: str
    name: str
    points: Sequence[Tuple[float, float]]


def _encode_polyline(points: Iterable[Tuple[float, float]]) -> str:
    result: List[str] = []
    prev_lat = 0
    prev_lon = 0

    for lat, lon in points:
        lat_e5 = int(round(lat * 1e5))
        lon_e5 = int(round(lon * 1e5))

        d_lat = lat_e5 - prev_lat
        d_lon = lon_e5 - prev_lon
        prev_lat = lat_e5
        prev_lon = lon_e5

        for value in (d_lat, d_lon):
            shifted = value << 1
            if value < 0:
                shifted = ~shifted
            chunks: List[str] = []
            while shifted >= 0x20:
                chunks.append(chr((0x20 | (shifted & 0x1F)) + 63))
                shifted >>= 5
            chunks.append(chr(shifted + 63))
            result.extend(chunks)

    return "".join(result)


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _route_length(points: Sequence[Tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points[:-1], points[1:]):
        total += _haversine_distance(start[0], start[1], end[0], end[1])
    return total


def build_routes() -> List[dict]:
    routes = [
        RouteDefinition(
            "R1",
            "Downtown to Brickell Key",
            [
                (25.77754, -80.19001),
                (25.77067, -80.19187),
                (25.76648, -80.18630),
                (25.76403, -80.18424),
            ],
        ),
        RouteDefinition(
            "R2",
            "Wynwood to Midtown Loop",
            [
                (25.80068, -80.20098),
                (25.80380, -80.19743),
                (25.80821, -80.19236),
                (25.80166, -80.19121),
                (25.79945, -80.19677),
            ],
        ),
        RouteDefinition(
            "R3",
            "South Beach Westward",
            [
                (25.79054, -80.13005),
                (25.78784, -80.13460),
                (25.78473, -80.14051),
                (25.78051, -80.14649),
            ],
        ),
        RouteDefinition(
            "R4",
            "Coconut Grove Waterfront",
            [
                (25.73043, -80.23528),
                (25.72801, -80.22977),
                (25.72519, -80.22324),
                (25.72213, -80.21662),
            ],
        ),
        RouteDefinition(
            "R5",
            "Little Havana Corridor",
            [
                (25.77018, -80.20537),
                (25.77192, -80.20216),
                (25.77354, -80.19887),
                (25.77719, -80.19580),
            ],
        ),
        RouteDefinition(
            "R6",
            "Edgewater North",
            [
                (25.79682, -80.18940),
                (25.80253, -80.18854),
                (25.80828, -80.18755),
                (25.81240, -80.18490),
            ],
        ),
        RouteDefinition(
            "R7",
            "Design District Grid",
            [
                (25.81468, -80.19396),
                (25.81642, -80.19032),
                (25.81983, -80.18901),
                (25.82249, -80.19188),
            ],
        ),
        RouteDefinition(
            "R8",
            "Coral Gables Miracle Mile",
            [
                (25.74973, -80.26358),
                (25.74962, -80.25953),
                (25.74946, -80.25545),
                (25.74841, -80.25148),
            ],
        ),
        RouteDefinition(
            "R9",
            "Miami Beach Collins Ave",
            [
                (25.81108, -80.12241),
                (25.81532, -80.12128),
                (25.81975, -80.12104),
                (25.82407, -80.12049),
            ],
        ),
        RouteDefinition(
            "R10",
            "Doral Downtown Loop",
            [
                (25.81311, -80.35602),
                (25.81379, -80.35157),
                (25.81453, -80.34689),
                (25.81541, -80.34235),
            ],
        ),
    ]

    payload: List[dict] = []
    for route in routes:
        encoded = _encode_polyline(route.points)
        distance_m = _route_length(route.points)
        duration_sec = distance_m / 1.4  # Approximate 1.4 m/s walking pace.
        payload.append(
            {
                "id": route.route_id,
                "name": route.name,
                "encoded_polyline": encoded,
                "duration_sec": int(round(duration_sec)),
                "distance_m": round(distance_m, 2),
                "points": [list(pt) for pt in route.points],
            }
        )

    return payload


def write_payload(destination: Path) -> None:
    data = build_routes()
    destination.write_text(json.dumps({"routes": data}, indent=2))


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "data" / "mock_routes.json"
    write_payload(target)
    print(f"Wrote {target}")
