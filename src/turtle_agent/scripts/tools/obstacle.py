#  Copyright (c) 2024. Jet Propulsion Laboratory. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""LangChain tools for obstacle CRUD backed by ``ObstacleStore``.

These tools intentionally do not expose HTTP. ``TurtleAgent`` injects the same
in-process store that static map loading and future collision checks use.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from langchain.agents import tool

from collision_geometry import (
    any_segment_intersects_disc,
    circle_intersects_aabb,
    segment_intersects_disc,
)
from obstacle_store import (
    AabbGeometry,
    CircleGeometry,
    Obstacle,
    ObstacleGeometry,
    ObstacleStore,
    SegmentsGeometry,
)

_STORE: Optional[ObstacleStore] = None


class ObstacleToolError(ValueError):
    """Invalid tool input or missing store configuration."""


def configure_obstacle_store(store: ObstacleStore) -> None:
    """Inject the process-local store used by obstacle tools."""
    global _STORE
    _STORE = store


def get_configured_obstacle_store() -> Optional[ObstacleStore]:
    """Return currently configured in-process ObstacleStore, if any."""
    return _STORE


def _require_store() -> ObstacleStore:
    if _STORE is None:
        raise ObstacleToolError("ObstacleStore is not configured.")
    return _STORE


def _coerce_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObstacleToolError(f"{key} must be a number.")
    return float(value)


def _parse_geometry_json(geometry_json: str) -> ObstacleGeometry:
    try:
        raw = json.loads(geometry_json)
    except json.JSONDecodeError as e:
        raise ObstacleToolError(f"Invalid geometry_json: {e}") from e
    if not isinstance(raw, dict):
        raise ObstacleToolError("geometry_json must decode to a JSON object.")
    return _parse_geometry(raw)


def _parse_geometry(raw: Dict[str, Any]) -> ObstacleGeometry:
    gtype = raw.get("type")
    if not isinstance(gtype, str):
        raise ObstacleToolError("geometry.type must be a string.")
    gtype_l = gtype.strip().lower()
    if gtype_l == "circle":
        for key in ("cx", "cy", "r"):
            if key not in raw:
                raise ObstacleToolError(f"geometry.{key} is required for circle.")
        return CircleGeometry(
            _coerce_float(raw["cx"], "cx"),
            _coerce_float(raw["cy"], "cy"),
            _coerce_float(raw["r"], "r"),
        )
    if gtype_l == "aabb":
        for key in ("min_x", "min_y", "max_x", "max_y"):
            if key not in raw:
                raise ObstacleToolError(f"geometry.{key} is required for aabb.")
        return AabbGeometry(
            _coerce_float(raw["min_x"], "min_x"),
            _coerce_float(raw["min_y"], "min_y"),
            _coerce_float(raw["max_x"], "max_x"),
            _coerce_float(raw["max_y"], "max_y"),
        )
    if gtype_l == "segments":
        segments = raw.get("segments")
        if not isinstance(segments, list):
            raise ObstacleToolError("geometry.segments must be a list.")
        parsed = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, list) or len(seg) != 2:
                raise ObstacleToolError(f"segments[{i}] must be [p1, p2].")
            p1, p2 = seg
            if not isinstance(p1, list) or len(p1) != 2:
                raise ObstacleToolError(f"segments[{i}][0] must be [x, y].")
            if not isinstance(p2, list) or len(p2) != 2:
                raise ObstacleToolError(f"segments[{i}][1] must be [x, y].")
            parsed.append(
                (
                    (_coerce_float(p1[0], "x1"), _coerce_float(p1[1], "y1")),
                    (_coerce_float(p2[0], "x2"), _coerce_float(p2[1], "y2")),
                )
            )
        return SegmentsGeometry(tuple(parsed))
    raise ObstacleToolError(
        f"unknown geometry.type {gtype!r}; expected circle, aabb, or segments."
    )


def _geometry_to_dict(geometry: ObstacleGeometry) -> Dict[str, Any]:
    if isinstance(geometry, CircleGeometry):
        return {"type": "circle", "cx": geometry.cx, "cy": geometry.cy, "r": geometry.r}
    if isinstance(geometry, AabbGeometry):
        return {
            "type": "aabb",
            "min_x": geometry.min_x,
            "min_y": geometry.min_y,
            "max_x": geometry.max_x,
            "max_y": geometry.max_y,
        }
    return {
        "type": "segments",
        "segments": [[[x1, y1], [x2, y2]] for (x1, y1), (x2, y2) in geometry.segments],
    }


def _segment_intersects_aabb(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> bool:
    """Return True when a line segment intersects an axis-aligned box."""
    dx = x2 - x1
    dy = y2 - y1
    t_min = 0.0
    t_max = 1.0

    for p, q in (
        (-dx, x1 - min_x),
        (dx, max_x - x1),
        (-dy, y1 - min_y),
        (dy, max_y - y1),
    ):
        if p == 0.0:
            if q < 0.0:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            t_min = max(t_min, ratio)
        else:
            t_max = min(t_max, ratio)
        if t_min > t_max:
            return False
    return True


def _point_on_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> bool:
    return (
        min(x1, x2) <= px <= max(x1, x2)
        and min(y1, y2) <= py <= max(y1, y2)
        and (py - y1) * (x2 - x1) == (px - x1) * (y2 - y1)
    )


def _orientation(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
) -> int:
    value = (by - ay) * (cx - bx) - (bx - ax) * (cy - by)
    if value == 0.0:
        return 0
    return 1 if value > 0.0 else 2


def _segments_intersect(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> bool:
    o1 = _orientation(ax1, ay1, ax2, ay2, bx1, by1)
    o2 = _orientation(ax1, ay1, ax2, ay2, bx2, by2)
    o3 = _orientation(bx1, by1, bx2, by2, ax1, ay1)
    o4 = _orientation(bx1, by1, bx2, by2, ax2, ay2)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _point_on_segment(bx1, by1, ax1, ay1, ax2, ay2))
        or (o2 == 0 and _point_on_segment(bx2, by2, ax1, ay1, ax2, ay2))
        or (o3 == 0 and _point_on_segment(ax1, ay1, bx1, by1, bx2, by2))
        or (o4 == 0 and _point_on_segment(ax2, ay2, bx1, by1, bx2, by2))
    )


def _segments_within_distance(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
    distance: float,
) -> bool:
    """Return True when two segments are within ``distance`` of each other."""
    return (
        _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
        or segment_intersects_disc(ax1, ay1, ax2, ay2, bx1, by1, distance)
        or segment_intersects_disc(ax1, ay1, ax2, ay2, bx2, by2, distance)
        or segment_intersects_disc(bx1, by1, bx2, by2, ax1, ay1, distance)
        or segment_intersects_disc(bx1, by1, bx2, by2, ax2, ay2, distance)
    )


def _path_hits_obstacle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    obstacle: Obstacle,
    clearance: float,
) -> Optional[str]:
    geometry = obstacle.geometry
    if isinstance(geometry, CircleGeometry):
        if segment_intersects_disc(
            x1, y1, x2, y2, geometry.cx, geometry.cy, geometry.r + clearance
        ):
            return "segment intersects circle inflated by turtle radius and safety margin"
        return None
    if isinstance(geometry, AabbGeometry):
        inflated = {
            "min_x": geometry.min_x - clearance,
            "min_y": geometry.min_y - clearance,
            "max_x": geometry.max_x + clearance,
            "max_y": geometry.max_y + clearance,
        }
        if (
            circle_intersects_aabb(
                x1,
                y1,
                clearance,
                geometry.min_x,
                geometry.min_y,
                geometry.max_x,
                geometry.max_y,
            )
            or circle_intersects_aabb(
                x2,
                y2,
                clearance,
                geometry.min_x,
                geometry.min_y,
                geometry.max_x,
                geometry.max_y,
            )
            or _segment_intersects_aabb(x1, y1, x2, y2, **inflated)
        ):
            return "segment crosses AABB inflated by turtle radius and safety margin"
        return None
    if isinstance(geometry, SegmentsGeometry):
        if (
            any_segment_intersects_disc(geometry.segments, x1, y1, clearance)
            or any_segment_intersects_disc(geometry.segments, x2, y2, clearance)
            or any(
                _segments_within_distance(
                    x1,
                    y1,
                    x2,
                    y2,
                    sx1,
                    sy1,
                    sx2,
                    sy2,
                    clearance,
                )
                for (sx1, sy1), (sx2, sy2) in geometry.segments
            )
        ):
            return "segment passes within turtle radius and safety margin of obstacle segment"
        return None
    raise TypeError(f"unsupported obstacle geometry: {type(geometry).__name__}")


@tool
def add_obstacle(
    obstacle_id: str,
    geometry_json: str,
    kind: str = "static",
    ttl_seconds: float = 0.0,
) -> str:
    """
    Add or replace an obstacle in the shared ObstacleStore.

    ``geometry_json`` uses the same shape as static map ``geometry`` entries:
    ``{"type":"circle","cx":1,"cy":2,"r":0.5}``,
    ``{"type":"aabb","min_x":1,"min_y":1,"max_x":2,"max_y":2}``, or
    ``{"type":"segments","segments":[[[0,0],[1,0]]]}``.

    ``kind`` is ``static`` for an obstacle that remains until removed, or
    ``temporary`` for an obstacle that expires after ``ttl_seconds``.
    """
    try:
        oid = obstacle_id.strip()
        if not oid:
            raise ObstacleToolError("obstacle_id must be a non-empty string.")
        kind_l = kind.strip().lower()
        if kind_l not in ("static", "temporary"):
            raise ObstacleToolError("kind must be 'static' or 'temporary'.")
        expires_at = None
        if kind_l == "temporary":
            ttl = _coerce_float(ttl_seconds, "ttl_seconds")
            if ttl <= 0:
                raise ObstacleToolError(
                    "ttl_seconds must be positive for temporary obstacles."
                )
            expires_at = time.monotonic() + ttl
        geometry = _parse_geometry_json(geometry_json)
        _require_store().upsert(
            Obstacle(
                id=oid,
                kind=kind_l,
                geometry=geometry,
                expires_at=expires_at,
            )
        )
        if kind_l == "temporary":
            return f"Added temporary obstacle '{oid}' with ttl_seconds={ttl}."
        return f"Added static obstacle '{oid}'."
    except (ObstacleToolError, ValueError) as e:
        return f"Failed to add obstacle: {e}"


@tool
def remove_obstacle(obstacle_id: str) -> str:
    """Remove an obstacle by id from the shared ObstacleStore."""
    try:
        oid = obstacle_id.strip()
        if not oid:
            raise ObstacleToolError("obstacle_id must be a non-empty string.")
        removed = _require_store().remove(oid)
        if removed:
            return f"Removed obstacle '{oid}'."
        return f"Obstacle '{oid}' was not found."
    except ObstacleToolError as e:
        return f"Failed to remove obstacle: {e}"


@tool
def list_obstacles(include_expired: bool = False) -> str:
    """List current obstacles from the shared ObstacleStore as JSON."""
    try:
        if include_expired:
            return (
                "Expired obstacles cannot be listed because ObstacleStore purges "
                "them during reads."
            )
        rows = []
        now = time.monotonic()
        for obstacle in _require_store().snapshot():
            expires_in = None
            if obstacle.expires_at is not None:
                expires_in = max(0.0, obstacle.expires_at - now)
            rows.append(
                {
                    "id": obstacle.id,
                    "kind": obstacle.kind,
                    "expires_in_seconds": expires_in,
                    "geometry": _geometry_to_dict(obstacle.geometry),
                }
            )
        return json.dumps({"obstacles": rows}, ensure_ascii=False, sort_keys=True)
    except ObstacleToolError as e:
        return f"Failed to list obstacles: {e}"


@tool
def check_path_against_obstacles(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    turtle_radius: float = 0.5,
    safety_margin: float = 0.2,
    obstacle_kinds: str = "temporary",
) -> str:
    """
    Check whether a straight turtle movement segment would collide with obstacles.

    Use this before moving through ``publish_twist_to_cmd_vel`` when the user asks
    to avoid obstacles. ``obstacle_kinds`` is a comma-separated filter such as
    ``temporary`` or ``temporary,static``.
    """
    try:
        start_x = _coerce_float(x1, "x1")
        start_y = _coerce_float(y1, "y1")
        end_x = _coerce_float(x2, "x2")
        end_y = _coerce_float(y2, "y2")
        radius = _coerce_float(turtle_radius, "turtle_radius")
        margin = _coerce_float(safety_margin, "safety_margin")
        if radius < 0:
            raise ObstacleToolError("turtle_radius must be non-negative.")
        if margin < 0:
            raise ObstacleToolError("safety_margin must be non-negative.")
        kinds = {
            item.strip().lower()
            for item in str(obstacle_kinds).split(",")
            if item.strip()
        }
        if not kinds:
            raise ObstacleToolError("obstacle_kinds must include at least one kind.")

        clearance = radius + margin
        blocked_by = []
        for obstacle in _require_store().snapshot():
            if obstacle.kind not in kinds:
                continue
            reason = _path_hits_obstacle(
                start_x,
                start_y,
                end_x,
                end_y,
                obstacle,
                clearance,
            )
            if reason:
                blocked_by.append(
                    {
                        "id": obstacle.id,
                        "kind": obstacle.kind,
                        "geometry": _geometry_to_dict(obstacle.geometry),
                        "reason": reason,
                    }
                )

        return json.dumps(
            {
                "safe": not blocked_by,
                "start": [start_x, start_y],
                "end": [end_x, end_y],
                "turtle_radius": radius,
                "safety_margin": margin,
                "clearance": clearance,
                "checked_kinds": sorted(kinds),
                "blocked_by": blocked_by,
                "recommendation": (
                    "Proceed with this segment."
                    if not blocked_by
                    else "Do not execute this segment; replan around blocked_by obstacles."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    except (ObstacleToolError, ValueError, TypeError) as e:
        return f"Failed to check path against obstacles: {e}"
