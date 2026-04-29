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

"""Navigation-goal progress evaluation without ROS or LangChain dependencies."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from collision_geometry import (
    any_segment_intersects_disc,
    circle_intersects_aabb,
    segment_intersects_disc,
)
from obstacle_store import (
    AabbGeometry,
    CircleGeometry,
    Obstacle,
    SegmentsGeometry,
)

Point = Tuple[float, float]


@dataclass(frozen=True)
class NavigationGoal:
    turtle: str
    start: Point
    final_goal: Point
    initial_goal_distance: float
    previous_goal_distance: float
    active_obstacles: Tuple[str, ...]
    previous_obstacle_distances: Dict[str, float]
    turtle_radius: float = 0.5
    safety_margin: float = 0.2
    goal_tolerance: float = 0.2
    obstacle_kinds: Tuple[str, ...] = ("temporary",)


class NavigationGoalStore:
    """Thread-safe in-memory registry of active navigation goals per turtle."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: Dict[str, NavigationGoal] = {}

    def upsert(self, goal: NavigationGoal) -> None:
        with self._lock:
            self._items[goal.turtle] = replace(
                goal,
                previous_obstacle_distances=dict(goal.previous_obstacle_distances),
            )

    def get(self, turtle: str) -> Optional[NavigationGoal]:
        with self._lock:
            goal = self._items.get(turtle)
            if goal is None:
                return None
            return replace(
                goal,
                previous_obstacle_distances=dict(goal.previous_obstacle_distances),
            )

    def remove(self, turtle: str) -> bool:
        with self._lock:
            return self._items.pop(turtle, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> Tuple[NavigationGoal, ...]:
        with self._lock:
            return tuple(
                replace(
                    goal,
                    previous_obstacle_distances=dict(goal.previous_obstacle_distances),
                )
                for goal in self._items.values()
            )


def make_navigation_goal(
    *,
    turtle: str,
    start: Point,
    final_goal: Point,
    obstacles: Iterable[Obstacle],
    turtle_radius: float = 0.5,
    safety_margin: float = 0.2,
    goal_tolerance: float = 0.2,
    obstacle_kinds: Iterable[str] = ("temporary",),
) -> NavigationGoal:
    """Create a goal state and identify obstacles affecting the direct segment."""

    turtle_name = str(turtle).replace("/", "").strip()
    if not turtle_name:
        raise ValueError("turtle must be a non-empty string")

    sx, sy = _coerce_point(start, "start")
    gx, gy = _coerce_point(final_goal, "final_goal")
    radius = _non_negative_float(turtle_radius, "turtle_radius")
    margin = _non_negative_float(safety_margin, "safety_margin")
    tolerance = _non_negative_float(goal_tolerance, "goal_tolerance")
    kinds = _normalize_kinds(obstacle_kinds)
    clearance = radius + margin

    active = []
    previous_obstacle_distances: Dict[str, float] = {}
    for obstacle in obstacles:
        if obstacle.kind not in kinds:
            continue
        if path_hits_obstacle((sx, sy), (gx, gy), obstacle, clearance):
            active.append(obstacle.id)
            previous_obstacle_distances[obstacle.id] = obstacle_clearance(
                (sx, sy), obstacle, radius
            )

    initial_goal_distance = point_distance((sx, sy), (gx, gy))
    return NavigationGoal(
        turtle=turtle_name,
        start=(sx, sy),
        final_goal=(gx, gy),
        initial_goal_distance=initial_goal_distance,
        previous_goal_distance=initial_goal_distance,
        active_obstacles=tuple(active),
        previous_obstacle_distances=previous_obstacle_distances,
        turtle_radius=radius,
        safety_margin=margin,
        goal_tolerance=tolerance,
        obstacle_kinds=tuple(sorted(kinds)),
    )


def evaluate_navigation_goal(
    *,
    goal: NavigationGoal,
    current: Point,
    obstacles: Iterable[Obstacle],
    progress_epsilon: float = 0.01,
) -> Tuple[NavigationGoal, Dict[str, Any]]:
    """Evaluate current pose against a stored goal and return updated state/result."""

    cx, cy = _coerce_point(current, "current")
    epsilon = _non_negative_float(progress_epsilon, "progress_epsilon")
    goal_distance = point_distance((cx, cy), goal.final_goal)
    goal_distance_delta = goal_distance - goal.previous_goal_distance

    obstacle_by_id = {obstacle.id: obstacle for obstacle in obstacles}
    obstacle_distances: Dict[str, float] = {}
    obstacle_distance_deltas: Dict[str, float] = {}
    missing_obstacles = []
    for obstacle_id in goal.active_obstacles:
        obstacle = obstacle_by_id.get(obstacle_id)
        if obstacle is None:
            missing_obstacles.append(obstacle_id)
            continue
        current_distance = obstacle_clearance((cx, cy), obstacle, goal.turtle_radius)
        previous_distance = goal.previous_obstacle_distances.get(obstacle_id)
        obstacle_distances[obstacle_id] = current_distance
        if previous_distance is not None:
            obstacle_distance_deltas[obstacle_id] = current_distance - previous_distance

    assessment, recommendation, reason = _assess_progress(
        goal_distance=goal_distance,
        goal_distance_delta=goal_distance_delta,
        obstacle_active=bool(goal.active_obstacles),
        obstacle_distance_deltas=obstacle_distance_deltas,
        goal_tolerance=goal.goal_tolerance,
        epsilon=epsilon,
    )

    updated_previous_obstacle_distances = dict(goal.previous_obstacle_distances)
    updated_previous_obstacle_distances.update(obstacle_distances)
    updated_goal = replace(
        goal,
        previous_goal_distance=goal_distance,
        previous_obstacle_distances=updated_previous_obstacle_distances,
    )
    result: Dict[str, Any] = {
        "turtle": goal.turtle,
        "start": list(goal.start),
        "final_goal": list(goal.final_goal),
        "current": [cx, cy],
        "initial_goal_distance": goal.initial_goal_distance,
        "goal_distance": goal_distance,
        "goal_distance_delta": goal_distance_delta,
        "obstacle_active": bool(goal.active_obstacles),
        "active_obstacles": list(goal.active_obstacles),
        "obstacle_distances": obstacle_distances,
        "obstacle_distance_deltas": obstacle_distance_deltas,
        "missing_obstacles": missing_obstacles,
        "assessment": assessment,
        "recommendation": recommendation,
        "reason": reason,
    }
    return updated_goal, result


def point_distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def obstacle_clearance(point: Point, obstacle: Obstacle, turtle_radius: float = 0.0) -> float:
    """Return signed clearance from turtle boundary to obstacle surface."""

    px, py = _coerce_point(point, "point")
    radius = _non_negative_float(turtle_radius, "turtle_radius")
    geometry = obstacle.geometry
    if isinstance(geometry, CircleGeometry):
        return math.hypot(px - geometry.cx, py - geometry.cy) - geometry.r - radius
    if isinstance(geometry, AabbGeometry):
        return (
            _point_aabb_signed_distance(
                px,
                py,
                geometry.min_x,
                geometry.min_y,
                geometry.max_x,
                geometry.max_y,
            )
            - radius
        )
    if isinstance(geometry, SegmentsGeometry):
        if not geometry.segments:
            return math.inf
        return (
            min(
                _point_segment_distance(px, py, x1, y1, x2, y2)
                for (x1, y1), (x2, y2) in geometry.segments
            )
            - radius
        )
    raise TypeError(f"unsupported obstacle geometry: {type(geometry).__name__}")


def path_hits_obstacle(
    start: Point,
    end: Point,
    obstacle: Obstacle,
    clearance: float,
) -> bool:
    sx, sy = _coerce_point(start, "start")
    ex, ey = _coerce_point(end, "end")
    margin = _non_negative_float(clearance, "clearance")
    geometry = obstacle.geometry
    if isinstance(geometry, CircleGeometry):
        return segment_intersects_disc(
            sx, sy, ex, ey, geometry.cx, geometry.cy, geometry.r + margin
        )
    if isinstance(geometry, AabbGeometry):
        return (
            circle_intersects_aabb(
                sx,
                sy,
                margin,
                geometry.min_x,
                geometry.min_y,
                geometry.max_x,
                geometry.max_y,
            )
            or circle_intersects_aabb(
                ex,
                ey,
                margin,
                geometry.min_x,
                geometry.min_y,
                geometry.max_x,
                geometry.max_y,
            )
            or _segment_intersects_aabb(
                sx,
                sy,
                ex,
                ey,
                geometry.min_x - margin,
                geometry.min_y - margin,
                geometry.max_x + margin,
                geometry.max_y + margin,
            )
        )
    if isinstance(geometry, SegmentsGeometry):
        return (
            any_segment_intersects_disc(geometry.segments, sx, sy, margin)
            or any_segment_intersects_disc(geometry.segments, ex, ey, margin)
            or any(
                _segments_within_distance(sx, sy, ex, ey, x1, y1, x2, y2, margin)
                for (x1, y1), (x2, y2) in geometry.segments
            )
        )
    raise TypeError(f"unsupported obstacle geometry: {type(geometry).__name__}")


def goal_to_record(goal: NavigationGoal) -> Dict[str, Any]:
    return {
        "turtle": goal.turtle,
        "start": list(goal.start),
        "final_goal": list(goal.final_goal),
        "initial_goal_distance": goal.initial_goal_distance,
        "previous_goal_distance": goal.previous_goal_distance,
        "obstacle_active": bool(goal.active_obstacles),
        "active_obstacles": list(goal.active_obstacles),
        "previous_obstacle_distances": dict(goal.previous_obstacle_distances),
        "turtle_radius": goal.turtle_radius,
        "safety_margin": goal.safety_margin,
        "goal_tolerance": goal.goal_tolerance,
        "obstacle_kinds": list(goal.obstacle_kinds),
    }


def _assess_progress(
    *,
    goal_distance: float,
    goal_distance_delta: float,
    obstacle_active: bool,
    obstacle_distance_deltas: Mapping[str, float],
    goal_tolerance: float,
    epsilon: float,
) -> Tuple[str, str, str]:
    if goal_distance <= goal_tolerance:
        return (
            "goal_reached",
            "clear_goal_or_start_next_goal",
            "최종 도착지 허용 오차 안에 도달함",
        )
    if goal_distance_delta < -epsilon:
        return (
            "good_progress",
            "continue",
            "최종 도착지와의 거리가 감소함",
        )
    if abs(goal_distance_delta) <= epsilon:
        return (
            "stalled",
            "check_heading",
            "최종 도착지와의 거리 변화가 충분하지 않음",
        )
    best_obstacle_delta = (
        max(obstacle_distance_deltas.values()) if obstacle_distance_deltas else 0.0
    )
    if obstacle_active and best_obstacle_delta > epsilon:
        return (
            "good_avoidance",
            "continue_avoidance",
            "목표와의 거리는 증가했지만 회피 대상 장애물과의 거리가 증가함",
        )
    return (
        "bad_regression",
        "stop_and_replan",
        "장애물 위험이 없는 상황에서 최종 도착지와의 거리가 증가함"
        if not obstacle_active
        else "최종 도착지와의 거리가 증가하고 회피 대상 장애물과의 거리도 개선되지 않음",
    )


def _coerce_point(value: Point, key: str) -> Point:
    try:
        x, y = value
    except (TypeError, ValueError) as e:
        raise ValueError(f"{key} must be a two-item point") from e
    return _finite_float(x, f"{key}.x"), _finite_float(y, f"{key}.y")


def _finite_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{key} must be finite")
    return out


def _non_negative_float(value: Any, key: str) -> float:
    out = _finite_float(value, key)
    if out < 0.0:
        raise ValueError(f"{key} must be non-negative")
    return out


def _normalize_kinds(kinds: Iterable[str]) -> set[str]:
    normalized = {str(kind).strip().lower() for kind in kinds if str(kind).strip()}
    if not normalized:
        raise ValueError("obstacle_kinds must include at least one kind")
    return normalized


def _point_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0.0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    qx = x1 + t * dx
    qy = y1 + t * dy
    return math.hypot(px - qx, py - qy)


def _point_aabb_signed_distance(
    px: float,
    py: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> float:
    if min_x <= px <= max_x and min_y <= py <= max_y:
        return -min(px - min_x, max_x - px, py - min_y, max_y - py)
    qx = min(max(px, min_x), max_x)
    qy = min(max(py, min_y), max_y)
    return math.hypot(px - qx, py - qy)


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
    return (
        _segments_intersect(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
        or segment_intersects_disc(ax1, ay1, ax2, ay2, bx1, by1, distance)
        or segment_intersects_disc(ax1, ay1, ax2, ay2, bx2, by2, distance)
        or segment_intersects_disc(bx1, by1, bx2, by2, ax1, ay1, distance)
        or segment_intersects_disc(bx1, by1, bx2, by2, ax2, ay2, distance)
    )


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
